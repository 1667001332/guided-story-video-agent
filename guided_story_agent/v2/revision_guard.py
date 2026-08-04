"""Fail-closed acceptance guard for revision candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .creative_analysis import CreativeAnalysisResult
from .creative_optimizer import CreativeOptimizerResult
from .models import MoviePlan
from .revision_candidate import RevisionCandidate
from .revision_diff import RevisionChange, RevisionDiff, RevisionDiffBuilder
from .revision_request import CreativeRevisionRequest
from .validation import ValidationIssue


@dataclass(frozen=True, slots=True)
class RevisionGuardPolicy:
    max_preserve_violations: int = 0
    allow_prompt_stuffing: bool = False
    allow_provider_leakage: bool = False
    allow_core_conflict_change: bool = False
    allow_resolution_change: bool = False
    require_target_response: bool = True


@dataclass(frozen=True, slots=True)
class RevisionDecision:
    decision: str
    reason: str
    accepted_candidate_id: str | None = None
    rejected_candidate_id: str | None = None
    rollback_to_movie_plan_id: str | None = None
    diagnostics: tuple[dict[str, object], ...] = ()
    diff_summary: dict[str, object] = field(default_factory=dict)
    next_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "accepted_candidate_id": self.accepted_candidate_id,
            "rejected_candidate_id": self.rejected_candidate_id,
            "rollback_to_movie_plan_id": self.rollback_to_movie_plan_id,
            "diagnostics": [dict(item) for item in self.diagnostics],
            "diff_summary": _plain(self.diff_summary),
            "next_action": self.next_action,
        }


class RevisionGuard:
    """Evaluate a candidate without applying it or touching external state."""

    def __init__(
        self,
        policy: RevisionGuardPolicy | None = None,
        diff_builder: RevisionDiffBuilder | None = None,
    ) -> None:
        self.policy = policy or RevisionGuardPolicy()
        self.diff_builder = diff_builder or RevisionDiffBuilder()

    def evaluate(
        self,
        original: MoviePlan,
        candidate: RevisionCandidate | None,
        diff: RevisionDiff | None = None,
        validation_issues: Sequence[ValidationIssue | dict[str, Any]] = (),
        analysis_results: Sequence[CreativeAnalysisResult | dict[str, Any]] = (),
        optimizer_result: CreativeOptimizerResult | None = None,
        requests: Sequence[CreativeRevisionRequest | dict[str, Any]] = (),
    ) -> RevisionDecision:
        if not isinstance(original, MoviePlan):
            return RevisionDecision(
                "reject",
                "original 不是 MoviePlan。",
                rejected_candidate_id=candidate.candidate_id if candidate else None,
                diagnostics=(_diagnostic("invalid_original_movie_plan", "RevisionGuard 只接受 MoviePlan。"),),
                next_action="discard_candidate",
            )
        if candidate is None or candidate.revised_movie_plan is None:
            return RevisionDecision(
                "pending_director",
                "没有可评估的 revised MoviePlan，等待真实 DirectorAgent candidate。",
                diagnostics=(_diagnostic("pending_director", "candidate.revised_movie_plan 为空。"),),
                diff_summary=_diff_summary(diff),
                next_action="await_director_candidate",
            )
        if diff is None:
            diff = self.diff_builder.build_diff(original, candidate, requests)
        if candidate.source_movie_plan_id != original.plan_id:
            return self._reject(
                candidate,
                "candidate 来源 MoviePlan 与当前计划不一致。",
                "source_movie_plan_mismatch",
                diff,
            )
        summary = _diff_summary(diff)

        if candidate.status == "rollback":
            return RevisionDecision(
                "rollback",
                "candidate 明确请求回滚，保留原 MoviePlan。",
                rejected_candidate_id=candidate.candidate_id,
                rollback_to_movie_plan_id=original.plan_id,
                diagnostics=(_diagnostic("candidate_requested_rollback", "不应用 candidate。"),),
                diff_summary=summary,
                next_action="restore_original_movie_plan",
            )
        if diff.provider_leakage_detected and not self.policy.allow_provider_leakage:
            return self._reject(candidate, "检测到 Provider/API 字段泄漏。", "provider_leakage", diff)
        if diff.prompt_stuffing_detected and not self.policy.allow_prompt_stuffing:
            return self._reject(candidate, "检测到 prompt stuffing。", "prompt_stuffing", diff)
        hard_validation = tuple(item for item in validation_issues if _severity(item) == "error")
        if hard_validation:
            return self._reject(
                candidate,
                "存在 hard validation issue，candidate 不得被接受。",
                "hard_validation_issue",
                diff,
                evidence=tuple(_code(item) for item in hard_validation),
            )
        if any(_character_removed(change) for change in diff.changes):
            return self._reject(candidate, "candidate 删除了主要人物。", "main_character_removed", diff)
        if len(diff.preserve_violations) > self.policy.max_preserve_violations:
            return self._reject(candidate, "candidate 违反 preserve 约束。", "preserve_violation", diff)
        if diff.avoid_violations:
            return self._reject(candidate, "candidate 违反 avoid 约束。", "avoid_violation", diff)
        if not self.policy.allow_core_conflict_change and any(
            _is_core_conflict_change(change) for change in diff.changes
        ):
            return self._reject(candidate, "核心冲突被无授权修改。", "core_conflict_change", diff)
        if not self.policy.allow_resolution_change and any(
            _is_resolution_change(change) for change in diff.changes
        ):
            return self._reject(candidate, "结局/解决结果被无授权修改。", "resolution_change", diff)
        if self.policy.require_target_response and requests and not diff.target_responses:
            return self._reject(candidate, "candidate 没有回应任何 revision request target。", "target_not_addressed", diff)

        unrelated = [change for change in diff.changes if change not in diff.target_responses]
        if requests and diff.target_responses and unrelated:
            return RevisionDecision(
                "accept_with_warning",
                "candidate 回应了 target，但还包含未被请求覆盖的改动。",
                accepted_candidate_id=candidate.candidate_id,
                diagnostics=(_diagnostic("unrelated_changes", "存在未被 target 覆盖的额外改动。"),),
                diff_summary=summary,
                next_action="review_before_apply",
            )
        return RevisionDecision(
            "accept" if diff.target_responses or not requests else "accept_with_warning",
            "candidate 通过 RevisionGuard。" if diff.target_responses else "没有 revision request，candidate 未引入 Guard 违规。",
            accepted_candidate_id=candidate.candidate_id,
            diff_summary=summary,
            next_action="apply_only_after_explicit_confirmation",
        )

    def _reject(
        self,
        candidate: RevisionCandidate,
        reason: str,
        code: str,
        diff: RevisionDiff,
        *,
        evidence: tuple[str, ...] = (),
    ) -> RevisionDecision:
        return RevisionDecision(
            "reject",
            reason,
            rejected_candidate_id=candidate.candidate_id,
            diagnostics=(_diagnostic(code, reason, evidence=evidence),),
            diff_summary=_diff_summary(diff),
            next_action="discard_candidate",
        )


def _diagnostic(code: str, message: str, *, evidence: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "code": code,
        "message": message,
        "severity": "error" if code not in {"pending_director", "unrelated_changes"} else "warning",
        "evidence": list(evidence),
    }


def _severity(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("severity", "warning"))
    return str(getattr(value, "severity", "warning"))


def _code(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("code", "issue"))
    return str(getattr(value, "code", "issue"))


def _character_removed(change: RevisionChange) -> bool:
    return "character" in change.path.lower() and change.change_type == "removed"


def _is_core_conflict_change(change: RevisionChange) -> bool:
    path = change.path.lower()
    return "story_plan.conflict" in path or path.endswith(".conflict")


def _is_resolution_change(change: RevisionChange) -> bool:
    path = change.path.lower()
    return "story_plan.resolution" in path or path.endswith(".resolution") or path.endswith("story.ending")


def _diff_summary(diff: RevisionDiff | None) -> dict[str, object]:
    if diff is None:
        return {}
    return {
        "candidate_id": diff.candidate_id,
        "succeeded": diff.succeeded,
        "metrics": dict(diff.metrics),
        "provider_leakage_detected": diff.provider_leakage_detected,
        "prompt_stuffing_detected": diff.prompt_stuffing_detected,
        "preserve_violation_count": len(diff.preserve_violations),
        "avoid_violation_count": len(diff.avoid_violations),
        "target_response_count": len(diff.target_responses),
    }


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = ["RevisionGuardPolicy", "RevisionDecision", "RevisionGuard"]

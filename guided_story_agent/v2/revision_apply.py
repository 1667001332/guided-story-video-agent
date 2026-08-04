"""Explicit MoviePlan Apply / Rollback services for Phase 4D.3.

Guard acceptance is only permission.  The services in this module require an
explicit serializable command, revalidate the proposed plan, update Session
state, archive a snapshot, and invalidate all active downstream artifacts.
They never rebuild IR, call a Provider, or generate media.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .models import CreativeBrief, MoviePlan, as_plain_data
from .revision_candidate import RevisionCandidate
from .revision_guard import RevisionDecision
from .revision_history import (
    MoviePlanVersionRecord,
    RevisionApplyRecord,
    RevisionRollbackRecord,
)
from .revision_invalidation import invalidate_downstream_after_movie_plan_change
from .validation import validate_movie_plan
from .fingerprint import ensure_movie_plan_provenance, movie_plan_fingerprint


_FORBIDDEN_KEYS = {
    "provider", "provider_key", "provider_name", "provider_profile", "api",
    "api_key", "payload", "provider_payload", "request_payload", "video_payload",
    "api_payload", "http_payload", "endpoint", "model", "task", "task_id",
    "video_id", "submit", "poll", "download",
}
_PROMPT_STUFFING_TERMS = (
    "masterpiece", "best quality", "ultra realistic", "ultra-realistic", "8k",
)


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _ensure_safe(value: Any, path: str = "revision_command") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"revision command contains forbidden field: {path}.{key}")
            _ensure_safe(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _ensure_safe(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for term in _PROMPT_STUFFING_TERMS:
            if term in lowered:
                raise ValueError(f"revision command contains prompt stuffing at {path}: {term}")


@dataclass(frozen=True, slots=True)
class ApplyRevisionCommand:
    command_id: str
    candidate_id: str
    source_movie_plan_id: str
    apply_reason: str
    confirmed_by: str
    decision_id: str | None = None
    require_accepted_decision: bool = True
    require_revalidation: bool = True
    invalidate_downstream: bool = True
    metadata: dict[str, object] = field(default_factory=dict)
    source_movie_plan_version: int | None = None
    source_movie_plan_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in ("command_id", "candidate_id", "source_movie_plan_id", "apply_reason", "confirmed_by"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        _ensure_safe(self.apply_reason, "apply_command.apply_reason")
        _ensure_safe(self.confirmed_by, "apply_command.confirmed_by")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a JSON object")
        _ensure_safe(self.metadata, "apply_command.metadata")

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class RevisionApplyResult:
    command_id: str
    applied: bool = False
    previous_movie_plan_id: str | None = None
    new_movie_plan_id: str | None = None
    applied_candidate_id: str | None = None
    decision_id: str | None = None
    invalidated_artifacts: tuple[str, ...] = ()
    revalidation_issues: tuple[dict[str, object], ...] = ()
    stop_reason: str | None = None
    diagnostics: tuple[dict[str, object], ...] = ()
    succeeded: bool = False
    previous_movie_plan_version: int | None = None
    new_movie_plan_version: int | None = None
    new_movie_plan_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class RollbackRevisionCommand:
    command_id: str
    rollback_to_movie_plan_id: str
    rollback_reason: str
    confirmed_by: str
    require_revalidation: bool = True
    invalidate_downstream: bool = True
    metadata: dict[str, object] = field(default_factory=dict)
    rollback_to_movie_plan_version: int | None = None
    rollback_to_movie_plan_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in ("command_id", "rollback_to_movie_plan_id", "rollback_reason", "confirmed_by"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        _ensure_safe(self.rollback_reason, "rollback_command.rollback_reason")
        _ensure_safe(self.confirmed_by, "rollback_command.confirmed_by")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a JSON object")
        _ensure_safe(self.metadata, "rollback_command.metadata")

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class RevisionRollbackResult:
    command_id: str
    rolled_back: bool = False
    previous_movie_plan_id: str | None = None
    restored_movie_plan_id: str | None = None
    invalidated_artifacts: tuple[str, ...] = ()
    revalidation_issues: tuple[dict[str, object], ...] = ()
    stop_reason: str | None = None
    diagnostics: tuple[dict[str, object], ...] = ()
    succeeded: bool = False
    previous_movie_plan_version: int | None = None
    restored_movie_plan_version: int | None = None
    restored_movie_plan_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


def _validation_issues(plan: MoviePlan, brief: CreativeBrief | None) -> tuple[dict[str, object], ...]:
    if brief is None:
        return (
            {
                "code": "missing_validation_brief",
                "message": "Apply/Rollback 必须提供 CreativeBrief 才能重新验证 MoviePlan。",
                "path": "brief",
                "severity": "error",
            },
        )
    report = validate_movie_plan(plan, brief)
    return tuple(
        {
            "code": "movie_plan_validation_error",
            "message": error,
            "path": "movie_plan",
            "severity": "error",
        }
        for error in report.errors
    )


def _apply_failure(
    command: ApplyRevisionCommand,
    stop_reason: str,
    message: str,
    *,
    candidate_id: str | None = None,
    previous_id: str | None = None,
    new_id: str | None = None,
    issues: tuple[dict[str, object], ...] = (),
) -> RevisionApplyResult:
    return RevisionApplyResult(
        command_id=command.command_id,
        applied=False,
        previous_movie_plan_id=previous_id,
        new_movie_plan_id=new_id,
        applied_candidate_id=candidate_id,
        decision_id=command.decision_id,
        revalidation_issues=issues,
        stop_reason=stop_reason,
        diagnostics=({"code": stop_reason, "message": message, "severity": "error"},),
        succeeded=False,
    )


def _rollback_failure(
    command: RollbackRevisionCommand,
    stop_reason: str,
    message: str,
    *,
    previous_id: str | None = None,
    issues: tuple[dict[str, object], ...] = (),
) -> RevisionRollbackResult:
    return RevisionRollbackResult(
        command_id=command.command_id,
        previous_movie_plan_id=previous_id,
        revalidation_issues=issues,
        stop_reason=stop_reason,
        diagnostics=({"code": stop_reason, "message": message, "severity": "error"},),
        succeeded=False,
    )


class RevisionApplyService:
    """Validate and authorize one explicit candidate application."""

    def __init__(self, brief: CreativeBrief | None = None) -> None:
        self.brief = brief

    def apply_revision(
        self,
        current_movie_plan: MoviePlan,
        candidate: RevisionCandidate | None,
        decision: RevisionDecision | None,
        command: ApplyRevisionCommand,
        *,
        brief: CreativeBrief | None = None,
    ) -> RevisionApplyResult:
        if not isinstance(current_movie_plan, MoviePlan):
            return _apply_failure(command, "source_movie_plan_mismatch", "当前对象不是 MoviePlan。")
        if candidate is None:
            return _apply_failure(command, "candidate_not_found", "没有找到可应用的 RevisionCandidate。")
        if candidate.status not in {"candidate", "accepted"}:
            return _apply_failure(
                command,
                "candidate_not_accepted",
                f"candidate 状态 {candidate.status} 不允许 Apply。",
                candidate_id=candidate.candidate_id,
                previous_id=current_movie_plan.plan_id,
            )
        if decision is None or decision.decision not in {"accept", "accept_with_warning"}:
            decision_value = decision.decision if decision else "no_decision"
            return _apply_failure(
                command,
                "candidate_not_accepted",
                f"candidate 不能应用，当前 decision={decision_value}。",
                candidate_id=candidate.candidate_id,
                previous_id=current_movie_plan.plan_id,
            )
        if command.candidate_id != candidate.candidate_id:
            return _apply_failure(
                command,
                "candidate_not_found",
                "ApplyRevisionCommand.candidate_id 与候选不一致。",
                candidate_id=candidate.candidate_id,
                previous_id=current_movie_plan.plan_id,
            )
        if decision.accepted_candidate_id != candidate.candidate_id:
            return _apply_failure(
                command,
                "candidate_not_accepted",
                "RevisionDecision 没有接受当前 candidate。",
                candidate_id=candidate.candidate_id,
                previous_id=current_movie_plan.plan_id,
            )
        if command.source_movie_plan_id != current_movie_plan.plan_id:
            return _apply_failure(
                command,
                "source_movie_plan_mismatch",
                "ApplyRevisionCommand 来源 MoviePlan 与当前计划不一致。",
                candidate_id=candidate.candidate_id,
                previous_id=current_movie_plan.plan_id,
            )
        current_provenance = ensure_movie_plan_provenance(current_movie_plan)
        if (
            command.source_movie_plan_version is not None
            and command.source_movie_plan_version != current_provenance.movie_plan_version
        ):
            return _apply_failure(
                command,
                "source_movie_plan_version_mismatch",
                "ApplyRevisionCommand 来源 MoviePlan version 与当前计划不一致。",
                candidate_id=candidate.candidate_id,
                previous_id=current_movie_plan.plan_id,
            )
        if (
            command.source_movie_plan_fingerprint
            and command.source_movie_plan_fingerprint != current_provenance.movie_plan_fingerprint
        ):
            return _apply_failure(
                command,
                "source_movie_plan_fingerprint_mismatch",
                "ApplyRevisionCommand 来源 MoviePlan fingerprint 与当前计划不一致。",
                candidate_id=candidate.candidate_id,
                previous_id=current_movie_plan.plan_id,
            )
        if candidate.source_movie_plan_id != current_movie_plan.plan_id:
            return _apply_failure(
                command,
                "source_movie_plan_mismatch",
                "candidate 来源 MoviePlan 与当前计划不一致。",
                candidate_id=candidate.candidate_id,
                previous_id=current_movie_plan.plan_id,
            )
        if candidate.revised_movie_plan is None:
            return _apply_failure(
                command,
                "candidate_has_no_movie_plan",
                "candidate.revised_movie_plan 为空。",
                candidate_id=candidate.candidate_id,
                previous_id=current_movie_plan.plan_id,
            )
        issues = _validation_issues(candidate.revised_movie_plan, brief or self.brief)
        if issues:
            return _apply_failure(
                command,
                "revalidation_failed",
                "candidate 重新验证失败，拒绝 Apply。",
                candidate_id=candidate.candidate_id,
                previous_id=current_movie_plan.plan_id,
                new_id=candidate.revised_movie_plan.plan_id,
                issues=issues,
            )
        return RevisionApplyResult(
            command_id=command.command_id,
            applied=True,
            previous_movie_plan_id=current_movie_plan.plan_id,
            new_movie_plan_id=candidate.revised_movie_plan.plan_id,
            applied_candidate_id=candidate.candidate_id,
            decision_id=command.decision_id,
            stop_reason="applied",
            succeeded=True,
        )

    def apply_revision_to_session(
        self,
        session: Any,
        command: ApplyRevisionCommand,
    ) -> RevisionApplyResult:
        return apply_revision_to_session(session, command, brief=self.brief)


class RevisionRollbackService:
    """Restore a validated MoviePlan snapshot through an explicit command."""

    def __init__(self, brief: CreativeBrief | None = None) -> None:
        self.brief = brief

    def validate_restored_plan(
        self,
        restored_movie_plan: MoviePlan,
        *,
        brief: CreativeBrief | None = None,
    ) -> tuple[dict[str, object], ...]:
        return _validation_issues(restored_movie_plan, brief or self.brief)

    def rollback_revision_to_session(
        self,
        session: Any,
        command: RollbackRevisionCommand,
    ) -> RevisionRollbackResult:
        return rollback_revision_to_session(session, command, brief=self.brief)


def _candidate_from_session(session: Any, candidate_id: str) -> RevisionCandidate | None:
    from .openai_director import movie_plan_from_data

    raw = next(
        (item for item in session.revision_candidates if item.get("candidate_id") == candidate_id),
        None,
    )
    if raw is None:
        return None
    raw_plan = raw.get("revised_movie_plan")
    revised = movie_plan_from_data(raw_plan) if isinstance(raw_plan, dict) else None
    return RevisionCandidate.from_dict(raw, revised_movie_plan=revised)


def _decision_from_session(session: Any, candidate_id: str, decision_id: str | None) -> RevisionDecision | None:
    raw = None
    if decision_id:
        raw = next(
            (
                item
                for item in reversed(session.revision_decisions)
                if item.get("decision_id") == decision_id
            ),
            None,
        )
    else:
        raw = next(
            (
                item
                for item in reversed(session.revision_decisions)
                if item.get("accepted_candidate_id") == candidate_id
            ),
            None,
        )
    if raw is None:
        return None
    return RevisionDecision(
        decision=str(raw.get("decision", "")),
        reason=str(raw.get("reason", "")),
        accepted_candidate_id=raw.get("accepted_candidate_id"),
        rejected_candidate_id=raw.get("rejected_candidate_id"),
        rollback_to_movie_plan_id=raw.get("rollback_to_movie_plan_id"),
        diagnostics=tuple(raw.get("diagnostics", ())),
        diff_summary=dict(raw.get("diff_summary", {})),
        next_action=raw.get("next_action"),
    )


def _next_version(history: list[dict[str, Any]]) -> int:
    return max((int(item.get("version", 0)) for item in history), default=0) + 1


def _ensure_version_record(session: Any, movie_plan: MoviePlan, *, reason: str) -> None:
    movie_plan = ensure_movie_plan_provenance(movie_plan)
    history = session.movie_plan_version_history
    snapshot = as_plain_data(movie_plan)
    if any(item.get("movie_plan_id") == movie_plan.plan_id and item.get("snapshot") == snapshot for item in history):
        return
    history.append(
        MoviePlanVersionRecord(
            movie_plan_id=movie_plan.plan_id,
            version=movie_plan.movie_plan_version,
            source="initial",
            created_by="system",
            reason=reason,
            snapshot=deepcopy(snapshot),
            movie_plan_fingerprint=movie_plan.movie_plan_fingerprint,
            movie_plan_lineage_token=movie_plan.movie_plan_lineage_token,
        ).to_dict()
    )


def apply_revision_to_session(
    session: Any,
    command: ApplyRevisionCommand,
    *,
    brief: CreativeBrief | None = None,
) -> RevisionApplyResult:
    """Resolve Session candidate/decision and apply only after explicit command."""

    current = getattr(session, "confirmed_movie_plan", None) or getattr(session, "movie_plan", None)
    if current is None:
        result = _apply_failure(command, "source_movie_plan_mismatch", "当前 Session 没有 MoviePlan。")
        session.revision_apply_results.append(result.to_dict())
        return result
    candidate = _candidate_from_session(session, command.candidate_id)
    decision = _decision_from_session(session, command.candidate_id, command.decision_id)
    if candidate is None:
        result = _apply_failure(command, "candidate_not_found", "Session 中没有该 candidate。", previous_id=current.plan_id)
        session.revision_apply_results.append(result.to_dict())
        return result
    validation_brief = brief if brief is not None else session._v2_brief()
    result = RevisionApplyService(validation_brief).apply_revision(
        current,
        candidate,
        decision,
        command,
    )
    if not result.succeeded:
        session.revision_apply_results.append(result.to_dict())
        return result

    old_id = current.plan_id
    current = ensure_movie_plan_provenance(current)
    candidate_plan = ensure_movie_plan_provenance(candidate.revised_movie_plan)
    candidate_fingerprint = movie_plan_fingerprint(candidate_plan)
    new_version = current.movie_plan_version + (1 if candidate_fingerprint != current.movie_plan_fingerprint else 0)
    new_plan = ensure_movie_plan_provenance(candidate_plan, version=new_version)
    _ensure_version_record(session, current, reason="archive before explicit Director revision apply")
    session.movie_plan = deepcopy(new_plan)
    session.confirmed_movie_plan = replace(deepcopy(new_plan), confirmed=True)
    session.movie_plan_revisions.append(deepcopy(new_plan))
    session.current_movie_plan_id = new_plan.plan_id
    session.current_movie_plan_version = new_plan.movie_plan_version
    session.current_movie_plan_fingerprint = new_plan.movie_plan_fingerprint
    session.current_movie_plan_lineage_token = new_plan.movie_plan_lineage_token
    session.previous_movie_plan_id = old_id
    apply_record = RevisionApplyRecord(
        command_id=command.command_id,
        candidate_id=candidate.candidate_id,
        decision_id=command.decision_id,
        previous_movie_plan_id=old_id,
        new_movie_plan_id=new_plan.plan_id,
        applied=True,
        reason=command.apply_reason,
        revalidation_succeeded=True,
    )
    session.movie_plan_version_history.append(
        MoviePlanVersionRecord(
            movie_plan_id=new_plan.plan_id,
            version=new_plan.movie_plan_version,
            source="director_revision_apply",
            parent_movie_plan_id=old_id,
            source_candidate_id=candidate.candidate_id,
            source_decision_id=command.decision_id,
            created_by=command.confirmed_by,
            reason=command.apply_reason,
            snapshot=deepcopy(as_plain_data(new_plan)),
            metadata=deepcopy(command.metadata),
            movie_plan_fingerprint=new_plan.movie_plan_fingerprint,
            movie_plan_lineage_token=new_plan.movie_plan_lineage_token,
        ).to_dict()
    )
    invalidation = invalidate_downstream_after_movie_plan_change(
        session,
        "MoviePlan changed by explicit Director revision apply",
        source_movie_plan_id=old_id,
    )
    apply_record = replace(apply_record, invalidated_artifacts=invalidation.invalidated)
    session.revision_apply_history.append(apply_record.to_dict())
    result = replace(
        result,
        invalidated_artifacts=invalidation.invalidated,
        previous_movie_plan_version=current.movie_plan_version,
        new_movie_plan_version=new_plan.movie_plan_version,
        new_movie_plan_fingerprint=new_plan.movie_plan_fingerprint,
    )
    session.revision_apply_results.append(result.to_dict())
    session.stage = session.stage.__class__.MOVIE_PLAN_REVISED
    session.user_action_count += 1
    session._snapshot("movie_plan", as_plain_data(new_plan), confirmed=True)
    return result


def _find_snapshot(
    session: Any,
    target_id: str,
    current: MoviePlan,
    *,
    target_version: int | None = None,
    target_fingerprint: str | None = None,
) -> MoviePlan | None:
    from .openai_director import movie_plan_from_data

    current_snapshot = as_plain_data(current)
    records = list(session.movie_plan_version_history)
    for item in reversed(records):
        if item.get("movie_plan_id") != target_id:
            continue
        if target_version is not None and int(item.get("version", 0) or 0) != target_version:
            continue
        if target_fingerprint and str(item.get("movie_plan_fingerprint", "")) != target_fingerprint:
            continue
        snapshot = item.get("snapshot")
        if not isinstance(snapshot, dict):
            continue
        if target_id == current.plan_id and snapshot == current_snapshot:
            continue
        return movie_plan_from_data(snapshot)
    return None


def rollback_revision_to_session(
    session: Any,
    command: RollbackRevisionCommand,
    *,
    brief: CreativeBrief | None = None,
) -> RevisionRollbackResult:
    current = getattr(session, "confirmed_movie_plan", None) or getattr(session, "movie_plan", None)
    if current is None:
        result = _rollback_failure(command, "target_not_found", "当前 Session 没有 MoviePlan。")
        session.revision_rollback_results.append(result.to_dict())
        return result
    restored = _find_snapshot(
        session,
        command.rollback_to_movie_plan_id,
        current,
        target_version=command.rollback_to_movie_plan_version,
        target_fingerprint=command.rollback_to_movie_plan_fingerprint,
    )
    if restored is None:
        result = _rollback_failure(
            command,
            "target_not_found",
            "MoviePlan version history 中没有指定目标。",
            previous_id=current.plan_id,
        )
        session.revision_rollback_results.append(result.to_dict())
        return result
    validation_brief = brief if brief is not None else session._v2_brief()
    issues = RevisionRollbackService(validation_brief).validate_restored_plan(restored)
    if issues:
        result = _rollback_failure(
            command,
            "revalidation_failed",
            "恢复的 MoviePlan 重新验证失败。",
            previous_id=current.plan_id,
            issues=issues,
        )
        session.revision_rollback_results.append(result.to_dict())
        return result

    old_id = current.plan_id
    current = ensure_movie_plan_provenance(current)
    restored = ensure_movie_plan_provenance(restored)
    session.movie_plan = deepcopy(restored)
    session.confirmed_movie_plan = replace(deepcopy(restored), confirmed=True)
    session.movie_plan_revisions.append(deepcopy(restored))
    session.current_movie_plan_id = restored.plan_id
    session.current_movie_plan_version = restored.movie_plan_version
    session.current_movie_plan_fingerprint = restored.movie_plan_fingerprint
    session.current_movie_plan_lineage_token = restored.movie_plan_lineage_token
    session.previous_movie_plan_id = old_id
    version_record = MoviePlanVersionRecord(
        movie_plan_id=restored.plan_id,
        version=restored.movie_plan_version,
        source="rollback",
        parent_movie_plan_id=old_id,
        created_by=command.confirmed_by,
        reason=command.rollback_reason,
        snapshot=deepcopy(as_plain_data(restored)),
        metadata=deepcopy(command.metadata),
        movie_plan_fingerprint=restored.movie_plan_fingerprint,
        movie_plan_lineage_token=restored.movie_plan_lineage_token,
    )
    session.movie_plan_version_history.append(version_record.to_dict())
    invalidation = invalidate_downstream_after_movie_plan_change(
        session,
        "MoviePlan restored by explicit rollback",
        source_movie_plan_id=old_id,
    )
    record = RevisionRollbackRecord(
        command_id=command.command_id,
        rollback_to_movie_plan_id=command.rollback_to_movie_plan_id,
        previous_movie_plan_id=old_id,
        restored_movie_plan_id=restored.plan_id,
        rolled_back=True,
        reason=command.rollback_reason,
        invalidated_artifacts=invalidation.invalidated,
        revalidation_succeeded=True,
    )
    session.revision_rollback_history.append(record.to_dict())
    result = RevisionRollbackResult(
        command_id=command.command_id,
        rolled_back=True,
        previous_movie_plan_id=old_id,
        restored_movie_plan_id=restored.plan_id,
        invalidated_artifacts=invalidation.invalidated,
        stop_reason="rolled_back",
        succeeded=True,
        previous_movie_plan_version=current.movie_plan_version,
        restored_movie_plan_version=restored.movie_plan_version,
        restored_movie_plan_fingerprint=restored.movie_plan_fingerprint,
    )
    session.revision_rollback_results.append(result.to_dict())
    session.stage = session.stage.__class__.MOVIE_PLAN_ROLLED_BACK
    session.user_action_count += 1
    session._snapshot("movie_plan", as_plain_data(restored), confirmed=True)
    return result


__all__ = [
    "ApplyRevisionCommand",
    "RevisionApplyResult",
    "RevisionApplyService",
    "RollbackRevisionCommand",
    "RevisionRollbackResult",
    "RevisionRollbackService",
    "apply_revision_to_session",
    "rollback_revision_to_session",
]

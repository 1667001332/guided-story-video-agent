"""Director-facing revision requests for Phase 4C.

Requests are policy-filtered recommendations.  They do not contain a revised
plan and are never applied by Python.  A future DirectorAgent adapter may
consume them and return a new MoviePlan through the normal validation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .creative_analysis import CreativeAnalysisDiagnostic
from .creative_optimizer import CreativeOptimizerResult, OptimizationSuggestion
from .models import MoviePlan


@dataclass(frozen=True, slots=True)
class CreativeRevisionRequest:
    request_id: str
    severity: str
    target: str
    instruction: str
    preserve: tuple[str, ...] = ()
    avoid: tuple[str, ...] = ()
    rationale: str = ""
    source_suggestion_codes: tuple[str, ...] = ()
    source_diagnostic_codes: tuple[str, ...] = ()
    requires_director: bool = True
    auto_apply_allowed: bool = False

    def __post_init__(self) -> None:
        if self.severity not in {"hard", "warning"}:
            raise ValueError("CreativeRevisionRequest.severity must be hard or warning")
        if self.auto_apply_allowed:
            raise ValueError("CreativeRevisionRequest cannot be auto-applied in Phase 4C")
        _ensure_safe(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "severity": self.severity,
            "target": self.target,
            "instruction": self.instruction,
            "preserve": list(self.preserve),
            "avoid": list(self.avoid),
            "rationale": self.rationale,
            "source_suggestion_codes": list(self.source_suggestion_codes),
            "source_diagnostic_codes": list(self.source_diagnostic_codes),
            "requires_director": bool(self.requires_director),
            "auto_apply_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class RevisionRequestBuilderResult:
    source_movie_plan_id: str
    requests: tuple[CreativeRevisionRequest, ...] = ()
    deferred_suggestions: tuple[OptimizationSuggestion, ...] = ()
    diagnostics: tuple[CreativeAnalysisDiagnostic, ...] = ()
    stop_reason: str = "no_revision_required"
    succeeded: bool = True

    def __post_init__(self) -> None:
        _ensure_safe(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_movie_plan_id": self.source_movie_plan_id,
            "requests": [item.to_dict() for item in self.requests],
            "deferred_suggestions": [item.to_dict() for item in self.deferred_suggestions],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "stop_reason": self.stop_reason,
            "succeeded": bool(self.succeeded),
        }


class RevisionPolicy:
    """Classify suggestions without changing their content."""

    hard_codes = frozenset(
        {
            "clarify_core_conflict",
            "clarify_protagonist_goal",
            "resolve_story_plan_legacy_conflict",
            "align_character_projection",
            "connect_climax_to_existing_conflict",
        }
    )
    warning_codes = frozenset(
        {
            "strengthen_climax_emotional_peak",
            "escalate_conflict_before_climax",
            "show_cost_in_resolution",
            "add_setup_before_reveal",
            "add_payoff_for_existing_setup",
            "strengthen_existing_character_arc",
        }
    )

    def classify(self, suggestion: OptimizationSuggestion) -> str:
        if suggestion.code in self.hard_codes or suggestion.severity == "hard":
            return "hard"
        if suggestion.code in self.warning_codes or suggestion.severity == "warning":
            return "warning"
        return "deferred"


_GROUPS = {
    "strengthen_climax_emotional_peak": "climax_conflict",
    "escalate_conflict_before_climax": "climax_conflict",
    "show_cost_in_resolution": "climax_conflict",
    "connect_climax_to_existing_conflict": "climax_conflict",
    "clarify_core_conflict": "climax_conflict",
    "add_setup_before_reveal": "audience_knowledge",
    "add_payoff_for_existing_setup": "audience_knowledge",
    "adjust_reveal_timing": "audience_knowledge",
    "adjust_withholding_timing": "audience_knowledge",
    "clarify_viewer_state_transition": "audience_knowledge",
    "clarify_audience_knowledge": "audience_knowledge",
    "clarify_protagonist_goal": "character_arc",
    "strengthen_existing_character_arc": "character_arc",
    "add_irreversible_choice": "character_arc",
    "maintain_character_presence_after_setup": "character_arc",
    "resolve_character_goal": "character_arc",
    "resolve_story_plan_legacy_conflict": "plan_layer_consistency",
    "align_character_projection": "plan_layer_consistency",
    "align_story_beat_projection": "plan_layer_consistency",
    "align_director_plan_with_legacy_visual_style": "plan_layer_consistency",
}


def _severity(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("severity", "warning"))
    return str(getattr(value, "severity", "warning"))


def _code(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("code", "issue"))
    return str(getattr(value, "code", "issue"))


def _ensure_safe(value: Any, path: str = "creative_revision_request") -> None:
    forbidden = {
        "provider", "provider_key", "provider_name", "provider_profile", "api",
        "api_key", "payload", "provider_payload", "request_payload", "video_payload",
        "api_payload", "http_payload", "endpoint", "model", "task", "task_id",
        "video_id", "submit", "poll", "download",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                raise ValueError(f"Revision request contains forbidden field: {path}.{key}")
            _ensure_safe(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _ensure_safe(child, f"{path}[{index}]")
    elif isinstance(value, str):
        for term in ("masterpiece", "best quality", "ultra realistic", "ultra-realistic", "8k"):
            if term in value.lower():
                raise ValueError(f"Revision request contains prompt stuffing at {path}: {term}")


class RevisionRequestBuilder:
    """Convert suggestions into grouped, director-facing requests."""

    def __init__(self, policy: RevisionPolicy | None = None) -> None:
        self.policy = policy or RevisionPolicy()

    def build(
        self,
        movie_plan: MoviePlan,
        optimizer_result: CreativeOptimizerResult,
        validation_issues: Iterable[Any] = (),
        analysis_diagnostics: Iterable[CreativeAnalysisDiagnostic | dict[str, Any]] = (),
    ) -> RevisionRequestBuilderResult:
        if not isinstance(movie_plan, MoviePlan):
            diagnostic = CreativeAnalysisDiagnostic(
                "invalid_movie_plan", "RevisionRequestBuilder 只接受 MoviePlan。", "movie_plan", "error"
            )
            return RevisionRequestBuilderResult("", diagnostics=(diagnostic,), stop_reason="hard_validation_error", succeeded=False)
        issues = tuple(validation_issues)
        hard_validation = tuple(item for item in issues if _severity(item) == "error")
        if hard_validation:
            diagnostic = CreativeAnalysisDiagnostic(
                "validation_error_requires_revision",
                "存在 Validator hard error，必须先由 DirectorAgent 修订 MoviePlan。",
                "validation",
                "error",
                tuple(_code(item) for item in hard_validation),
            )
            return RevisionRequestBuilderResult(
                movie_plan.plan_id,
                diagnostics=(diagnostic,),
                stop_reason="hard_validation_error",
                succeeded=False,
            )
        suggestions = tuple(optimizer_result.suggestions)
        actionable = tuple(item for item in suggestions if self.policy.classify(item) in {"hard", "warning"})
        deferred = tuple(item for item in suggestions if self.policy.classify(item) == "deferred")
        grouped: dict[str, list[OptimizationSuggestion]] = {}
        for suggestion in actionable:
            grouped.setdefault(_GROUPS.get(suggestion.code, suggestion.code), []).append(suggestion)
        requests = tuple(
            self._request(movie_plan, target, items, index)
            for index, (target, items) in enumerate(grouped.items(), 1)
        )
        diagnostics = tuple(
            item
            for item in analysis_diagnostics
            if _severity(item) == "error"
        )
        if requests:
            stop_reason = "pending_director_revision"
        elif deferred:
            stop_reason = "policy_deferred"
        else:
            stop_reason = "no_revision_required"
        return RevisionRequestBuilderResult(
            movie_plan.plan_id,
            requests,
            deferred,
            diagnostics,
            stop_reason,
            True,
        )

    def _request(
        self,
        movie_plan: MoviePlan,
        target: str,
        suggestions: list[OptimizationSuggestion],
        index: int,
    ) -> CreativeRevisionRequest:
        codes = tuple(item.code for item in suggestions)
        original_codes = tuple(
            code for item in suggestions for code in item.source_diagnostic_codes
        )
        severity = "hard" if any(self.policy.classify(item) == "hard" for item in suggestions) else "warning"
        if target == "climax_conflict":
            instruction = "请导演强化高潮段落，使核心冲突在高潮处升级，并让解决结果体现可见代价。"
        elif target == "audience_knowledge":
            instruction = "请导演重新检查信息揭示与观众理解节奏，补足已有铺垫和回收关系。"
        elif target == "character_arc":
            instruction = "请导演明确现有角色目标与弧线变化，保持角色在既有故事中的因果连续性。"
        elif target == "plan_layer_consistency":
            instruction = "请导演解决 StoryPlan、DirectorPlan 与既有兼容投影之间的不一致。"
        else:
            instruction = "请导演根据创意分析诊断重新审视该创作决策。"
        rationale = "；".join(item.reason or item.message for item in suggestions)
        return CreativeRevisionRequest(
            request_id=f"revision-{movie_plan.plan_id}-{index}",
            severity=severity,
            target=target,
            instruction=instruction,
            preserve=(
                "保留现有主要人物",
                "保留现有核心冲突方向",
                "保留既有 StoryPlan story beats",
                "不改变既有世界观与结局基调",
            ),
            avoid=(
                "不新增主要人物",
                "不改变世界观规则",
                "不生成 Provider payload",
                "不使用 prompt stuffing",
            ),
            rationale=rationale,
            source_suggestion_codes=codes,
            source_diagnostic_codes=original_codes,
            requires_director=True,
            auto_apply_allowed=False,
        )


__all__ = [
    "CreativeRevisionRequest",
    "RevisionRequestBuilderResult",
    "RevisionPolicy",
    "RevisionRequestBuilder",
]

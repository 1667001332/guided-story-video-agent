"""Deterministic creative optimization over read-only analysis results.

The creative optimizer is deliberately not a compiler pass and not a second
director.  It translates analysis diagnostics into traceable suggestions and
future transformation candidates.  It never mutates a MoviePlan, StoryPlan,
DirectorPlan, FilmIR, MovieIR, prompt, or Provider request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .creative_analysis import CreativeAnalysisDiagnostic, CreativeAnalysisResult
from .film_ir import FilmIR
from .models import MoviePlan


_FORBIDDEN_KEYS = {
    "provider", "provider_key", "provider_name", "provider_profile", "api",
    "api_key", "payload", "provider_payload", "request_payload", "video_payload",
    "api_payload", "http_payload", "endpoint", "model", "task", "task_id",
    "video_id", "submit", "poll", "download",
}
_FORBIDDEN_TERMS = (
    "masterpiece", "best quality", "ultra realistic", "ultra-realistic",
    "cinematic masterpiece", "8k", "award winning", "photorealistic masterpiece",
)


def _ensure_safe(value: Any, path: str = "creative_optimizer") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in _FORBIDDEN_KEYS:
                raise ValueError(f"Creative optimizer contains forbidden field: {path}.{key}")
            _ensure_safe(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _ensure_safe(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for term in _FORBIDDEN_TERMS:
            if term in lowered:
                raise ValueError(f"Creative optimizer contains prompt stuffing at {path}: {term}")


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class OptimizationSuggestion:
    """A director-facing suggestion; it is never an automatic rewrite."""

    code: str
    message: str
    severity: str
    path: str = ""
    reason: str = ""
    evidence: tuple[str, ...] = ()
    suggested_direction: str = ""
    preserve_constraints: tuple[str, ...] = ()
    risk: str = "medium"
    confidence: float = 1.0
    source_analysis_types: tuple[str, ...] = ()
    source_diagnostic_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in {"hard", "warning", "deferred"}:
            raise ValueError("OptimizationSuggestion.severity must be hard, warning, or deferred")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("OptimizationSuggestion.confidence must be between 0 and 1")
        _ensure_safe(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "path": self.path,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "suggested_direction": self.suggested_direction,
            "preserve_constraints": list(self.preserve_constraints),
            "risk": self.risk,
            "confidence": float(self.confidence),
            "source_analysis_types": list(self.source_analysis_types),
            "source_diagnostic_codes": list(self.source_diagnostic_codes),
        }


@dataclass(frozen=True, slots=True)
class TransformationCandidate:
    """A future transformation proposal, intentionally non-executable now."""

    code: str
    message: str
    severity: str
    path: str = ""
    before: Any = None
    after: Any = None
    reason: str = ""
    risk: str = "medium"
    confidence: float = 1.0
    executable: bool = False

    def __post_init__(self) -> None:
        if self.executable:
            raise ValueError("Creative transformation candidates are not executable in Phase 4C")
        _ensure_safe(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "path": self.path,
            "before": _plain(self.before),
            "after": _plain(self.after),
            "reason": self.reason,
            "risk": self.risk,
            "confidence": float(self.confidence),
            "executable": False,
        }


@dataclass(frozen=True, slots=True)
class CreativeOptimizerResult:
    source_movie_plan_id: str
    source_film_ir_id: str | None
    suggestions: tuple[OptimizationSuggestion, ...] = ()
    transformation_candidates: tuple[TransformationCandidate, ...] = ()
    diagnostics: tuple[CreativeAnalysisDiagnostic, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    succeeded: bool = True

    def __post_init__(self) -> None:
        _ensure_safe(self.to_dict())

    @property
    def errors(self) -> tuple[CreativeAnalysisDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_movie_plan_id": self.source_movie_plan_id,
            "source_film_ir_id": self.source_film_ir_id,
            "suggestions": [item.to_dict() for item in self.suggestions],
            "transformation_candidates": [item.to_dict() for item in self.transformation_candidates],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "metrics": {str(key): float(value) for key, value in self.metrics.items()},
            "succeeded": bool(self.succeeded),
        }


_MAPPINGS: dict[str, dict[str, str]] = {
    "emotion_flow": {
        "climax_not_emotional_peak": "strengthen_climax_emotional_peak",
        "emotion_curve_missing": "clarify_emotion_curve",
        "ending_tone_missing": "clarify_ending_tone",
        "emotional_intention_unlinked": "link_emotional_intention_to_story_beat",
    },
    "audience_knowledge": {
        "reveal_without_setup": "add_setup_before_reveal",
        "setup_without_payoff": "add_payoff_for_existing_setup",
        "audience_knows_too_late": "adjust_reveal_timing",
        "audience_knows_too_early": "adjust_withholding_timing",
        "viewer_state_missing": "clarify_viewer_state_transition",
        "audience_knowledge_missing": "clarify_audience_knowledge",
    },
    "conflict_progression": {
        "conflict_missing": "clarify_core_conflict",
        "stakes_missing": "make_stakes_visible",
        "conflict_not_escalating": "escalate_conflict_before_climax",
        "climax_resolves_no_conflict": "connect_climax_to_existing_conflict",
        "resolution_without_cost": "show_cost_in_resolution",
    },
    "character_arc": {
        "protagonist_goal_missing": "clarify_protagonist_goal",
        "character_arc_flat": "strengthen_existing_character_arc",
        "irreversible_choice_missing": "add_irreversible_choice",
        "character_disappears_after_setup": "maintain_character_presence_after_setup",
        "character_goal_not_resolved": "resolve_character_goal",
    },
    "plan_layer_consistency": {
        "story_plan_legacy_conflict": "resolve_story_plan_legacy_conflict",
        "character_projection_mismatch": "align_character_projection",
        "film_beat_projection_mismatch": "align_story_beat_projection",
        "director_plan_visual_style_mismatch": "align_director_plan_with_legacy_visual_style",
    },
}

_HARD_CODES = {
    "conflict_missing", "protagonist_goal_missing", "story_plan_legacy_conflict",
    "character_projection_mismatch", "climax_resolves_no_conflict",
}
_WARNING_CODES = {
    "climax_not_emotional_peak", "conflict_not_escalating", "resolution_without_cost",
    "reveal_without_setup", "setup_without_payoff", "character_arc_flat",
}
_DEFERRED_CODES = {
    "ending_tone_missing", "emotional_intention_unlinked", "audience_knowledge_missing",
    "viewer_state_missing", "audience_knows_too_late", "audience_knows_too_early",
    "stakes_missing", "irreversible_choice_missing", "character_disappears_after_setup",
    "character_goal_not_resolved", "film_beat_projection_mismatch",
    "director_plan_visual_style_mismatch", "emotion_curve_missing",
}


def _severity_for(code: str, original: str = "warning") -> str:
    if code in _HARD_CODES or original == "error":
        return "hard"
    if code in _WARNING_CODES:
        return "warning"
    if code in _DEFERRED_CODES:
        return "deferred"
    return "warning"


def _analysis_field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _diagnostics(value: Any) -> tuple[Any, ...]:
    raw = _analysis_field(value, "diagnostics", ())
    return tuple(raw or ())


def _diagnostic_field(value: Any, key: str, default: Any = "") -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _preserve_constraints(movie_plan: MoviePlan) -> tuple[str, ...]:
    characters = getattr(movie_plan.story_plan, "characters", ())
    names = tuple(item.character_id for item in characters if getattr(item, "character_id", ""))
    return (
        "保留现有核心冲突方向",
        "保留现有 StoryPlan story beats",
        "保留现有主要人物" + (f"（{', '.join(names)}）" if names else ""),
        "不改变既有世界观与结局基调",
    )


class _DiagnosticOptimizer:
    analysis_type = ""

    def run(
        self,
        movie_plan: MoviePlan,
        analysis_results: Sequence[CreativeAnalysisResult | dict[str, Any]],
        film_ir: FilmIR | None = None,
    ) -> CreativeOptimizerResult:
        """Compatibility alias for callers that model an optimizer as a pass."""

        return self.optimize(movie_plan, analysis_results, film_ir)

    def optimize(
        self,
        movie_plan: MoviePlan,
        analysis_results: Sequence[CreativeAnalysisResult | dict[str, Any]],
        film_ir: FilmIR | None = None,
    ) -> CreativeOptimizerResult:
        if not isinstance(movie_plan, MoviePlan):
            diagnostic = CreativeAnalysisDiagnostic(
                "invalid_movie_plan", "Creative Optimizer 只接受 MoviePlan。", "movie_plan", "error"
            )
            return CreativeOptimizerResult("", None, diagnostics=(diagnostic,), succeeded=False)
        if isinstance(analysis_results, (CreativeAnalysisResult, dict)):
            analysis_results = (analysis_results,)
        suggestions: list[OptimizationSuggestion] = []
        diagnostics: list[CreativeAnalysisDiagnostic] = []
        for result in analysis_results:
            if _analysis_field(result, "analysis_type", "") != self.analysis_type:
                continue
            for diagnostic in _diagnostics(result):
                code = str(_diagnostic_field(diagnostic, "code", ""))
                mapped = _MAPPINGS.get(self.analysis_type, {}).get(code)
                if not mapped:
                    continue
                severity = _severity_for(code, str(_diagnostic_field(diagnostic, "severity", "warning")))
                evidence = tuple(str(item) for item in (_diagnostic_field(diagnostic, "evidence", ()) or ()))
                message = str(_diagnostic_field(diagnostic, "message", code))
                suggestion = OptimizationSuggestion(
                    code=mapped,
                    message=message,
                    severity=severity,
                    path=str(_diagnostic_field(diagnostic, "path", "")),
                    reason=f"分析诊断 {code} 表明需要导演层重新审视该创作决策。",
                    evidence=evidence,
                    suggested_direction=self._direction(mapped, code),
                    preserve_constraints=_preserve_constraints(movie_plan),
                    risk="high" if severity == "hard" else "medium" if severity == "warning" else "low",
                    confidence=0.9 if evidence else 0.75,
                    source_analysis_types=(self.analysis_type,),
                    source_diagnostic_codes=(code,),
                )
                suggestions.append(suggestion)
                diagnostics.append(
                    CreativeAnalysisDiagnostic(
                        f"optimizer_{mapped}",
                        f"已生成创意优化建议：{mapped}。",
                        suggestion.path,
                        # Creative diagnostics never fail compilation.  The
                        # policy severity remains on OptimizationSuggestion.
                        "warning",
                        evidence,
                    )
                )
        candidates = tuple(
            TransformationCandidate(
                code=f"candidate_{item.code}",
                message=f"未来可评估：{item.code}",
                severity=item.severity,
                path=item.path,
                reason="Phase 4C 只记录候选，不执行自动变换。",
                risk=item.risk,
                confidence=item.confidence,
            )
            for item in suggestions
        )
        return CreativeOptimizerResult(
            movie_plan.plan_id,
            film_ir.ir_id if film_ir is not None else None,
            tuple(suggestions),
            candidates,
            tuple(diagnostics),
            {"suggestion_count": float(len(suggestions)), "candidate_count": float(len(candidates))},
            True,
        )

    @staticmethod
    def _direction(mapped: str, original: str) -> str:
        directions = {
            "strengthen_climax_emotional_peak": "让已有高潮承载更明确的情绪峰值，不新增剧情线。",
            "escalate_conflict_before_climax": "在现有冲突链中提高代价与压力，再进入高潮。",
            "show_cost_in_resolution": "让已有解决结果呈现可见代价或后果。",
            "connect_climax_to_existing_conflict": "把高潮动作连接回已有核心冲突。",
            "clarify_core_conflict": "请导演明确已有故事中的核心冲突与对立关系。",
            "clarify_protagonist_goal": "请导演明确现有主角目标，不凭空新增角色或世界规则。",
        }
        return directions.get(mapped, f"请导演重新审视 {original} 对观众体验的影响。")


class EmotionOptimizer(_DiagnosticOptimizer):
    analysis_type = "emotion_flow"


class AudienceOptimizer(_DiagnosticOptimizer):
    analysis_type = "audience_knowledge"


class ConflictOptimizer(_DiagnosticOptimizer):
    analysis_type = "conflict_progression"


class CharacterArcOptimizer(_DiagnosticOptimizer):
    analysis_type = "character_arc"


class PlanLayerConsistencyOptimizer(_DiagnosticOptimizer):
    analysis_type = "plan_layer_consistency"


class CreativeOptimizer:
    """Aggregate read-only optimizers for all creative analysis domains."""

    def __init__(self, optimizers: Iterable[_DiagnosticOptimizer] | None = None) -> None:
        self.optimizers = tuple(
            optimizers
            or (
                EmotionOptimizer(),
                AudienceOptimizer(),
                ConflictOptimizer(),
                CharacterArcOptimizer(),
                PlanLayerConsistencyOptimizer(),
            )
        )

    def optimize(
        self,
        movie_plan: MoviePlan,
        analysis_results: Sequence[CreativeAnalysisResult | dict[str, Any]],
        film_ir: FilmIR | None = None,
    ) -> CreativeOptimizerResult:
        if not isinstance(movie_plan, MoviePlan):
            return _DiagnosticOptimizer().optimize(movie_plan, analysis_results, film_ir)
        if isinstance(analysis_results, (CreativeAnalysisResult, dict)):
            analysis_results = (analysis_results,)
        results = [optimizer.optimize(movie_plan, analysis_results, film_ir) for optimizer in self.optimizers]
        suggestions = tuple(item for result in results for item in result.suggestions)
        candidates = tuple(item for result in results for item in result.transformation_candidates)
        diagnostics = tuple(item for result in results for item in result.diagnostics)
        succeeded = all(result.succeeded for result in results)
        return CreativeOptimizerResult(
            movie_plan.plan_id,
            film_ir.ir_id if film_ir is not None else None,
            suggestions,
            candidates,
            diagnostics,
            {
                "suggestion_count": float(len(suggestions)),
                "candidate_count": float(len(candidates)),
                "hard_suggestion_count": float(sum(item.severity == "hard" for item in suggestions)),
                "warning_suggestion_count": float(sum(item.severity == "warning" for item in suggestions)),
                "deferred_suggestion_count": float(sum(item.severity == "deferred" for item in suggestions)),
            },
            succeeded,
        )

    run = optimize


def creative_optimizer(
    optimizers: Iterable[_DiagnosticOptimizer] | None = None,
) -> CreativeOptimizer:
    return CreativeOptimizer(optimizers)


__all__ = [
    "OptimizationSuggestion",
    "TransformationCandidate",
    "CreativeOptimizerResult",
    "CreativeOptimizer",
    "EmotionOptimizer",
    "AudienceOptimizer",
    "ConflictOptimizer",
    "CharacterArcOptimizer",
    "PlanLayerConsistencyOptimizer",
    "creative_optimizer",
]

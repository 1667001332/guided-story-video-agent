"""Deterministic, provider-neutral optimization diagnostics for V2 IRs.

The optimizers intentionally start in *suggestion mode*: they identify a
possible improvement and preserve the original IR.  A future phase may apply
the recorded transformation after an explicit policy decision.  No optimizer
is allowed to invent story content or call a Provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from .film_ir import FilmIR
from .ir import MovieIR


OptimizerSeverity = Literal["error", "warning"]
T = TypeVar("T", FilmIR, MovieIR)


@dataclass(frozen=True, slots=True)
class OptimizationDiagnostic:
    code: str
    message: str
    path: str = ""
    severity: OptimizerSeverity = "warning"


@dataclass(frozen=True, slots=True)
class OptimizationTransformation:
    code: str
    message: str
    path: str = ""
    before: Any = None
    after: Any = None
    reason: str = ""
    severity: OptimizerSeverity = "warning"


@dataclass(frozen=True, slots=True)
class OptimizerResult(Generic[T]):
    before_ir: T
    after_ir: T | None
    diagnostics: tuple[OptimizationDiagnostic, ...] = ()
    transformations: tuple[OptimizationTransformation, ...] = ()

    @property
    def ok(self) -> bool:
        return self.after_ir is not None and not any(
            item.severity == "error" for item in self.diagnostics
        )

    @property
    def errors(self) -> tuple[OptimizationDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "error")

    @property
    def suggestions(self) -> tuple[OptimizationDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "warning")


class DetectOverloadedBeatOptimizer:
    name = "detect_overloaded_beat"

    def optimize(self, ir: FilmIR) -> OptimizerResult[FilmIR]:
        diagnostics: list[OptimizationDiagnostic] = []
        if not ir.beats:
            diagnostics.append(
                OptimizationDiagnostic(
                    "missing_film_beats",
                    "没有可优化的 film beats。",
                    "beats",
                    "error",
                )
            )
            return OptimizerResult(ir, None, tuple(diagnostics))
        total = float(ir.target_duration_seconds)
        if total <= 0:
            diagnostics.append(
                OptimizationDiagnostic(
                    "invalid_target_duration",
                    "FilmIR target duration 必须为正数。",
                    "target_duration_seconds",
                    "error",
                )
            )
            return OptimizerResult(ir, None, tuple(diagnostics))
        for index, beat in enumerate(ir.beats):
            ratio = beat.duration_seconds / total
            if len(ir.beats) > 1 and ratio >= 0.75:
                diagnostics.append(
                    OptimizationDiagnostic(
                        "overloaded_beat_candidate",
                        "该 beat 占据过高时长比例，建议由导演重新审视节奏。",
                        f"beats[{index}].duration_seconds",
                    )
                )
        return OptimizerResult(ir, ir, tuple(diagnostics))

    run = optimize


class DetectWeakClimaxOptimizer:
    name = "detect_weak_climax"

    def optimize(self, ir: FilmIR) -> OptimizerResult[FilmIR]:
        if len(ir.beats) < 2:
            return OptimizerResult(ir, ir)
        climax = ir.beats[-1]
        previous_max = max(beat.tension_level for beat in ir.beats[:-1])
        if climax.tension_level < previous_max and "climax" not in climax.narrative_function.lower():
            return OptimizerResult(
                ir,
                ir,
                (
                    OptimizationDiagnostic(
                        "weak_climax_candidate",
                        "末 beat 的张力低于前序峰值，建议导演复核结尾。",
                        f"beats[{len(ir.beats) - 1}].tension_level",
                    ),
                ),
            )
        return OptimizerResult(ir, ir)


class DetectMissingAudienceUnderstandingOptimizer:
    name = "detect_missing_audience_understanding"

    def optimize(self, ir: FilmIR) -> OptimizerResult[FilmIR]:
        diagnostics = tuple(
            OptimizationDiagnostic(
                "missing_audience_understanding",
                "beat 缺少观众理解目标，不能安全进入 lowering。",
                f"beats[{index}].required_audience_understanding",
                "error",
            )
            for index, beat in enumerate(ir.beats)
            if not beat.required_audience_understanding.strip()
        )
        return OptimizerResult(ir, None if diagnostics else ir, diagnostics)


class FilmIROptimizer:
    """Run FilmIR optimization diagnostics without rewriting the IR."""

    def __init__(self, optimizers: tuple[Any, ...] | None = None) -> None:
        self.optimizers = optimizers or (
            DetectOverloadedBeatOptimizer(),
            DetectWeakClimaxOptimizer(),
            DetectMissingAudienceUnderstandingOptimizer(),
        )

    def optimize(self, ir: FilmIR) -> OptimizerResult[FilmIR]:
        diagnostics: list[OptimizationDiagnostic] = []
        transformations: list[OptimizationTransformation] = []
        after: FilmIR | None = ir
        for optimizer in self.optimizers:
            if after is None:
                break
            result = optimizer.optimize(after)
            diagnostics.extend(result.diagnostics)
            transformations.extend(result.transformations)
            if not result.ok:
                after = None
                break
            after = result.after_ir
        return OptimizerResult(ir, after, tuple(diagnostics), tuple(transformations))

    run = optimize


class MergeAdjacentCompatibleShotsOptimizer:
    name = "merge_adjacent_compatible_shots"

    def optimize(self, ir: MovieIR) -> OptimizerResult[MovieIR]:
        candidates: list[OptimizationTransformation] = []
        for index, (left, right) in enumerate(zip(ir.shots, ir.shots[1:])):
            compatible = (
                left.scene_id == right.scene_id
                and left.camera == right.camera
                and left.motion == right.motion
                and left.lighting == right.lighting
                and left.composition == right.composition
            )
            if compatible:
                candidates.append(
                    OptimizationTransformation(
                        "merge_candidate",
                        "相邻镜头的场景和摄影语言兼容，可由后续策略评估是否合并。",
                        f"shots[{index}:{index + 2}]",
                        before=[left.shot_id, right.shot_id],
                        after=[left.shot_id],
                        reason="减少重复切换，但本阶段不自动合并镜头。",
                    )
                )
        diagnostics = tuple(
            OptimizationDiagnostic(item.code, item.message, item.path)
            for item in candidates
        )
        return OptimizerResult(ir, ir, diagnostics, tuple(candidates))

    run = optimize


class DetectRedundantShotOptimizer:
    name = "detect_redundant_shot"

    def optimize(self, ir: MovieIR) -> OptimizerResult[MovieIR]:
        diagnostics: list[OptimizationDiagnostic] = []
        for index, (left, right) in enumerate(zip(ir.shots, ir.shots[1:])):
            if (
                left.scene_id == right.scene_id
                and left.visible_action.strip() == right.visible_action.strip()
                and left.subject.strip() == right.subject.strip()
            ):
                diagnostics.append(
                    OptimizationDiagnostic(
                        "redundant_shot_candidate",
                        "相邻镜头的可见动作和主体完全重复。",
                        f"shots[{index + 1}]",
                    )
                )
        return OptimizerResult(ir, ir, tuple(diagnostics))

    run = optimize


class DetectTimelineBudgetRiskOptimizer:
    name = "detect_timeline_budget_risk"

    def optimize(self, ir: MovieIR) -> OptimizerResult[MovieIR]:
        diagnostics: list[OptimizationDiagnostic] = []
        total = sum(float(shot.duration_seconds) for shot in ir.shots)
        if total > float(ir.target_duration_seconds) + 1e-6:
            diagnostics.append(
                OptimizationDiagnostic(
                    "timeline_budget_overflow",
                    "shot duration 总和超过 MovieIR 目标时长。",
                    "shots",
                )
            )
        if len(ir.shots) > 1:
            minimum_readable = max(0.5, ir.target_duration_seconds * 0.02)
            for index, shot in enumerate(ir.shots):
                if shot.duration_seconds < minimum_readable:
                    diagnostics.append(
                        OptimizationDiagnostic(
                            "timeline_budget_risk",
                            "镜头时长过短，可能不足以承载可识别视觉证据。",
                            f"shots[{index}].duration_seconds",
                        )
                    )
        return OptimizerResult(ir, ir, tuple(diagnostics))

    run = optimize


class MovieIROptimizer:
    """Run MovieIR diagnostics and preserve transformation candidates."""

    def __init__(self, optimizers: tuple[Any, ...] | None = None) -> None:
        self.optimizers = optimizers or (
            MergeAdjacentCompatibleShotsOptimizer(),
            DetectRedundantShotOptimizer(),
            DetectTimelineBudgetRiskOptimizer(),
        )

    def optimize(self, ir: MovieIR) -> OptimizerResult[MovieIR]:
        diagnostics: list[OptimizationDiagnostic] = []
        transformations: list[OptimizationTransformation] = []
        after: MovieIR | None = ir
        for optimizer in self.optimizers:
            if after is None:
                break
            result = optimizer.optimize(after)
            diagnostics.extend(result.diagnostics)
            transformations.extend(result.transformations)
            if not result.ok:
                after = None
                break
            after = result.after_ir
        return OptimizerResult(ir, after, tuple(diagnostics), tuple(transformations))

    run = optimize

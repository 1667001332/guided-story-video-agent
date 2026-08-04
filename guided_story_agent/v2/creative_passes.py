"""Creative, film-language diagnostics for the V2 compiler.

These passes deliberately sit above the provider-neutral compiler passes.  A
creative pass can inspect the audience contract and dramatic beats, but it
cannot access a Provider, mutate a MoviePlan, or manufacture missing story
content.  Any hard diagnostic returns ``ir=None`` so callers cannot continue
silently with an invalid creative contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

from .film_ir import FilmIR
from .passes import Diagnostic


@dataclass(frozen=True, slots=True)
class CreativePassResult:
    """Result of one pure FilmIR creative pass."""

    ir: FilmIR | None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def film_ir(self) -> FilmIR | None:
        """Named alias for callers that want to make the layer explicit."""

        return self.ir

    @property
    def ok(self) -> bool:
        return self.ir is not None and not any(
            item.severity == "error" for item in self.diagnostics
        )

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "error")


class CreativePass(Protocol):
    name: str

    def run(self, ir: FilmIR) -> CreativePassResult: ...


class CreativePassPipeline:
    """Compose pure creative diagnostics and fail closed on hard errors."""

    def __init__(self, passes: Sequence[CreativePass]) -> None:
        self.passes = tuple(passes)

    def run(self, ir: FilmIR) -> CreativePassResult:
        if not isinstance(ir, FilmIR):
            return CreativePassResult(
                None,
                (Diagnostic("invalid_film_ir", "Creative Pass 只接受 FilmIR。"),),
            )
        current: FilmIR | None = ir
        diagnostics: list[Diagnostic] = []
        for current_pass in self.passes:
            if current is None:
                break
            try:
                result = current_pass.run(current)
            except Exception as exc:  # pragma: no cover - defensive boundary
                diagnostics.append(
                    Diagnostic(
                        "creative_pass_exception",
                        f"Creative Pass {current_pass.name} 执行失败：{exc}",
                        severity="error",
                    )
                )
                return CreativePassResult(None, tuple(diagnostics))
            diagnostics.extend(result.diagnostics)
            if not result.ok or result.ir is None:
                return CreativePassResult(None, tuple(diagnostics))
            current = result.ir
        return CreativePassResult(current, tuple(diagnostics))


class PacingDiagnosticsPass:
    """Report beat pacing risks without allocating or changing durations."""

    name = "pacing_diagnostics"

    def run(self, ir: FilmIR) -> CreativePassResult:
        if not ir.beats:
            return CreativePassResult(
                None,
                (Diagnostic("missing_film_beats", "FilmIR 没有可诊断的 beats。", "beats"),),
            )
        total = float(ir.target_duration_seconds)
        if not math.isfinite(total) or total <= 0:
            return CreativePassResult(
                None,
                (
                    Diagnostic(
                        "invalid_target_duration",
                        "FilmIR target_duration_seconds 必须为正数。",
                        "target_duration_seconds",
                    ),
                ),
            )
        diagnostics: list[Diagnostic] = []
        if len(ir.beats) > 1:
            for index, beat in enumerate(ir.beats):
                if beat.duration_seconds / total >= 0.75:
                    diagnostics.append(
                        Diagnostic(
                            "overloaded_beat",
                            "单个 beat 占据过高时长比例，建议交由 Director 重新审视节奏。",
                            f"beats[{index}].duration_seconds",
                            "warning",
                        )
                    )
        return CreativePassResult(ir, tuple(diagnostics))


class EmotionalContinuityPass:
    """Check that adjacent beats communicate a coherent emotional handoff."""

    name = "emotional_continuity_diagnostics"

    def run(self, ir: FilmIR) -> CreativePassResult:
        diagnostics: list[Diagnostic] = []
        for index, (current, following) in enumerate(zip(ir.beats, ir.beats[1:])):
            if not current.viewer_state_after.strip() or not following.viewer_state_before.strip():
                diagnostics.append(
                    Diagnostic(
                        "missing_emotional_handoff",
                        "相邻 beat 缺少完整的 viewer state handoff。",
                        f"beats[{index}:{index + 2}]",
                        "error",
                    )
                )
            elif current.viewer_state_after.strip() != following.viewer_state_before.strip():
                diagnostics.append(
                    Diagnostic(
                        "emotional_continuity_gap",
                        "前一 beat 的 viewer_state_after 与后一 beat 的 viewer_state_before 不一致。",
                        f"beats[{index + 1}].viewer_state_before",
                        "warning",
                    )
                )
        return CreativePassResult(
            None if any(item.severity == "error" for item in diagnostics) else ir,
            tuple(diagnostics),
        )


class VisualMotifDiagnosticsPass:
    """Detect visual-focus drift; it never invents or rewrites motifs."""

    name = "visual_motif_diagnostics"

    def run(self, ir: FilmIR) -> CreativePassResult:
        focuses = [beat.visual_focus.strip() for beat in ir.beats if beat.visual_focus.strip()]
        diagnostics: list[Diagnostic] = []
        if len(ir.beats) >= 3 and len(set(focuses)) == len(focuses):
            diagnostics.append(
                Diagnostic(
                    "visual_motif_drift",
                    "每个 beat 都使用不同 visual_focus，可能缺少可识别的视觉母题。",
                    "beats",
                    "warning",
                )
            )
        return CreativePassResult(ir, tuple(diagnostics))


class AudienceUnderstandingDiagnosticsPass:
    """Ensure every beat states what the audience must understand."""

    name = "audience_understanding_diagnostics"

    def run(self, ir: FilmIR) -> CreativePassResult:
        diagnostics: list[Diagnostic] = []
        for index, beat in enumerate(ir.beats):
            path = f"beats[{index}]"
            if not beat.required_audience_understanding.strip():
                diagnostics.append(
                    Diagnostic(
                        "missing_audience_understanding",
                        "beat 必须声明 required_audience_understanding。",
                        f"{path}.required_audience_understanding",
                        "error",
                    )
                )
            if not beat.required_evidence:
                diagnostics.append(
                    Diagnostic(
                        "missing_audience_evidence",
                        "beat 必须声明 audience evidence。",
                        f"{path}.required_evidence",
                        "warning",
                    )
                )
        return CreativePassResult(
            None if any(item.severity == "error" for item in diagnostics) else ir,
            tuple(diagnostics),
        )


class ConflictClarityDiagnosticsPass:
    """Report beats whose dramatic purpose is not explicit enough to review."""

    name = "conflict_clarity_diagnostics"

    def run(self, ir: FilmIR) -> CreativePassResult:
        diagnostics: list[Diagnostic] = []
        for index, beat in enumerate(ir.beats):
            if not beat.dramatic_purpose.strip() or not beat.narrative_function.strip():
                diagnostics.append(
                    Diagnostic(
                        "conflict_clarity_missing",
                        "beat 必须同时声明 dramatic_purpose 与 narrative_function。",
                        f"beats[{index}]",
                        "error",
                    )
                )
        return CreativePassResult(
            None if any(item.severity == "error" for item in diagnostics) else ir,
            tuple(diagnostics),
        )


def creative_pass_pipeline() -> CreativePassPipeline:
    """Return the default film-language diagnostics pipeline."""

    return CreativePassPipeline(
        (
            PacingDiagnosticsPass(),
            EmotionalContinuityPass(),
            VisualMotifDiagnosticsPass(),
            AudienceUnderstandingDiagnosticsPass(),
            ConflictClarityDiagnosticsPass(),
        )
    )


CreativeDiagnostic = Diagnostic

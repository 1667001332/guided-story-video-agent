"""Pure, composable validation and normalization passes for V2 IRs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, Sequence

from .film_ir import FilmIR
from .ir import MovieIR, TimelineEntry
from .validation import FilmIRValidator, MovieIRValidator, ValidationIssue


DiagnosticSeverity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    message: str
    path: str = ""
    severity: DiagnosticSeverity = "error"


@dataclass(frozen=True, slots=True)
class PassResult:
    ir: FilmIR | MovieIR | None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.ir is not None and not any(
            item.severity == "error" for item in self.diagnostics
        )

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "error")


class IRPass(Protocol):
    name: str

    def run(self, ir: FilmIR | MovieIR) -> PassResult: ...


class PassPipeline:
    """Run pure IR passes in order and fail closed on hard diagnostics."""

    def __init__(self, passes: Sequence[IRPass]) -> None:
        self.passes = tuple(passes)

    def run(self, ir: FilmIR | MovieIR) -> PassResult:
        current: FilmIR | MovieIR | None = ir
        diagnostics: list[Diagnostic] = []
        for current_pass in self.passes:
            if current is None:
                break
            try:
                result = current_pass.run(current)
            except Exception as exc:  # pragma: no cover - defensive boundary
                diagnostics.append(
                    Diagnostic(
                        "pass_exception",
                        f"Pass {current_pass.name} 执行失败：{exc}",
                        severity="error",
                    )
                )
                return PassResult(None, tuple(diagnostics))
            diagnostics.extend(result.diagnostics)
            if not result.ok or result.ir is None:
                return PassResult(None, tuple(diagnostics))
            current = result.ir
        return PassResult(current, tuple(diagnostics))


class CompilerPassPipeline(PassPipeline):
    """Named boundary for passes that operate on executable MovieIR."""


class ValidateFilmIRPass:
    name = "validate_film_ir"

    def run(self, ir: FilmIR | MovieIR) -> PassResult:
        if not isinstance(ir, FilmIR):
            return PassResult(
                None,
                (Diagnostic("invalid_film_ir", "该 Pass 只接受 FilmIR。"),),
            )
        return _validation_pass_result(ir, FilmIRValidator().validate(ir))


class NormalizeFilmBeatOrderPass:
    name = "normalize_film_beat_order"

    def run(self, ir: FilmIR | MovieIR) -> PassResult:
        if not isinstance(ir, FilmIR):
            return PassResult(
                None,
                (Diagnostic("invalid_film_ir", "该 Pass 只接受 FilmIR。"),),
            )
        ordered = tuple(sorted(ir.beats, key=lambda beat: (beat.order, beat.beat_id)))
        changed = (
            tuple(beat.beat_id for beat in ir.beats)
            != tuple(beat.beat_id for beat in ordered)
            or any(beat.order != index for index, beat in enumerate(ordered, start=1))
        )
        beats = tuple(
            replace(beat, order=index)
            for index, beat in enumerate(ordered, start=1)
        )
        by_id = {beat.beat_id: beat for beat in beats}
        cursor = 0.0
        timeline: list[Any] = []
        for index, entry in enumerate(
            sorted(ir.beat_timeline, key=lambda item: (by_id.get(item.beat_id).order if item.beat_id in by_id else item.order, item.beat_id)),
            start=1,
        ):
            beat = by_id.get(entry.beat_id)
            if beat is None:
                timeline.append(entry)
                continue
            normalized = replace(
                entry,
                order=index,
                scene_id=beat.scene_id,
                shot_ids=beat.shot_ids,
                start_seconds=cursor,
                duration_seconds=beat.duration_seconds,
            )
            timeline.append(normalized)
            cursor += beat.duration_seconds
        diagnostics: tuple[Diagnostic, ...] = (
            (
                Diagnostic(
                    "beat_order_normalized",
                    "FilmIR beat order/timeline 已按声明顺序规范化。",
                    "beats",
                    "warning",
                ),
            )
            if changed
            else ()
        )
        return PassResult(
            replace(ir, beats=beats, beat_timeline=tuple(timeline)),
            diagnostics,
        )


class EnsureAudienceUnderstandingPass:
    name = "ensure_audience_understanding"

    def run(self, ir: FilmIR | MovieIR) -> PassResult:
        if not isinstance(ir, FilmIR):
            return PassResult(
                None,
                (Diagnostic("invalid_film_ir", "该 Pass 只接受 FilmIR。"),),
            )
        diagnostics = tuple(
            Diagnostic(
                "missing_audience_understanding",
                "FilmIR beat 缺少 required_audience_understanding。",
                f"beats[{index}].required_audience_understanding",
                "error",
            )
            for index, beat in enumerate(ir.beats)
            if not beat.required_audience_understanding.strip()
        )
        return PassResult(None if diagnostics else ir, diagnostics)


class ValidateMovieIRPass:
    name = "validate_movie_ir"

    def run(self, ir: FilmIR | MovieIR) -> PassResult:
        if not isinstance(ir, MovieIR):
            return PassResult(
                None,
                (Diagnostic("invalid_movie_ir", "该 Pass 只接受 MovieIR。"),),
            )
        return _validation_pass_result(ir, MovieIRValidator().validate(ir))


class NormalizeShotTimelinePass:
    name = "normalize_shot_timeline"

    def run(self, ir: FilmIR | MovieIR) -> PassResult:
        if not isinstance(ir, MovieIR):
            return PassResult(
                None,
                (Diagnostic("invalid_movie_ir", "该 Pass 只接受 MovieIR。"),),
            )
        shots = tuple(
            replace(shot, order=index)
            for index, shot in enumerate(
                sorted(ir.shots, key=lambda shot: (shot.order, shot.shot_id)),
                start=1,
            )
        )
        timeline: list[TimelineEntry] = []
        cursor = 0.0
        changed = False
        for index, shot in enumerate(shots, start=1):
            old = next(
                (entry for entry in ir.timeline if entry.shot_id == shot.shot_id),
                None,
            )
            if old is None or old.order != index or old.start_seconds != cursor:
                changed = True
            timeline.append(
                TimelineEntry(
                    shot_id=shot.shot_id,
                    order=index,
                    start_seconds=cursor,
                    duration_seconds=shot.duration_seconds,
                )
            )
            cursor += shot.duration_seconds
        diagnostics: tuple[Diagnostic, ...] = (
            (
                Diagnostic(
                    "shot_timeline_normalized",
                    "MovieIR shot order/timeline 已规范化。",
                    "timeline",
                    "warning",
                ),
            )
            if changed
            else ()
        )
        return PassResult(
            replace(ir, shots=shots, timeline=tuple(timeline)),
            diagnostics,
        )


class ContinuityDiagnosticsPass:
    name = "continuity_diagnostics"

    def run(self, ir: FilmIR | MovieIR) -> PassResult:
        if not isinstance(ir, MovieIR):
            return PassResult(
                None,
                (Diagnostic("invalid_movie_ir", "该 Pass 只接受 MovieIR。"),),
            )
        shot_ids = {shot.shot_id for shot in ir.shots}
        diagnostics: list[Diagnostic] = []
        for index, anchor in enumerate(ir.continuity_anchors):
            missing = set(anchor.applies_to_shots) - shot_ids
            if missing:
                diagnostics.append(
                    Diagnostic(
                        "untracked_continuity_anchor",
                        "continuity anchor 引用了不存在的 shot。",
                        f"continuity_anchors[{index}].applies_to_shots",
                        "error",
                    )
                )
        return PassResult(None if any(item.severity == "error" for item in diagnostics) else ir, tuple(diagnostics))


class PromptLeakageDiagnosticsPass:
    name = "prompt_leakage_diagnostics"

    def run(self, ir: FilmIR | MovieIR) -> PassResult:
        if not isinstance(ir, MovieIR):
            return PassResult(
                None,
                (Diagnostic("invalid_movie_ir", "该 Pass 只接受 MovieIR。"),),
            )
        issues = MovieIRValidator().validate(ir).issues
        diagnostics = tuple(
            Diagnostic(item.code, item.message, item.path, item.severity)
            for item in issues
            if item.code in {"prompt_leakage", "provider_field_in_ir"}
        )
        return PassResult(
            None if any(item.severity == "error" for item in diagnostics) else ir,
            diagnostics,
        )


def film_ir_pass_pipeline() -> PassPipeline:
    return PassPipeline(
        (
            NormalizeFilmBeatOrderPass(),
            EnsureAudienceUnderstandingPass(),
            ValidateFilmIRPass(),
        )
    )


def movie_ir_pass_pipeline() -> CompilerPassPipeline:
    return CompilerPassPipeline(
        (
            NormalizeShotTimelinePass(),
            ContinuityDiagnosticsPass(),
            PromptLeakageDiagnosticsPass(),
            ValidateMovieIRPass(),
        )
    )


def compiler_pass_pipeline() -> CompilerPassPipeline:
    """Explicit alias for the MovieIR compiler pass pipeline."""

    return movie_ir_pass_pipeline()


def _validation_pass_result(ir: FilmIR | MovieIR, result) -> PassResult:
    diagnostics = tuple(_diagnostic(issue) for issue in result.issues)
    return PassResult(
        None if not result.ok else ir,
        diagnostics,
    )


def _diagnostic(issue: ValidationIssue) -> Diagnostic:
    return Diagnostic(issue.code, issue.message, issue.path, issue.severity)

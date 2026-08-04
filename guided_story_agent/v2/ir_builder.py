"""Fail-closed FilmIR → MovieIR conversion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import warnings

from .film_ir import FilmIR
from .ir import (
    AcceptanceCriterion,
    CharacterAnchor,
    ContinuityAnchor,
    MovieIR,
    MusicCue,
    NarrationCue,
    ShotIR,
    SubtitleCue,
    TimelineEntry,
    TransitionCue,
)
from .validation import MovieIRValidator
from .fingerprint import content_fingerprint


@dataclass(frozen=True, slots=True)
class IRBuildError:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class IRBuildDiagnostic:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class IRBuildResult:
    movie_ir: MovieIR | None
    errors: tuple[IRBuildError, ...] = ()
    diagnostics: tuple[IRBuildDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.movie_ir is not None and not self.errors


class MovieIRBuilder:
    """Lower explicit film-language decisions into executable shot IR."""

    def __init__(self, *, aspect_ratio: str = "16:9") -> None:
        self.aspect_ratio = aspect_ratio.strip() or "16:9"

    def build(self, film_ir: FilmIR) -> IRBuildResult:
        if not isinstance(film_ir, FilmIR):
            return self._failure(
                IRBuildError(
                    "invalid_film_ir_state",
                    "MovieIRBuilder 只接受 FilmIR；请先执行 FilmIRBuilder。",
                )
            )
        errors = self._preflight(film_ir)
        if errors:
            return self._failure(*errors)

        beat_by_shot = {
            shot_id: beat
            for beat in film_ir.beats
            for shot_id in beat.shot_ids
        }
        start = 0.0
        timeline: list[TimelineEntry] = []
        shots: list[ShotIR] = []
        continuity_map: dict[str, ContinuityAnchor] = {}
        narration_track: list[NarrationCue] = []
        subtitle_track: list[SubtitleCue] = []
        transition_cues: list[TransitionCue] = []
        acceptance_criteria: list[AcceptanceCriterion] = []

        for shot in sorted(film_ir.shots, key=lambda item: item.order):
            beat = beat_by_shot[shot.shot_id]
            required_evidence = _merge_strings(
                shot.required_visual_evidence,
                beat.required_evidence,
            )
            shot_acceptance = _merge_strings(
                shot.acceptance_criteria,
                beat.acceptance_criteria,
            )
            shot_ir = ShotIR(
                shot_id=shot.shot_id,
                scene_id=shot.scene_id,
                order=shot.order,
                duration_seconds=float(shot.duration_seconds),
                purpose=shot.purpose,
                visible_action=shot.visible_action,
                subject=shot.subject,
                location=shot.location,
                emotion=shot.emotion,
                camera=shot.camera,
                motion=shot.motion,
                lighting=shot.lighting,
                composition=shot.composition,
                characters=shot.characters,
                props=shot.props,
                narration=shot.narration,
                dialogue=shot.dialogue,
                subtitles=shot.subtitles,
                transition_in=shot.transition_in,
                transition_out=shot.transition_out,
                continuity_anchors=shot.continuity_anchors,
                required_visual_evidence=required_evidence,
                acceptance_criteria=shot_acceptance,
                references=shot.references,
                character_identity_anchors=shot.character_identity_anchors,
                metadata={
                    "source_scene_id": shot.scene_id,
                    "source_film_beat_id": beat.beat_id,
                },
            )
            shots.append(shot_ir)
            timeline.append(
                TimelineEntry(
                    shot_id=shot.shot_id,
                    order=shot.order,
                    start_seconds=start,
                    duration_seconds=float(shot.duration_seconds),
                )
            )
            start += float(shot.duration_seconds)
            for description in shot.continuity_anchors:
                _add_continuity_anchor(
                    continuity_map,
                    f"continuity-{_slug(description)}",
                    "shot",
                    description,
                    shot.shot_id,
                )
            _add_continuity_anchor(
                continuity_map,
                f"film-beat-{_slug(beat.beat_id)}",
                "film-beat",
                beat.continuity_intent,
                shot.shot_id,
                required_consistency="按 FilmIR 的连续性意图保持一致",
            )
            if shot.narration.strip():
                narration_track.append(NarrationCue(shot.shot_id, shot.narration))
            if shot.subtitles.strip():
                subtitle_track.append(SubtitleCue(shot.shot_id, shot.subtitles))
            for criterion_index, description in enumerate(shot_acceptance, start=1):
                acceptance_criteria.append(
                    AcceptanceCriterion(
                        criterion_id=f"criterion-{shot.shot_id}-{criterion_index}",
                        target_id=shot.shot_id,
                        criterion_type="shot",
                        description=description,
                        severity="required",
                    )
                )

        for criterion in film_ir.acceptance_criteria:
            acceptance_criteria.append(
                AcceptanceCriterion(
                    criterion_id=criterion.criterion_id,
                    target_id=criterion.target_id,
                    criterion_type=(
                        "movie"
                        if criterion.target_id == film_ir.source_movie_plan_id
                        else "film-beat"
                    ),
                    description=criterion.description,
                    severity=criterion.severity,
                )
            )
        for intent in film_ir.transition_intents:
            from_beat = _beat_by_id(film_ir, intent.from_beat_id)
            to_beat = _beat_by_id(film_ir, intent.to_beat_id)
            transition_cues.append(
                TransitionCue(
                    from_beat.shot_ids[-1],
                    to_beat.shot_ids[0],
                    intent.description,
                    f"FilmIR transition {intent.from_beat_id} → {intent.to_beat_id}",
                )
            )

        character_anchors = tuple(
            CharacterAnchor(
                character_id=item.character_id,
                name=item.name,
                role=item.role,
                visual_identity=item.visual_identity,
                costume=item.costume,
                continuity_notes=item.continuity_notes,
                reference_keys=item.reference_keys,
            )
            for item in film_ir.character_anchors
        )
        music_cues = tuple(
            MusicCue(target_id=item.beat_id, direction=item.description)
            for item in film_ir.music_intents
        )
        movie_ir = MovieIR(
            ir_id=f"ir-{film_ir.ir_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            source_movie_plan_id=film_ir.source_movie_plan_id,
            version=film_ir.version,
            title=film_ir.title,
            target_duration_seconds=float(film_ir.target_duration_seconds),
            aspect_ratio=self.aspect_ratio,
            visual_style=film_ir.visual_style,
            timeline=tuple(timeline),
            shots=tuple(shots),
            continuity_anchors=tuple(continuity_map.values()),
            character_anchors=character_anchors,
            narration_track=tuple(narration_track),
            subtitle_track=tuple(subtitle_track),
            music_cues=music_cues,
            transition_cues=tuple(transition_cues),
            acceptance_criteria=tuple(acceptance_criteria),
            source_film_ir_id=film_ir.ir_id,
            source_movie_plan_version=film_ir.source_movie_plan_version,
            source_movie_plan_fingerprint=film_ir.source_movie_plan_fingerprint,
            source_movie_plan_lineage_token=film_ir.source_movie_plan_lineage_token,
            source_film_ir_fingerprint=content_fingerprint(film_ir.to_dict()),
            metadata={
                "source": "film_ir",
                "source_movie_plan_id": film_ir.source_movie_plan_id,
                "source_film_ir_id": film_ir.ir_id,
                "source_movie_plan_version": film_ir.source_movie_plan_version,
                "source_movie_plan_fingerprint": film_ir.source_movie_plan_fingerprint,
                "built_at": datetime.now(timezone.utc).isoformat(),
                "scene_ids": sorted({shot.scene_id for shot in film_ir.shots}),
                "shot_ids": [shot.shot_id for shot in sorted(film_ir.shots, key=lambda item: item.order)],
                "beat_ids": [beat.beat_id for beat in sorted(film_ir.beats, key=lambda item: item.order)],
            },
        )
        validation = MovieIRValidator().validate(movie_ir)
        if not validation.ok:
            return self._failure(
                *tuple(
                    IRBuildError(issue.code, issue.message, issue.path)
                    for issue in validation.errors
                )
            )
        diagnostics = tuple(
            IRBuildDiagnostic(issue.code, issue.message, issue.path)
            for issue in validation.warnings
        )
        return IRBuildResult(movie_ir=movie_ir, diagnostics=diagnostics)

    @staticmethod
    def _preflight(film_ir: FilmIR) -> list[IRBuildError]:
        errors: list[IRBuildError] = []
        if not film_ir.ir_id.strip() or not film_ir.source_movie_plan_id.strip():
            errors.append(IRBuildError("invalid_film_ir_state", "FilmIR 缺少稳定来源标识。"))
        if not film_ir.source_movie_plan_id.strip():
            errors.append(IRBuildError("invalid_film_ir_state", "FilmIR 缺少 source_movie_plan_id。"))
        if not film_ir.beats:
            errors.append(IRBuildError("missing_film_level_beats", "FilmIR 缺少 cinematic beats。", "beats"))
        if not film_ir.shots:
            errors.append(IRBuildError("invalid_film_ir_state", "FilmIR 必须包含 shots。", "shots"))
            return _unique(errors)
        shot_ids = {shot.shot_id for shot in film_ir.shots}
        beat_shots = [shot_id for beat in film_ir.beats for shot_id in beat.shot_ids]
        if set(beat_shots) != shot_ids or len(beat_shots) != len(set(beat_shots)):
            errors.append(IRBuildError("invalid_beat_coverage", "FilmIR beats 必须覆盖每个 shot 一次。", "beats"))
        return _unique(errors)

    @staticmethod
    def _failure(*errors: IRBuildError) -> IRBuildResult:
        return IRBuildResult(movie_ir=None, errors=tuple(errors))


def build_movie_ir(film_ir: FilmIR, *, aspect_ratio: str = "16:9") -> IRBuildResult:
    warnings.warn(
        "build_movie_ir() is deprecated; use MovieIRBuilder(...).build()",
        DeprecationWarning,
        stacklevel=2,
    )
    return MovieIRBuilder(aspect_ratio=aspect_ratio).build(film_ir)


def _merge_strings(*groups: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for group in groups:
        for value in group:
            cleaned = str(value).strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
    return tuple(values)


def _add_continuity_anchor(
    mapping: dict[str, ContinuityAnchor],
    anchor_id: str,
    anchor_type: str,
    description: str,
    shot_id: str,
    *,
    required_consistency: str = "跨引用镜头保持一致",
) -> None:
    existing = mapping.get(anchor_id)
    applies = tuple(existing.applies_to_shots) if existing else ()
    if shot_id not in applies:
        applies += (shot_id,)
    mapping[anchor_id] = ContinuityAnchor(
        anchor_id=anchor_id,
        type=anchor_type,
        description=description,
        applies_to_shots=applies,
        required_consistency=(existing.required_consistency if existing else required_consistency),
    )


def _beat_by_id(film_ir: FilmIR, beat_id: str):
    for beat in film_ir.beats:
        if beat.beat_id == beat_id:
            return beat
    raise ValueError(f"FilmIR transition 引用了不存在的 beat：{beat_id}")


def _slug(value: str) -> str:
    return "-".join(value.lower().split())[:40] or "anchor"


def _unique(errors: list[IRBuildError]) -> list[IRBuildError]:
    seen: set[tuple[str, str, str]] = set()
    result: list[IRBuildError] = []
    for error in errors:
        key = (error.code, error.path, error.message)
        if key not in seen:
            seen.add(key)
            result.append(error)
    return result

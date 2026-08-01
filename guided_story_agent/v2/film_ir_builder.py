"""Fail-closed MoviePlan → FilmIR conversion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from .film_ir import (
    BeatTimelineEntry,
    CharacterEmotionAnchor,
    EmotionCurvePoint,
    FilmAcceptanceCriterion,
    FilmBeat,
    FilmCharacterAnchor,
    FilmContinuityIntent,
    FilmIR,
    FilmMusicIntent,
    FilmNarrationIntent,
    FilmShot,
    FilmTransitionIntent,
    TensionCurvePoint,
)
from .models import CreativeBrief, MoviePlan
from .validation import FilmIRValidator, validate_movie_plan


@dataclass(frozen=True, slots=True)
class FilmIRBuildError:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class FilmIRBuildDiagnostic:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class FilmIRBuildResult:
    film_ir: FilmIR | None
    errors: tuple[FilmIRBuildError, ...] = ()
    diagnostics: tuple[FilmIRBuildDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.film_ir is not None and not self.errors


class FilmIRBuilder:
    """Project explicit director decisions into a film-language IR.

    This builder validates and copies.  It never invents a beat, shot,
    duration, character state, or audience requirement.
    """

    def build(self, movie_plan: MoviePlan) -> FilmIRBuildResult:
        if not isinstance(movie_plan, MoviePlan):
            return self._failure(
                FilmIRBuildError("invalid_movie_plan_state", "输入不是 MoviePlan。")
            )
        errors = self._preflight(movie_plan)
        if errors:
            return self._failure(*errors)

        scene_by_id = {scene.scene_id: scene for scene in movie_plan.script.scenes}
        camera_by_id = {item.scene_id: item for item in movie_plan.camera_plan.instructions}
        character_by_id = {
            item.character_id: item for item in movie_plan.character_sheet.characters
        }
        ordered_shots = tuple(sorted(movie_plan.shot_plan, key=lambda item: item.order))
        film_shots = tuple(
            FilmShot(
                shot_id=shot.shot_id,
                scene_id=shot.scene_id,
                order=shot.order,
                duration_seconds=float(shot.duration_seconds),
                purpose=shot.purpose,
                visible_action=shot.visible_action,
                subject=shot.subject,
                location=scene_by_id[shot.scene_id].location,
                emotion=scene_by_id[shot.scene_id].emotion,
                camera="；".join(
                    (
                        camera_by_id[shot.scene_id].shot_size,
                        camera_by_id[shot.scene_id].angle,
                        camera_by_id[shot.scene_id].lens,
                        camera_by_id[shot.scene_id].movement,
                        camera_by_id[shot.scene_id].composition,
                        camera_by_id[shot.scene_id].language,
                        shot.camera_intent,
                    )
                ),
                motion=shot.motion_intent,
                lighting=shot.lighting,
                composition=shot.composition,
                characters=tuple(shot.characters),
                props=tuple(shot.props),
                narration=shot.narration,
                dialogue=shot.dialogue,
                subtitles=shot.subtitles,
                transition_in=shot.transition_in,
                transition_out=shot.transition_out,
                continuity_anchors=tuple(shot.continuity_anchors),
                required_visual_evidence=tuple(shot.required_visual_evidence),
                acceptance_criteria=tuple(shot.acceptance_criteria),
                references=tuple(
                    reference
                    for character_id in shot.characters
                    for reference in character_by_id[character_id].reference_keys
                ),
                character_identity_anchors=tuple(shot.characters),
            )
            for shot in ordered_shots
        )
        shot_by_id = {shot.shot_id: shot for shot in film_shots}

        ordered_beats = tuple(sorted(movie_plan.film_beats, key=lambda item: item.order))
        beats: list[FilmBeat] = []
        beat_timeline: list[BeatTimelineEntry] = []
        emotion_curve: list[EmotionCurvePoint] = []
        tension_curve: list[TensionCurvePoint] = []
        character_emotions: list[CharacterEmotionAnchor] = []
        continuity_intents: list[FilmContinuityIntent] = []
        narration_intents: list[FilmNarrationIntent] = []
        music_intents: list[FilmMusicIntent] = []
        acceptance_criteria: list[FilmAcceptanceCriterion] = []
        transition_intents: list[FilmTransitionIntent] = []
        start = 0.0
        for index, plan_beat in enumerate(ordered_beats):
            beat_shots = tuple(shot_by_id[item] for item in plan_beat.shot_ids)
            duration = sum(item.duration_seconds for item in beat_shots)
            beat = FilmBeat(
                beat_id=plan_beat.beat_id,
                order=plan_beat.order,
                scene_id=plan_beat.scene_id,
                shot_ids=tuple(plan_beat.shot_ids),
                duration_seconds=duration,
                dramatic_purpose=plan_beat.dramatic_purpose,
                narrative_function=plan_beat.narrative_function,
                viewer_state_before=plan_beat.viewer_state_before,
                viewer_state_after=plan_beat.viewer_state_after,
                emotion=plan_beat.emotion,
                tension_level=float(plan_beat.tension_level),
                visual_focus=plan_beat.visual_focus,
                required_audience_understanding=plan_beat.required_audience_understanding,
                required_evidence=tuple(plan_beat.required_evidence),
                character_emotional_state=tuple(plan_beat.character_emotional_state),
                continuity_intent=plan_beat.continuity_intent,
                transition_intent=plan_beat.transition_intent,
                narration_intent=plan_beat.narration_intent,
                music_intent=plan_beat.music_intent,
                acceptance_criteria=tuple(plan_beat.acceptance_criteria),
                timing_weight=float(plan_beat.timing_weight),
            )
            beats.append(beat)
            beat_timeline.append(
                BeatTimelineEntry(
                    beat_id=beat.beat_id,
                    order=beat.order,
                    scene_id=beat.scene_id,
                    shot_ids=beat.shot_ids,
                    start_seconds=start,
                    duration_seconds=duration,
                )
            )
            start += duration
            emotion_curve.append(
                EmotionCurvePoint(beat.beat_id, beat.emotion, beat.tension_level)
            )
            tension_curve.append(TensionCurvePoint(beat.beat_id, beat.tension_level))
            for state in beat.character_emotional_state:
                character_id, emotional_state = _split_character_state(state)
                character_emotions.append(
                    CharacterEmotionAnchor(beat.beat_id, character_id, emotional_state)
                )
            continuity_intents.append(
                FilmContinuityIntent(
                    beat.beat_id,
                    beat.continuity_intent,
                    "按导演声明保持跨 beat 一致",
                )
            )
            narration_intents.append(FilmNarrationIntent(beat.beat_id, beat.narration_intent))
            music_intents.append(FilmMusicIntent(beat.beat_id, beat.music_intent))
            for criterion_index, description in enumerate(beat.acceptance_criteria, start=1):
                acceptance_criteria.append(
                    FilmAcceptanceCriterion(
                        criterion_id=f"film-criterion-{beat.beat_id}-{criterion_index}",
                        target_id=beat.beat_id,
                        description=description,
                        severity="required",
                    )
                )
            next_beat = (
                ordered_beats[index + 1].beat_id
                if index + 1 < len(ordered_beats)
                else beat.beat_id
            )
            transition_intents.append(
                FilmTransitionIntent(beat.beat_id, next_beat, beat.transition_intent)
            )
        for index, description in enumerate(movie_plan.review_criteria, start=1):
            acceptance_criteria.append(
                FilmAcceptanceCriterion(
                    criterion_id=f"film-movie-criterion-{index}",
                    target_id=movie_plan.plan_id,
                    description=description,
                    severity="required",
                )
            )

        character_anchors = tuple(
            FilmCharacterAnchor(
                character_id=item.character_id,
                name=item.name,
                role=item.role or "supporting",
                visual_identity=item.visual_signature,
                costume=item.costume,
                continuity_notes=item.performance_notes,
                reference_keys=item.reference_keys,
            )
            for item in movie_plan.character_sheet.characters
        )
        film_ir = FilmIR(
            ir_id=f"film-ir-{uuid4().hex}",
            source_movie_plan_id=movie_plan.plan_id,
            version=movie_plan.revision,
            title=movie_plan.story.title,
            target_duration_seconds=float(movie_plan.timing_plan.target_duration_seconds),
            visual_style=movie_plan.visual_style,
            beats=tuple(beats),
            beat_timeline=tuple(beat_timeline),
            shots=film_shots,
            character_anchors=character_anchors,
            emotion_curve=tuple(emotion_curve),
            tension_curve=tuple(tension_curve),
            character_emotion_anchors=tuple(character_emotions),
            continuity_intents=tuple(continuity_intents),
            transition_intents=tuple(transition_intents),
            narration_intents=tuple(narration_intents),
            music_intents=tuple(music_intents),
            acceptance_criteria=tuple(acceptance_criteria),
            source_story_plan_id=(
                f"{movie_plan.plan_id}:story_plan"
                if movie_plan.story_plan is not None
                else ""
            ),
            source_director_plan_id=(
                f"{movie_plan.plan_id}:director_plan"
                if movie_plan.director_plan is not None
                else ""
            ),
            metadata={
                "source": "movie_plan",
                "source_movie_plan_id": movie_plan.plan_id,
                "source_story_plan_id": (
                    f"{movie_plan.plan_id}:story_plan"
                    if movie_plan.story_plan is not None
                    else ""
                ),
                "source_director_plan_id": (
                    f"{movie_plan.plan_id}:director_plan"
                    if movie_plan.director_plan is not None
                    else ""
                ),
                "built_at": datetime.now(timezone.utc).isoformat(),
                "scene_ids": list(movie_plan.scene_ids),
                "shot_ids": [item.shot_id for item in ordered_shots],
                "beat_ids": [item.beat_id for item in ordered_beats],
            },
        )
        validation = FilmIRValidator().validate(film_ir)
        if not validation.ok:
            return self._failure(
                *tuple(
                    FilmIRBuildError(
                        issue.code,
                        issue.message,
                        issue.path,
                    )
                    for issue in validation.errors
                )
            )
        diagnostics = tuple(
            FilmIRBuildDiagnostic(issue.code, issue.message, issue.path)
            for issue in validation.warnings
        )
        return FilmIRBuildResult(film_ir=film_ir, diagnostics=diagnostics)

    def _preflight(self, plan: MoviePlan) -> list[FilmIRBuildError]:
        errors: list[FilmIRBuildError] = []
        if not plan.confirmed:
            errors.append(
                FilmIRBuildError(
                    "invalid_movie_plan_state",
                    "只有 confirmed MoviePlan 才能构建 FilmIR。",
                )
            )
        brief = CreativeBrief(
            target_duration_seconds=plan.timing_plan.target_duration_seconds,
            video_type="film_ir",
            visual_style=plan.visual_style,
            audience="unspecified",
        )
        report = validate_movie_plan(plan, brief)
        errors.extend(
            FilmIRBuildError(_validation_error_code(item), item)
            for item in report.errors
        )
        if not plan.shot_plan:
            errors.append(
                FilmIRBuildError(
                    "missing_shot_level_plan",
                    "MoviePlan lacks shot-level execution decisions. DirectorAgent must regenerate MoviePlan with shot-level planning.",
                    "shot_plan",
                )
            )
        if not plan.film_beats:
            errors.append(
                FilmIRBuildError(
                    "missing_film_level_beats",
                    "MoviePlan lacks film-level cinematic beat decisions. DirectorAgent must regenerate MoviePlan with cinematic beat planning.",
                    "film_beats",
                )
            )
            return _unique(errors)
        scene_ids = set(plan.scene_ids)
        character_ids = {item.character_id for item in plan.character_sheet.characters}
        camera_ids = {item.scene_id for item in plan.camera_plan.instructions}
        shot_ids = {shot.shot_id for shot in plan.shot_plan}
        for shot in plan.shot_plan:
            path = f"shot_plan.{shot.shot_id or '<empty>'}"
            if shot.scene_id not in scene_ids or shot.scene_id not in camera_ids:
                errors.append(
                    FilmIRBuildError(
                        "unfilmable_scene",
                        "shot 引用了缺少场景或摄影决策的 scene。",
                        path,
                    )
                )
            if not shot.characters or any(item not in character_ids for item in shot.characters):
                errors.append(
                    FilmIRBuildError(
                        "missing_character_anchor",
                        "shot 必须引用已声明的人物。",
                        f"{path}.characters",
                    )
                )
        for beat in plan.film_beats:
            path = f"film_beats.{beat.beat_id or '<empty>'}"
            if not set(beat.shot_ids).issubset(shot_ids):
                errors.append(
                    FilmIRBuildError(
                        "unfilmable_beat",
                        "film beat 引用了不存在的 shot。",
                        f"{path}.shot_ids",
                    )
                )
        return _unique(errors)

    @staticmethod
    def _failure(*errors: FilmIRBuildError) -> FilmIRBuildResult:
        return FilmIRBuildResult(film_ir=None, errors=tuple(errors))


def build_film_ir(movie_plan: MoviePlan) -> FilmIRBuildResult:
    return FilmIRBuilder().build(movie_plan)


CinematicIRBuilder = FilmIRBuilder
build_cinematic_ir = build_film_ir


def _split_character_state(value: str) -> tuple[str, str]:
    character_id, state = value.split(":", 1)
    return character_id.strip(), state.strip()


def _unique(errors: list[FilmIRBuildError]) -> list[FilmIRBuildError]:
    seen: set[tuple[str, str, str]] = set()
    result: list[FilmIRBuildError] = []
    for error in errors:
        key = (error.code, error.path, error.message)
        if key not in seen:
            seen.add(key)
            result.append(error)
    return result


def _validation_error_code(message: str) -> str:
    lowered = message.lower()
    if "timingplan" in lowered or "timing plan" in lowered:
        return "missing_timing_plan"
    if "duration" in lowered:
        return "missing_scene_duration"
    return "missing_required_field"

"""Film-language intermediate representation (FilmIR).

FilmIR is the deterministic, provider-neutral bridge between the director's
MoviePlan and the executable MovieIR.  It records cinematic beats and the
audience contract while retaining the director-authored shot evidence needed
for the next lowering step.  It deliberately contains no Provider, HTTP,
prompt, task, polling, or billing fields.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class FilmShot:
    """Director-authored shot evidence carried through FilmIR."""

    shot_id: str
    scene_id: str
    order: int
    duration_seconds: float
    purpose: str
    visible_action: str
    subject: str
    location: str
    emotion: str
    camera: str
    motion: str
    lighting: str
    composition: str
    characters: tuple[str, ...]
    props: tuple[str, ...]
    narration: str
    dialogue: str
    subtitles: str
    transition_in: str
    transition_out: str
    continuity_anchors: tuple[str, ...]
    required_visual_evidence: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    references: tuple[str, ...] = ()
    character_identity_anchors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FilmCharacterAnchor:
    character_id: str
    name: str
    role: str
    visual_identity: str
    costume: str
    continuity_notes: str
    reference_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FilmBeat:
    """One director-authored cinematic beat."""

    beat_id: str
    order: int
    scene_id: str
    shot_ids: tuple[str, ...]
    duration_seconds: float
    dramatic_purpose: str
    narrative_function: str
    viewer_state_before: str
    viewer_state_after: str
    emotion: str
    tension_level: float
    visual_focus: str
    required_audience_understanding: str
    required_evidence: tuple[str, ...]
    character_emotional_state: tuple[str, ...]
    continuity_intent: str
    transition_intent: str
    narration_intent: str
    music_intent: str
    acceptance_criteria: tuple[str, ...]
    timing_weight: float


@dataclass(frozen=True, slots=True)
class BeatTimelineEntry:
    beat_id: str
    order: int
    scene_id: str
    shot_ids: tuple[str, ...]
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class EmotionCurvePoint:
    beat_id: str
    label: str
    intensity: float


@dataclass(frozen=True, slots=True)
class TensionCurvePoint:
    beat_id: str
    level: float


@dataclass(frozen=True, slots=True)
class CharacterEmotionAnchor:
    beat_id: str
    character_id: str
    emotional_state: str


@dataclass(frozen=True, slots=True)
class FilmContinuityIntent:
    beat_id: str
    description: str
    required_consistency: str


@dataclass(frozen=True, slots=True)
class FilmTransitionIntent:
    from_beat_id: str
    to_beat_id: str
    description: str


@dataclass(frozen=True, slots=True)
class FilmNarrationIntent:
    beat_id: str
    description: str


@dataclass(frozen=True, slots=True)
class FilmMusicIntent:
    beat_id: str
    description: str


@dataclass(frozen=True, slots=True)
class FilmAcceptanceCriterion:
    criterion_id: str
    target_id: str
    description: str
    severity: str


# These names are intentionally lightweight aliases: the wire contract uses
# explicit strings while callers may import semantic field types by name.
DramaticPurpose = str
ViewerState = str
NarrativeFunction = str
VisualFocus = str
RequiredAudienceUnderstanding = str
RequiredEvidence = str


@dataclass(frozen=True, slots=True)
class FilmIR:
    """Stable film-language IR produced from one confirmed MoviePlan."""

    ir_id: str
    source_movie_plan_id: str
    version: int
    title: str
    target_duration_seconds: float
    visual_style: str
    beats: tuple[FilmBeat, ...]
    beat_timeline: tuple[BeatTimelineEntry, ...]
    shots: tuple[FilmShot, ...]
    character_anchors: tuple[FilmCharacterAnchor, ...]
    emotion_curve: tuple[EmotionCurvePoint, ...]
    tension_curve: tuple[TensionCurvePoint, ...]
    character_emotion_anchors: tuple[CharacterEmotionAnchor, ...]
    continuity_intents: tuple[FilmContinuityIntent, ...]
    transition_intents: tuple[FilmTransitionIntent, ...]
    narration_intents: tuple[FilmNarrationIntent, ...]
    music_intents: tuple[FilmMusicIntent, ...]
    acceptance_criteria: tuple[FilmAcceptanceCriterion, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Source lineage is a compiler boundary, not Provider Runtime data.
    # Empty values are accepted only for old persisted sessions; the
    # SourceLineageGuard marks them unknown/stale before reuse.
    source_story_plan_id: str = ""
    source_director_plan_id: str = ""
    source_movie_plan_version: int = 0
    source_movie_plan_fingerprint: str = ""
    source_movie_plan_lineage_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilmIR":
        if not isinstance(data, dict):
            raise ValueError("FilmIR 顶层必须是 JSON 对象。")
        _reject_provider_fields(data)
        allowed = {item.name for item in fields(cls)}
        unexpected = sorted(set(data) - allowed)
        if unexpected:
            raise ValueError("FilmIR 包含未允许字段：" + ", ".join(unexpected))
        try:
            return cls(
                ir_id=_string(data, "ir_id"),
                source_movie_plan_id=str(data.get("source_movie_plan_id", "")).strip(),
                version=_positive_int(data, "version"),
                title=_string(data, "title"),
                target_duration_seconds=_number(data, "target_duration_seconds"),
                visual_style=_string(data, "visual_style"),
                beats=tuple(_beat(item) for item in _list(data, "beats")),
                beat_timeline=tuple(
                    _beat_timeline(item) for item in _list(data, "beat_timeline")
                ),
                shots=tuple(_film_shot(item) for item in _list(data, "shots")),
                character_anchors=tuple(
                    _film_character(item) for item in _list(data, "character_anchors")
                ),
                emotion_curve=tuple(
                    _emotion_curve(item) for item in _list(data, "emotion_curve")
                ),
                tension_curve=tuple(
                    _tension_curve(item) for item in _list(data, "tension_curve")
                ),
                character_emotion_anchors=tuple(
                    _character_emotion(item)
                    for item in _list(data, "character_emotion_anchors")
                ),
                continuity_intents=tuple(
                    _continuity_intent(item)
                    for item in _list(data, "continuity_intents")
                ),
                transition_intents=tuple(
                    _transition_intent(item)
                    for item in _list(data, "transition_intents")
                ),
                narration_intents=tuple(
                    _narration_intent(item)
                    for item in _list(data, "narration_intents")
                ),
                music_intents=tuple(
                    _music_intent(item) for item in _list(data, "music_intents")
                ),
                acceptance_criteria=tuple(
                    _film_criterion(item)
                    for item in _list(data, "acceptance_criteria")
                ),
                source_story_plan_id=str(data.get("source_story_plan_id", "")).strip(),
                source_director_plan_id=str(data.get("source_director_plan_id", "")).strip(),
                source_movie_plan_version=int(data.get("source_movie_plan_version", 0) or 0),
                source_movie_plan_fingerprint=str(data.get("source_movie_plan_fingerprint", "")).strip(),
                source_movie_plan_lineage_token=str(data.get("source_movie_plan_lineage_token", "")).strip(),
                metadata=dict(data.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"FilmIR 字段不完整：{exc}") from exc


# CinematicIR is a semantic alias used by architecture documents and future
# adapters; the persisted contract remains named FilmIR.
CinematicIR = FilmIR
CinematicBeat = FilmBeat


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _string(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须是非空字符串")
    return value.strip()


def _text(data: dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} 必须是字符串")
    return value.strip()


def _number(data: dict[str, Any], key: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} 必须是数字")
    return float(value)


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} 必须是正整数")
    return value


def _list(data: dict[str, Any], key: str) -> list[Any]:
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} 必须是数组")
    return value


def _strings(data: dict[str, Any], key: str) -> tuple[str, ...]:
    return tuple(_string_value(item, key) for item in _list(data, key))


def _string_value(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须是非空字符串数组")
    return value.strip()


def _film_shot(data: Any) -> FilmShot:
    return FilmShot(
        shot_id=_string(data, "shot_id"),
        scene_id=_string(data, "scene_id"),
        order=_positive_int(data, "order"),
        duration_seconds=_number(data, "duration_seconds"),
        purpose=_string(data, "purpose"),
        visible_action=_string(data, "visible_action"),
        subject=_string(data, "subject"),
        location=_string(data, "location"),
        emotion=_string(data, "emotion"),
        camera=_string(data, "camera"),
        motion=_string(data, "motion"),
        lighting=_string(data, "lighting"),
        composition=_string(data, "composition"),
        characters=_strings(data, "characters"),
        props=_strings(data, "props"),
        narration=_text(data, "narration"),
        dialogue=_text(data, "dialogue"),
        subtitles=_text(data, "subtitles"),
        transition_in=_string(data, "transition_in"),
        transition_out=_string(data, "transition_out"),
        continuity_anchors=_strings(data, "continuity_anchors"),
        required_visual_evidence=_strings(data, "required_visual_evidence"),
        acceptance_criteria=_strings(data, "acceptance_criteria"),
        references=_strings(data, "references"),
        character_identity_anchors=_strings(data, "character_identity_anchors"),
    )


def _film_character(data: Any) -> FilmCharacterAnchor:
    return FilmCharacterAnchor(
        character_id=_string(data, "character_id"),
        name=_string(data, "name"),
        role=_string(data, "role"),
        visual_identity=_string(data, "visual_identity"),
        costume=_string(data, "costume"),
        continuity_notes=_string(data, "continuity_notes"),
        reference_keys=_strings(data, "reference_keys"),
    )


def _beat(data: Any) -> FilmBeat:
    return FilmBeat(
        beat_id=_string(data, "beat_id"),
        order=_positive_int(data, "order"),
        scene_id=_string(data, "scene_id"),
        shot_ids=_strings(data, "shot_ids"),
        duration_seconds=_number(data, "duration_seconds"),
        dramatic_purpose=_string(data, "dramatic_purpose"),
        narrative_function=_string(data, "narrative_function"),
        viewer_state_before=_string(data, "viewer_state_before"),
        viewer_state_after=_string(data, "viewer_state_after"),
        emotion=_string(data, "emotion"),
        tension_level=_number(data, "tension_level"),
        visual_focus=_string(data, "visual_focus"),
        required_audience_understanding=_string(
            data, "required_audience_understanding"
        ),
        required_evidence=_strings(data, "required_evidence"),
        character_emotional_state=_strings(data, "character_emotional_state"),
        continuity_intent=_string(data, "continuity_intent"),
        transition_intent=_string(data, "transition_intent"),
        narration_intent=_string(data, "narration_intent"),
        music_intent=_string(data, "music_intent"),
        acceptance_criteria=_strings(data, "acceptance_criteria"),
        timing_weight=_number(data, "timing_weight"),
    )


def _beat_timeline(data: Any) -> BeatTimelineEntry:
    return BeatTimelineEntry(
        beat_id=_string(data, "beat_id"),
        order=_positive_int(data, "order"),
        scene_id=_string(data, "scene_id"),
        shot_ids=_strings(data, "shot_ids"),
        start_seconds=_number(data, "start_seconds"),
        duration_seconds=_number(data, "duration_seconds"),
    )


def _emotion_curve(data: Any) -> EmotionCurvePoint:
    return EmotionCurvePoint(
        beat_id=_string(data, "beat_id"),
        label=_string(data, "label"),
        intensity=_number(data, "intensity"),
    )


def _tension_curve(data: Any) -> TensionCurvePoint:
    return TensionCurvePoint(_string(data, "beat_id"), _number(data, "level"))


def _character_emotion(data: Any) -> CharacterEmotionAnchor:
    return CharacterEmotionAnchor(
        _string(data, "beat_id"),
        _string(data, "character_id"),
        _string(data, "emotional_state"),
    )


def _continuity_intent(data: Any) -> FilmContinuityIntent:
    return FilmContinuityIntent(
        _string(data, "beat_id"),
        _string(data, "description"),
        _string(data, "required_consistency"),
    )


def _transition_intent(data: Any) -> FilmTransitionIntent:
    return FilmTransitionIntent(
        _string(data, "from_beat_id"),
        _string(data, "to_beat_id"),
        _string(data, "description"),
    )


def _narration_intent(data: Any) -> FilmNarrationIntent:
    return FilmNarrationIntent(_string(data, "beat_id"), _string(data, "description"))


def _music_intent(data: Any) -> FilmMusicIntent:
    return FilmMusicIntent(_string(data, "beat_id"), _string(data, "description"))


def _film_criterion(data: Any) -> FilmAcceptanceCriterion:
    return FilmAcceptanceCriterion(
        _string(data, "criterion_id"),
        _string(data, "target_id"),
        _string(data, "description"),
        _string(data, "severity"),
    )


def _reject_provider_fields(value: Any, path: str = "") -> None:
    forbidden = {
        "provider",
        "provider_key",
        "provider_name",
        "provider_profile",
        "model",
        "provider_prompt",
        "negative_prompt",
        "prompt",
        "api",
        "api_key",
        "endpoint",
        "payload",
        "api_payload",
        "http_payload",
        "task_id",
        "task",
        "task_url",
        "provider_task_id",
        "poll_url",
        "download_url",
        "billing",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                raise ValueError(f"FilmIR 不允许 Provider/API 字段：{path}.{key}")
            _reject_provider_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_provider_fields(child, f"{path}[{index}]")

"""Provider-neutral executable movie intermediate representation (MovieIR).

MovieIR is lowered from FilmIR.  It contains concrete shot/timeline evidence,
but no API payloads, task identifiers, URLs, billing data, or provider-specific
fields.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    shot_id: str
    order: int
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class CharacterAnchor:
    character_id: str
    name: str
    role: str
    visual_identity: str
    costume: str
    continuity_notes: str
    reference_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContinuityAnchor:
    anchor_id: str
    type: str
    description: str
    applies_to_shots: tuple[str, ...]
    required_consistency: str


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion:
    criterion_id: str
    target_id: str
    criterion_type: str
    description: str
    severity: str


@dataclass(frozen=True, slots=True)
class NarrationCue:
    shot_id: str
    text: str
    delivery: str = ""


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    shot_id: str
    text: str
    language: str = ""


@dataclass(frozen=True, slots=True)
class MusicCue:
    target_id: str
    direction: str
    intensity: str = ""
    beat_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TransitionCue:
    from_id: str
    to_id: str
    transition: str
    reason: str


@dataclass(frozen=True, slots=True)
class ShotIR:
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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MovieIR:
    ir_id: str
    source_movie_plan_id: str
    version: int
    title: str
    target_duration_seconds: float
    aspect_ratio: str
    visual_style: str
    timeline: tuple[TimelineEntry, ...]
    shots: tuple[ShotIR, ...]
    continuity_anchors: tuple[ContinuityAnchor, ...]
    character_anchors: tuple[CharacterAnchor, ...]
    narration_track: tuple[NarrationCue, ...]
    subtitle_track: tuple[SubtitleCue, ...]
    music_cues: tuple[MusicCue, ...]
    transition_cues: tuple[TransitionCue, ...]
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_film_ir_id: str = ""
    source_movie_plan_version: int = 0
    source_movie_plan_fingerprint: str = ""
    source_movie_plan_lineage_token: str = ""
    source_film_ir_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MovieIR":
        if not isinstance(data, dict):
            raise ValueError("MovieIR 顶层必须是 JSON 对象。")
        _reject_provider_fields(data)
        allowed = {item.name for item in fields(cls)}
        unexpected = sorted(set(data) - allowed)
        if unexpected:
            raise ValueError("MovieIR 包含未允许字段：" + ", ".join(unexpected))
        try:
            return cls(
                ir_id=_string(data, "ir_id"),
                # Old sessions may not have lineage fields.  Preserve the
                # object as unknown lineage so the Stale Guard can reject
                # reuse without making load itself fail.
                source_movie_plan_id=str(data.get("source_movie_plan_id", "")).strip(),
                source_film_ir_id=str(data.get("source_film_ir_id", "")).strip(),
                source_movie_plan_version=int(data.get("source_movie_plan_version", 0) or 0),
                source_movie_plan_fingerprint=str(data.get("source_movie_plan_fingerprint", "")).strip(),
                source_movie_plan_lineage_token=str(data.get("source_movie_plan_lineage_token", "")).strip(),
                source_film_ir_fingerprint=str(data.get("source_film_ir_fingerprint", "")).strip(),
                version=_positive_int(data, "version"),
                title=_string(data, "title"),
                target_duration_seconds=_number(data, "target_duration_seconds"),
                aspect_ratio=_string(data, "aspect_ratio"),
                visual_style=_string(data, "visual_style"),
                timeline=tuple(_timeline(item) for item in _list(data, "timeline")),
                shots=tuple(_shot(item) for item in _list(data, "shots")),
                continuity_anchors=tuple(
                    _continuity(item) for item in _list(data, "continuity_anchors")
                ),
                character_anchors=tuple(
                    _character(item) for item in _list(data, "character_anchors")
                ),
                narration_track=tuple(
                    _narration(item) for item in _list(data, "narration_track")
                ),
                subtitle_track=tuple(
                    _subtitle(item) for item in _list(data, "subtitle_track")
                ),
                music_cues=tuple(_music(item) for item in _list(data, "music_cues")),
                transition_cues=tuple(
                    _transition(item) for item in _list(data, "transition_cues")
                ),
                acceptance_criteria=tuple(
                    _criterion(item) for item in _list(data, "acceptance_criteria")
                ),
                metadata=dict(data.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"MovieIR 字段不完整：{exc}") from exc


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


def _timeline(data: Any) -> TimelineEntry:
    return TimelineEntry(
        _string(data, "shot_id"),
        _positive_int(data, "order"),
        _number(data, "start_seconds"),
        _number(data, "duration_seconds"),
    )


def _character(data: Any) -> CharacterAnchor:
    return CharacterAnchor(
        _string(data, "character_id"),
        _string(data, "name"),
        _string(data, "role"),
        _string(data, "visual_identity"),
        _string(data, "costume"),
        _string(data, "continuity_notes"),
        _strings(data, "reference_keys"),
    )


def _continuity(data: Any) -> ContinuityAnchor:
    return ContinuityAnchor(
        _string(data, "anchor_id"),
        _string(data, "type"),
        _string(data, "description"),
        _strings(data, "applies_to_shots"),
        _string(data, "required_consistency"),
    )


def _criterion(data: Any) -> AcceptanceCriterion:
    return AcceptanceCriterion(
        _string(data, "criterion_id"),
        _string(data, "target_id"),
        _string(data, "criterion_type"),
        _string(data, "description"),
        _string(data, "severity"),
    )


def _narration(data: Any) -> NarrationCue:
    return NarrationCue(_string(data, "shot_id"), _string(data, "text"), str(data.get("delivery", "")))


def _subtitle(data: Any) -> SubtitleCue:
    return SubtitleCue(_string(data, "shot_id"), _string(data, "text"), str(data.get("language", "")))


def _music(data: Any) -> MusicCue:
    return MusicCue(
        _string(data, "target_id"),
        _string(data, "direction"),
        str(data.get("intensity", "")),
        _strings(data, "beat_notes"),
    )


def _transition(data: Any) -> TransitionCue:
    return TransitionCue(
        _string(data, "from_id"),
        _string(data, "to_id"),
        _string(data, "transition"),
        _string(data, "reason"),
    )


def _shot(data: Any) -> ShotIR:
    return ShotIR(
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
        narration=str(data.get("narration", "")),
        dialogue=str(data.get("dialogue", "")),
        subtitles=str(data.get("subtitles", "")),
        transition_in=_string(data, "transition_in"),
        transition_out=_string(data, "transition_out"),
        continuity_anchors=_strings(data, "continuity_anchors"),
        required_visual_evidence=_strings(data, "required_visual_evidence"),
        acceptance_criteria=_strings(data, "acceptance_criteria"),
        references=_strings(data, "references"),
        character_identity_anchors=_strings(data, "character_identity_anchors"),
        metadata=dict(data.get("metadata", {})),
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
                raise ValueError(f"MovieIR 不允许 Provider/API 字段：{path}.{key}")
            _reject_provider_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_provider_fields(child, f"{path}[{index}]")

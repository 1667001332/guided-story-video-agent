from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Stage(str, Enum):
    COLLECTING = "collecting"
    OUTLINE_REVIEW = "outline_review"
    DETAILING = "detailing"
    SCRIPT_REVIEW = "script_review"
    STORYBOARD_REVIEW = "storyboard_review"
    RENDER_READY = "render_ready"
    COMPLETED = "completed"


@dataclass(slots=True)
class CreativeBrief:
    target_seconds: int = 45
    genre: str = "短片"
    visual_style: str = "电影感写实"
    language: str = "zh-CN"
    narration_enabled: bool = True

    def validate(self) -> None:
        if not 30 <= int(self.target_seconds) <= 60:
            raise ValueError("目标视频时长必须在 30 到 60 秒之间。")


@dataclass(slots=True)
class StoryFacts:
    premise: str = ""
    genre: str = ""
    tone: str = ""
    theme: str = ""
    audience: str = ""
    opening: str = ""
    protagonist: str = ""
    protagonist_goal: str = ""
    motivation: str = ""
    conflict: str = ""
    stakes: str = ""
    development: str = ""
    turning_point: str = ""
    ending: str = ""
    character_visuals: str = ""
    scene_details: str = ""
    props: str = ""
    narration_style: str = ""
    dialogue_style: str = ""
    camera_style: str = ""
    visual_anchors: str = ""
    transitions: str = ""

    def missing_outline_fields(self) -> list[str]:
        missing = [
            name
            for name in ("opening", "protagonist_goal", "conflict", "ending")
            if not getattr(self, name).strip()
        ]
        if not self.development.strip() and not self.turning_point.strip():
            missing.append("development_or_turning_point")
        return missing

    def missing_detail_fields(self) -> list[str]:
        return [
            name
            for name in (
                "character_visuals",
                "scene_details",
                "props",
                "narration_style",
                "transitions",
            )
            if not getattr(self, name).strip()
        ]


@dataclass(slots=True)
class CreatorContribution:
    turn_id: int
    text: str
    source: str = "human"
    extracted_facts: dict[str, str] = field(default_factory=dict)
    fact_evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class FactEvidence:
    field: str
    value: str
    evidence: str
    confidence: float = 1.0


@dataclass(slots=True)
class StoryConflict:
    field: str
    existing_value: str
    proposed_value: str
    reason: str


@dataclass(slots=True)
class CreativeSuggestion:
    suggestion_id: str
    label: str
    content: str
    target_field: str


@dataclass(slots=True)
class ReadinessReport:
    score: float
    missing_critical_fields: list[str] = field(default_factory=list)
    unresolved_conflicts: list[StoryConflict] = field(default_factory=list)
    recommended_action: str = "continue"


@dataclass(slots=True)
class GuideTurnResult:
    accepted: bool
    assistant_message: str
    next_question: str
    suggestions: list[str]
    valid_turns: int
    missing_fields: list[str]
    can_build_outline: bool
    extracted_facts: list[FactEvidence] = field(default_factory=list)
    conflicts: list[StoryConflict] = field(default_factory=list)
    readiness_score: float = 0.0
    missing_critical_fields: list[str] = field(default_factory=list)
    recommended_action: str = "continue"
    used_fallback: bool = False


@dataclass(slots=True)
class StoryBeat:
    beat_id: int
    purpose: str
    event: str
    causal_link: str
    emotional_change: str
    duration: int
    source_turn_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class StoryOutline:
    title: str
    logline: str
    opening: str
    protagonist_goal: str
    conflict: str
    development: str
    turning_point: str
    ending: str
    source_turn_ids: list[int]
    confirmed: bool = False
    beats: list[StoryBeat] = field(default_factory=list)


@dataclass(slots=True)
class StoryScene:
    scene_id: int
    title: str
    location: str
    time_of_day: str
    characters: list[str]
    action: str
    narration: str
    duration: int
    dialogue: str = ""
    props: list[str] = field(default_factory=list)
    visible_action: str = ""
    start_state: str = ""
    end_state: str = ""
    emotional_change: str = ""


@dataclass(slots=True)
class StoryScript:
    title: str
    target_seconds: int
    scenes: list[StoryScene]
    confirmed: bool = False

    @property
    def total_duration(self) -> int:
        return sum(scene.duration for scene in self.scenes)


@dataclass(slots=True)
class StoryboardShot:
    shot_id: int
    scene_id: int
    duration: int
    character: str
    location: str
    visual: str
    action: str
    camera: str
    lighting: str
    mood: str
    narration: str
    video_prompt: str
    negative_prompt: str
    aspect_ratio: str = "16:9"
    continuity_notes: list[str] = field(default_factory=list)
    shot_purpose: str = ""
    composition: str = ""
    camera_movement: str = ""
    start_frame: str = ""
    end_frame: str = ""
    visual_anchors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VideoArtifact:
    artifact_id: str
    shot_id: int
    provider: str
    model: str
    status: str
    local_path: str
    remote_url: str
    duration: int
    prompt: str
    created_at: str
    request_id: str | None = None
    attempt: int = 1
    error_message: str = ""


@dataclass(slots=True)
class StoryboardPlan:
    title: str
    target_seconds: int
    shots: list[StoryboardShot]
    narration_text: str
    confirmed: bool = False
    audio_path: str = ""
    subtitle_path: str = ""
    artifacts: list[VideoArtifact] = field(default_factory=list)

    @property
    def total_duration(self) -> int:
        return sum(shot.duration for shot in self.shots)


@dataclass(slots=True)
class RenderManifest:
    status: str
    output_dir: str
    generated_shots: list[int] = field(default_factory=list)
    reused_shots: list[int] = field(default_factory=list)
    failed_shots: list[int] = field(default_factory=list)
    artifacts: list[VideoArtifact] = field(default_factory=list)
    final_video_path: str = ""
    audio_path: str = ""
    subtitle_path: str = ""
    error: str = ""


@dataclass(slots=True)
class ArtifactRevision:
    artifact_type: str
    version: int
    payload: dict[str, Any]
    parent_version: int | None = None
    user_feedback: str = ""
    source_turn_ids: list[int] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    confirmed: bool = False


@dataclass(slots=True)
class ArtifactReview:
    artifact_type: str
    hard_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def can_confirm(self) -> bool:
        return not self.hard_errors


def to_plain_data(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    return value

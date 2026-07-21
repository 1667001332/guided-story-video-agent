from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    opening: str = ""
    protagonist_goal: str = ""
    conflict: str = ""
    development: str = ""
    turning_point: str = ""
    ending: str = ""
    character_visuals: str = ""
    scene_details: str = ""
    props: str = ""
    narration_style: str = ""
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


@dataclass(slots=True)
class GuideTurnResult:
    accepted: bool
    assistant_message: str
    next_question: str
    suggestions: list[str]
    valid_turns: int
    missing_fields: list[str]
    can_build_outline: bool


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

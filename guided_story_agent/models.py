from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Stage(str, Enum):
    IDEATING = "ideating"
    STORY_REVIEW = "story_review"
    DRAFT_REVIEW = "draft_review"
    COLLECTING = "collecting"
    OUTLINE_REVIEW = "outline_review"
    DETAILING = "detailing"
    SCRIPT_REVIEW = "script_review"
    STORYBOARD_REVIEW = "storyboard_review"
    RENDER_READY = "render_ready"
    COMPLETED = "completed"


@dataclass(slots=True)
class CreativeBrief:
    target_seconds: int | None = None
    duration_mode: str = "auto"
    resolved_target_seconds: int | None = None
    genre: str = "短片"
    visual_style: str = "电影感写实"
    language: str = "zh-CN"
    narration_enabled: bool = True

    def __post_init__(self) -> None:
        # v0.4.1 and older only stored target_seconds. Preserve those saved
        # values as an explicit custom duration during migration.
        if self.target_seconds is not None and self.duration_mode == "auto":
            self.duration_mode = "custom"

    def validate(self) -> None:
        if self.duration_mode not in {"auto", "custom"}:
            raise ValueError("时长模式必须是 auto 或 custom。")
        if self.duration_mode == "custom" and self.target_seconds is None:
            raise ValueError("自定义时长模式必须填写目标秒数。")
        for value in (self.target_seconds, self.resolved_target_seconds):
            if value is not None and not 15 <= int(value) <= 300:
                raise ValueError("目标视频时长必须在 15 到 300 秒之间。")


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
class IdeaCard:
    idea_id: str
    title: str
    logline: str
    hook: str
    protagonist: str
    central_conflict: str
    tone: str
    ending_direction: str
    source_idea_ids: list[str] = field(default_factory=list)
    generation_kind: str = "diverge"

    @property
    def fingerprint(self) -> str:
        return " ".join(f"{self.title} {self.logline} {self.hook}".lower().split())


@dataclass(slots=True)
class IdeaBatch:
    round: int
    cards: list[IdeaCard]
    recommended_id: str = ""
    feedback: str = ""
    generation_kind: str = "diverge"


@dataclass(slots=True)
class ElementOption:
    option_id: str
    kind: str
    title: str
    content: str
    source_idea_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ElementPalette:
    options: dict[str, list[ElementOption]] = field(default_factory=dict)


@dataclass(slots=True)
class SelectionState:
    selected_idea_ids: list[str] = field(default_factory=list)
    selected_elements: dict[str, str] = field(default_factory=dict)
    can_generate_story: bool = False
    can_generate_draft: bool = False


@dataclass(slots=True)
class SourceAttribution:
    field: str
    source_type: str
    value: str
    source_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StoryCharacter:
    name: str
    description: str
    visual_identity: str = ""


@dataclass(slots=True)
class StoryLocation:
    name: str
    description: str
    visual_identity: str = ""


@dataclass(slots=True)
class StoryDraft:
    title: str
    logline: str
    story_text: str
    characters: list[StoryCharacter] = field(default_factory=list)
    locations: list[StoryLocation] = field(default_factory=list)
    tone: str = ""
    theme: str = ""
    core_conflict: str = ""
    ending: str = ""
    visual_anchors: list[str] = field(default_factory=list)
    field_sources: dict[str, SourceAttribution] = field(default_factory=dict)
    ai_filled_fields: list[str] = field(default_factory=list)
    version: int = 1
    confirmed: bool = False


@dataclass(slots=True)
class DraftBundle:
    outline: StoryOutline
    script: StoryScript
    field_sources: dict[str, SourceAttribution] = field(default_factory=dict)
    ai_filled_fields: list[str] = field(default_factory=list)
    version: int = 1


@dataclass(slots=True)
class IdeationTurnResult:
    message: str
    batch: IdeaBatch | None
    selection: SelectionState
    available_actions: list[str] = field(default_factory=list)
    used_fallback: bool = False


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
class VisualReference:
    """One confirmed image and its explicit, provider-independent purpose."""

    reference_id: str
    path: str
    usage: str
    content_digest: str = ""
    content_summary: str = ""
    confirmed: bool = False
    binding_kind: str = ""
    binding_id: str = ""


@dataclass(slots=True)
class VisualAsset:
    """A planned identity anchor that can later receive one or more reference images."""

    asset_id: str
    kind: str
    name: str
    description: str
    reference_images: list[str] = field(default_factory=list)
    references: list[VisualReference] = field(default_factory=list)


@dataclass(slots=True)
class VisualBible:
    """Provider-independent visual source of truth for every generated shot."""

    visual_style: str = "电影感写实"
    color_palette: str = "统一、克制的电影色彩"
    lighting_rules: str = "光源方向和时段连续，避免镜头间突变"
    camera_language: str = "镜头服务于动作和情绪，不为变化而变化"
    assets: list[VisualAsset] = field(default_factory=list)
    continuity_rules: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ContinuityState:
    """Structured visual and narrative state at one side of a shot boundary."""

    character_appearance: dict[str, str] = field(default_factory=dict)
    character_clothing: dict[str, str] = field(default_factory=dict)
    character_positions: dict[str, str] = field(default_factory=dict)
    character_emotions: dict[str, str] = field(default_factory=dict)
    character_knowledge: dict[str, list[str]] = field(default_factory=dict)
    character_injuries: dict[str, str] = field(default_factory=dict)
    character_held_props: dict[str, list[str]] = field(default_factory=dict)
    prop_positions: dict[str, str] = field(default_factory=dict)
    location: str = ""
    time_of_day: str = ""
    weather: str = ""
    key_light_direction: str = ""


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Visual-input features that a concrete provider adapter can actually use."""

    supports_text_to_video: bool = True
    supports_image_to_video: bool = False
    supports_reference_images: bool = False
    supports_seed: bool = False
    requires_public_image_url: bool = False


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
    shot_kind: str = "action"
    duration_reason: str = ""
    duration_weight: float = 0.0
    estimated_duration: float = 0.0
    first_frame_prompt: str = ""
    motion_prompt: str = ""
    end_frame_prompt: str = ""
    reference_asset_ids: list[str] = field(default_factory=list)
    confirmed_visual_inputs: list[VisualReference] = field(default_factory=list)
    reference_image_paths: list[str] = field(default_factory=list)
    initial_frame_source_path: str = ""
    initial_frame_path: str = ""
    initial_frame_url: str = ""
    previous_shot_id: int | None = None
    continuity_mode: str = "independent"
    continuity_state: dict[str, Any] = field(default_factory=dict)
    continuity_start_state: ContinuityState = field(default_factory=ContinuityState)
    continuity_end_state: ContinuityState = field(default_factory=ContinuityState)
    continuity_diagnostics: list[str] = field(default_factory=list)
    seed: int | None = None
    generated_first_frame_path: str = ""
    generated_last_frame_path: str = ""


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
    reference_image_paths: list[str] = field(default_factory=list)
    confirmed_visual_inputs: list[VisualReference] = field(default_factory=list)
    initial_frame_source_path: str = ""
    initial_frame_path: str = ""
    initial_frame_url: str = ""
    previous_shot_id: int | None = None
    continuity_mode: str = "independent"
    input_fingerprint: str = ""
    seed: int | None = None
    generated_first_frame_path: str = ""
    generated_last_frame_path: str = ""
    published_last_frame_path: str = ""
    published_last_frame_url: str = ""
    continuity_diagnostics: list[str] = field(default_factory=list)
    used_unreferenced_fallback: bool = False


@dataclass(slots=True)
class StoryboardPlan:
    title: str
    target_seconds: int
    shots: list[StoryboardShot]
    narration_text: str
    visual_bible: VisualBible = field(default_factory=VisualBible)
    base_seed: int = 0
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
    render_run_id: str = ""
    generated_shots: list[int] = field(default_factory=list)
    reused_shots: list[int] = field(default_factory=list)
    failed_shots: list[int] = field(default_factory=list)
    dependency_failed_shots: list[int] = field(default_factory=list)
    unreferenced_fallback_shots: list[int] = field(default_factory=list)
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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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

"""Provider-independent V2 planning contracts.

These dataclasses describe what the DirectorAgent must decide.  They contain
no duration allocation, prompt construction, or provider-specific repair code.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


def _required(value: str, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


@dataclass(frozen=True, slots=True)
class CreativeBrief:
    """Minimal user intent passed to the director.

    Duration is a request, not a scene allocation instruction.  The director
    returns the complete timing plan and the validator only checks it.
    """

    target_duration_seconds: int
    video_type: str
    visual_style: str
    audience: str
    narration_requirement: str = "optional"
    output_format: str = "mp4"

    def validate(self) -> None:
        if (
            isinstance(self.target_duration_seconds, bool)
            or not isinstance(self.target_duration_seconds, (int, float))
            or not math.isfinite(float(self.target_duration_seconds))
            or self.target_duration_seconds <= 0
        ):
            raise ValueError("target_duration_seconds must be a positive integer")
        _required(self.video_type, "video_type")
        _required(self.visual_style, "visual_style")
        _required(self.audience, "audience")
        _required(self.narration_requirement, "narration_requirement")
        _required(self.output_format, "output_format")


@dataclass(frozen=True, slots=True)
class Story:
    title: str
    logline: str
    synopsis: str
    theme: str = ""
    ending: str = ""


@dataclass(frozen=True, slots=True)
class CharacterSheetEntry:
    character_id: str
    name: str
    identity: str
    role: str = ""
    visual_signature: str = ""
    performance_notes: str = ""
    reference_keys: tuple[str, ...] = ()
    costume: str = ""


@dataclass(frozen=True, slots=True)
class CharacterSheet:
    characters: tuple[CharacterSheetEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class StoryCharacterGoal:
    """A story-level goal, independent from how it is filmed."""

    character_id: str
    goal: str
    obstacle: str = ""
    outcome: str = ""


@dataclass(frozen=True, slots=True)
class StoryPlan:
    """The narrative contract: what happens and why it matters.

    This layer deliberately contains no camera, prompt, provider, or runtime
    data.  The legacy ``Story``/``Script`` fields remain on ``MoviePlan`` for
    compatibility; this object is the explicit story boundary for new code.
    """

    title: str = ""
    logline: str = ""
    synopsis: str = ""
    theme: str = ""
    ending: str = ""
    characters: tuple[CharacterSheetEntry, ...] = ()
    events: tuple[str, ...] = ()
    causality: tuple[str, ...] = ()
    conflict: str = ""
    stakes: str = ""
    resolution: str = ""
    story_beats: tuple[str, ...] = ()
    character_goals: tuple[StoryCharacterGoal, ...] = ()

    @classmethod
    def from_legacy(
        cls,
        story: Story,
        script: "Script",
        character_sheet: CharacterSheet,
    ) -> "StoryPlan":
        """Project legacy story fields without inventing new story content."""

        scenes = tuple(script.scenes)
        events = tuple((scene.action or scene.goal).strip() for scene in scenes)
        beats = tuple(scene.goal.strip() for scene in scenes)
        character_ids = tuple(
            character_id
            for scene in scenes
            for character_id in scene.characters
        )
        goals = tuple(
            StoryCharacterGoal(character_id, goal)
            for character_id, goal in _first_character_goals(scenes, character_ids)
        )
        return cls(
            title=story.title,
            logline=story.logline,
            synopsis=story.synopsis,
            theme=story.theme,
            ending=story.ending,
            characters=tuple(character_sheet.characters),
            events=events,
            # Legacy MoviePlan has no explicit causality/conflict/stakes
            # contract.  Leave those fields empty instead of guessing them.
            causality=(),
            conflict="",
            stakes="",
            resolution=story.ending,
            story_beats=beats,
            character_goals=goals,
        )


@dataclass(frozen=True, slots=True)
class ScriptScene:
    """A complete LLM-authored scene contract.

    ``minimum_duration`` and ``estimated_duration_weight`` are declarations
    from the director.  Python may reject them, but must never recompute them.
    """

    scene_id: str
    goal: str
    emotion: str
    importance: str
    estimated_duration_weight: float
    minimum_duration: float
    camera_language: str
    motion_type: str
    dialogue: str
    narration: str
    characters: tuple[str, ...]
    location: str
    continuity_requirements: tuple[str, ...]
    transition: str
    timing_reason: str
    action: str = ""


@dataclass(frozen=True, slots=True)
class Script:
    title: str
    logline: str
    scenes: tuple[ScriptScene, ...]
    confirmed: bool = False
    revision: int = 1


@dataclass(frozen=True, slots=True)
class ScenePlanEntry:
    scene_id: str
    visual_intent: str
    blocking: str
    props: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScenePlan:
    scenes: tuple[ScenePlanEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class CameraInstruction:
    scene_id: str
    shot_size: str
    angle: str
    lens: str
    movement: str
    composition: str
    language: str


@dataclass(frozen=True, slots=True)
class CameraPlan:
    instructions: tuple[CameraInstruction, ...] = ()


@dataclass(frozen=True, slots=True)
class TimingEntry:
    scene_id: str
    duration_seconds: float
    reason: str


@dataclass(frozen=True, slots=True)
class TimingPlan:
    target_duration_seconds: float
    entries: tuple[TimingEntry, ...]

    @property
    def declared_total_seconds(self) -> float:
        return sum(entry.duration_seconds for entry in self.entries)


@dataclass(frozen=True, slots=True)
class ContinuityEntry:
    scene_id: str
    requirements: tuple[str, ...]
    prior_state: str = ""
    resulting_state: str = ""


@dataclass(frozen=True, slots=True)
class ContinuityPlan:
    entries: tuple[ContinuityEntry, ...] = ()
    global_rules: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NarrationSegment:
    scene_id: str
    text: str
    delivery: str = ""


@dataclass(frozen=True, slots=True)
class NarrationPlan:
    enabled: bool
    language: str
    style: str
    segments: tuple[NarrationSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class MusicPlan:
    direction: str
    intensity: str = ""
    beat_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TransitionInstruction:
    from_scene_id: str
    to_scene_id: str
    transition: str
    reason: str


@dataclass(frozen=True, slots=True)
class TransitionPlan:
    transitions: tuple[TransitionInstruction, ...] = ()


@dataclass(frozen=True, slots=True)
class ShotPlan:
    """Director-authored shot-level execution decision.

    The IR builder may translate this structure, but it may never invent a
    missing shot, duration, action, or acceptance requirement.
    """

    shot_id: str
    scene_id: str
    order: int
    duration_seconds: float
    purpose: str
    visible_action: str
    subject: str
    camera_intent: str
    motion_intent: str
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


@dataclass(frozen=True, slots=True)
class FilmBeatPlan:
    """Director-authored film-language decision for a cinematic beat.

    A beat is not a Provider request and it is not a replacement for a shot
    plan.  It explains what the audience must understand and feel while
    pointing at the already planned scene/shot evidence.
    """

    beat_id: str
    order: int
    scene_id: str
    shot_ids: tuple[str, ...]
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
class EmotionPoint:
    """A director-authored point on the film's emotional curve."""

    label: str
    intensity: float
    scene_id: str = ""


@dataclass(frozen=True, slots=True)
class DirectorPlan:
    """The audience-experience contract: how the story should be felt."""

    pacing_strategy: str = ""
    suspense_strategy: str = ""
    audience_knowledge: str = ""
    emotional_intention: str = ""
    reveal_timing: str = ""
    withholding_strategy: str = ""
    visual_motif_strategy: str = ""
    silence_pause_intention: str = ""
    climax_emphasis: str = ""
    ending_tone: str = ""

    @classmethod
    def from_legacy(
        cls,
        story: Story,
        script: Script,
        *,
        visual_style: str,
        film_beats: tuple[FilmBeatPlan, ...],
    ) -> "DirectorPlan":
        """Project already-declared cinematic fields without local decisions."""

        emotions = tuple(scene.emotion.strip() for scene in script.scenes if scene.emotion.strip())
        audience = tuple(
            beat.required_audience_understanding.strip()
            for beat in film_beats
            if beat.required_audience_understanding.strip()
        )
        reveals = tuple(
            beat.transition_intent.strip()
            for beat in film_beats
            if beat.transition_intent.strip()
        )
        climax = film_beats[-1].dramatic_purpose if film_beats else ""
        return cls(
            audience_knowledge="; ".join(audience),
            emotional_intention="; ".join(emotions),
            reveal_timing="; ".join(reveals),
            visual_motif_strategy=visual_style,
            climax_emphasis=climax,
            ending_tone=story.ending,
        )


def _first_character_goals(
    scenes: tuple[ScriptScene, ...],
    character_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Return one deterministic legacy goal projection per character."""

    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for character_id in character_ids:
        if character_id in seen:
            continue
        seen.add(character_id)
        goal = next(
            (scene.goal.strip() for scene in scenes if character_id in scene.characters),
            "",
        )
        result.append((character_id, goal))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class MoviePlan:
    """Compatibility aggregate and single source of truth for the movie.

    ``story_plan`` and ``director_plan`` are the explicit Phase 4A layers;
    the existing fields remain until downstream lowering is migrated.
    """

    plan_id: str
    story: Story
    script: Script
    scene_plan: ScenePlan
    camera_plan: CameraPlan
    timing_plan: TimingPlan
    continuity_plan: ContinuityPlan
    character_sheet: CharacterSheet
    narration_plan: NarrationPlan
    music_plan: MusicPlan
    transition_plan: TransitionPlan
    shot_plan: tuple[ShotPlan, ...] = ()
    film_beats: tuple[FilmBeatPlan, ...] = ()
    visual_style: str = ""
    emotion_curve: tuple[EmotionPoint, ...] = ()
    review_criteria: tuple[str, ...] = ()
    revision: int = 1
    confirmed: bool = False
    # Phase 4A explicit layers.  ``None`` means migrate the legacy fields in
    # ``__post_init__``; callers may provide authored plans explicitly.
    story_plan: StoryPlan | None = None
    director_plan: DirectorPlan | None = None

    def __post_init__(self) -> None:
        if self.story_plan is None:
            object.__setattr__(
                self,
                "story_plan",
                StoryPlan.from_legacy(self.story, self.script, self.character_sheet),
            )
        if self.director_plan is None:
            object.__setattr__(
                self,
                "director_plan",
                DirectorPlan.from_legacy(
                    self.story,
                    self.script,
                    visual_style=self.visual_style,
                    film_beats=self.film_beats,
                ),
            )

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.script.scenes)

def as_plain_data(value: Any) -> Any:
    """Serialize V2 dataclasses without importing legacy model helpers."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: as_plain_data(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [as_plain_data(item) for item in value]
    if isinstance(value, list):
        return [as_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): as_plain_data(item) for key, item in value.items()}
    return value

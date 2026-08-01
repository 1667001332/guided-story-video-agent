"""OpenAI-compatible DirectorAgent adapter for the V2 MoviePlan contract."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Callable

from ..provider_config import TextProviderConfig
from .models import (
    CameraInstruction,
    CameraPlan,
    CharacterSheet,
    CharacterSheetEntry,
    ContinuityEntry,
    ContinuityPlan,
    CreativeBrief,
    EmotionPoint,
    FilmBeatPlan,
    MoviePlan,
    MusicPlan,
    NarrationPlan,
    NarrationSegment,
    ScenePlan,
    ScenePlanEntry,
    Script,
    ScriptScene,
    ShotPlan,
    Story,
    StoryCharacterGoal,
    StoryPlan,
    TimingEntry,
    TimingPlan,
    TransitionInstruction,
    TransitionPlan,
    DirectorPlan,
    as_plain_data,
)


class DirectorGenerationError(ValueError):
    """The model response cannot be parsed as a complete MoviePlan."""


_PLAN_KEYS = {
    "plan_id",
    "story",
    "script",
    "scene_plan",
    "camera_plan",
    "timing_plan",
    "continuity_plan",
    "character_sheet",
    "narration_plan",
    "music_plan",
    "transition_plan",
    "shot_plan",
    "film_beats",
    "visual_style",
    "emotion_curve",
    "review_criteria",
    "revision",
    "confirmed",
    "story_plan",
    "director_plan",
}

_STORY_PLAN_KEYS = {
    "title",
    "logline",
    "synopsis",
    "theme",
    "ending",
    "characters",
    "events",
    "causality",
    "conflict",
    "stakes",
    "resolution",
    "story_beats",
    "character_goals",
}
_DIRECTOR_PLAN_KEYS = {
    "pacing_strategy",
    "suspense_strategy",
    "audience_knowledge",
    "emotional_intention",
    "reveal_timing",
    "withholding_strategy",
    "visual_motif_strategy",
    "silence_pause_intention",
    "climax_emphasis",
    "ending_tone",
}
_PROMPT_STUFFING_TERMS = (
    "masterpiece",
    "best quality",
    "ultra realistic",
    "ultra-realistic",
    "8k",
    "4k highly detailed",
)


class OpenAIDirectorAgent:
    """A strict JSON DirectorAgent with no local content repair."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        prompt_path: str | Path | None = None,
        completion_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        config_source: str = "constructor",
        json_mode: str = "auto",
    ) -> None:
        self.client = client
        self.model = model
        self.prompt_path = Path(prompt_path) if prompt_path else _default_prompt_path()
        self.completion_fn = completion_fn
        self.provider_name = "openai_director"
        self.config_source = config_source
        self.json_mode = json_mode

    @classmethod
    def from_env(cls) -> "OpenAIDirectorAgent":
        config = TextProviderConfig.from_env()
        if not config.configured:
            raise DirectorGenerationError(config.error or "文本模型未配置")
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {
                "api_key": config.api_key,
                "timeout": config.timeout,
                "max_retries": 0,
            }
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return cls(
                OpenAI(**kwargs),
                config.model,
                config_source=config.source,
                json_mode=config.json_mode,
            )
        except Exception as exc:
            raise DirectorGenerationError(f"初始化 DirectorAgent 失败：{exc}") from exc

    def create_movie_plan(
        self,
        brief: CreativeBrief,
        direction: str,
        *,
        feedback: str = "",
    ) -> MoviePlan:
        payload = {
            "creative_brief": as_plain_data(brief),
            "direction": direction.strip(),
            "feedback_from_previous_attempt": feedback.strip(),
        }
        return movie_plan_from_data(self._complete(payload))

    def revise_movie_plan(
        self,
        brief: CreativeBrief,
        plan: MoviePlan,
        feedback: str,
    ) -> MoviePlan:
        payload = {
            "creative_brief": as_plain_data(brief),
            "current_movie_plan": as_plain_data(plan),
            "revision_request": feedback.strip(),
        }
        return movie_plan_from_data(self._complete(payload))

    def _complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.completion_fn is not None:
            result = self.completion_fn(payload)
            if not isinstance(result, dict):
                raise DirectorGenerationError("completion_fn 必须返回 JSON 对象")
            return result
        if self.client is None:
            raise DirectorGenerationError("DirectorAgent 没有可用的文本模型客户端")
        system = self.prompt_path.read_text(encoding="utf-8")
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            "temperature": 0.7,
        }
        if self.json_mode in {"auto", "json_object"}:
            request["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            if "response_format" not in request:
                raise DirectorGenerationError(f"DirectorAgent 调用失败：{exc}") from exc
            request.pop("response_format", None)
            try:
                response = self.client.chat.completions.create(**request)
            except Exception as retry_exc:
                raise DirectorGenerationError(
                    f"DirectorAgent 调用失败：{retry_exc}"
                ) from retry_exc
        try:
            content = response.choices[0].message.content
            data = json.loads(content)
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise DirectorGenerationError("DirectorAgent 返回的不是有效 JSON MoviePlan") from exc
        if not isinstance(data, dict):
            raise DirectorGenerationError("DirectorAgent 顶层输出必须是 JSON 对象")
        return data


class RuleBasedDirectorAgent:
    """Deterministic offline fixture; never used as a production LLM fallback."""

    def create_movie_plan(
        self,
        brief: CreativeBrief,
        direction: str,
        *,
        feedback: str = "",
    ) -> MoviePlan:
        del feedback
        brief.validate()
        subject = direction.strip() or "一个尚未命名的短片"
        scene_id = "scene-1"
        plan_id = "offline-" + hashlib.sha1(subject.encode("utf-8")).hexdigest()[:12]
        return MoviePlan(
            plan_id=plan_id,
            story=Story(subject, subject, f"围绕“{subject}”展开一个完整的连续事件。"),
            script=Script(
                subject,
                subject,
                (
                    ScriptScene(
                        scene_id=scene_id,
                        goal="完成方向中的核心事件",
                        emotion="克制的期待",
                        importance="high",
                        estimated_duration_weight=1.0,
                        minimum_duration=float(brief.target_duration_seconds),
                        camera_language="连续观察",
                        motion_type="motivated movement",
                        dialogue="",
                        narration="",
                        characters=("protagonist",),
                        location="由导演方向决定的主要空间",
                        continuity_requirements=("人物身份和空间关系保持一致",),
                        transition="opening",
                        timing_reason="完整事件需要一个连续时间窗口",
                        action=f"主角围绕“{subject}”完成一个可见的连续行动",
                    ),
                ),
            ),
            scene_plan=ScenePlan(
                (ScenePlanEntry(scene_id, "呈现核心事件", "主角在主要空间内完成连续行动"),)
            ),
            camera_plan=CameraPlan(
                (
                    CameraInstruction(
                        scene_id,
                        "medium-wide",
                        "eye-level",
                        "natural",
                        "motivated",
                        "保持主体与环境关系",
                        "连续观察",
                    ),
                )
            ),
            timing_plan=TimingPlan(
                float(brief.target_duration_seconds),
                (TimingEntry(scene_id, float(brief.target_duration_seconds), "完整连续事件"),),
            ),
            continuity_plan=ContinuityPlan(
                (ContinuityEntry(scene_id, ("人物和空间状态保持一致",)),),
                ("不跳变人物身份", "不跳变空间关系"),
            ),
            character_sheet=CharacterSheet(
                (
                    CharacterSheetEntry(
                        "protagonist",
                        "主角",
                        "由方向定义的行动者",
                        "protagonist",
                        "由方向定义的稳定外观",
                        "动作连续、表演克制",
                        ("protagonist",),
                        "由导演方向确定的服装",
                    ),
                )
            ),
            narration_plan=NarrationPlan(False, "zh-CN", "none"),
            music_plan=MusicPlan("与核心事件同步的低干预环境音乐"),
            transition_plan=TransitionPlan(),
            shot_plan=(
                ShotPlan(
                    shot_id="shot-1",
                    scene_id=scene_id,
                    order=1,
                    duration_seconds=float(brief.target_duration_seconds),
                    purpose="完整呈现核心事件",
                    visible_action=f"主角围绕“{subject}”完成一个可见的连续行动",
                    subject="主角与关键行动",
                    camera_intent="连续观察并保持主体与环境关系",
                    motion_intent="motivated movement",
                    lighting="雨夜环境光与车站实用光连续",
                    composition="主体与环境关系清晰",
                    characters=("protagonist",),
                    props=("方向中的关键物件",),
                    narration="",
                    dialogue="",
                    subtitles="",
                    transition_in="opening",
                    transition_out="ending",
                    continuity_anchors=("人物身份和空间关系保持一致",),
                    required_visual_evidence=("核心行动清晰可见", "人物与空间关系稳定"),
                    acceptance_criteria=("观众能识别核心行动", "人物身份连续"),
                ),
            ),
            film_beats=(
                FilmBeatPlan(
                    beat_id="beat-1",
                    order=1,
                    scene_id=scene_id,
                    shot_ids=("shot-1",),
                    dramatic_purpose="让观众理解核心事件并保持期待",
                    narrative_function="setup",
                    viewer_state_before="不知道方向中的核心问题",
                    viewer_state_after="理解核心行动并期待结果",
                    emotion="克制的期待",
                    tension_level=0.5,
                    visual_focus="主角与方向中的关键物件",
                    required_audience_understanding="观众能识别主角正在完成的核心行动",
                    required_evidence=("核心行动清晰可见", "人物与空间关系稳定"),
                    character_emotional_state=("protagonist:克制的期待",),
                    continuity_intent="人物身份、服装与空间关系在整个事件中保持一致",
                    transition_intent="从建立情境自然过渡到事件结束",
                    narration_intent="无旁白，以可见行动传达信息",
                    music_intent="低干预环境音乐托住期待，不遮盖行动",
                    acceptance_criteria=("观众理解核心行动", "人物状态连续"),
                    timing_weight=1.0,
                ),
            ),
            visual_style=brief.visual_style,
            emotion_curve=(EmotionPoint("期待", 0.5, scene_id),),
            review_criteria=("人物连续", "核心行动可辨识", "空间关系稳定"),
            story_plan=StoryPlan(
                title=subject,
                logline=subject,
                synopsis=f"围绕“{subject}”展开一个完整的连续事件。",
                characters=(
                    CharacterSheetEntry(
                        "protagonist",
                        "主角",
                        "由方向定义的行动者",
                        "protagonist",
                    ),
                ),
                events=(f"主角围绕“{subject}”完成一个可见的连续行动",),
                story_beats=("完成核心事件",),
                character_goals=(StoryCharacterGoal("protagonist", "完成核心事件"),),
            ),
            director_plan=DirectorPlan(
                audience_knowledge="观众能够识别核心行动与主角身份",
                emotional_intention="克制的期待",
                visual_motif_strategy=brief.visual_style,
                climax_emphasis="完成核心事件",
            ),
        )

    def revise_movie_plan(
        self,
        brief: CreativeBrief,
        plan: MoviePlan,
        feedback: str,
    ) -> MoviePlan:
        del plan
        return self.create_movie_plan(brief, feedback)


def _default_prompt_path() -> Path:
    return Path(__file__).resolve().parent.parent / "prompts" / "v2" / "director" / "movie_plan.md"


def movie_plan_from_data(data: dict[str, Any]) -> MoviePlan:
    """Parse, but never repair, a DirectorAgent JSON response."""

    if not isinstance(data, dict):
        raise DirectorGenerationError("MoviePlan 顶层必须是对象")
    unexpected = sorted(set(data) - _PLAN_KEYS)
    if unexpected:
        raise DirectorGenerationError(
            "MoviePlan 包含禁止的 Provider/API 字段：" + ", ".join(unexpected)
        )
    _reject_prompt_stuffing(data)
    try:
        scenes = tuple(_scene(item) for item in _list(data, "script.scenes", data["script"]))
        return MoviePlan(
            plan_id=_string(data, "plan_id"),
            story=_story(_mapping(data, "story")),
            script=Script(
                title=_string(data["script"], "title"),
                logline=_string(data["script"], "logline"),
                scenes=scenes,
            ),
            scene_plan=ScenePlan(
                tuple(_scene_plan(item) for item in _list(data, "scene_plan.scenes", data["scene_plan"]))
            ),
            camera_plan=CameraPlan(
                tuple(
                    _camera(item)
                    for item in _list(data, "camera_plan.instructions", data["camera_plan"])
                )
            ),
            timing_plan=TimingPlan(
                target_duration_seconds=_number(
                    _mapping(data, "timing_plan"), "target_duration_seconds"
                ),
                entries=tuple(
                    _timing(item)
                    for item in _list(data, "timing_plan.entries", data["timing_plan"])
                ),
            ),
            continuity_plan=ContinuityPlan(
                entries=tuple(
                    _continuity(item)
                    for item in _list(data, "continuity_plan.entries", data["continuity_plan"])
                ),
                global_rules=tuple(
                    _string_value(item, "continuity_plan.global_rules")
                    for item in _list(data, "continuity_plan.global_rules", data["continuity_plan"])
                ),
            ),
            character_sheet=CharacterSheet(
                tuple(
                    _character(item)
                    for item in _list(data, "character_sheet.characters", data["character_sheet"])
                )
            ),
            narration_plan=_narration(_mapping(data, "narration_plan")),
            music_plan=_music(_mapping(data, "music_plan")),
            transition_plan=TransitionPlan(
                tuple(
                    _transition(item)
                    for item in _list(data, "transition_plan.transitions", data["transition_plan"])
                )
            ),
            shot_plan=tuple(
                _shot_plan(item)
                for item in _list(data, "shot_plan.shots", data.get("shot_plan", []))
            ),
            film_beats=tuple(
                _film_beat(item)
                for item in _list(
                    data,
                    "film_beats",
                    data.get("film_beats", []),
                )
            ),
            visual_style=_string(data, "visual_style"),
            emotion_curve=tuple(
                EmotionPoint(
                    label=_string(item, "label"),
                    intensity=_number(item, "intensity"),
                    scene_id=str(item.get("scene_id", "")).strip(),
                )
                for item in _list(data, "emotion_curve", data)
            ),
            review_criteria=tuple(
                _string_value(item, "review_criteria")
                for item in _list(data, "review_criteria", data)
            ),
            revision=_revision(data),
            confirmed=bool(data.get("confirmed", False)),
            story_plan=(
                _story_plan(data["story_plan"])
                if data.get("story_plan") is not None
                else None
            ),
            director_plan=(
                _director_plan(data["director_plan"])
                if data.get("director_plan") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError, DirectorGenerationError) as exc:
        if isinstance(exc, DirectorGenerationError):
            raise
        raise DirectorGenerationError(f"MoviePlan 字段不完整：{exc}") from exc


def _story(data: dict[str, Any]) -> Story:
    return Story(
        title=_string(data, "title"),
        logline=_string(data, "logline"),
        synopsis=_string(data, "synopsis"),
        theme=_string(data, "theme"),
        ending=_string(data, "ending"),
    )


def _story_plan(data: Any) -> StoryPlan:
    mapping = _strict_mapping(data, "story_plan", _STORY_PLAN_KEYS)
    return StoryPlan(
        title=_string(mapping, "title"),
        logline=_string(mapping, "logline"),
        synopsis=_string(mapping, "synopsis"),
        theme=_string(mapping, "theme"),
        ending=_string(mapping, "ending"),
        characters=tuple(
            _character(item)
            for item in _list(mapping, "story_plan.characters", mapping)
        ),
        events=tuple(
            _string_value(item, "story_plan.events")
            for item in _list(mapping, "story_plan.events", mapping)
        ),
        causality=tuple(
            _string_value(item, "story_plan.causality")
            for item in _list(mapping, "story_plan.causality", mapping)
        ),
        conflict=_string(mapping, "conflict"),
        stakes=_string(mapping, "stakes"),
        resolution=_string(mapping, "resolution"),
        story_beats=tuple(
            _string_value(item, "story_plan.story_beats")
            for item in _list(mapping, "story_plan.story_beats", mapping)
        ),
        character_goals=tuple(
            _story_character_goal(item)
            for item in _list(mapping, "story_plan.character_goals", mapping)
        ),
    )


def _story_character_goal(data: Any) -> StoryCharacterGoal:
    mapping = _mapping_value(data, "story_plan.character_goals")
    return StoryCharacterGoal(
        character_id=_string(mapping, "character_id"),
        goal=_string(mapping, "goal"),
        obstacle=_string(mapping, "obstacle"),
        outcome=_string(mapping, "outcome"),
    )


def _director_plan(data: Any) -> DirectorPlan:
    mapping = _strict_mapping(data, "director_plan", _DIRECTOR_PLAN_KEYS)
    return DirectorPlan(
        pacing_strategy=_string(mapping, "pacing_strategy"),
        suspense_strategy=_string(mapping, "suspense_strategy"),
        audience_knowledge=_string(mapping, "audience_knowledge"),
        emotional_intention=_string(mapping, "emotional_intention"),
        reveal_timing=_string(mapping, "reveal_timing"),
        withholding_strategy=_string(mapping, "withholding_strategy"),
        visual_motif_strategy=_string(mapping, "visual_motif_strategy"),
        silence_pause_intention=_string(mapping, "silence_pause_intention"),
        climax_emphasis=_string(mapping, "climax_emphasis"),
        ending_tone=_string(mapping, "ending_tone"),
    )


def _scene(data: Any) -> ScriptScene:
    if not isinstance(data, dict):
        raise DirectorGenerationError("script.scenes 必须是对象数组")
    return ScriptScene(
        scene_id=_string(data, "scene_id"),
        goal=_string(data, "goal"),
        emotion=_string(data, "emotion"),
        importance=_string(data, "importance"),
        estimated_duration_weight=_number(data, "estimated_duration_weight"),
        minimum_duration=_number(data, "minimum_duration"),
        camera_language=_string(data, "camera_language"),
        motion_type=_string(data, "motion_type"),
        dialogue=_string(data, "dialogue"),
        narration=_string(data, "narration"),
        characters=tuple(_string_value(item, "scene.characters") for item in _list(data, "characters", data)),
        location=_string(data, "location"),
        continuity_requirements=tuple(
            _string_value(item, "scene.continuity_requirements")
            for item in _list(data, "continuity_requirements", data)
        ),
        transition=_string(data, "transition"),
        timing_reason=_string(data, "timing_reason"),
        action=_string(data, "action"),
    )


def _scene_plan(data: Any) -> ScenePlanEntry:
    return ScenePlanEntry(
        scene_id=_string(data, "scene_id"),
        visual_intent=_string(data, "visual_intent"),
        blocking=_string(data, "blocking"),
        props=tuple(_string_value(item, "scene_plan.props") for item in _list(data, "props", data)),
    )


def _camera(data: Any) -> CameraInstruction:
    return CameraInstruction(
        scene_id=_string(data, "scene_id"),
        shot_size=_string(data, "shot_size"),
        angle=_string(data, "angle"),
        lens=_string(data, "lens"),
        movement=_string(data, "movement"),
        composition=_string(data, "composition"),
        language=_string(data, "language"),
    )


def _timing(data: Any) -> TimingEntry:
    return TimingEntry(
        scene_id=_string(data, "scene_id"),
        duration_seconds=_number(data, "duration_seconds"),
        reason=_string(data, "reason"),
    )


def _continuity(data: Any) -> ContinuityEntry:
    return ContinuityEntry(
        scene_id=_string(data, "scene_id"),
        requirements=tuple(
            _string_value(item, "continuity.requirements")
            for item in _list(data, "requirements", data)
        ),
        prior_state=_string(data, "prior_state"),
        resulting_state=_string(data, "resulting_state"),
    )


def _character(data: Any) -> CharacterSheetEntry:
    return CharacterSheetEntry(
        character_id=_string(data, "character_id"),
        name=_string(data, "name"),
        identity=_string(data, "identity"),
        role=_string(data, "role"),
        visual_signature=_string(data, "visual_signature"),
        performance_notes=_string(data, "performance_notes"),
        reference_keys=tuple(
            _string_value(item, "character.reference_keys")
            for item in _list(data, "reference_keys", {"reference_keys": data.get("reference_keys", [])})
        ),
        costume=str(data.get("costume", "")).strip(),
    )


def _shot_plan(data: Any) -> ShotPlan:
    return ShotPlan(
        shot_id=_string(data, "shot_id"),
        scene_id=_string(data, "scene_id"),
        order=int(_number(data, "order")),
        duration_seconds=_number(data, "duration_seconds"),
        purpose=_string(data, "purpose"),
        visible_action=_string(data, "visible_action"),
        subject=_string(data, "subject"),
        camera_intent=_string(data, "camera_intent"),
        motion_intent=_string(data, "motion_intent"),
        lighting=_string(data, "lighting"),
        composition=_string(data, "composition"),
        characters=tuple(
            _string_value(item, "shot.characters")
            for item in _list(data, "characters", data)
        ),
        props=tuple(
            _string_value(item, "shot.props") for item in _list(data, "props", data)
        ),
        narration=_string(data, "narration"),
        dialogue=_string(data, "dialogue"),
        subtitles=_string(data, "subtitles"),
        transition_in=_string(data, "transition_in"),
        transition_out=_string(data, "transition_out"),
        continuity_anchors=tuple(
            _string_value(item, "shot.continuity_anchors")
            for item in _list(data, "continuity_anchors", data)
        ),
        required_visual_evidence=tuple(
            _string_value(item, "shot.required_visual_evidence")
            for item in _list(data, "required_visual_evidence", data)
        ),
        acceptance_criteria=tuple(
            _string_value(item, "shot.acceptance_criteria")
            for item in _list(data, "acceptance_criteria", data)
        ),
    )


def _film_beat(data: Any) -> FilmBeatPlan:
    return FilmBeatPlan(
        beat_id=_string(data, "beat_id"),
        order=int(_number(data, "order")),
        scene_id=_string(data, "scene_id"),
        shot_ids=tuple(
            _string_value(item, "film_beats.shot_ids")
            for item in _list(data, "shot_ids", data)
        ),
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
        required_evidence=tuple(
            _string_value(item, "film_beats.required_evidence")
            for item in _list(data, "required_evidence", data)
        ),
        character_emotional_state=tuple(
            _string_value(item, "film_beats.character_emotional_state")
            for item in _list(data, "character_emotional_state", data)
        ),
        continuity_intent=_string(data, "continuity_intent"),
        transition_intent=_string(data, "transition_intent"),
        narration_intent=_string(data, "narration_intent"),
        music_intent=_string(data, "music_intent"),
        acceptance_criteria=tuple(
            _string_value(item, "film_beats.acceptance_criteria")
            for item in _list(data, "acceptance_criteria", data)
        ),
        timing_weight=_number(data, "timing_weight"),
    )


def _narration(data: dict[str, Any]) -> NarrationPlan:
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        raise DirectorGenerationError("narration_plan.enabled 必须是布尔值")
    return NarrationPlan(
        enabled=enabled,
        language=_string(data, "language"),
        style=_string(data, "style"),
        segments=tuple(
            NarrationSegment(
                scene_id=_string(item, "scene_id"),
                text=_string(item, "text"),
                delivery=_string(item, "delivery"),
            )
            for item in _list(data, "segments", data)
        ),
    )


def _music(data: dict[str, Any]) -> MusicPlan:
    return MusicPlan(
        direction=_string(data, "direction"),
        intensity=_string(data, "intensity"),
        beat_notes=tuple(
            _string_value(item, "music_plan.beat_notes")
            for item in _list(data, "beat_notes", data)
        ),
    )


def _transition(data: Any) -> TransitionInstruction:
    return TransitionInstruction(
        from_scene_id=_string(data, "from_scene_id"),
        to_scene_id=_string(data, "to_scene_id"),
        transition=_string(data, "transition"),
        reason=_string(data, "reason"),
    )


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, dict):
        raise DirectorGenerationError(f"{key} 必须是对象")
    return value


def _mapping_value(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DirectorGenerationError(f"{key} must be an object")
    return value


def _strict_mapping(
    value: Any,
    key: str,
    allowed: set[str],
) -> dict[str, Any]:
    mapping = _mapping_value(value, key)
    unexpected = sorted(set(mapping) - allowed)
    if unexpected:
        raise DirectorGenerationError(
            f"{key} contains unsupported fields: " + ", ".join(unexpected)
        )
    return mapping


def _reject_prompt_stuffing(value: Any, path: str = "movie_plan") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_prompt_stuffing(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_prompt_stuffing(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for term in _PROMPT_STUFFING_TERMS:
            if term in lowered:
                raise DirectorGenerationError(
                    f"{path} contains forbidden prompt stuffing: {term}"
                )


def _list(parent: dict[str, Any], key: str, owner: dict[str, Any]) -> list[Any]:
    del parent
    if isinstance(owner, list):
        return owner
    simple_key = key.rsplit(".", 1)[-1]
    if simple_key in owner:
        value = owner[simple_key]
    elif "beats" in owner and key == "film_beats":
        value = owner["beats"]
    else:
        raise DirectorGenerationError(f"{key} 必须是数组")
    if not isinstance(value, list):
        raise DirectorGenerationError(f"{key} 必须是数组")
    return value


def _string(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise DirectorGenerationError(f"{key} 必须是字符串")
    return value.strip()


def _string_value(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise DirectorGenerationError(f"{field_name} 必须是字符串数组")
    return value.strip()


def _number(data: dict[str, Any], key: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectorGenerationError(f"{key} 必须是数字")
    return float(value)


def _revision(data: dict[str, Any]) -> int:
    value = data.get("revision", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DirectorGenerationError("revision 必须是正整数")
    return value

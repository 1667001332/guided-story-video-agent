from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass

from .continuity import assign_continuity_modes, continuity_state_to_dict
from .models import (
    ContinuityState,
    StoryFacts,
    StoryScene,
    StoryScript,
    StoryboardPlan,
    StoryboardShot,
    VisualAsset,
    VisualBible,
    to_plain_data,
)
from .timing import (
    ReadableShotMinimum,
    ShotTimingDemand,
    allocate_weighted_durations,
    assess_shot_readable_minimum,
    estimate_shot_duration,
    estimate_scene_duration_weight,
)


@dataclass(frozen=True, slots=True)
class StoryboardTimingAssessment:
    """Auditable hard floors and soft preferences for one shot plan."""

    target_seconds: int
    minimum_durations: tuple[int, ...]
    preferred_durations: tuple[float, ...]
    minimum_reasons: tuple[str, ...]

    @property
    def minimum_total(self) -> int:
        return sum(self.minimum_durations)

    @property
    def preferred_total(self) -> float:
        return round(sum(self.preferred_durations), 3)

    @property
    def over_capacity_shots(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, duration in enumerate(self.minimum_durations, start=1)
            if duration > 15
        )

    @property
    def feasible(self) -> bool:
        return (
            bool(self.minimum_durations)
            and not self.over_capacity_shots
            and self.minimum_total <= self.target_seconds <= 15 * len(self.minimum_durations)
        )

    def feedback(self) -> str:
        details = "、".join(
            f"镜头{index}至少{duration}秒"
            for index, duration in enumerate(self.minimum_durations, start=1)
        )
        if self.over_capacity_shots:
            shots = "、".join(str(index) for index in self.over_capacity_shots)
            return (
                f"上次分镜中镜头 {shots} 的单镜内容超过15秒容量；{details}。"
                "请拆分过载动作，但同时删去低信息增量镜头，重新使总预算成立。"
            )
        return (
            f"上次分镜的内容可读下限合计 {self.minimum_total} 秒，"
            f"但成片只有 {self.target_seconds} 秒；{details}。"
            "请减少镜头数或简化每镜动作；不得靠把所有镜头统一压成3秒解决。"
        )


class StoryboardTimingBudgetError(ValueError):
    """Raised before rendering when shot content cannot fit the film budget."""

    def __init__(self, assessment: StoryboardTimingAssessment) -> None:
        self.assessment = assessment
        super().__init__(assessment.feedback())


@dataclass(slots=True)
class _ShotUnit:
    scene: StoryScene
    kind: str
    action: str
    purpose: str
    priority: int
    required: bool = False
    transition_type: str = "same_scene_cut"
    transition_reason: str = ""
    inherit_previous_frame: bool = False
    camera: str = ""
    camera_movement: str = ""
    composition: str = ""


CAMERA_BY_KIND = {
    "establish": ("wide establishing shot", "slow push-in", "主体与环境关系清晰"),
    "action": ("medium shot", "motivated tracking", "动作方向明确，保留空间参照"),
    "detail": ("insert close-up", "locked camera", "关键道具和手部动作占据视觉中心"),
    "dialogue": ("over-the-shoulder two-shot", "subtle lateral move", "对话双方保持视线匹配"),
    "reaction": ("medium close-up", "subtle handheld", "人物反应与压力来源同时可读"),
    "transition": ("wide reveal shot", "controlled pull-back", "展示动作结果和空间变化"),
}


def build_visual_bible(script: StoryScript, facts: StoryFacts) -> VisualBible:
    """Compile story facts into provider-independent reference assets."""
    characters = _unique(
        name.strip() for scene in script.scenes for name in scene.characters if name.strip()
    )
    locations = _unique(scene.location.strip() for scene in script.scenes if scene.location.strip())
    props = _unique(
        [
            *(item.strip() for scene in script.scenes for item in scene.props if item.strip()),
            *_split_anchors(facts.props),
        ]
    )
    assets: list[VisualAsset] = []
    for index, name in enumerate(characters, start=1):
        assets.append(
            VisualAsset(
                asset_id=f"character-{index:02d}",
                kind="character",
                name=name,
                description=_matching_anchor(
                    name,
                    facts.character_visuals,
                    f"{name}的脸部、发型、服装和标志性物件保持统一",
                ),
            )
        )
    for index, name in enumerate(locations, start=1):
        assets.append(
            VisualAsset(
                asset_id=f"location-{index:02d}",
                kind="location",
                name=name,
                description=_matching_anchor(
                    name,
                    facts.scene_details,
                    f"{name}的空间结构、主色、光线方向和时段保持统一",
                ),
            )
        )
    for index, name in enumerate(props, start=1):
        assets.append(
            VisualAsset(
                asset_id=f"prop-{index:02d}",
                kind="prop",
                name=name,
                description=f"{name}的材质、尺寸、颜色和磨损状态保持统一",
            )
        )
    continuity_rules = [
        "同一角色在所有镜头中保持脸部、发型、服装和年龄一致",
        "同一地点保持空间布局、光源方向、天气和主色一致",
        "关键道具的外形、持有者和所在位置必须承接上一镜头",
        facts.transitions or "动作方向、人物视线和关键物件使用匹配剪辑",
    ]
    return VisualBible(
        visual_style=facts.camera_style or "写实电影短片，真实材质和自然景深",
        color_palette=f"{facts.tone or '克制情绪'}，统一场景主色和肤色表现",
        lighting_rules="遵守剧本时段和场景光源，镜头间不改变主光方向",
        camera_language="根据可见动作、对话、道具和情绪选择镜头，不套固定景别顺序",
        assets=assets,
        continuity_rules=continuity_rules,
    )


def build_storyboard(
    script: StoryScript,
    facts: StoryFacts,
    visual_bible: VisualBible | None = None,
    director_plan: list[dict[str, object]] | None = None,
) -> StoryboardPlan:
    if not script.confirmed:
        raise RuntimeError("请先确认剧本，再生成分镜。")
    if not script.scenes:
        raise ValueError("剧本没有可转换的场景。")
    normalized = _normalized_storyboard_script(script)
    bible = visual_bible or build_visual_bible(normalized, facts)
    units = (
        _units_from_director_plan(normalized, director_plan)
        if director_plan
        else _plan_shot_units(normalized)
    )
    assessment, timing, readable, dialogues, narrations = _assess_units_timing(
        normalized,
        units,
    )
    if not assessment.feasible:
        raise StoryboardTimingBudgetError(assessment)
    durations = allocate_weighted_durations(
        normalized.target_seconds,
        [item[1] for item in timing],
        minimum=3,
        maximum=15,
        minimums=list(assessment.minimum_durations),
        keys=[
            f"{unit.scene.scene_id}:{unit.kind}:{unit.purpose}:{index}"
            for index, unit in enumerate(units, start=1)
        ],
    )
    states = _derive_continuity_states(units, bible)
    base_seed = derive_storyboard_seed(normalized, facts)
    shots = [
        _build_shot(
            index,
            unit,
            duration,
            bible,
            facts,
            len(units),
            start_state=states[index - 1][0],
            end_state=states[index - 1][1],
            estimated_duration=timing[index - 1][0],
            duration_weight=timing[index - 1][1],
            duration_reason=f"{timing[index - 1][2]}；{readable[index - 1].reason}",
            minimum_readable_duration=readable[index - 1].minimum_seconds,
            dialogue=dialogues[index - 1],
            seed=derive_shot_seed(
                base_seed,
                unit.scene.scene_id,
                index,
            ),
        )
        for index, (unit, duration) in enumerate(zip(units, durations), start=1)
    ]
    _assign_narration_timeline(shots, narrations)
    assign_continuity_modes(shots)
    apply_transition_prompt_context(shots)
    plan = StoryboardPlan(
        title=normalized.title,
        target_seconds=normalized.target_seconds,
        shots=shots,
        narration_text="\n".join(shot.narration for shot in shots if shot.narration),
        visual_bible=bible,
        base_seed=base_seed,
    )
    if plan.total_duration != normalized.target_seconds:
        raise ValueError("分镜总时长与目标时长不一致。")
    return plan


def assess_director_plan_timing(
    script: StoryScript,
    director_plan: list[dict[str, object]],
    *,
    dialogue_overrides: list[str] | None = None,
    narration_overrides: list[str] | None = None,
) -> StoryboardTimingAssessment:
    """Assess a model director plan before accepting or retrying it."""

    normalized = _normalized_storyboard_script(script)
    units = _units_from_director_plan(normalized, director_plan)
    assessment, _, _, _, _ = _assess_units_timing(
        normalized,
        units,
        dialogue_overrides=dialogue_overrides,
        narration_overrides=narration_overrides,
    )
    return assessment


def _normalized_storyboard_script(script: StoryScript) -> StoryScript:
    normalized = deepcopy(script)
    normalized.scenes = fit_scenes_to_duration(
        normalized.scenes,
        normalized.target_seconds,
        minimum=3,
    )
    normalized_durations = allocate_weighted_durations(
        normalized.target_seconds,
        [_scene_duration_weight(scene) for scene in normalized.scenes],
        minimum=3,
        maximum=normalized.target_seconds,
        keys=[f"scene-{scene.scene_id}-{scene.title}" for scene in normalized.scenes],
    )
    for scene, duration in zip(normalized.scenes, normalized_durations):
        scene.duration = duration
    return normalized


def _assess_units_timing(
    script: StoryScript,
    units: list[_ShotUnit],
    *,
    dialogue_overrides: list[str] | None = None,
    narration_overrides: list[str] | None = None,
) -> tuple[
    StoryboardTimingAssessment,
    list[tuple[float, float, str]],
    list[ReadableShotMinimum],
    list[str],
    list[str],
]:
    if not units:
        raise ValueError("分镜至少需要一个镜头单元。")
    scene_shot_counts = {
        scene_id: sum(unit.scene.scene_id == scene_id for unit in units)
        for scene_id in {unit.scene.scene_id for unit in units}
    }
    dialogues = (
        _validate_spoken_overrides(
            dialogue_overrides,
            units,
            script.scenes,
            field_name="对白",
            source_field="dialogue",
        )
        if dialogue_overrides is not None
        else _dialogue_assignments_for_units(units, script.scenes)
    )
    narrations = (
        _validate_spoken_overrides(
            narration_overrides,
            units,
            script.scenes,
            field_name="旁白",
            source_field="narration",
        )
        if narration_overrides is not None
        else _narration_assignments_for_units(units, script.scenes)
    )
    demands = [
        ShotTimingDemand(
            shot_kind=unit.kind,
            purpose=unit.purpose,
            priority=unit.priority,
            action=unit.action,
            dialogue=dialogues[index],
            narration=narrations[index],
            emotional_change=unit.scene.emotional_change,
            scene_duration=unit.scene.duration,
            scene_shot_count=scene_shot_counts[unit.scene.scene_id],
            narration_is_per_shot=True,
        )
        for index, unit in enumerate(units)
    ]
    timing = [estimate_shot_duration(demand) for demand in demands]
    readable = [assess_shot_readable_minimum(demand) for demand in demands]
    assessment = StoryboardTimingAssessment(
        target_seconds=script.target_seconds,
        minimum_durations=tuple(item.minimum_seconds for item in readable),
        preferred_durations=tuple(item[0] for item in timing),
        minimum_reasons=tuple(item.reason for item in readable),
    )
    return assessment, timing, readable, dialogues, narrations


def fit_scenes_to_duration(
    scenes: list[StoryScene],
    target_seconds: int,
    *,
    minimum: int = 3,
) -> list[StoryScene]:
    """Fit scenes without pretending that different places or times are one scene."""

    if not scenes:
        raise ValueError("剧本至少需要一个可拍摄场景。")
    target = int(target_seconds)
    maximum_count = target // int(minimum)
    if maximum_count < 1:
        raise ValueError("目标时长不足以容纳一个可拍摄场景。")
    if len(scenes) <= maximum_count:
        result = deepcopy(scenes)
        for index, scene in enumerate(result, start=1):
            scene.scene_id = index
        return result

    result = deepcopy(scenes)
    while len(result) > maximum_count:
        compatible = next(
            (
                index
                for index in range(len(result) - 1)
                if _scenes_share_physical_context(result[index], result[index + 1])
            ),
            None,
        )
        if compatible is None:
            raise ValueError(
                "剧本场景超过目标时长容量，且剩余场景跨地点或跨时段；"
                "禁止机械合并，请让文本模型压缩或重写剧本。"
            )
        merged = _merge_scene_group(
            compatible + 1,
            result[compatible : compatible + 2],
        )
        result[compatible : compatible + 2] = [merged]
    for index, scene in enumerate(result, start=1):
        scene.scene_id = index
    return result


def _merge_scene_group(scene_id: int, group: list[StoryScene]) -> StoryScene:
    if not group:
        raise ValueError("不能合并空场景组。")
    if any(
        not _scenes_share_physical_context(group[0], item)
        for item in group[1:]
    ):
        raise ValueError("不同地点或不同时段的场景不能机械合并。")

    def joined(values, separator: str = "；随后，") -> str:
        return separator.join(str(value).strip() for value in values if str(value).strip())

    return StoryScene(
        scene_id=scene_id,
        title=joined((scene.title for scene in group), " / "),
        location=group[0].location,
        time_of_day=group[0].time_of_day,
        characters=_unique(character for scene in group for character in scene.characters),
        action=joined((scene.visible_action or scene.action for scene in group)),
        visible_action=joined((scene.visible_action or scene.action for scene in group)),
        narration=joined((scene.narration for scene in group), "\n"),
        duration=sum(max(0, int(scene.duration)) for scene in group),
        dialogue=joined((scene.dialogue for scene in group), "\n"),
        props=_unique(prop for scene in group for prop in scene.props),
        start_state=group[0].start_state,
        end_state=group[-1].end_state,
        emotional_change=joined((scene.emotional_change for scene in group)),
        duration_weight=sum(max(0.0, float(scene.duration_weight)) for scene in group),
        duration_reason=joined((scene.duration_reason for scene in group)),
    )


def _scenes_share_physical_context(left: StoryScene, right: StoryScene) -> bool:
    return (
        left.location.strip() == right.location.strip()
        and left.time_of_day.strip() == right.time_of_day.strip()
    )


def _units_from_director_plan(
    script: StoryScript,
    director_plan: list[dict[str, object]],
) -> list[_ShotUnit]:
    scenes = {scene.scene_id: scene for scene in script.scenes}
    units: list[_ShotUnit] = []
    for index, item in enumerate(director_plan):
        scene_id = int(item.get("scene_id", 0))
        scene = scenes.get(scene_id)
        if scene is None:
            raise ValueError(f"导演分镜引用了不存在的场景：{scene_id}")
        kind = str(item.get("kind", "action")).strip()
        if kind not in CAMERA_BY_KIND:
            raise ValueError(f"导演分镜使用了不支持的镜头类型：{kind}")
        action = str(item.get("action", "")).strip()
        purpose = str(item.get("purpose", "")).strip()
        if not action or not purpose:
            raise ValueError("导演分镜的 action 和 purpose 不能为空。")
        transition_type = str(item.get("transition_type", "same_scene_cut")).strip()
        inherit = item.get("inherit_previous_frame", False)
        if not isinstance(inherit, bool):
            raise ValueError("inherit_previous_frame 必须是布尔值。")
        if inherit:
            if index == 0 or transition_type != "continuous_action":
                raise ValueError("只有非开场的连续动作镜头可以继承上一镜头末帧。")
            previous_scene = units[-1].scene
            if not _scenes_share_physical_context(previous_scene, scene):
                raise ValueError("跨地点或跨时段镜头不能继承上一镜头末帧。")
        units.append(
            _ShotUnit(
                scene=scene,
                kind=kind,
                action=action,
                purpose=purpose,
                priority=6 if kind == "action" else 4,
                required=True,
                transition_type=transition_type,
                transition_reason=str(item.get("transition_reason", "")).strip(),
                inherit_previous_frame=inherit,
                camera=str(item.get("camera", "")).strip(),
                camera_movement=str(item.get("camera_movement", "")).strip(),
                composition=str(item.get("composition", "")).strip(),
            )
        )
    minimum = max(1, math.ceil(script.target_seconds / 15))
    maximum = max(1, script.target_seconds // 3)
    if not minimum <= len(units) <= maximum:
        raise ValueError(f"导演分镜数量必须在 {minimum} 到 {maximum} 之间。")
    if {unit.scene.scene_id for unit in units} != set(scenes):
        raise ValueError("导演分镜必须覆盖每个剧本场景。")
    return units


def _plan_shot_units(script: StoryScript) -> list[_ShotUnit]:
    """Derive non-repeating, atomic fallback shots when no model director is available."""
    units: list[_ShotUnit] = []
    previous_context: tuple[str, str] | None = None
    for scene in script.scenes:
        action = (scene.visible_action or scene.action).strip()
        if not action:
            action = "人物完成一个清晰可见的动作"
        context = (scene.location.strip(), scene.time_of_day.strip())
        if previous_context is None or context != previous_context:
            units.append(
                _ShotUnit(
                    scene,
                    "establish",
                    scene.start_state.strip()
                    or f"人物位于{scene.location}，动作即将开始",
                    f"交代《{scene.title}》的空间、人物位置和行动起点",
                    priority=4 if previous_context is None else 2,
                    transition_type="scene_change",
                    transition_reason="地点或时段变化，需要重新建立空间关系",
                )
            )
        beats = _split_action_beats(action)
        for beat_index, beat in enumerate(beats):
            continuous = beat_index > 0 and _is_direct_action_continuation(beat)
            units.append(
                _ShotUnit(
                    scene,
                    "action",
                    beat,
                    f"呈现《{scene.title}》中第{beat_index + 1}个不可省略的动作阶段",
                    priority=6,
                    required=True,
                    transition_type=(
                        "continuous_action" if continuous else "same_scene_cut"
                    ),
                    transition_reason=(
                        "同一物理动作的直接下一阶段"
                        if continuous
                        else "动作阶段变化，正常切换机位重新构图"
                    ),
                    inherit_previous_frame=continuous,
                )
            )
        if scene.props:
            units.append(
                _ShotUnit(
                    scene,
                    "detail",
                    f"呈现{_join(scene.props)}当前状态、所在位置及其与人物动作的关系",
                    f"让关键道具成为《{scene.title}》的叙事证据",
                    priority=4,
                    transition_type="insert_shot",
                    transition_reason="插入道具细节，不继承上一镜头机位",
                )
            )
        if scene.dialogue.strip():
            units.append(
                _ShotUnit(
                    scene,
                    "dialogue",
                    f"人物说出对白时仍有可见反应和动作：{scene.dialogue}",
                    "通过视线、停顿和身体反应呈现对话关系",
                    priority=3,
                    required=True,
                    transition_type="reverse_shot",
                    transition_reason="对白关系需要独立的正反打或过肩构图",
                )
            )
        if scene.emotional_change.strip():
            units.append(
                _ShotUnit(
                    scene,
                    "reaction",
                    f"呈现情绪变化：{scene.emotional_change}",
                    f"让《{scene.title}》的情绪变化可以被看见",
                    priority=4,
                    transition_type="reaction_cut",
                    transition_reason="切到人物反应以呈现新的情绪信息",
                )
            )
        if scene.end_state.strip() and scene.end_state.strip() != scene.start_state.strip():
            units.append(
                _ShotUnit(
                    scene,
                    "transition",
                    scene.end_state.strip(),
                    f"明确《{scene.title}》结束时已经发生的状态变化",
                    priority=5,
                    transition_type="same_scene_cut",
                    transition_reason="动作结果需要独立构图确认，不强制沿用上一机位",
                )
            )
        previous_context = context

    maximum = max(1, script.target_seconds // 3)
    minimum = max(
        math.ceil(script.target_seconds / 15),
        min(len(script.scenes), maximum),
    )
    required_indexes = [index for index, unit in enumerate(units) if unit.required]
    if len(required_indexes) > maximum:
        raise ValueError(
            f"剧本包含 {len(required_indexes)} 个不可省略动作阶段，"
            f"但 {script.target_seconds} 秒最多容纳 {maximum} 个镜头；"
            "请压缩剧本内容，禁止把多个阶段机械塞进同一镜头。"
        )
    selected = set(required_indexes)
    extras = sorted(
        (index for index, unit in enumerate(units) if not unit.required),
        key=lambda index: (-units[index].priority, index),
    )

    def candidate_fits(indexes: set[int]) -> bool:
        candidate = [unit for index, unit in enumerate(units) if index in indexes]
        assessment, _, _, _, _ = _assess_units_timing(script, candidate)
        return (
            not assessment.over_capacity_shots
            and assessment.minimum_total <= script.target_seconds
        )

    for index in extras:
        if len(selected) >= minimum:
            break
        candidate = {*selected, index}
        if candidate_fits(candidate):
            selected = candidate
    if len(selected) < minimum:
        raise ValueError(
            f"剧本只有 {len(selected)} 个能在内容预算内成立的镜头单元，"
            f"不足以支撑 {script.target_seconds} 秒（至少需要 {minimum} 个）；"
            "请扩充剧本的可见动作，而不是用重复镜头填时长。"
        )
    for index in extras:
        if index in selected or len(selected) >= maximum:
            continue
        candidate = {*selected, index}
        if candidate_fits(candidate):
            selected = candidate
    result = [unit for index, unit in enumerate(units) if index in selected]
    assessment, _, _, _, _ = _assess_units_timing(script, result)
    if not assessment.feasible:
        raise StoryboardTimingBudgetError(assessment)
    return result


def _split_action_beats(action: str) -> list[str]:
    pieces = [
        piece.strip(" ，,；;")
        for piece in re.split(
            (
                r"(?<=[。！？；;])|(?<=[^\d])\.(?!\d)"
                r"|(?=随后|接着|紧接着|然后|随即|继而|之后|转而|并且)"
                r"|(?=最后(?!一?(?:个|次|步|项|段|场|镜|句|刻|天|年|人|动作)))"
                r"|(?=最终(?!一?(?:个|次|步|项|段|场|镜|句|刻|天|年|人|动作)))"
                r"|(?=(?<!不)再(?:次|度)?)"
            ),
            action,
        )
        if piece.strip(" ，,；;")
    ]
    return pieces or [action.strip()]


def _is_direct_action_continuation(action: str) -> bool:
    value = action.strip()
    return value.startswith(
        (
            "随后",
            "接着",
            "紧接着",
            "然后",
            "随即",
            "继而",
            "之后",
            "最后",
            "最终",
            "转而",
            "并且",
            "再",
            "继续",
        )
    )


def _trim_units(units: list[_ShotUnit], maximum: int) -> list[_ShotUnit]:
    if len(units) <= maximum:
        return units
    required_indexes = [index for index, unit in enumerate(units) if unit.required]
    if len(required_indexes) > maximum:
        return _collapse_required_units(units, required_indexes, maximum)
    if len(required_indexes) == maximum:
        chosen = set(_evenly_spaced(required_indexes, maximum))
    else:
        slots = maximum - len(required_indexes)
        extras = sorted(
            (index for index, unit in enumerate(units) if not unit.required),
            key=lambda index: (-units[index].priority, index),
        )
        chosen = set(required_indexes + extras[:slots])
    return [unit for index, unit in enumerate(units) if index in chosen]


def _collapse_required_units(
    units: list[_ShotUnit],
    required_indexes: list[int],
    maximum: int,
) -> list[_ShotUnit]:
    """Keep every required action by grouping adjacent beats inside each script scene."""
    by_scene: dict[int, list[int]] = {}
    scene_order: list[int] = []
    for index in required_indexes:
        scene_id = units[index].scene.scene_id
        if scene_id not in by_scene:
            by_scene[scene_id] = []
            scene_order.append(scene_id)
        by_scene[scene_id].append(index)
    if len(scene_order) > maximum:
        raise ValueError("剧本场景数超过可用镜头数，无法在不丢失动作的前提下生成分镜。")
    capacities = {scene_id: 1 for scene_id in scene_order}
    remaining = maximum - len(scene_order)
    while remaining:
        expandable = [
            scene_id
            for scene_id in scene_order
            if capacities[scene_id] < len(by_scene[scene_id])
        ]
        if not expandable:
            break
        scene_id = max(
            expandable,
            key=lambda item: len(by_scene[item]) - capacities[item],
        )
        capacities[scene_id] += 1
        remaining -= 1

    collapsed: list[tuple[int, _ShotUnit]] = []
    for scene_id in scene_order:
        indexes = by_scene[scene_id]
        capacity = capacities[scene_id]
        for group_number in range(capacity):
            start = group_number * len(indexes) // capacity
            end = (group_number + 1) * len(indexes) // capacity
            group = indexes[start:end]
            first = units[group[0]]
            actions = [units[index].action.rstrip("。") for index in group]
            purposes = [units[index].purpose for index in group]
            collapsed.append(
                (
                    group[0],
                    _ShotUnit(
                        scene=first.scene,
                        kind="action",
                        action="；随后，".join(actions),
                        purpose="；".join(purposes),
                        priority=max(units[index].priority for index in group),
                        required=True,
                        transition_type="same_scene_cut",
                        transition_reason=(
                            "目标时长不足以拆成更多镜头，同一场景内的相邻动作阶段"
                            "合并在一个镜头中完整呈现"
                        ),
                        inherit_previous_frame=False,
                    ),
                )
            )
    return [unit for _, unit in sorted(collapsed, key=lambda item: item[0])]


def _build_shot(
    shot_id: int,
    unit: _ShotUnit,
    duration: int,
    bible: VisualBible,
    facts: StoryFacts,
    total_shots: int,
    *,
    start_state: ContinuityState,
    end_state: ContinuityState,
    estimated_duration: float,
    duration_weight: float,
    duration_reason: str,
    minimum_readable_duration: int,
    dialogue: str,
    seed: int,
) -> StoryboardShot:
    scene = unit.scene
    default_camera, default_movement, default_composition = CAMERA_BY_KIND[unit.kind]
    camera = unit.camera or default_camera
    movement = unit.camera_movement or default_movement
    composition = unit.composition or default_composition
    character = scene.characters[0] if scene.characters else "主角"
    start_frame = _state_frame_summary(
        start_state,
        scene.start_state or f"{character}位于{scene.location}，动作尚未完成",
    )
    end_frame = _state_frame_summary(
        end_state,
        scene.end_state or unit.action,
    )
    reference_ids = _reference_ids(bible, scene)
    anchors = [asset.description for asset in bible.assets if asset.asset_id in reference_ids]
    first_frame_prompt = _first_frame_prompt_text(
        location=scene.location,
        time_of_day=scene.time_of_day,
        character=_join(scene.characters) or character,
        start_frame=start_frame,
        composition=composition,
        camera=camera,
        visual_style=bible.visual_style,
        color_palette=bible.color_palette,
        lighting=bible.lighting_rules,
        anchors=anchors,
        retake_instruction="",
    )
    motion_prompt = _motion_prompt_text(
        duration=duration,
        action=unit.action,
        camera_movement=movement,
        dialogue=dialogue,
    )
    end_frame_prompt = _end_frame_prompt_text(
        end_frame=end_frame,
        location=scene.location,
    )
    prompt = _video_prompt_text(
        purpose=unit.purpose,
        camera=camera,
        composition=composition,
        first_frame_prompt=first_frame_prompt,
        motion_prompt=motion_prompt,
        end_frame_prompt=end_frame_prompt,
    )
    continuity = [
        *bible.continuity_rules,
        f"本镜头引用资产：{_join(reference_ids) or '暂无参考图资产'}",
    ]
    return StoryboardShot(
        shot_id=shot_id,
        scene_id=scene.scene_id,
        duration=duration,
        character=character,
        location=scene.location,
        visual=unit.action,
        action=unit.action,
        camera=camera,
        lighting=bible.lighting_rules,
        mood=facts.tone or scene.emotional_change or "克制、连贯、电影感",
        narration="",
        dialogue=dialogue,
        source_action=unit.action,
        retake_instruction="",
        time_of_day=scene.time_of_day,
        visual_style=bible.visual_style,
        color_palette=bible.color_palette,
        video_prompt=prompt,
        negative_prompt=_negative_prompt_text(),
        continuity_notes=continuity,
        shot_purpose=unit.purpose,
        composition=composition,
        camera_movement=movement,
        start_frame=start_frame,
        end_frame=end_frame,
        visual_anchors=anchors,
        shot_kind=unit.kind,
        duration_reason=duration_reason,
        duration_weight=duration_weight,
        estimated_duration=estimated_duration,
        minimum_readable_duration=minimum_readable_duration,
        first_frame_prompt=first_frame_prompt,
        motion_prompt=motion_prompt,
        end_frame_prompt=end_frame_prompt,
        reference_asset_ids=reference_ids,
        transition_type=unit.transition_type,
        transition_reason=unit.transition_reason,
        inherit_previous_frame=unit.inherit_previous_frame,
        continuity_state={
            "start": continuity_state_to_dict(start_state),
            "end": continuity_state_to_dict(end_state),
        },
        continuity_start_state=start_state,
        continuity_end_state=end_state,
        seed=seed,
    )


def _motion_prompt_text(
    *,
    duration: int,
    action: str,
    camera_movement: str,
    dialogue: str,
    retake_instruction: str = "",
) -> str:
    spoken = f"对白按剧本完整说出：{dialogue}。" if dialogue.strip() else ""
    retake = (
        f"Retake 保留本镜剧情动作，仅应用以下约束：{retake_instruction}。"
        if retake_instruction.strip()
        else ""
    )
    return (
        f"在{duration}秒内完成一个连续动作：{action}。"
        f"{spoken}{retake}摄影机运动：{camera_movement or 'static'}。"
        "动作遵守真实物理过程，不跳步，不瞬移。"
        "人物视线、移动方向、道具位置和光线方向承接首帧。"
    )


def _first_frame_prompt_text(
    *,
    location: str,
    time_of_day: str,
    character: str,
    start_frame: str,
    composition: str,
    camera: str,
    visual_style: str,
    color_palette: str,
    lighting: str,
    anchors: list[str],
    retake_instruction: str = "",
) -> str:
    retake = (
        f"Retake 画面约束：{retake_instruction}。"
        if retake_instruction.strip()
        else ""
    )
    return (
        f"电影分镜首帧。地点与时段：{location}，{time_of_day or '连续时间'}。"
        f"人物：{character or '主角'}。起始状态：{start_frame}。"
        f"构图：{composition}。摄影：{camera}。"
        f"视觉风格：{visual_style or '电影感写实'}；"
        f"色彩：{color_palette or '统一、克制的电影色彩'}；"
        f"光线：{lighting}。身份与场景锚点：{_join(anchors)}。"
        f"{retake}画面静止、结构清楚、可作为图生视频首帧。"
    )


def _end_frame_prompt_text(
    *,
    end_frame: str,
    location: str,
    retake_instruction: str = "",
) -> str:
    retake = (
        f"Retake 画面约束：{retake_instruction}。"
        if retake_instruction.strip()
        else ""
    )
    return (
        f"结束帧必须明确到达：{end_frame}。保持{location}的空间结构、"
        f"角色身份、服装、道具状态和色彩连续。{retake}可自然衔接下一镜头。"
    )


def _negative_prompt_text() -> str:
    return (
        "inconsistent identity, changed face, changed costume, duplicated character, "
        "broken anatomy, teleporting object, unreadable text, watermark, flicker, jump cut"
    )


def _video_prompt_text(
    *,
    purpose: str,
    camera: str,
    composition: str,
    first_frame_prompt: str,
    motion_prompt: str,
    end_frame_prompt: str,
) -> str:
    return (
        f"Cinematic narrative shot. Purpose: {purpose}. "
        f"CAMERA: {camera}. COMPOSITION: {composition}. "
        f"FIRST FRAME: {first_frame_prompt} "
        f"MOTION: {motion_prompt} "
        f"END FRAME: {end_frame_prompt}"
    )


def refresh_shot_prompts(shot: StoryboardShot) -> None:
    """Rebuild Provider-bound prompts only from reviewed structured shot fields."""

    shot.first_frame_prompt = _first_frame_prompt_text(
        location=shot.location,
        time_of_day=shot.time_of_day,
        character=shot.character,
        start_frame=shot.start_frame,
        composition=shot.composition,
        camera=shot.camera,
        visual_style=shot.visual_style,
        color_palette=shot.color_palette,
        lighting=shot.lighting,
        anchors=shot.visual_anchors,
        retake_instruction=shot.retake_instruction,
    )
    shot.motion_prompt = _motion_prompt_text(
        duration=shot.duration,
        action=shot.action,
        camera_movement=shot.camera_movement,
        dialogue=shot.dialogue,
        retake_instruction=shot.retake_instruction,
    )
    shot.end_frame_prompt = _end_frame_prompt_text(
        end_frame=shot.end_frame,
        location=shot.location,
        retake_instruction=shot.retake_instruction,
    )
    shot.negative_prompt = _negative_prompt_text()
    shot.video_prompt = _video_prompt_text(
        purpose=shot.shot_purpose,
        camera=shot.camera,
        composition=shot.composition,
        first_frame_prompt=shot.first_frame_prompt,
        motion_prompt=shot.motion_prompt,
        end_frame_prompt=shot.end_frame_prompt,
    )
    apply_transition_prompt_context([shot])


def shot_prompts_match_content(shot: StoryboardShot) -> bool:
    expected = deepcopy(shot)
    refresh_shot_prompts(expected)
    return (
        shot.first_frame_prompt == expected.first_frame_prompt
        and shot.motion_prompt == expected.motion_prompt
        and shot.end_frame_prompt == expected.end_frame_prompt
        and shot.video_prompt == expected.video_prompt
        and shot.negative_prompt == expected.negative_prompt
    )


def apply_transition_prompt_context(shots: list[StoryboardShot]) -> None:
    """Idempotently describe the edit relationship in each one-shot prompt."""
    for shot in shots:
        if shot.continuity_mode == "same_scene_chain":
            instruction = (
                f"转场关系：连续动作，承接镜头 {shot.previous_shot_id} 的真实结束画面；"
                "保持机位运动和动作方向连续，不重新建立构图。"
            )
        elif shot.continuity_mode == "same_scene_reference":
            instruction = (
                f"转场关系：{shot.transition_type}。这是同一场景内的正常切镜；"
                "重新建立本镜头的景别、角度和构图，不沿用上一镜头画面，"
                "但保持人物身份、服装、伤势、道具位置、地点、时段和主光方向一致。"
            )
        elif shot.continuity_mode == "new_scene_reference":
            instruction = (
                "转场关系：场景切换。使用新地点和本镜头自己的机位构图，"
                "不继承上一镜头画面；只保留跨场景仍成立的人物身份和剧情状态。"
            )
        else:
            instruction = "转场关系：独立镜头。按照本镜头描述重新建立机位和构图。"
        marker = f"[TRANSITION]{instruction}[/TRANSITION] "
        first_frame = _strip_transition_prompt_context(shot.first_frame_prompt)
        video_prompt = _strip_transition_prompt_context(shot.video_prompt)
        shot.first_frame_prompt = f"{marker}{first_frame}"
        shot.video_prompt = f"{marker}{video_prompt}"


def _strip_transition_prompt_context(value: str) -> str:
    return re.sub(
        r"\[TRANSITION\].*?\[/TRANSITION\]\s*",
        "",
        str(value or ""),
        count=1,
    )


def _dialogue_assignments_for_units(
    units: list[_ShotUnit],
    scenes: list[StoryScene],
) -> list[str]:
    assignments = [""] * len(units)
    for scene in scenes:
        positions = [
            index
            for index, unit in enumerate(units)
            if unit.scene.scene_id == scene.scene_id and unit.kind == "dialogue"
        ]
        chunks = _split_spoken_text(scene.dialogue)
        if chunks and not positions:
            raise ValueError(
                f"场景 {scene.scene_id} 含对白，但导演分镜没有 dialogue 镜头。"
            )
        _distribute_spoken_chunks(assignments, positions, chunks)
        empty_positions = [
            index + 1
            for index in positions
            if not assignments[index].strip()
        ]
        if empty_positions:
            raise ValueError(
                f"场景 {scene.scene_id} 的 dialogue 镜头多于可分配对白段落：镜头"
                + "、".join(str(index) for index in empty_positions)
            )
    return assignments


def _narration_assignments_for_units(
    units: list[_ShotUnit],
    scenes: list[StoryScene],
) -> list[str]:
    return _build_narration_assignments(
        [unit.scene.scene_id for unit in units],
        scenes,
    )


def _build_narration_assignments(
    scene_ids: list[int],
    scenes: list[StoryScene],
) -> list[str]:
    """Distribute each scene narration once before timing floors are calculated."""

    assignments = [""] * len(scene_ids)
    for scene in scenes:
        positions = [
            index
            for index, scene_id in enumerate(scene_ids)
            if scene_id == scene.scene_id
        ]
        chunks = _split_spoken_text(scene.narration)
        _distribute_spoken_chunks(assignments, positions, chunks)
    return assignments


def _validate_spoken_overrides(
    values: list[str],
    units: list[_ShotUnit],
    scenes: list[StoryScene],
    *,
    field_name: str,
    source_field: str,
) -> list[str]:
    if len(values) != len(units):
        raise ValueError(f"逐镜{field_name}数量必须与镜头数量一致。")
    cleaned = [" ".join(str(value or "").split()) for value in values]
    if source_field == "dialogue":
        invalid = [
            index
            for index, (unit, value) in enumerate(zip(units, cleaned), start=1)
            if value and unit.kind != "dialogue"
        ]
        if invalid:
            raise ValueError(
                "对白只能分配给 dialogue 镜头：镜头"
                + "、".join(str(index) for index in invalid)
            )
        empty_dialogue_shots = [
            index
            for index, (unit, value) in enumerate(zip(units, cleaned), start=1)
            if unit.kind == "dialogue" and not value
        ]
        if empty_dialogue_shots:
            raise ValueError(
                "dialogue 镜头缺少对应对白：镜头"
                + "、".join(str(index) for index in empty_dialogue_shots)
            )
    for scene in scenes:
        positions = [
            index
            for index, unit in enumerate(units)
            if unit.scene.scene_id == scene.scene_id
        ]
        expected = str(getattr(scene, source_field, "") or "")
        actual = " ".join(cleaned[index] for index in positions if cleaned[index])
        if _normalize_spoken_text(actual) != _normalize_spoken_text(expected):
            raise ValueError(
                f"场景 {scene.scene_id} 的逐镜{field_name}与已确认剧本不一致。"
            )
        if (
            source_field == "dialogue"
            and expected.strip()
            and not any(units[index].kind == "dialogue" for index in positions)
        ):
            raise ValueError(
                f"场景 {scene.scene_id} 含对白，但导演分镜没有 dialogue 镜头。"
            )
    return cleaned


def _distribute_spoken_chunks(
    assignments: list[str],
    positions: list[int],
    chunks: list[str],
) -> None:
    if not positions or not chunks:
        return
    if len(chunks) <= len(positions):
        for index, chunk in enumerate(chunks):
            position_index = min(
                len(positions) - 1,
                index * len(positions) // len(chunks),
            )
            assignments[positions[position_index]] = chunk
        return
    for position_index, position in enumerate(positions):
        start = position_index * len(chunks) // len(positions)
        end = (position_index + 1) * len(chunks) // len(positions)
        assignments[position] = " ".join(chunks[start:end])


def _assign_narration_timeline(
    shots: list[StoryboardShot],
    narrations: list[str],
) -> None:
    if len(shots) != len(narrations):
        raise ValueError("旁白时间轴数量必须与镜头数量一致。")
    for shot, narration in zip(shots, narrations):
        shot.narration = narration


def _split_spoken_text(text: str) -> list[str]:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return []
    chunks = [
        item.strip()
        for item in re.findall(
            r".+?(?:[。！？!?；;]|(?<!\d)\.(?!\d)|$)",
            cleaned,
        )
        if item.strip()
    ]
    return chunks or [cleaned]


def _normalize_spoken_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def derive_storyboard_seed(script: StoryScript, facts: StoryFacts) -> int:
    """Derive a stable base seed from confirmed story-to-script inputs."""
    payload = {
        "script": to_plain_data(script),
        "facts": to_plain_data(facts),
    }
    return _stable_seed(payload)


def derive_shot_seed(base_seed: int, scene_id: int, shot_id: int) -> int:
    """Derive a stable positive 31-bit seed for one confirmed shot."""
    return _stable_seed(
        {
            "base_seed": int(base_seed),
            "scene_id": int(scene_id),
            "shot_id": int(shot_id),
        }
    )


def derive_retake_seed(
    previous_seed: int | None,
    shot_id: int,
    revision_payload: object,
) -> int:
    """Retakes deliberately move to a new deterministic seed."""
    return _stable_seed(
        {
            "previous_seed": previous_seed,
            "shot_id": int(shot_id),
            "revision": revision_payload,
        }
    )


def _derive_continuity_states(
    units: list[_ShotUnit],
    bible: VisualBible,
) -> list[tuple[ContinuityState, ContinuityState]]:
    """Advance formal state shot by shot while preserving cross-scene identity."""
    result: list[tuple[ContinuityState, ContinuityState]] = []
    current: ContinuityState | None = None
    previous_scene: StoryScene | None = None
    for unit in units:
        if current is None:
            start = _new_scene_state(current, unit.scene, bible)
        elif unit.transition_type == "scene_change":
            start = _new_scene_state(current, unit.scene, bible)
        elif previous_scene is not None and unit.scene.scene_id == previous_scene.scene_id:
            start = deepcopy(current)
        elif previous_scene is not None and _scenes_share_physical_context(
            previous_scene,
            unit.scene,
        ):
            start = _continue_physical_scene_state(current, unit.scene, bible)
        else:
            start = _new_scene_state(current, unit.scene, bible)
        end = _apply_shot_transition(start, unit)
        result.append((start, end))
        current = deepcopy(end)
        previous_scene = unit.scene
    return result


def _new_scene_state(
    previous: ContinuityState | None,
    scene: StoryScene,
    bible: VisualBible,
) -> ContinuityState:
    if previous is None:
        state = ContinuityState()
    else:
        state = ContinuityState(
            character_appearance=deepcopy(previous.character_appearance),
            character_clothing=deepcopy(previous.character_clothing),
            character_knowledge=deepcopy(previous.character_knowledge),
            character_injuries=deepcopy(previous.character_injuries),
            character_held_props=deepcopy(previous.character_held_props),
            prop_positions=deepcopy(previous.prop_positions),
        )
    descriptions = {
        asset.name: asset.description
        for asset in bible.assets
        if asset.kind == "character"
    }
    start_description = (
        scene.start_state.strip()
        or f"{_join(scene.characters) or '人物'}位于{scene.location}"
    )
    for name in scene.characters:
        anchor = descriptions.get(name, f"{name}外观沿用已确认人物设定")
        state.character_appearance.setdefault(name, anchor)
        state.character_clothing.setdefault(name, anchor)
        state.character_positions[name] = start_description
        state.character_emotions.pop(name, None)
    state.location = scene.location
    state.time_of_day = scene.time_of_day or "未说明时段"
    state.weather = _weather_from_scene(scene)
    state.key_light_direction = _light_from_scene(scene, bible)
    for prop in scene.props:
        state.prop_positions.setdefault(prop, start_description)
        if any(marker in start_description for marker in ("手中", "拿着", "握着", "口袋")):
            holder = scene.characters[0] if scene.characters else "主角"
            held = state.character_held_props.setdefault(holder, [])
            if prop not in held:
                held.append(prop)
    _update_injuries(state, scene.characters, start_description)
    return state


def _continue_physical_scene_state(
    previous: ContinuityState,
    scene: StoryScene,
    bible: VisualBible,
) -> ContinuityState:
    state = deepcopy(previous)
    descriptions = {
        asset.name: asset.description
        for asset in bible.assets
        if asset.kind == "character"
    }
    for name in scene.characters:
        anchor = descriptions.get(name, f"{name}外观沿用已确认人物设定")
        state.character_appearance.setdefault(name, anchor)
        state.character_clothing.setdefault(name, anchor)
        state.character_positions.setdefault(
            name,
            scene.start_state.strip() or f"{name}仍位于{scene.location}",
        )
    state.location = scene.location
    state.time_of_day = scene.time_of_day or state.time_of_day
    weather = _weather_from_scene(scene)
    if weather != "天气未说明":
        state.weather = weather
    state.key_light_direction = _light_from_scene(scene, bible) or state.key_light_direction
    for prop in scene.props:
        state.prop_positions.setdefault(
            prop,
            scene.start_state.strip() or f"{prop}仍位于上一镜头记录的位置",
        )
    _update_injuries(state, scene.characters, scene.start_state)
    return state


def _apply_shot_transition(
    start: ContinuityState,
    unit: _ShotUnit,
) -> ContinuityState:
    end = deepcopy(start)
    scene = unit.scene
    action = unit.action.strip()
    position = (
        scene.end_state.strip()
        if unit.kind == "transition" and scene.end_state.strip()
        else action
    )
    for name in scene.characters:
        end.character_positions[name] = position
    if unit.kind == "reaction" and scene.emotional_change.strip():
        for name in scene.characters:
            end.character_emotions[name] = scene.emotional_change.strip()
    if unit.kind == "dialogue" or any(
        marker in f"{unit.purpose}{action}"
        for marker in ("揭示", "证据", "发现", "确认", "真相", "得知")
    ):
        learned = scene.dialogue.strip() or action
        for name in scene.characters:
            knowledge = end.character_knowledge.setdefault(name, [])
            if learned and learned not in knowledge:
                knowledge.append(learned)
    _update_prop_state(end, scene, action)
    _update_injuries(end, scene.characters, action)
    return end


def _update_prop_state(
    state: ContinuityState,
    scene: StoryScene,
    action: str,
) -> None:
    if not scene.props:
        return
    actor = scene.characters[0] if scene.characters else "主角"
    receiver = scene.characters[1] if len(scene.characters) > 1 else ""
    for prop in scene.props:
        if prop not in action:
            continue
        if any(marker in action for marker in ("拿", "握", "捡", "拾", "接过", "取出", "掏出")):
            held = state.character_held_props.setdefault(actor, [])
            if prop not in held:
                held.append(prop)
            state.prop_positions[prop] = f"{actor}手中：{action}"
        if any(marker in action for marker in ("放", "推", "扔", "丢", "藏")):
            for props in state.character_held_props.values():
                if prop in props:
                    props.remove(prop)
            state.prop_positions[prop] = action
        if any(marker in action for marker in ("递给", "交给")):
            for name, props in state.character_held_props.items():
                if name != receiver and prop in props:
                    props.remove(prop)
            if receiver:
                held = state.character_held_props.setdefault(receiver, [])
                if prop not in held:
                    held.append(prop)
                state.prop_positions[prop] = f"{receiver}手中：{action}"
            else:
                state.prop_positions[prop] = action


def _update_injuries(
    state: ContinuityState,
    characters: list[str],
    text: str,
) -> None:
    if not any(
        marker in text
        for marker in ("受伤", "伤口", "流血", "中弹", "割伤", "擦伤", "骨折")
    ):
        return
    for name in characters:
        if name in text or len(characters) == 1:
            state.character_injuries[name] = text


def _weather_from_scene(scene: StoryScene) -> str:
    source = f"{scene.location} {scene.time_of_day} {scene.action} {scene.start_state}"
    for marker, value in (
        ("暴雨", "暴雨"),
        ("雨", "雨"),
        ("暴雪", "暴雪"),
        ("雪", "雪"),
        ("雾", "雾"),
        ("晴", "晴"),
        ("阴", "阴"),
    ):
        if marker in source:
            return value
    if any(marker in scene.location for marker in ("室", "房", "厅", "车内", "仓库")):
        return "室内，天气不适用"
    return "天气未说明"


def _light_from_scene(scene: StoryScene, bible: VisualBible) -> str:
    source = f"{scene.time_of_day} {scene.start_state} {bible.lighting_rules}"
    direction = next(
        (
            label
            for marker, label in (
                ("左", "画面左侧"),
                ("右", "画面右侧"),
                ("逆光", "人物后方"),
                ("顶光", "人物上方"),
                ("正面", "镜头方向"),
            )
            if marker in source
        ),
        "延续场景既定主光方向",
    )
    return f"{direction}（{scene.time_of_day or '未说明时段'}）"


def _state_frame_summary(state: ContinuityState, fallback: str) -> str:
    positions = "；".join(
        f"{name}:{value}" for name, value in state.character_positions.items()
    )
    props = "；".join(f"{name}:{value}" for name, value in state.prop_positions.items())
    emotions = "；".join(
        f"{name}:{value}" for name, value in state.character_emotions.items()
    )
    parts = [
        fallback.strip(),
        f"人物位置[{positions}]" if positions else "",
        f"道具位置[{props}]" if props else "",
        f"情绪[{emotions}]" if emotions else "",
    ]
    return "；".join(item for item in parts if item)


def _scene_duration_weight(scene: StoryScene) -> float:
    try:
        provided = float(scene.duration_weight)
    except (TypeError, ValueError):
        provided = 0.0
    if math.isfinite(provided) and provided > 0:
        return provided
    weight, _ = estimate_scene_duration_weight(
        title=scene.title,
        action=scene.visible_action or scene.action,
        dialogue=scene.dialogue,
        narration=scene.narration,
        emotional_change=scene.emotional_change,
    )
    return max(0.1, weight)


def _stable_seed(payload: object) -> int:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")
    return value & 0x7FFFFFFF


def _reference_ids(bible: VisualBible, scene: StoryScene) -> list[str]:
    values: list[str] = []
    scene_names = {*scene.characters, scene.location, *scene.props}
    for asset in bible.assets:
        if asset.name in scene_names:
            values.append(asset.asset_id)
    return values


def _matching_anchor(name: str, source: str, fallback: str) -> str:
    segments = [item.strip() for item in re.split(r"[；\n]", source) if item.strip()]
    return next((item for item in segments if name in item), source.strip() or fallback)


def _split_anchors(source: str) -> list[str]:
    return [item.strip() for item in re.split(r"[；，,、\n]", source) if item.strip()]


def _unique(values) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _join(values) -> str:
    return "、".join(str(value) for value in values if str(value).strip())


def _evenly_spaced(indexes: list[int], count: int) -> list[int]:
    if count >= len(indexes):
        return indexes
    if count == 1:
        return [indexes[0]]
    return [indexes[round(i * (len(indexes) - 1) / (count - 1))] for i in range(count)]

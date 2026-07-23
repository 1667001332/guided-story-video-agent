from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .models import (
    StoryFacts,
    StoryScene,
    StoryScript,
    StoryboardPlan,
    StoryboardShot,
    VisualAsset,
    VisualBible,
)
from .timing import allocate_durations


@dataclass(slots=True)
class _ShotUnit:
    scene: StoryScene
    kind: str
    action: str
    purpose: str
    priority: int
    required: bool = False


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
) -> StoryboardPlan:
    if not script.confirmed:
        raise RuntimeError("请先确认剧本，再生成分镜。")
    if not script.scenes:
        raise ValueError("剧本没有可转换的场景。")
    bible = visual_bible or build_visual_bible(script, facts)
    units = _plan_shot_units(script)
    durations = allocate_durations(script.target_seconds, len(units), minimum=3, maximum=15)
    shots = [
        _build_shot(index, unit, duration, bible, facts, len(units))
        for index, (unit, duration) in enumerate(zip(units, durations), start=1)
    ]
    plan = StoryboardPlan(
        title=script.title,
        target_seconds=script.target_seconds,
        shots=shots,
        narration_text="\n".join(
            scene.narration for scene in script.scenes if scene.narration.strip()
        ),
        visual_bible=bible,
    )
    if abs(plan.total_duration - script.target_seconds) > 1:
        raise ValueError("分镜总时长与目标时长不一致。")
    return plan


def _plan_shot_units(script: StoryScript) -> list[_ShotUnit]:
    """Derive shots from filmable events instead of a fixed camera sequence."""
    units: list[_ShotUnit] = []
    previous_location = ""
    for scene in script.scenes:
        action = (scene.visible_action or scene.action).strip()
        if not action:
            action = "人物完成一个清晰可见的动作"
        if not previous_location or scene.location != previous_location:
            units.append(
                _ShotUnit(
                    scene,
                    "establish",
                    f"建立{scene.location}的空间关系，并让动作开始：{action}",
                    f"交代《{scene.title}》的空间、人物位置和行动起点",
                    priority=4 if not previous_location else 2,
                )
            )
        units.append(
            _ShotUnit(
                scene,
                "action",
                action,
                f"完整呈现《{scene.title}》中推动故事的可见行动",
                priority=6,
                required=True,
            )
        )
        if scene.props:
            units.append(
                _ShotUnit(
                    scene,
                    "detail",
                    f"突出{_join(scene.props)}与人物动作之间的关系",
                    f"让关键道具成为《{scene.title}》的叙事证据",
                    priority=4,
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
                )
            )
        previous_location = scene.location

    maximum = max(1, min(12, script.target_seconds // 3))
    minimum = max(
        math.ceil(script.target_seconds / 15),
        min(len(script.scenes), maximum),
        min(math.ceil(script.target_seconds / 12), maximum),
    )
    units = _trim_units(units, maximum)
    while len(units) < minimum:
        source = max(
            (item for item in units if item.kind == "action"),
            key=lambda item: item.scene.duration,
            default=units[-1],
        )
        insertion = units.index(source) + 1
        units.insert(
            insertion,
            _ShotUnit(
                source.scene,
                "action",
                f"{source.action}过程中出现的关键变化，动作方向与前后镜头连续",
                f"补足《{source.scene.title}》中不能被一个镜头省略的行动过程",
                priority=5,
                required=True,
            ),
        )
    return units


def _trim_units(units: list[_ShotUnit], maximum: int) -> list[_ShotUnit]:
    if len(units) <= maximum:
        return units
    required_indexes = [index for index, unit in enumerate(units) if unit.required]
    if len(required_indexes) >= maximum:
        chosen = set(_evenly_spaced(required_indexes, maximum))
    else:
        slots = maximum - len(required_indexes)
        extras = sorted(
            (index for index, unit in enumerate(units) if not unit.required),
            key=lambda index: (-units[index].priority, index),
        )
        chosen = set(required_indexes + extras[:slots])
    return [unit for index, unit in enumerate(units) if index in chosen]


def _build_shot(
    shot_id: int,
    unit: _ShotUnit,
    duration: int,
    bible: VisualBible,
    facts: StoryFacts,
    total_shots: int,
) -> StoryboardShot:
    scene = unit.scene
    camera, movement, composition = CAMERA_BY_KIND[unit.kind]
    if shot_id == total_shots:
        camera, movement, composition = (
            "wide closing shot",
            "slow pull-back",
            "保留动作结果、人物关系和情绪余韵",
        )
    character = scene.characters[0] if scene.characters else "主角"
    start_frame = scene.start_state or f"{character}位于{scene.location}，动作尚未完成"
    end_frame = scene.end_state or unit.action
    reference_ids = _reference_ids(bible, scene)
    anchors = [
        asset.description for asset in bible.assets if asset.asset_id in reference_ids
    ]
    first_frame_prompt = (
        f"电影分镜首帧。地点与时段：{scene.location}，{scene.time_of_day}。"
        f"人物：{_join(scene.characters) or character}。起始状态：{start_frame}。"
        f"构图：{composition}。摄影：{camera}。"
        f"视觉风格：{bible.visual_style}；色彩：{bible.color_palette}。"
        f"身份与场景锚点：{_join(anchors)}。画面静止、结构清楚、可作为图生视频首帧。"
    )
    motion_prompt = (
        f"在{duration}秒内完成一个连续动作：{unit.action}。"
        f"摄影机运动：{movement}。动作遵守真实物理过程，不跳步，不瞬移。"
        f"人物视线、移动方向、道具位置和光线方向承接首帧。"
    )
    end_frame_prompt = (
        f"结束帧必须明确到达：{end_frame}。保持{scene.location}的空间结构、"
        f"角色身份、服装、道具状态和色彩连续，可自然衔接下一镜头。"
    )
    prompt = (
        f"Cinematic narrative shot. Purpose: {unit.purpose}. "
        f"FIRST FRAME: {first_frame_prompt} "
        f"MOTION: {motion_prompt} "
        f"END FRAME: {end_frame_prompt}"
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
        narration=scene.narration,
        video_prompt=prompt,
        negative_prompt=(
            "inconsistent identity, changed face, changed costume, duplicated character, "
            "broken anatomy, teleporting object, unreadable text, watermark, flicker, jump cut"
        ),
        continuity_notes=continuity,
        shot_purpose=unit.purpose,
        composition=composition,
        camera_movement=movement,
        start_frame=start_frame,
        end_frame=end_frame,
        visual_anchors=anchors,
        shot_kind=unit.kind,
        first_frame_prompt=first_frame_prompt,
        motion_prompt=motion_prompt,
        end_frame_prompt=end_frame_prompt,
        reference_asset_ids=reference_ids,
    )


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
    return [
        item.strip()
        for item in re.split(r"[；，,、\n]", source)
        if item.strip()
    ]


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

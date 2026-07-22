from __future__ import annotations

from .models import StoryFacts, StoryScript, StoryboardPlan, StoryboardShot
from .timing import allocate_durations


CAMERA_PLAN = (
    ("wide establishing shot", "slow push-in", "主体位于环境中，保留空间关系"),
    ("medium tracking shot", "lateral tracking", "主角沿画面运动方向推进"),
    ("insert close-up", "locked camera", "关键道具占据画面中心"),
    ("medium close-up", "subtle handheld", "人物反应与环境压力同框"),
    ("over-the-shoulder shot", "slow pan", "前景遮挡制造对峙关系"),
    ("low-angle medium shot", "dolly forward", "强化阻力和失控感"),
    ("extreme close-up", "rack focus", "从细节揭示转折信息"),
    ("wide reveal shot", "pull back", "扩大空间展示真相"),
    ("profile two-shot", "controlled orbit", "让最终选择发生在同一空间"),
    ("wide closing shot", "slow pull-back", "复现开场构图并留出情绪余韵"),
)


def build_storyboard(script: StoryScript, facts: StoryFacts) -> StoryboardPlan:
    if not script.confirmed:
        raise RuntimeError("请先确认剧本，再生成分镜。")
    if not script.scenes:
        raise ValueError("剧本没有可转换的场景。")
    shot_count = min(10, max(5, round(script.target_seconds / 6)))
    durations = allocate_durations(script.target_seconds, shot_count, minimum=3, maximum=15)
    anchors = _visual_anchors(facts)
    shots: list[StoryboardShot] = []
    for index, duration in enumerate(durations):
        scene_index = min(len(script.scenes) - 1, index * len(script.scenes) // shot_count)
        scene = script.scenes[scene_index]
        character = scene.characters[0] if scene.characters else "主角"
        camera, movement, composition = CAMERA_PLAN[index]
        purpose = _shot_purpose(index, shot_count, scene.title)
        start_frame = scene.start_state or f"{character}处于{scene.location}，动作尚未完成"
        end_frame = scene.end_state or scene.visible_action or scene.action
        continuity = [
            f"人物外观锁定：{facts.character_visuals}",
            f"场景规则锁定：{facts.scene_details}",
            f"关键道具锁定：{facts.props}",
            f"承接方式：{facts.transitions}",
        ]
        prompt = (
            f"Cinematic short film shot. Narrative purpose: {purpose}. "
            f"Location and time: {scene.location}, {scene.time_of_day}. "
            f"Character identity lock: {character}; {facts.character_visuals}. "
            f"Visible action: {scene.visible_action or scene.action}. "
            f"Composition: {composition}. Camera: {camera}; movement: {movement}. "
            f"Start frame: {start_frame}. End frame: {end_frame}. "
            f"Props and visual anchors: {', '.join(anchors)}. "
            f"Style: {facts.camera_style or 'realistic cinematic lighting'}, "
            f"{facts.tone or 'restrained emotional tone'}, coherent spatial continuity."
        )
        shots.append(
            StoryboardShot(
                shot_id=index + 1,
                scene_id=scene.scene_id,
                duration=duration,
                character=character,
                location=scene.location,
                visual=scene.visible_action or scene.action,
                action=scene.visible_action or scene.action,
                camera=camera,
                lighting="motivated cinematic lighting, stable scene color palette",
                mood=facts.tone or scene.emotional_change or "克制、连贯、电影感",
                narration=scene.narration,
                video_prompt=prompt,
                negative_prompt=(
                    "inconsistent face, changed costume, duplicated character, broken anatomy, "
                    "unreadable text, watermark, abrupt location change, jump cut, flicker"
                ),
                continuity_notes=continuity,
                shot_purpose=purpose,
                composition=composition,
                camera_movement=movement,
                start_frame=start_frame,
                end_frame=end_frame,
                visual_anchors=anchors,
            )
        )
    plan = StoryboardPlan(
        title=script.title,
        target_seconds=script.target_seconds,
        shots=shots,
        narration_text="\n".join(scene.narration for scene in script.scenes if scene.narration.strip()),
    )
    if abs(plan.total_duration - script.target_seconds) > 1:
        raise ValueError("分镜总时长与目标时长不一致。")
    return plan


def _visual_anchors(facts: StoryFacts) -> list[str]:
    values = [
        facts.character_visuals,
        facts.scene_details,
        facts.props,
        facts.visual_anchors,
    ]
    return [value.strip() for value in values if value.strip()]


def _shot_purpose(index: int, count: int, scene_title: str) -> str:
    if index == 0:
        return f"建立世界并呈现开场异常：{scene_title}"
    if index == count - 1:
        return f"完成选择并形成结尾回响：{scene_title}"
    progress = index / max(1, count - 1)
    if progress < 0.4:
        return f"明确目标并推动行动：{scene_title}"
    if progress < 0.7:
        return f"升级阻力并压缩选择空间：{scene_title}"
    return f"揭示转折并逼近最终选择：{scene_title}"

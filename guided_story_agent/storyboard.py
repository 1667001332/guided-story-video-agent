from __future__ import annotations

from .models import StoryFacts, StoryScript, StoryboardPlan, StoryboardShot


def build_storyboard(script: StoryScript, facts: StoryFacts) -> StoryboardPlan:
    if not script.confirmed:
        raise RuntimeError("请先确认剧本，再生成分镜。")
    shots: list[StoryboardShot] = []
    for scene in script.scenes:
        character = scene.characters[0] if scene.characters else "主角"
        continuity = [
            f"人物外观保持一致：{facts.character_visuals}",
            f"关键道具保持一致：{facts.props}",
            f"承接方式：{facts.transitions}",
        ]
        prompt = (
            f"Cinematic short film, {scene.location}, {scene.time_of_day}. "
            f"Character: {character}. Visible action: {scene.action}. "
            f"Visual style: consistent {facts.character_visuals}; props: {facts.props}. "
            "Clear subject action, coherent spatial continuity, realistic cinematic lighting."
        )
        shots.append(
            StoryboardShot(
                shot_id=len(shots) + 1,
                scene_id=scene.scene_id,
                duration=scene.duration,
                character=character,
                location=scene.location,
                visual=scene.action,
                action=scene.action,
                camera=_camera_for_scene(scene.scene_id),
                lighting="motivated cinematic lighting, stable color palette",
                mood="克制、连贯、电影感",
                narration=scene.narration,
                video_prompt=prompt,
                negative_prompt=(
                    "inconsistent face, changed costume, duplicated character, broken anatomy, "
                    "unreadable text, watermark, abrupt location change"
                ),
                continuity_notes=continuity,
            )
        )
    plan = StoryboardPlan(
        title=script.title,
        target_seconds=script.target_seconds,
        shots=shots,
        narration_text="\n".join(scene.narration for scene in script.scenes),
    )
    if abs(plan.total_duration - script.target_seconds) > 1:
        raise ValueError("分镜总时长与目标时长不一致。")
    return plan


def _camera_for_scene(scene_id: int) -> str:
    cameras = {
        1: "wide establishing shot, slow push-in",
        2: "medium tracking shot",
        3: "handheld medium close-up",
        4: "close-up then reveal",
        5: "wide closing shot, slow pull-back",
    }
    return cameras.get(scene_id, "cinematic medium shot")

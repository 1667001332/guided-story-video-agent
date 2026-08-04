"""Build the provider-independent whole-video request from a confirmed script."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from .models import StoryDraft, StoryFacts, StoryScript, VideoJob


def build_video_job(
    script: StoryScript,
    *,
    story: StoryDraft | None = None,
    facts: StoryFacts | None = None,
    visual_style: str = "",
    include_narration_in_prompt: bool = False,
) -> VideoJob:
    """Convert a script into one complete VideoJob.

    Scene boundaries remain semantic context in the prompt and metadata.  No
    fixed per-scene duration, shot count, or 3–15 second rule is introduced
    here; those decisions belong to the selected provider adapter.

    ``include_narration_in_prompt`` is off by default: video models that read
    the prompt aloud produce clumsy voiceovers, and narration is best served
    by a dedicated TTS track (edge-tts) in the multi-shot render path.
    Character dialogue stays in the prompt because the actors must speak it.
    """

    if not isinstance(script, StoryScript) or not script.confirmed:
        raise RuntimeError("只能从已确认的剧本创建视频任务。")
    if not script.scenes:
        raise ValueError("剧本至少需要一个可拍摄场景。")
    if isinstance(script.target_seconds, bool) or int(script.target_seconds) <= 0:
        raise ValueError("视频目标时长必须是正整数。")

    lines = [
        "生成一支完整、连续的叙事视频，不要把每个场景当作独立短片。",
        f"总时长约 {int(script.target_seconds)} 秒；场景时长只用于叙事节奏，不是固定分镜切片。",
    ]
    if story is not None:
        if story.logline.strip():
            lines.append(f"故事梗概：{story.logline.strip()}")
        if story.story_text.strip():
            lines.append(f"故事正文：{story.story_text.strip()}")
        if story.characters:
            identities = "；".join(
                f"{item.name}：{item.visual_identity or item.description}"
                for item in story.characters
            )
            lines.append(f"人物身份连续性：{identities}")
        if story.locations:
            locations = "；".join(
                f"{item.name}：{item.visual_identity or item.description}"
                for item in story.locations
            )
            lines.append(f"地点连续性：{locations}")
    if facts is not None:
        narration_style = facts.narration_style.strip()
        for label, value in (
            ("视觉锚点", facts.visual_anchors),
            ("镜头语言", facts.camera_style),
            ("转场要求", facts.transitions),
        ):
            if value.strip():
                lines.append(f"{label}：{value.strip()}")
        if include_narration_in_prompt and narration_style:
            lines.append(f"旁白风格：{narration_style}")

    lines.append("按以下叙事顺序自然完成动作、情绪和空间连续性：")
    scene_metadata: list[dict[str, Any]] = []
    for scene in script.scenes:
        action = (scene.visible_action or scene.action).strip()
        scene_line = f"场景 {scene.scene_id}（叙事时长 {scene.duration} 秒）："
        scene_line += f"{scene.title}；地点：{scene.location}；时间：{scene.time_of_day}；"
        scene_line += f"人物：{'、'.join(scene.characters)}；动作：{action}"
        if scene.dialogue.strip():
            scene_line += f"；对白：{scene.dialogue.strip()}"
        if include_narration_in_prompt and scene.narration.strip():
            scene_line += f"；旁白：{scene.narration.strip()}"
        if scene.emotional_change.strip():
            scene_line += f"；情绪变化：{scene.emotional_change.strip()}"
        lines.append(scene_line)
        scene_metadata.append(
            {
                "scene_id": scene.scene_id,
                "title": scene.title,
                "duration": scene.duration,
                "location": scene.location,
                "time_of_day": scene.time_of_day,
                "characters": list(scene.characters),
                "action": action,
                "dialogue": scene.dialogue,
                "narration": scene.narration,
            }
        )

    prompt = "\n".join(lines)
    negative = (
        "人物身份、服装、道具和空间位置前后不一致；突然跳切；"
        "无意义的镜头拼贴；乱码字幕；水印；肢体畸变；低清晰度"
    )
    return VideoJob(
        title=script.title,
        prompt=prompt,
        negative_prompt=negative,
        target_seconds=int(script.target_seconds),
        visual_style=visual_style.strip(),
        narration="\n".join(scene.narration.strip() for scene in script.scenes if scene.narration.strip()),
        dialogue="\n".join(scene.dialogue.strip() for scene in script.scenes if scene.dialogue.strip()),
        metadata={"scenes": deepcopy(scene_metadata), "source": "confirmed_script"},
        job_id=f"video-job-{uuid4().hex[:16]}",
        confirmed=True,
    )

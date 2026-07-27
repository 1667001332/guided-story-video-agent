from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .models import StoryboardPlan


class NarrationUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class NarrationArtifact:
    audio_path: str
    subtitle_path: str


@dataclass(frozen=True, slots=True)
class NarrationCue:
    shot_id: int
    start_seconds: int
    end_seconds: int
    text: str


class EdgeNarrationSynthesizer:
    """Create narration before any paid video call and always export an SRT timeline."""

    def __init__(self, voice: str | None = None, rate: str = "+0%") -> None:
        self.voice = voice or os.getenv("NARRATION_VOICE", "zh-CN-XiaoxiaoNeural")
        self.rate = rate

    def synthesize(self, plan: StoryboardPlan, output_dir: str | Path) -> NarrationArtifact:
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        normalize_narration_timeline(plan)
        timeline_text = narration_text_from_timeline(plan)
        subtitle_path = target / "narration.srt"
        subtitle_path.write_text(build_srt(plan), encoding="utf-8")
        plan.subtitle_path = str(subtitle_path)
        plan.audio_path = ""
        if not timeline_text:
            return NarrationArtifact(audio_path="", subtitle_path=str(subtitle_path))
        try:
            import edge_tts
        except ImportError as exc:
            raise NarrationUnavailable("未安装 edge-tts；已生成字幕，但暂时没有旁白音频。") from exc
        audio_path = target / "narration.mp3"
        try:
            communicate = edge_tts.Communicate(
                timeline_text,
                self.voice,
                rate=self.rate,
            )
            communicate.save_sync(str(audio_path))
        except Exception as exc:
            audio_path.unlink(missing_ok=True)
            raise NarrationUnavailable(f"Edge TTS 旁白生成失败：{exc}") from exc
        if not audio_path.is_file() or audio_path.stat().st_size == 0:
            raise NarrationUnavailable("Edge TTS 没有生成可用音频。")
        plan.audio_path = str(audio_path)
        return NarrationArtifact(str(audio_path), str(subtitle_path))


def build_srt(plan: StoryboardPlan) -> str:
    lines: list[str] = []
    for index, cue in enumerate(narration_timeline(plan), start=1):
        lines.extend(
            [
                str(index),
                f"{_timestamp(cue.start_seconds)} --> {_timestamp(cue.end_seconds)}",
                cue.text,
                "",
            ]
        )
    return "\n".join(lines)


def normalize_narration_timeline(plan: StoryboardPlan) -> None:
    """Migrate duplicate scene narration into one canonical shot timeline."""
    seen_by_scene: dict[int, set[str]] = {}
    has_shot_narration = False
    for shot in plan.shots:
        text = _clean_text(shot.narration)
        seen = seen_by_scene.setdefault(shot.scene_id, set())
        if text and text in seen:
            shot.narration = ""
            continue
        shot.narration = text
        if text:
            seen.add(text)
            has_shot_narration = True
    if not has_shot_narration and plan.shots:
        legacy_text = _clean_text(plan.narration_text)
        if legacy_text:
            plan.shots[0].narration = legacy_text
    plan.narration_text = narration_text_from_timeline(plan)


def narration_timeline(plan: StoryboardPlan) -> list[NarrationCue]:
    cues: list[NarrationCue] = []
    cursor = 0
    seen_by_scene: dict[int, set[str]] = {}
    for shot in plan.shots:
        start = cursor
        cursor += shot.duration
        text = _clean_text(shot.narration)
        seen = seen_by_scene.setdefault(shot.scene_id, set())
        if not text or text in seen:
            continue
        seen.add(text)
        cues.append(
            NarrationCue(
                shot_id=shot.shot_id,
                start_seconds=start,
                end_seconds=cursor,
                text=text,
            )
        )
    return cues


def narration_text_from_timeline(plan: StoryboardPlan) -> str:
    return "\n".join(cue.text for cue in narration_timeline(plan))


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _timestamp(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},000"

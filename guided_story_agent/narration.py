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


class EdgeNarrationSynthesizer:
    """Create narration before any paid video call and always export an SRT timeline."""

    def __init__(self, voice: str | None = None, rate: str = "+0%") -> None:
        self.voice = voice or os.getenv("NARRATION_VOICE", "zh-CN-XiaoxiaoNeural")
        self.rate = rate

    def synthesize(self, plan: StoryboardPlan, output_dir: str | Path) -> NarrationArtifact:
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        subtitle_path = target / "narration.srt"
        subtitle_path.write_text(build_srt(plan), encoding="utf-8")
        plan.subtitle_path = str(subtitle_path)
        if not plan.narration_text.strip():
            return NarrationArtifact(audio_path="", subtitle_path=str(subtitle_path))
        try:
            import edge_tts
        except ImportError as exc:
            raise NarrationUnavailable(
                "未安装 edge-tts；已生成字幕，但暂时没有旁白音频。"
            ) from exc
        audio_path = target / "narration.mp3"
        try:
            communicate = edge_tts.Communicate(
                plan.narration_text,
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
    cursor = 0
    for index, shot in enumerate(plan.shots, start=1):
        start = cursor
        cursor += shot.duration
        lines.extend(
            [
                str(index),
                f"{_timestamp(start)} --> {_timestamp(cursor)}",
                shot.narration.strip() or shot.action.strip(),
                "",
            ]
        )
    return "\n".join(lines)


def _timestamp(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},000"

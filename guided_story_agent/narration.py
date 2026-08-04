from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

    def __init__(
        self,
        voice: str | None = None,
        rate: str = "+0%",
        *,
        timeline_assembler: Callable[
            [list[tuple[NarrationCue, Path]], Path, int],
            str,
        ]
        | None = None,
    ) -> None:
        self.voice = voice or os.getenv("NARRATION_VOICE", "zh-CN-XiaoxiaoNeural")
        self.rate = rate
        self.timeline_assembler = timeline_assembler or assemble_timed_narration

    def synthesize(self, plan: StoryboardPlan, output_dir: str | Path) -> NarrationArtifact:
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        cues = spoken_timeline(plan)
        timeline_text = "\n".join(cue.text for cue in cues)
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
        audio_path = target / "narration.m4a"
        segment_dir = target / "narration_segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        segments: list[tuple[NarrationCue, Path]] = []
        try:
            for index, cue in enumerate(cues, start=1):
                segment_path = segment_dir / f"{index:03d}_shot_{cue.shot_id:03d}.mp3"
                communicate = edge_tts.Communicate(
                    cue.text,
                    self.voice,
                    rate=self._cue_rate(cue),
                )
                communicate.save_sync(str(segment_path))
                if not segment_path.is_file() or segment_path.stat().st_size == 0:
                    raise NarrationUnavailable(f"镜头 {cue.shot_id} 没有生成可用旁白音频。")
                segments.append((cue, segment_path))
            self.timeline_assembler(segments, audio_path, plan.total_duration)
        except Exception as exc:
            audio_path.unlink(missing_ok=True)
            if isinstance(exc, NarrationUnavailable):
                raise
            raise NarrationUnavailable(f"Edge TTS 旁白生成失败：{exc}") from exc
        if not audio_path.is_file() or audio_path.stat().st_size == 0:
            raise NarrationUnavailable("Edge TTS 没有生成可用音频。")
        plan.audio_path = str(audio_path)
        return NarrationArtifact(str(audio_path), str(subtitle_path))

    def _cue_rate(self, cue: NarrationCue) -> str:
        duration = max(1, cue.end_seconds - cue.start_seconds)
        spoken_units = len(re.sub(r"\s+", "", cue.text))
        estimated_seconds = max(1.0, spoken_units / 4.2)
        required_delta = round((estimated_seconds / duration - 1.0) * 100)
        base_match = re.fullmatch(r"([+-]?)(\d+)%", self.rate.strip())
        base_delta = 0
        if base_match:
            base_delta = int(base_match.group(2))
            if base_match.group(1) == "-":
                base_delta *= -1
        delta = max(-30, min(100, base_delta + required_delta))
        return f"{delta:+d}%"


def build_srt(plan: StoryboardPlan) -> str:
    lines: list[str] = []
    for index, cue in enumerate(spoken_timeline(plan), start=1):
        lines.extend(
            [
                str(index),
                f"{_timestamp(cue.start_seconds)} --> {_timestamp(cue.end_seconds)}",
                cue.text,
                "",
            ]
        )
    return "\n".join(lines)


def normalize_narration_timeline(
    plan: StoryboardPlan,
    *,
    deduplicate_legacy: bool = False,
) -> None:
    """Clean narration, with opt-in deduplication only for legacy migration."""

    seen_by_scene: dict[int, set[str]] = {}
    has_shot_narration = False
    for shot in plan.shots:
        text = _clean_text(shot.narration)
        seen = seen_by_scene.setdefault(shot.scene_id, set())
        if deduplicate_legacy and text and text in seen:
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
    for shot in plan.shots:
        start = cursor
        cursor += shot.duration
        text = _clean_text(shot.narration)
        if not text:
            continue
        cues.append(
            NarrationCue(
                shot_id=shot.shot_id,
                start_seconds=start,
                end_seconds=cursor,
                text=text,
            )
        )
    return cues


def spoken_timeline(plan: StoryboardPlan) -> list[NarrationCue]:
    """Return every confirmed narration/dialogue cue on its shot window."""

    cues: list[NarrationCue] = []
    cursor = 0
    for shot in plan.shots:
        start = cursor
        cursor += shot.duration
        text = "\n".join(
            value
            for value in (
                _clean_text(shot.narration),
                _clean_text(shot.dialogue),
            )
            if value
        )
        if text:
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


def assemble_timed_narration(
    segments: list[tuple[NarrationCue, Path]],
    output_path: Path,
    total_duration: int,
) -> str:
    """Place every TTS segment on its exact cue window and produce one fixed-length track."""
    if not segments:
        raise NarrationUnavailable("没有可合成的旁白片段。")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise NarrationUnavailable("未找到 ffmpeg/ffprobe，无法按镜头对齐旁白。")
    command = [ffmpeg, "-y"]
    filters: list[str] = []
    labels: list[str] = []
    for index, (cue, path) in enumerate(segments):
        command.extend(["-i", str(path)])
        source_duration = _probe_audio_duration(path, ffprobe)
        window = max(0.1, float(cue.end_seconds - cue.start_seconds))
        tempo = source_duration / window if source_duration > window else 1.0
        tempo_filter = _atempo_chain(tempo)
        label = f"cue{index}"
        filters.append(
            f"[{index}:a]{tempo_filter}"
            f"apad=pad_dur={window:.3f},atrim=duration={window:.3f},"
            f"adelay={cue.start_seconds * 1000}:all=1[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0,"
        + f"atrim=duration={float(total_duration):.3f}[mixed]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[mixed]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise NarrationUnavailable("旁白时间轴合成失败：" + completed.stderr[-1000:])
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise NarrationUnavailable("ffmpeg 没有生成可用的旁白音轨。")
    return str(output_path)


def _probe_audio_duration(path: Path, ffprobe: str) -> float:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise NarrationUnavailable("无法读取旁白片段时长：" + completed.stderr[-500:])
    try:
        value = float(completed.stdout.strip())
    except ValueError as exc:
        raise NarrationUnavailable("旁白片段没有有效时长。") from exc
    if value <= 0:
        raise NarrationUnavailable("旁白片段时长必须大于零。")
    return value


def _atempo_chain(factor: float) -> str:
    if factor <= 1.0001:
        return ""
    parts: list[float] = []
    remaining = factor
    while remaining > 2.0:
        parts.append(2.0)
        remaining /= 2.0
    if remaining > 1.0001:
        parts.append(remaining)
    return "".join(f"atempo={value:.5f}," for value in parts)

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .models import RenderManifest, StoryboardPlan, StoryboardShot, VideoArtifact, to_plain_data
from .narration import EdgeNarrationSynthesizer, NarrationUnavailable


class ShotProvider(Protocol):
    endpoint: str

    def generate_shot(
        self,
        shot: StoryboardShot,
        output_dir: str | Path,
        *,
        attempt: int = 1,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> VideoArtifact: ...


class StoryRenderer:
    def __init__(
        self,
        provider: ShotProvider,
        *,
        narration=None,
        assembler: Callable[[list[str], str | Path], str] | None = None,
        finalizer: Callable[[str, str, str, str | Path], str] | None = None,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> None:
        self.provider = provider
        self.narration = narration or EdgeNarrationSynthesizer()
        self.assembler = assembler or concatenate_mp4_clips
        self.finalizer = finalizer or mux_audio_and_subtitles
        self.progress_callback = progress_callback

    def render(self, plan: StoryboardPlan, output_dir: str | Path) -> RenderManifest:
        if not plan.confirmed:
            raise RuntimeError("分镜尚未确认，禁止调用视频生成。")
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        manifest = RenderManifest(status="running", output_dir=str(target))
        narration_warning = ""
        self._progress("narration", 0.01, "正在预生成旁白和字幕")
        try:
            narration_artifact = self.narration.synthesize(plan, target)
            manifest.audio_path = narration_artifact.audio_path
            manifest.subtitle_path = narration_artifact.subtitle_path
        except NarrationUnavailable as exc:
            narration_warning = str(exc)
            manifest.subtitle_path = plan.subtitle_path

        for index, shot in enumerate(plan.shots):
            reusable = self._find_reusable(plan, shot)
            if reusable:
                manifest.reused_shots.append(shot.shot_id)
                manifest.artifacts.append(reusable)
                self._progress(
                    "reused",
                    0.1 + 0.75 * ((index + 1) / len(plan.shots)),
                    f"复用镜头 {shot.shot_id}",
                )
                continue
            attempt = 1 + sum(item.shot_id == shot.shot_id for item in plan.artifacts)
            try:
                artifact = self.provider.generate_shot(
                    shot,
                    target,
                    attempt=attempt,
                    progress_callback=lambda stage, fraction, message, i=index: self._progress(
                        stage,
                        0.1 + 0.75 * ((i + fraction) / len(plan.shots)),
                        f"镜头 {shot.shot_id}/{len(plan.shots)}：{message}",
                    ),
                )
            except Exception as exc:
                artifact = VideoArtifact(
                    artifact_id=f"failed_shot_{shot.shot_id}_{attempt}",
                    shot_id=shot.shot_id,
                    provider=str(getattr(self.provider, "provider_name", "unknown")),
                    model=str(getattr(self.provider, "endpoint", type(self.provider).__name__)),
                    status="failed",
                    local_path="",
                    remote_url="",
                    duration=shot.duration,
                    prompt=shot.video_prompt,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    attempt=attempt,
                    error_message=str(exc),
                )
                plan.artifacts.append(artifact)
                manifest.artifacts.append(artifact)
                manifest.failed_shots.append(shot.shot_id)
                manifest.status = "failed"
                manifest.error = f"镜头 {shot.shot_id} 生成失败：{exc}"
                self._write_manifest(manifest, target)
                return manifest
            plan.artifacts.append(artifact)
            manifest.artifacts.append(artifact)
            manifest.generated_shots.append(shot.shot_id)

        clips = [item.local_path for item in manifest.artifacts if item.status == "succeeded"]
        try:
            self._progress("assembling", 0.9, "正在拼接全部镜头")
            silent = self.assembler(clips, target / "silent_final.mp4")
            if manifest.audio_path:
                final = self.finalizer(
                    silent,
                    manifest.audio_path,
                    manifest.subtitle_path,
                    target / "final_video.mp4",
                )
            else:
                final = silent
            manifest.final_video_path = final
            manifest.status = "succeeded"
            manifest.error = narration_warning
            self._progress("completed", 1.0, "成片已经生成")
        except Exception as exc:
            manifest.status = "failed"
            manifest.error = f"视频拼接失败：{exc}"
        self._write_manifest(manifest, target)
        return manifest

    def _find_reusable(
        self, plan: StoryboardPlan, shot: StoryboardShot
    ) -> VideoArtifact | None:
        model = str(getattr(self.provider, "endpoint", type(self.provider).__name__))
        for artifact in reversed(plan.artifacts):
            if (
                artifact.shot_id == shot.shot_id
                and artifact.status == "succeeded"
                and artifact.model == model
                and artifact.prompt == shot.video_prompt
                and Path(artifact.local_path).is_file()
            ):
                return artifact
        return None

    def _progress(self, stage: str, fraction: float, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(stage, min(1.0, max(0.0, fraction)), message)

    @staticmethod
    def _write_manifest(manifest: RenderManifest, target: Path) -> None:
        (target / "render_manifest.json").write_text(
            json.dumps(to_plain_data(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def concatenate_mp4_clips(paths: list[str], output_path: str | Path) -> str:
    if not paths:
        raise ValueError("没有可拼接的镜头。")
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("未找到 ffmpeg。")
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    concat_file = target.with_suffix(".concat.txt")
    resolved = [Path(path).expanduser().resolve() for path in paths]
    if any(not path.is_file() for path in resolved):
        raise FileNotFoundError("部分镜头文件不存在。")
    concat_file.write_text(
        "\n".join(f"file '{path.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for path in resolved) + "\n",
        encoding="utf-8",
    )
    try:
        copy_result = subprocess.run(
            [executable, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(target)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if copy_result.returncode != 0:
            encoded = subprocess.run(
                [executable, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(target)],
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            if encoded.returncode != 0:
                raise RuntimeError(encoded.stderr[-1000:])
    finally:
        concat_file.unlink(missing_ok=True)
    return str(target)


def mux_audio_and_subtitles(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_path: str | Path,
) -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("未找到 ffmpeg。")
    target = Path(output_path).expanduser().resolve()
    command = [
        executable, "-y", "-i", video_path, "-i", audio_path,
    ]
    if subtitle_path and Path(subtitle_path).is_file():
        command.extend(["-i", subtitle_path, "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0", "-c:s", "mov_text"])
    else:
        command.extend(["-map", "0:v:0", "-map", "1:a:0"])
    command.extend(["-c:v", "copy", "-c:a", "aac", "-shortest", str(target)])
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-1000:])
    return str(target)

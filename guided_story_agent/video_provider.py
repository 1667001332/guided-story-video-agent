from __future__ import annotations

import json
import hashlib
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import ProviderCapabilities, StoryboardShot, VideoArtifact, VideoJob
from .provider_config import VideoProviderConfig


ProgressCallback = Callable[[str, float, str], None]
DEFAULT_API_ROOT = "https://apihub.agnes-ai.com"
DEFAULT_MODEL = "agnes-video-v2.0"


class VideoGenerationError(RuntimeError):
    pass


class VideoProviderNotConfigured(VideoGenerationError):
    pass


class VideoTaskTerminalError(VideoGenerationError):
    """A remote task reached a terminal failure and may be submitted again."""

    def __init__(self, message: str, *, request_id: str) -> None:
        super().__init__(message)
        self.request_id = request_id


class VideoSubmissionUncertainError(VideoGenerationError):
    """The submit response was lost, so retrying automatically may charge twice."""

    def __init__(self, message: str, *, operation_id: str) -> None:
        super().__init__(message)
        self.operation_id = operation_id


@dataclass(frozen=True, slots=True)
class PublishedImage:
    source_path: str
    published_path: str
    public_url: str
    content_digest: str


class LocalPublicImagePublisher:
    """Atomically stage a single visual input inside an explicitly public root."""

    def __init__(self, reference_root: str | Path, reference_base_url: str) -> None:
        self.root = Path(reference_root).expanduser().resolve()
        self.resolver = build_public_image_url_resolver(
            self.root,
            reference_base_url,
        )

    def publish(
        self,
        source_path: str | Path,
        *,
        run_id: str,
        label: str,
    ) -> PublishedImage:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"待发布视觉输入不存在或为空：{source}")
        digest = _sha256_file(source)
        safe_run = _safe_component(run_id, "render")
        safe_label = _safe_component(label, "frame")
        run_dir = (self.root / safe_run).resolve()
        try:
            run_dir.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("视觉输入暂存目录越过 VIDEO_REFERENCE_ROOT。") from exc
        run_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower() or ".png"
        target = (run_dir / f"{safe_label}_{digest[:16]}{suffix}").resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("视觉输入暂存文件越过 VIDEO_REFERENCE_ROOT。") from exc
        if not target.is_file() or _sha256_file(target) != digest:
            temporary = run_dir / f".{target.name}.{uuid4().hex}.tmp"
            try:
                shutil.copyfile(source, temporary)
                if _sha256_file(temporary) != digest:
                    raise OSError("暂存文件复制后内容摘要不一致")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return PublishedImage(
            source_path=str(source),
            published_path=str(target),
            public_url=self.resolver(str(target)),
            content_digest=digest,
        )


def sanitize_remote_url(url: str) -> str:
    """Persist a useful URL without credentials, signatures, or fragments."""
    if not url:
        return ""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def build_public_image_url_resolver(
    reference_root: str | Path,
    reference_base_url: str,
) -> Callable[[str], str]:
    """Map an already-hosted local file tree to its matching public URL tree."""
    root = Path(reference_root).expanduser().resolve()
    parsed_base = urllib.parse.urlsplit(reference_base_url.strip())
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc:
        raise ValueError("VIDEO_REFERENCE_BASE_URL 必须是有效的 http(s) URL。")
    base = reference_base_url.rstrip("/")

    def resolve(local_path: str) -> str:
        candidate = Path(local_path).expanduser().resolve()
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise FileNotFoundError(f"待发布参考图不存在或为空：{candidate}")
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"参考图不在 VIDEO_REFERENCE_ROOT 内：{candidate}"
            ) from exc
        encoded = urllib.parse.quote(relative.as_posix(), safe="/")
        return f"{base}/{encoded}"

    return resolve


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("._-")
    return cleaned[:80] or fallback


def validate_mp4_file(path: str | Path) -> bool:
    """Reject empty, HTML, and structurally invalid video files before reuse."""
    target = Path(path)
    try:
        if not target.is_file() or not _has_required_mp4_boxes(target):
            return False
    except OSError:
        return False

    executable = shutil.which("ffprobe")
    if executable is None:
        return True
    try:
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(target),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and "video" in completed.stdout.lower()


def _has_required_mp4_boxes(path: Path) -> bool:
    """Check complete top-level ISO-BMFF boxes without loading a video into memory."""
    file_size = path.stat().st_size
    if file_size < 24:
        return False
    found: set[bytes] = set()
    offset = 0
    with path.open("rb") as handle:
        while offset + 8 <= file_size:
            handle.seek(offset)
            header = handle.read(8)
            if len(header) != 8:
                return False
            box_size = int.from_bytes(header[:4], byteorder="big")
            box_type = header[4:8]
            header_size = 8
            if box_size == 1:
                extended_size = handle.read(8)
                if len(extended_size) != 8:
                    return False
                box_size = int.from_bytes(extended_size, byteorder="big")
                header_size = 16
            elif box_size == 0:
                box_size = file_size - offset
            if box_size < header_size or offset + box_size > file_size:
                return False
            if box_type in {b"ftyp", b"moov", b"moof", b"mdat"}:
                found.add(box_type)
            offset += box_size
    return (
        offset == file_size
        and b"ftyp" in found
        and b"moov" in found
        and bool({b"moof", b"mdat"} & found)
    )


class AgnesVideoProvider:
    provider_name = "agnes"

    def __init__(
        self,
        api_key: str | None = None,
        api_root: str = DEFAULT_API_ROOT,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = 120,
        poll_interval: float = 5,
        max_poll_seconds: float = 900,
        network_retries: int = 2,
        retry_backoff: float = 1,
        submit_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        status_fn: Callable[[str], dict[str, Any]] | None = None,
        download_fn: Callable[[str, Path], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
        image_url_resolver: Callable[[str], str] | None = None,
        image_publisher: LocalPublicImagePublisher | None = None,
        config_source: str = "constructor",
        configuration_error: str = "",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.api_root = api_root.rstrip("/") or DEFAULT_API_ROOT
        self.model = model.strip() or DEFAULT_MODEL
        self.endpoint = self.model
        self.timeout = max(1.0, float(timeout))
        self.poll_interval = max(0.0, float(poll_interval))
        self.max_poll_seconds = max(1.0, float(max_poll_seconds))
        self.network_retries = max(0, min(10, int(network_retries)))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.config_source = config_source
        self.configuration_error = configuration_error
        self._submit_fn = submit_fn
        self._status_fn = status_fn
        self._download_fn = download_fn or self._download_file
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn
        self._image_url_resolver = image_url_resolver
        self._image_publisher = image_publisher

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_text_to_video=True,
            supports_image_to_video=(
                self._image_url_resolver is not None
                or self._image_publisher is not None
            ),
            supports_reference_images=False,
            supports_seed=True,
            requires_public_image_url=True,
            min_duration_seconds=3,
            max_duration_seconds=15,
            supports_long_video=False,
            supports_multi_scene_prompt=True,
        )

    @classmethod
    def from_env(cls) -> AgnesVideoProvider:
        config = VideoProviderConfig.from_env(
            default_api_root=DEFAULT_API_ROOT,
            default_model=DEFAULT_MODEL,
        )
        image_publisher = (
            LocalPublicImagePublisher(
                config.reference_root,
                config.reference_base_url,
            )
            if config.reference_root and config.reference_base_url
            else None
        )
        return cls(
            api_key=config.api_key,
            api_root=config.api_root,
            model=config.model,
            timeout=config.timeout,
            poll_interval=config.poll_interval,
            max_poll_seconds=config.max_poll_seconds,
            network_retries=config.network_retries,
            retry_backoff=config.retry_backoff,
            image_publisher=image_publisher,
            config_source=config.source,
            configuration_error=config.error,
        )

    def prepare_image_input(
        self,
        source_path: str | Path,
        *,
        run_id: str,
        label: str,
    ) -> PublishedImage:
        """Publish when configured, otherwise validate an existing public mapping."""
        if self._image_publisher is not None:
            return self._image_publisher.publish(
                source_path,
                run_id=run_id,
                label=label,
            )
        source = Path(source_path).expanduser().resolve()
        if self._image_url_resolver is None:
            raise VideoProviderNotConfigured(
                "Agnes 图生视频需要公网 image URL。请同时配置 "
                "VIDEO_REFERENCE_ROOT 与 VIDEO_REFERENCE_BASE_URL；"
                "项目只会向该公开根目录暂存单张视觉输入。"
            )
        image_url = str(self._image_url_resolver(str(source))).strip()
        return PublishedImage(
            source_path=str(source),
            published_path=str(source),
            public_url=image_url,
            content_digest=_sha256_file(source),
        )

    def generate_shot(
        self,
        shot: StoryboardShot,
        output_dir: str | Path,
        *,
        attempt: int = 1,
        progress_callback: ProgressCallback | None = None,
        resume_request_id: str | None = None,
    ) -> VideoArtifact:
        if self.configuration_error:
            raise VideoProviderNotConfigured(self.configuration_error)
        if not self.api_key:
            raise VideoProviderNotConfigured(
                "未配置 VIDEO_API_KEY（旧版可使用 AGNES_API_KEY），尚未发起视频任务。"
            )
        if shot.duration not in range(3, 16):
            raise ValueError("单镜头时长必须是 3 到 15 秒之间的整数。")
        payload = self._build_payload(shot)
        video_id = (resume_request_id or "").strip()
        if video_id:
            self._notify(
                progress_callback,
                "resuming",
                0.05,
                f"正在继续查询镜头 {shot.shot_id} 的已有任务",
            )
        else:
            self._notify(
                progress_callback,
                "submitting",
                0.05,
                f"正在提交镜头 {shot.shot_id}",
            )
            operation_id = f"local-submit-{uuid4().hex}"
            try:
                submitted = self._submit(payload)
            except VideoProviderNotConfigured:
                raise
            except Exception as exc:
                raise VideoSubmissionUncertainError(
                    "Agnes 提交响应未能确认，服务端可能已经受理。"
                    "为避免重复付费，系统不会自动重提；请先到 Provider 后台核对任务。"
                    f"原始错误：{exc}",
                    operation_id=operation_id,
                ) from exc
            # Agnes currently documents both ``video_id`` and ``task_id``;
            # some gateway responses expose only the latter.  Treat all
            # documented identifiers as opaque and prefer the video id.
            video_id = self._extract_text(submitted, ("video_id", "task_id", "id")) or ""
            if not video_id:
                raise VideoSubmissionUncertainError(
                    "Agnes 提交结果缺少 video_id/task_id，无法判断服务端是否已经受理。"
                    "为避免重复付费，系统不会自动重提；请先到 Provider 后台核对任务。",
                    operation_id=operation_id,
                )
        try:
            result = self._poll(video_id, progress_callback)
        except VideoTaskTerminalError:
            raise
        except Exception as exc:
            return self._pending_artifact(
                shot,
                video_id,
                attempt,
                error_message=str(exc),
            )
        remote_url = self._extract_url(result)
        if not remote_url:
            return self._pending_artifact(
                shot,
                video_id,
                attempt,
                error_message="Agnes 完成任务但未返回视频 URL。",
            )
        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = target_dir / f"shot_{shot.shot_id:03d}_{stamp}.mp4"
        self._notify(progress_callback, "downloading", 0.95, f"正在下载镜头 {shot.shot_id}")
        try:
            self._download_fn(remote_url, target)
            if not validate_mp4_file(target):
                target.unlink(missing_ok=True)
                raise VideoGenerationError("下载结果不是可用的 MP4 视频。")
        except Exception as exc:
            safe_error = str(exc).replace(remote_url, sanitize_remote_url(remote_url))
            return self._pending_artifact(
                shot,
                video_id,
                attempt,
                error_message=safe_error,
                remote_url=remote_url,
            )
        artifact = VideoArtifact(
            artifact_id=f"video_shot_{shot.shot_id:03d}_{stamp}",
            shot_id=shot.shot_id,
            provider=self.provider_name,
            model=self.model,
            status="succeeded",
            local_path=str(target),
            remote_url=sanitize_remote_url(remote_url),
            duration=shot.duration,
            prompt=shot.video_prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
            request_id=video_id,
            attempt=max(1, int(attempt)),
        )
        self._notify(progress_callback, "completed", 1.0, f"镜头 {shot.shot_id} 已保存")
        return artifact

    def generate_video(
        self,
        job: VideoJob,
        output_dir: str | Path,
        *,
        attempt: int = 1,
        progress_callback: ProgressCallback | None = None,
        resume_request_id: str | None = None,
    ) -> VideoArtifact:
        """Compatibility adapter for the whole-video core.

        Agnes currently accepts only a single 3–15 second generation.  That
        limitation is declared here, inside the Agnes adapter; the generic
        VideoJob pipeline does not split or validate against it.  A future
        long-video adapter can implement this method without changing the
        session or renderer.
        """
        if not isinstance(job, VideoJob):
            raise TypeError("AgnesVideoProvider.generate_video 需要 VideoJob。")
        shot = StoryboardShot(
            shot_id=1,
            scene_id=1,
            duration=int(job.target_seconds),
            character="",
            location="",
            visual=job.prompt,
            action=job.prompt,
            camera="",
            lighting="",
            mood="",
            narration=job.narration,
            video_prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            aspect_ratio=job.aspect_ratio,
            dialogue=job.dialogue,
            visual_style=job.visual_style,
            reference_image_paths=list(job.reference_image_paths),
            initial_frame_path=job.initial_frame_path,
            initial_frame_url=job.initial_frame_url,
            seed=job.seed,
        )
        return self.generate_shot(
            shot,
            output_dir,
            attempt=attempt,
            progress_callback=progress_callback,
            resume_request_id=resume_request_id,
        )

    def _build_payload(self, shot: StoryboardShot) -> dict[str, Any]:
        width, height = self._dimensions(shot.aspect_ratio)
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": shot.video_prompt,
            "negative_prompt": shot.negative_prompt,
            "width": width,
            "height": height,
            "num_frames": shot.duration * 24 + 1,
            "frame_rate": 24,
        }
        if shot.seed is not None:
            payload["seed"] = int(shot.seed)
        if shot.reference_image_paths and not shot.initial_frame_path:
            raise VideoProviderNotConfigured(
                "Agnes 官方接口没有通用身份参考图字段；"
                "请先由渲染器选择一张首帧，或改用声明支持 reference images 的 Provider。"
            )
        if shot.initial_frame_path or shot.initial_frame_url:
            if shot.initial_frame_url:
                image_url = shot.initial_frame_url.strip()
            elif self._image_url_resolver is not None:
                image_url = str(self._image_url_resolver(shot.initial_frame_path)).strip()
            else:
                raise VideoProviderNotConfigured(
                    "Agnes 图生视频要求公网可访问的 image URL，"
                    "当前未配置本地图片到公网 URL 的解析器，未提交任务。"
                )
            parsed = urllib.parse.urlsplit(image_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise VideoProviderNotConfigured(
                    "Agnes image 必须解析为有效的 http(s) 公网 URL，未提交任务。"
                )
            payload["image"] = image_url
        return payload

    def _pending_artifact(
        self,
        shot: StoryboardShot,
        video_id: str,
        attempt: int,
        *,
        error_message: str,
        remote_url: str = "",
    ) -> VideoArtifact:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return VideoArtifact(
            artifact_id=f"pending_shot_{shot.shot_id:03d}_{stamp}",
            shot_id=shot.shot_id,
            provider=self.provider_name,
            model=self.model,
            status="pending",
            local_path="",
            remote_url=sanitize_remote_url(remote_url),
            duration=shot.duration,
            prompt=shot.video_prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
            request_id=video_id,
            attempt=max(1, int(attempt)),
            error_message=error_message,
        )

    def _submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._submit_fn:
            return self._submit_fn(payload)
        return self._request_json("POST", f"{self.api_root}/v1/videos", payload)

    def _status(self, video_id: str) -> dict[str, Any]:
        if self._status_fn:
            return self._status_fn(video_id)
        query = urllib.parse.urlencode({"video_id": video_id})
        return self._request_json("GET", f"{self.api_root}/agnesapi?{query}")

    def _poll(self, video_id: str, callback: ProgressCallback | None) -> dict[str, Any]:
        deadline = self._monotonic_fn() + self.max_poll_seconds
        last = "unknown"
        while self._monotonic_fn() <= deadline:
            result = self._status(video_id)
            last = (self._extract_text(result, ("status", "state")) or "unknown").lower()
            progress = 0.5
            raw = self._extract_text(result, ("progress",))
            if raw:
                try:
                    progress = min(0.9, max(0.1, float(raw) / 100))
                except ValueError:
                    pass
            self._notify(callback, last, progress, f"Agnes 状态：{last}")
            if last in {"completed", "succeeded", "success", "done"}:
                return result
            if last in {"failed", "error", "cancelled", "canceled"}:
                detail = self._extract_text(result, ("error", "message", "detail")) or last
                raise VideoTaskTerminalError(
                    f"Agnes 视频任务失败：{detail}",
                    request_id=video_id,
                )
            self._sleep_fn(self.poll_interval)
        raise VideoGenerationError(f"Agnes 视频任务超时，最后状态：{last}")

    def _request_json(
        self, method: str, url: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "guided-story-video-agent/0.1",
            },
        )
        safe_retry = method.upper() != "POST"
        attempts = self.network_retries + 1 if safe_retry else 1
        result: Any = None
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code == 401:
                    raise VideoProviderNotConfigured(
                        "Agnes 鉴权失败，请检查 API Key。"
                    ) from exc
                retryable = exc.code in {408, 425, 429, 500, 502, 503, 504}
                if safe_retry and retryable and attempt + 1 < attempts:
                    self._retry_wait(attempt)
                    continue
                raise VideoGenerationError(f"Agnes HTTP {exc.code}: {detail}") from exc
            except (OSError, TimeoutError, json.JSONDecodeError) as exc:
                if safe_retry and attempt + 1 < attempts:
                    self._retry_wait(attempt)
                    continue
                raise VideoGenerationError(
                    f"Agnes 请求失败：{type(exc).__name__}: {exc}"
                ) from exc
        if not isinstance(result, dict):
            raise VideoGenerationError("Agnes 返回了无法识别的 JSON。")
        return result

    def _retry_wait(self, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        if delay:
            self._sleep_fn(delay)

    @staticmethod
    def _notify(
        callback: ProgressCallback | None, stage: str, fraction: float, message: str
    ) -> None:
        if callback:
            try:
                callback(stage, min(1.0, max(0.0, fraction)), message)
            except Exception:
                pass

    @classmethod
    def _extract_text(cls, data: Any, keys: tuple[str, ...]) -> str | None:
        if isinstance(data, dict):
            for key in keys:
                value = data.get(key)
                if isinstance(value, (str, int, float)) and str(value).strip():
                    return str(value)
            for value in data.values():
                found = cls._extract_text(value, keys)
                if found:
                    return found
        elif isinstance(data, list):
            for value in data:
                found = cls._extract_text(value, keys)
                if found:
                    return found
        return None

    @classmethod
    def _extract_url(cls, data: Any) -> str | None:
        value = cls._extract_text(data, ("video_url", "download_url", "output_url", "url"))
        return value if value and value.startswith(("https://", "http://")) else None

    @staticmethod
    def _dimensions(ratio: str) -> tuple[int, int]:
        dimensions = {
            "16:9": (1152, 648),
            "9:16": (648, 1152),
            "1:1": (768, 768),
        }
        try:
            return dimensions[ratio]
        except KeyError as exc:
            raise ValueError(
                "Agnes 画幅比例只支持 16:9、9:16 或 1:1。"
            ) from exc

    def _download_file(self, url: str, target: Path) -> None:
        temporary = target.with_suffix(".mp4.part")
        request = urllib.request.Request(
            url, headers={"User-Agent": "guided-story-video-agent/0.1"}
        )
        attempts = self.network_retries + 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(response, handle)
                temporary.replace(target)
                return
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                if attempt + 1 < attempts:
                    self._retry_wait(attempt)
                    continue
                raise VideoGenerationError(f"MP4 下载失败：{exc}") from exc

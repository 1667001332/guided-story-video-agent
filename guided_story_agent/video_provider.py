from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import StoryboardShot, VideoArtifact
from .provider_config import VideoProviderConfig


ProgressCallback = Callable[[str, float, str], None]
DEFAULT_API_ROOT = "https://apihub.agnes-ai.com"
DEFAULT_MODEL = "agnes-video-v2.0"


class VideoGenerationError(RuntimeError):
    pass


class VideoProviderNotConfigured(VideoGenerationError):
    pass


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
        retry_backoff: float = 1.0,
        submit_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        status_fn: Callable[[str], dict[str, Any]] | None = None,
        download_fn: Callable[[str, Path], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
        config_source: str = "constructor",
        configuration_error: str = "",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.api_root = api_root.rstrip("/") or DEFAULT_API_ROOT
        self.model = model.strip() or DEFAULT_MODEL
        self.endpoint = self.model
        self.timeout = max(1.0, float(timeout))
        self.poll_interval = max(0.1, float(poll_interval))
        self.max_poll_seconds = max(1.0, float(max_poll_seconds))
        self.network_retries = max(0, int(network_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.config_source = config_source
        self.configuration_error = configuration_error
        self._submit_fn = submit_fn
        self._status_fn = status_fn
        self._download_fn = download_fn or self._download_file
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn

    @classmethod
    def from_env(cls) -> AgnesVideoProvider:
        config = VideoProviderConfig.from_env(
            default_api_root=DEFAULT_API_ROOT,
            default_model=DEFAULT_MODEL,
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
            config_source=config.source,
            configuration_error=config.error,
        )

    def generate_shot(
        self,
        shot: StoryboardShot,
        output_dir: str | Path,
        *,
        attempt: int = 1,
        progress_callback: ProgressCallback | None = None,
    ) -> VideoArtifact:
        if self.configuration_error:
            raise VideoProviderNotConfigured(self.configuration_error)
        if not self.api_key:
            raise VideoProviderNotConfigured(
                "未配置 VIDEO_API_KEY（旧版可使用 AGNES_API_KEY），尚未发起视频任务。"
            )
        if shot.duration not in range(3, 16):
            raise ValueError("单镜头时长必须是 3 到 15 秒之间的整数。")
        width, height = self._dimensions(shot.aspect_ratio)
        payload = {
            "model": self.model,
            "prompt": shot.video_prompt,
            "negative_prompt": shot.negative_prompt,
            "width": width,
            "height": height,
            "num_frames": shot.duration * 24 + 1,
            "frame_rate": 24,
        }
        self._notify(progress_callback, "submitting", 0.05, f"正在提交镜头 {shot.shot_id}")
        submitted = self._submit(payload)
        video_id = self._extract_text(submitted, ("video_id", "id"))
        if not video_id:
            raise VideoGenerationError("Agnes 提交结果缺少 video_id。")
        result = self._poll(video_id, progress_callback)
        remote_url = self._extract_url(result)
        if not remote_url:
            raise VideoGenerationError("Agnes 完成任务但未返回视频 URL。")
        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = target_dir / f"shot_{shot.shot_id:03d}_{stamp}.mp4"
        self._notify(progress_callback, "downloading", 0.95, f"正在下载镜头 {shot.shot_id}")
        self._download_fn(remote_url, target)
        if not target.is_file() or target.stat().st_size == 0:
            raise VideoGenerationError("下载的 MP4 为空。")
        artifact = VideoArtifact(
            artifact_id=f"video_shot_{shot.shot_id:03d}_{stamp}",
            shot_id=shot.shot_id,
            provider=self.provider_name,
            model=self.model,
            status="succeeded",
            local_path=str(target),
            remote_url=remote_url,
            duration=shot.duration,
            prompt=shot.video_prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
            request_id=video_id,
            attempt=max(1, int(attempt)),
        )
        self._notify(progress_callback, "completed", 1.0, f"镜头 {shot.shot_id} 已保存")
        return artifact

    def _submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._submit_fn:
            return self._submit_fn(payload)
        return self._request_json("POST", f"{self.api_root}/v1/videos", payload)

    def _status(self, video_id: str) -> dict[str, Any]:
        if self._status_fn:
            return self._status_fn(video_id)
        query = urllib.parse.urlencode({"video_id": video_id})
        return self._request_json(
            "GET",
            f"{self.api_root}/agnesapi?{query}",
            retry=self.network_retries,
        )

    def _poll(
        self, video_id: str, callback: ProgressCallback | None
    ) -> dict[str, Any]:
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
                raise VideoGenerationError(f"Agnes 视频任务失败：{detail}")
            self._sleep_fn(self.poll_interval)
        raise VideoGenerationError(f"Agnes 视频任务超时，最后状态：{last}")

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        *,
        retry: int = 0,
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
        attempts = max(0, int(retry)) + 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code == 401:
                    raise VideoProviderNotConfigured("Agnes 鉴权失败，请检查 API Key。") from exc
                raise VideoGenerationError(f"Agnes HTTP {exc.code}: {detail}") from exc
            except (OSError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt + 1 < attempts:
                    self._sleep_fn(self.retry_backoff * (2**attempt))
                    continue
                raise VideoGenerationError(f"Agnes 请求失败：{type(exc).__name__}: {exc}") from exc
        if not isinstance(result, dict):
            raise VideoGenerationError("Agnes 返回了无法识别的 JSON。")
        return result

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
        return {"16:9": (1152, 648), "9:16": (648, 1152), "1:1": (768, 768)}.get(
            ratio, (1152, 648)
        )

    def _download_file(self, url: str, target: Path) -> None:
        temporary = target.with_suffix(".mp4.part")
        request = urllib.request.Request(
            url, headers={"User-Agent": "guided-story-video-agent/0.1"}
        )
        attempts = max(0, int(self.network_retries)) + 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(response, handle)
                temporary.replace(target)
                return
            except urllib.error.HTTPError as exc:
                temporary.unlink(missing_ok=True)
                raise VideoGenerationError(f"MP4 下载失败：HTTP {exc.code}: {exc.reason}") from exc
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                if attempt + 1 < attempts:
                    self._sleep_fn(self.retry_backoff * (2**attempt))
                    continue
                raise VideoGenerationError(f"MP4 下载失败：{exc}") from exc

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .continuity import (
    CONTINUITY_MODES,
    build_input_fingerprint,
    confirmed_start_frame,
    resolve_reference_assets,
    validate_continuity_boundary,
    verify_confirmed_visual_inputs,
)
from .models import (
    ProviderCapabilities,
    RenderManifest,
    StoryboardPlan,
    StoryboardShot,
    VideoJob,
    VideoArtifact,
    VisualReference,
    to_plain_data,
)
from .narration import EdgeNarrationSynthesizer, NarrationUnavailable
from .storyboard import refresh_shot_prompts
from .video_provider import (
    PublishedImage,
    VideoSubmissionUncertainError,
    VideoTaskTerminalError,
    sanitize_remote_url,
    validate_mp4_file,
)


class ShotProvider(Protocol):
    endpoint: str
    capabilities: ProviderCapabilities

    def generate_shot(
        self,
        shot: StoryboardShot,
        output_dir: str | Path,
        *,
        attempt: int = 1,
        progress_callback: Callable[[str, float, str], None] | None = None,
        resume_request_id: str | None = None,
    ) -> VideoArtifact: ...


class VideoProvider(Protocol):
    """Provider contract for one complete video request.

    Providers that need internal chunking may implement it behind this
    boundary.  The story/session layer never has to manufacture storyboard
    shots merely to satisfy an adapter's API.
    """

    endpoint: str
    capabilities: ProviderCapabilities

    def generate_video(
        self,
        job: VideoJob,
        output_dir: str | Path,
        *,
        attempt: int = 1,
        progress_callback: Callable[[str, float, str], None] | None = None,
        resume_request_id: str | None = None,
    ) -> VideoArtifact: ...


class VideoJobRenderer:
    """Render one VideoJob without a storyboard or local clip assembly."""

    def __init__(
        self,
        provider: VideoProvider,
        *,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> None:
        self.provider = provider
        self.progress_callback = progress_callback

    def render(self, job: VideoJob, output_dir: str | Path) -> RenderManifest:
        if not isinstance(job, VideoJob) or not job.confirmed:
            raise RuntimeError("视频任务尚未确认，禁止调用视频生成。")
        if (
            isinstance(job.target_seconds, bool)
            or not isinstance(job.target_seconds, int)
            or job.target_seconds <= 0
        ):
            raise ValueError("视频任务目标时长必须是正整数。")
        # Keep the caller's lexical path for persisted artifacts. ``resolve``
        # can expand Windows 8.3 aliases (for example ``RUNNER~1``) on CI,
        # which makes a path look outside the caller-provided temp directory
        # even though it points to the same directory. Input validation still
        # resolves paths where canonicalization is semantically required.
        target = Path(output_dir).expanduser().absolute()
        target.mkdir(parents=True, exist_ok=True)
        run_id = f"video-job-{uuid4().hex[:16]}"
        manifest = RenderManifest(
            status="running",
            output_dir=str(target),
            render_run_id=run_id,
        )
        capabilities = getattr(self.provider, "capabilities", ProviderCapabilities())
        minimum = capabilities.min_duration_seconds
        maximum = capabilities.max_duration_seconds
        if minimum is not None and job.target_seconds < minimum:
            raise ValueError(
                f"当前 Provider 的最短时长为 {minimum} 秒；"
                "这是 Provider 适配器限制，不是脚本或核心流程限制。"
            )
        if maximum is not None and job.target_seconds > maximum:
            raise ValueError(
                f"当前 Provider 的最长时长为 {maximum} 秒；"
                "请更换支持长视频的 Provider，或让该适配器自行分段。"
            )
        existing = self._read_manifest(target)
        if existing is not None:
            if existing.status == "submission_uncertain":
                return existing
        self._write_manifest(manifest, target)
        self._progress("submitting", 0.05, "正在提交完整视频任务")
        resume_request_id = None
        attempt = 1
        if existing is not None and existing.status == "pending" and existing.artifacts:
            pending = existing.artifacts[-1]
            resume_request_id = pending.request_id or None
            attempt = max(1, pending.attempt)
        intent = VideoArtifact(
            artifact_id=f"submit-intent-video-{uuid4().hex[:12]}",
            shot_id=1,
            provider=str(getattr(self.provider, "provider_name", "unknown")),
            model=str(getattr(self.provider, "endpoint", type(self.provider).__name__)),
            status="submission_uncertain",
            local_path="",
            remote_url="",
            duration=job.target_seconds,
            prompt=job.prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
            request_id=f"submit-intent-{uuid4().hex}",
            attempt=attempt,
            error_message=(
                "完整视频提交意图已持久化；若进程中断，必须先核对 Provider 后台，"
                "不会自动重复提交。"
            ),
        )
        manifest.artifacts = [intent]
        manifest.status = "submission_uncertain"
        manifest.error = intent.error_message
        self._write_manifest(manifest, target)
        try:
            provider_kwargs = {
                "attempt": attempt,
                "progress_callback": self._provider_progress,
            }
            if resume_request_id:
                provider_kwargs["resume_request_id"] = resume_request_id
            artifact = self.provider.generate_video(
                job,
                target,
                **provider_kwargs,
            )
            if artifact.status == "succeeded" and not validate_mp4_file(artifact.local_path):
                raise RuntimeError("Provider 返回的文件不是可用的 MP4 视频。")
            if artifact.status not in {"succeeded", "pending", "submission_uncertain", "failed"}:
                raise RuntimeError(f"Provider 返回了无法识别的状态：{artifact.status}")
        except VideoTaskTerminalError as exc:
            artifact = VideoArtifact(
                artifact_id=f"failed-video-{uuid4().hex[:12]}",
                shot_id=1,
                provider=str(getattr(self.provider, "provider_name", "unknown")),
                model=str(getattr(self.provider, "endpoint", type(self.provider).__name__)),
                status="failed",
                local_path="",
                remote_url="",
                duration=job.target_seconds,
                prompt=job.prompt,
                created_at=datetime.now(timezone.utc).isoformat(),
                request_id=exc.request_id,
                error_message=str(exc),
            )
        except VideoSubmissionUncertainError as exc:
            artifact = VideoArtifact(
                artifact_id=f"uncertain-video-{uuid4().hex[:12]}",
                shot_id=1,
                provider=str(getattr(self.provider, "provider_name", "unknown")),
                model=str(getattr(self.provider, "endpoint", type(self.provider).__name__)),
                status="submission_uncertain",
                local_path="",
                remote_url="",
                duration=job.target_seconds,
                prompt=job.prompt,
                created_at=datetime.now(timezone.utc).isoformat(),
                request_id=exc.operation_id,
                error_message=str(exc),
            )
        except Exception as exc:
            artifact = VideoArtifact(
                artifact_id=f"failed-video-{uuid4().hex[:12]}",
                shot_id=1,
                provider=str(getattr(self.provider, "provider_name", "unknown")),
                model=str(getattr(self.provider, "endpoint", type(self.provider).__name__)),
                status="failed",
                local_path="",
                remote_url="",
                duration=job.target_seconds,
                prompt=job.prompt,
                created_at=datetime.now(timezone.utc).isoformat(),
                error_message=str(exc),
            )

        manifest.artifacts = [artifact]
        if artifact.status == "succeeded":
            manifest.status = "succeeded"
            manifest.generated_shots = [1]
            manifest.final_video_path = artifact.local_path
            self._progress("completed", 1.0, "完整视频已经生成")
        elif artifact.status == "pending":
            manifest.status = "pending"
            manifest.error = artifact.error_message or "远端视频任务仍在处理中。"
            self._progress("pending", 0.6, manifest.error)
        elif artifact.status == "submission_uncertain":
            manifest.status = "submission_uncertain"
            manifest.error = artifact.error_message
            self._progress("submission_uncertain", 0.6, manifest.error)
        else:
            manifest.status = "failed"
            manifest.failed_shots = [1]
            manifest.error = artifact.error_message or "完整视频生成失败。"
            self._progress("failed", 1.0, manifest.error)
        self._write_manifest(manifest, target)
        return manifest

    def _provider_progress(self, stage: str, fraction: float, message: str) -> None:
        self._progress(stage, 0.05 + 0.9 * max(0.0, min(1.0, fraction)), message)

    def _progress(self, stage: str, fraction: float, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(stage, fraction, message)

    @staticmethod
    def _write_manifest(manifest: RenderManifest, target: Path) -> None:
        (target / "render_manifest.json").write_text(
            json.dumps(to_plain_data(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _read_manifest(target: Path) -> RenderManifest | None:
        path = target / "render_manifest.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            artifacts = []
            fields = set(VideoArtifact.__dataclass_fields__)
            for raw in payload.get("artifacts", []):
                if isinstance(raw, dict):
                    artifacts.append(
                        VideoArtifact(
                            **{key: value for key, value in raw.items() if key in fields}
                        )
                    )
            return RenderManifest(
                status=str(payload.get("status", "")),
                output_dir=str(payload.get("output_dir", str(target))),
                render_run_id=str(payload.get("render_run_id", "")),
                generated_shots=[int(item) for item in payload.get("generated_shots", [])],
                reused_shots=[int(item) for item in payload.get("reused_shots", [])],
                failed_shots=[int(item) for item in payload.get("failed_shots", [])],
                dependency_failed_shots=[
                    int(item) for item in payload.get("dependency_failed_shots", [])
                ],
                unreferenced_fallback_shots=[
                    int(item) for item in payload.get("unreferenced_fallback_shots", [])
                ],
                artifacts=artifacts,
                final_video_path=str(payload.get("final_video_path", "")),
                audio_path=str(payload.get("audio_path", "")),
                subtitle_path=str(payload.get("subtitle_path", "")),
                error=str(payload.get("error", "")),
            )
        except (OSError, ValueError, TypeError, KeyError):
            return None


class StoryRenderer:
    def __init__(
        self,
        provider: ShotProvider,
        *,
        narration=None,
        assembler: Callable[[list[str], str | Path], str] | None = None,
        finalizer: Callable[[str, str, str, str | Path], str] | None = None,
        progress_callback: Callable[[str, float, str], None] | None = None,
        asset_base_dir: str | Path | None = None,
        frame_extractor: Callable[[str | Path, str | Path], str] | None = None,
    ) -> None:
        self.provider = provider
        self.narration = narration or EdgeNarrationSynthesizer()
        self.assembler = assembler or concatenate_mp4_clips
        self.finalizer = finalizer or mux_audio_and_subtitles
        self.progress_callback = progress_callback
        self.asset_base_dir = asset_base_dir
        self.frame_extractor = frame_extractor or extract_last_frame

    def render(self, plan: StoryboardPlan, output_dir: str | Path) -> RenderManifest:
        if not plan.confirmed:
            raise RuntimeError("分镜尚未确认，禁止调用视频生成。")
        # See the VideoJobRenderer note above: artifact paths must remain
        # relative to the exact output root supplied by the caller.
        target = Path(output_dir).expanduser().absolute()
        target.mkdir(parents=True, exist_ok=True)
        run_id = self._render_run_id(plan, target)
        manifest = RenderManifest(
            status="running",
            output_dir=str(target),
            render_run_id=run_id,
        )
        self._recover_submission_journal(plan, target, run_id)
        visual_errors = verify_confirmed_visual_inputs(plan)
        if visual_errors:
            raise RuntimeError(
                "已确认视觉输入失效，禁止调用 Provider：" + "；".join(visual_errors)
            )
        resolve_reference_assets(plan, base_dir=self.asset_base_dir)
        capabilities, legacy_provider = self._provider_capabilities()
        provider_name = str(getattr(self.provider, "provider_name", "unknown"))
        model = str(getattr(self.provider, "endpoint", type(self.provider).__name__))
        current_artifacts: dict[int, VideoArtifact] = {}
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
            previous = self._previous_shot(plan, shot)
            provider_shot, dependency_error, used_fallback, diagnostics = (
                self._prepare_provider_shot(
                    shot,
                    previous,
                    current_artifacts,
                    capabilities,
                    legacy_provider=legacy_provider,
                    run_id=run_id,
                )
            )
            shot.continuity_diagnostics = self._unique(
                [*shot.continuity_diagnostics, *diagnostics]
            )
            refresh_shot_prompts(provider_shot)
            fingerprint = build_input_fingerprint(
                provider_shot,
                provider=provider_name,
                model=model,
            )
            if dependency_error:
                attempt = 1 + sum(
                    item.shot_id == shot.shot_id for item in plan.artifacts
                )
                artifact = self._failed_artifact(
                    provider_shot,
                    attempt,
                    RuntimeError(dependency_error),
                )
                self._attach_evidence(
                    artifact,
                    provider_shot,
                    fingerprint,
                    diagnostics,
                    used_fallback=used_fallback,
                )
                self._store_artifact(plan, None, artifact)
                manifest.artifacts.append(artifact)
                manifest.failed_shots.append(shot.shot_id)
                manifest.dependency_failed_shots.append(shot.shot_id)
                manifest.error = self._append_error(
                    manifest.error,
                    f"镜头 {shot.shot_id} 依赖失败：{dependency_error}",
                )
                self._progress(
                    "dependency_failed",
                    0.1 + 0.75 * (index / len(plan.shots)),
                    manifest.error,
                )
                self._write_manifest(manifest, target)
                continue

            reusable = self._find_reusable(plan, provider_shot, fingerprint)
            if reusable:
                self._attach_evidence(
                    reusable,
                    provider_shot,
                    fingerprint,
                    diagnostics,
                    used_fallback=used_fallback,
                )
                self._ensure_last_frame(
                    reusable,
                    shot,
                    target,
                    required=self._has_chain_dependent(plan, shot.shot_id),
                    enabled=not legacy_provider,
                    run_id=run_id,
                    capabilities=capabilities,
                )
                current_artifacts[shot.shot_id] = reusable
                manifest.reused_shots.append(shot.shot_id)
                manifest.artifacts.append(reusable)
                if reusable.used_unreferenced_fallback:
                    manifest.unreferenced_fallback_shots.append(shot.shot_id)
                self._progress(
                    "reused",
                    0.1 + 0.75 * ((index + 1) / len(plan.shots)),
                    f"复用镜头 {shot.shot_id}",
                )
                self._write_manifest(manifest, target)
                continue
            pending = self._find_pending(plan, provider_shot)
            if pending and not self._artifact_matches_input(
                pending,
                provider_shot,
                fingerprint,
            ):
                manifest.status = "pending"
                manifest.artifacts.append(pending)
                manifest.error = (
                    f"镜头 {shot.shot_id} 仍有旧版本远端任务在处理中"
                    f"（任务 ID：{pending.request_id}）。为避免重复付费，本次没有提交新任务；"
                    "请先等待旧任务完成，再生成当前 Retake 版本。"
                )
                self._progress(
                    "pending",
                    0.1 + 0.75 * (index / len(plan.shots)),
                    manifest.error,
                )
                self._write_manifest(manifest, target)
                return manifest
            uncertain = self._find_uncertain(plan, provider_shot)
            if uncertain is not None:
                manifest.status = "submission_uncertain"
                manifest.artifacts.append(uncertain)
                manifest.error = (
                    f"镜头 {shot.shot_id} 上次提交的结果无法确认"
                    f"（本地操作 ID：{uncertain.request_id}）。"
                    "为避免重复付费，本次没有自动重提；请先到 Provider 后台核对任务。"
                )
                self._progress(
                    "submission_uncertain",
                    0.1 + 0.75 * (index / len(plan.shots)),
                    manifest.error,
                )
                self._write_manifest(manifest, target)
                return manifest
            attempt = (
                pending.attempt
                if pending
                else 1 + sum(item.shot_id == shot.shot_id for item in plan.artifacts)
            )
            submission_intent: VideoArtifact | None = None
            if pending is None:
                submission_intent = self._submission_intent_artifact(
                    provider_shot,
                    attempt,
                )
                self._attach_evidence(
                    submission_intent,
                    provider_shot,
                    fingerprint,
                    diagnostics,
                    used_fallback=used_fallback,
                )
                plan.artifacts.append(submission_intent)
                manifest.artifacts.append(submission_intent)
                manifest.status = "submission_uncertain"
                manifest.error = submission_intent.error_message
                self._write_manifest(manifest, target)
            try:
                provider_kwargs = {
                    "attempt": attempt,
                    "progress_callback": (
                        lambda stage, fraction, message, i=index: self._progress(
                            stage,
                            0.1 + 0.75 * ((i + fraction) / len(plan.shots)),
                            f"镜头 {shot.shot_id}/{len(plan.shots)}：{message}",
                        )
                    ),
                }
                if pending and pending.request_id:
                    provider_kwargs["resume_request_id"] = pending.request_id
                artifact = self.provider.generate_shot(
                    provider_shot,
                    target,
                    **provider_kwargs,
                )
                if artifact.status == "succeeded" and not validate_mp4_file(artifact.local_path):
                    raise RuntimeError("Provider 返回的文件不是可用的 MP4 视频。")
                if artifact.status not in {"succeeded", "pending"}:
                    raise RuntimeError(f"Provider 返回了无法识别的状态：{artifact.status}")
            except VideoTaskTerminalError as exc:
                artifact = self._failed_artifact(
                    provider_shot,
                    attempt,
                    exc,
                    request_id=exc.request_id,
                )
            except VideoSubmissionUncertainError as exc:
                artifact = self._uncertain_artifact(
                    provider_shot,
                    attempt,
                    exc,
                    operation_id=exc.operation_id,
                )
            except Exception as exc:
                artifact = self._failed_artifact(provider_shot, attempt, exc)
            self._attach_evidence(
                artifact,
                provider_shot,
                fingerprint,
                diagnostics,
                used_fallback=used_fallback,
            )
            previous_artifact = pending or submission_intent
            self._store_artifact(plan, previous_artifact, artifact)
            if submission_intent is not None:
                manifest.artifacts[manifest.artifacts.index(submission_intent)] = artifact
            else:
                manifest.artifacts.append(artifact)
            manifest.status = "running"
            manifest.error = ""
            if artifact.status == "pending":
                manifest.status = "pending"
                manifest.error = (
                    f"镜头 {shot.shot_id} 的远端任务仍在处理中"
                    f"（任务 ID：{artifact.request_id}）。再次生成时会继续查询，不会重复提交。"
                )
                self._progress("pending", 0.1 + 0.75 * (index / len(plan.shots)), manifest.error)
                self._write_manifest(manifest, target)
                return manifest
            if artifact.status == "submission_uncertain":
                manifest.status = "submission_uncertain"
                manifest.error = artifact.error_message
                self._progress(
                    "submission_uncertain",
                    0.1 + 0.75 * (index / len(plan.shots)),
                    manifest.error,
                )
                self._write_manifest(manifest, target)
                return manifest
            if artifact.status == "failed":
                manifest.failed_shots.append(shot.shot_id)
                manifest.error = self._append_error(
                    manifest.error,
                    f"镜头 {shot.shot_id} 生成失败：{artifact.error_message}",
                )
                self._write_manifest(manifest, target)
                continue
            self._ensure_last_frame(
                artifact,
                shot,
                target,
                required=self._has_chain_dependent(plan, shot.shot_id),
                enabled=not legacy_provider,
                run_id=run_id,
                capabilities=capabilities,
            )
            current_artifacts[shot.shot_id] = artifact
            manifest.generated_shots.append(shot.shot_id)
            if artifact.used_unreferenced_fallback:
                manifest.unreferenced_fallback_shots.append(shot.shot_id)
            self._write_manifest(manifest, target)

        if manifest.failed_shots:
            manifest.status = "failed"
            self._progress(
                "incomplete",
                0.88,
                "部分镜头生成失败；成功镜头已经保存，下次只需重试失败镜头",
            )
            self._write_manifest(manifest, target)
            return manifest

        clips = [item.local_path for item in manifest.artifacts if item.status == "succeeded"]
        try:
            self._progress("assembling", 0.9, "正在拼接全部镜头")
            silent = self.assembler(clips, target / "silent_final.mp4")
            if manifest.audio_path or manifest.subtitle_path:
                final = self.finalizer(
                    silent,
                    manifest.audio_path,
                    manifest.subtitle_path,
                    target / "final_video.mp4",
                )
            else:
                final = silent
            manifest.final_video_path = final
            continuity_warnings = self._unique(
                [
                    message
                    for artifact in manifest.artifacts
                    for message in artifact.continuity_diagnostics
                    if message.startswith("警告：")
                    or "无参考文本回退" in message
                    or "后续同场景镜头将依赖失败" in message
                    or "Provider 不支持通用" in message
                    or "Provider 未声明视觉能力" in message
                ]
            )
            warning_parts = [
                item
                for item in (
                    narration_warning,
                    *continuity_warnings,
                )
                if item
            ]
            if warning_parts:
                manifest.status = "succeeded_with_warnings"
                manifest.error = "；".join(warning_parts)
                self._progress("completed_with_warnings", 1.0, manifest.error)
            else:
                manifest.status = "succeeded"
                self._progress("completed", 1.0, "成片已经生成")
        except Exception as exc:
            manifest.status = "failed"
            manifest.error = f"视频拼接失败：{exc}"
        self._write_manifest(manifest, target)
        return manifest

    def _find_reusable(
        self,
        plan: StoryboardPlan,
        shot: StoryboardShot,
        fingerprint: str,
    ) -> VideoArtifact | None:
        model = str(getattr(self.provider, "endpoint", type(self.provider).__name__))
        for artifact in reversed(plan.artifacts):
            if (
                artifact.shot_id == shot.shot_id
                and artifact.status == "succeeded"
                and artifact.model == model
            ):
                if not validate_mp4_file(artifact.local_path):
                    artifact.status = "failed"
                    artifact.error_message = "本地视频文件缺失或不是可用的 MP4，已取消复用。"
                    continue
                if artifact.input_fingerprint == fingerprint:
                    return artifact
        return None

    def _find_pending(self, plan: StoryboardPlan, shot: StoryboardShot) -> VideoArtifact | None:
        model = str(getattr(self.provider, "endpoint", type(self.provider).__name__))
        for artifact in reversed(plan.artifacts):
            if (
                artifact.shot_id == shot.shot_id
                and artifact.status == "pending"
                and artifact.model == model
                and artifact.request_id
            ):
                return artifact
        return None

    def _find_uncertain(
        self,
        plan: StoryboardPlan,
        shot: StoryboardShot,
    ) -> VideoArtifact | None:
        model = str(getattr(self.provider, "endpoint", type(self.provider).__name__))
        for artifact in reversed(plan.artifacts):
            if (
                artifact.shot_id == shot.shot_id
                and artifact.status == "submission_uncertain"
                and artifact.model == model
            ):
                return artifact
        return None

    def _provider_capabilities(self) -> tuple[ProviderCapabilities, bool]:
        raw = getattr(self.provider, "capabilities", None)
        if raw is None:
            return (
                ProviderCapabilities(
                    supports_text_to_video=True,
                    supports_image_to_video=False,
                    supports_reference_images=False,
                    supports_seed=False,
                ),
                True,
            )
        if callable(raw):
            raw = raw()
        if not isinstance(raw, ProviderCapabilities):
            raise TypeError("Provider capabilities 必须是 ProviderCapabilities。")
        return raw, False

    def _prepare_provider_shot(
        self,
        shot: StoryboardShot,
        previous: StoryboardShot | None,
        current_artifacts: dict[int, VideoArtifact],
        capabilities: ProviderCapabilities,
        *,
        legacy_provider: bool,
        run_id: str,
    ) -> tuple[StoryboardShot, str, bool, list[str]]:
        provider_shot = deepcopy(shot)
        diagnostics: list[str] = []
        used_fallback = False
        image_input_label = f"shot_{shot.shot_id:03d}_start"

        if shot.continuity_mode not in CONTINUITY_MODES:
            return (
                provider_shot,
                f"不支持的 continuity_mode：{shot.continuity_mode}",
                False,
                diagnostics,
            )
        if not capabilities.supports_text_to_video:
            return provider_shot, "Provider 不支持当前项目所需的视频生成", False, diagnostics
        if legacy_provider and (
            shot.continuity_mode != "independent"
            or bool(shot.confirmed_visual_inputs)
        ):
            diagnostics.append(
                "Provider 未声明视觉能力，按文本能力最小集处理；不会静默伪造跨镜头链"
            )
        fixed_start = confirmed_start_frame(shot)
        fixed_start_path = (
            fixed_start.path
            if fixed_start is not None and self._is_ready_file(fixed_start.path)
            else ""
        )
        provider_shot.reference_image_paths = [
            reference.path
            for reference in shot.confirmed_visual_inputs
            if reference.confirmed
            and reference.usage != "start_frame"
            and self._is_ready_file(reference.path)
        ]
        if previous is not None:
            check = validate_continuity_boundary(previous, shot)
            diagnostics.extend(f"警告：{message}" for message in check.warnings)
            if check.hard_errors:
                return (
                    provider_shot,
                    "；".join(check.hard_errors),
                    False,
                    diagnostics,
                )

        if shot.continuity_mode == "same_scene_chain":
            if previous is None or shot.previous_shot_id is None:
                return provider_shot, "同场景链缺少 previous_shot_id", False, diagnostics
            previous_artifact = current_artifacts.get(shot.previous_shot_id)
            upstream_frame = ""
            if previous_artifact is not None:
                if capabilities.requires_public_image_url:
                    upstream_frame = (
                        previous_artifact.generated_last_frame_path
                        if self._is_ready_file(
                            previous_artifact.generated_last_frame_path
                        )
                        else previous_artifact.published_last_frame_path
                    )
                else:
                    upstream_frame = previous_artifact.generated_last_frame_path
            if self._is_ready_file(upstream_frame):
                shot.initial_frame_source_path = upstream_frame
                shot.initial_frame_path = upstream_frame
                shot.initial_frame_url = ""
                provider_shot.initial_frame_source_path = shot.initial_frame_source_path
                provider_shot.initial_frame_path = upstream_frame
                provider_shot.initial_frame_url = ""
                image_input_label = f"shot_{previous.shot_id:03d}_last"
            elif fixed_start_path:
                shot.initial_frame_source_path = fixed_start_path
                shot.initial_frame_path = fixed_start_path
                shot.initial_frame_url = ""
                provider_shot.initial_frame_source_path = fixed_start_path
                provider_shot.initial_frame_path = fixed_start_path
                provider_shot.initial_frame_url = ""
                provider_shot.continuity_mode = "new_scene_reference"
                provider_shot.transition_type = "scene_change"
                provider_shot.transition_reason = (
                    "上游末帧不可用，改用已确认 start_frame 重新建立镜头"
                )
                provider_shot.inherit_previous_frame = False
                provider_shot.previous_shot_id = None
                diagnostics.append(
                    "上游末帧不可用，已明确降级为已确认的 start_frame，不使用文本裸生成"
                )
            else:
                return (
                    provider_shot,
                    "上一个镜头没有安全可用的已发布末帧，"
                    "且本镜头没有已确认 start_frame",
                    False,
                    diagnostics,
                )
        elif shot.continuity_mode in {
            "same_scene_reference",
            "new_scene_reference",
        }:
            shot.initial_frame_source_path = fixed_start_path
            shot.initial_frame_path = fixed_start_path
            shot.initial_frame_url = ""
            provider_shot.initial_frame_source_path = fixed_start_path
            provider_shot.initial_frame_path = shot.initial_frame_path
            provider_shot.initial_frame_url = ""
            if not provider_shot.initial_frame_path:
                if provider_shot.reference_image_paths:
                    diagnostics.append(
                        "本镜头正常切换机位，不继承上一镜头末帧；"
                        "将使用已确认人物/地点/道具参考图重新构图"
                    )
                else:
                    used_fallback = True
                    diagnostics.append(
                        "本镜头没有已确认 start_frame 或固定参考图，"
                        "已显式标记为无参考文本回退；不会继承上一镜头构图"
                    )
        else:
            provider_shot.previous_shot_id = None
            shot.initial_frame_source_path = fixed_start_path
            shot.initial_frame_path = fixed_start_path
            shot.initial_frame_url = ""
            provider_shot.initial_frame_source_path = fixed_start_path
            provider_shot.initial_frame_path = fixed_start_path
            provider_shot.initial_frame_url = ""

        if provider_shot.initial_frame_path:
            if not capabilities.supports_image_to_video:
                if (
                    shot.continuity_mode == "same_scene_chain"
                    and capabilities.supports_reference_images
                ):
                    provider_shot.reference_image_paths = self._unique(
                        [
                            provider_shot.initial_frame_path,
                            *provider_shot.reference_image_paths,
                        ]
                    )
                    provider_shot.initial_frame_path = ""
                    provider_shot.initial_frame_url = ""
                    diagnostics.append(
                        "Provider 不支持首帧输入，已把上游末帧作为显式参考图传入"
                    )
                elif shot.continuity_mode == "same_scene_chain":
                    return (
                        provider_shot,
                        "Provider 适配器不能使用本地首帧，禁止把同场景链静默降级为文生视频",
                        False,
                        diagnostics,
                    )
                else:
                    provider_shot.initial_frame_path = ""
                    provider_shot.initial_frame_url = ""
                    provider_shot.reference_image_paths = []
                    provider_shot.continuity_mode = "independent"
                    provider_shot.transition_type = "independent"
                    provider_shot.transition_reason = (
                        "Provider 不支持已确认的本地首帧，改为独立文本镜头"
                    )
                    provider_shot.inherit_previous_frame = False
                    provider_shot.previous_shot_id = None
                    used_fallback = True
                    diagnostics.append(
                        "Provider 不能使用本地首帧，本镜头已显式标记为无参考文本回退"
                    )
            elif capabilities.requires_public_image_url and not provider_shot.initial_frame_url:
                try:
                    publication = self._prepare_image_input(
                        provider_shot.initial_frame_path,
                        run_id=run_id,
                        label=image_input_label,
                    )
                    shot.initial_frame_path = publication.published_path
                    shot.initial_frame_url = publication.public_url
                    provider_shot.initial_frame_path = publication.published_path
                    provider_shot.initial_frame_url = publication.public_url
                except Exception as exc:
                    return (
                        provider_shot,
                        f"首帧安全发布失败：{exc}",
                        False,
                        diagnostics,
                    )

        if provider_shot.reference_image_paths:
            if capabilities.supports_reference_images:
                pass
            else:
                provider_shot.reference_image_paths = []
                used_fallback = used_fallback or not bool(
                    provider_shot.initial_frame_path
                )
                diagnostics.append(
                    "Provider 不支持通用身份/地点/道具参考图；这些图片仅保留为审计输入，"
                    "不会转换成 Agnes image"
                )

        if shot.seed is not None and not capabilities.supports_seed:
            provider_shot.seed = None
            diagnostics.append("Provider 不支持 seed；未向 Provider 发送该字段")
        return provider_shot, "", used_fallback, diagnostics

    def _ensure_last_frame(
        self,
        artifact: VideoArtifact,
        shot: StoryboardShot,
        target: Path,
        *,
        required: bool,
        enabled: bool,
        run_id: str,
        capabilities: ProviderCapabilities,
    ) -> None:
        if not enabled or artifact.status != "succeeded":
            return
        existing = artifact.generated_last_frame_path
        if self._is_ready_file(existing):
            shot.generated_last_frame_path = existing
        else:
            output = target / "frames" / f"shot_{shot.shot_id:03d}_last.png"
            try:
                extracted = self.frame_extractor(artifact.local_path, output)
                if not self._is_ready_file(extracted):
                    raise RuntimeError("末帧文件不存在或为空")
                shot.generated_last_frame_path = str(
                    Path(extracted).expanduser().absolute()
                )
                artifact.generated_last_frame_path = shot.generated_last_frame_path
            except Exception as exc:
                message = f"末帧提取失败：{exc}"
                if required:
                    message = f"{message}；后续同场景镜头将依赖失败"
                shot.continuity_diagnostics = self._unique(
                    [*shot.continuity_diagnostics, message]
                )
                artifact.continuity_diagnostics = self._unique(
                    [*artifact.continuity_diagnostics, message]
                )
                return
        if not required or not capabilities.requires_public_image_url:
            return
        try:
            publication = self._prepare_image_input(
                artifact.generated_last_frame_path,
                run_id=run_id,
                label=f"shot_{shot.shot_id:03d}_last",
            )
            artifact.published_last_frame_path = publication.published_path
            artifact.published_last_frame_url = publication.public_url
        except Exception as exc:
            message = (
                f"末帧公网暂存失败：{exc}；后续同场景镜头将 dependency_failed"
            )
            shot.continuity_diagnostics = self._unique(
                [*shot.continuity_diagnostics, message]
            )
            artifact.continuity_diagnostics = self._unique(
                [*artifact.continuity_diagnostics, message]
            )

    def _prepare_image_input(
        self,
        source_path: str | Path,
        *,
        run_id: str,
        label: str,
    ) -> PublishedImage:
        prepare = getattr(self.provider, "prepare_image_input", None)
        if not callable(prepare):
            raise RuntimeError(
                "Provider 要求公网图片，但适配器没有实现安全暂存/映射。"
                "请配置 VIDEO_REFERENCE_ROOT 与 VIDEO_REFERENCE_BASE_URL，"
                "或在 Provider 中实现 prepare_image_input。"
            )
        publication = prepare(source_path, run_id=run_id, label=label)
        if not isinstance(publication, PublishedImage):
            raise TypeError("Provider prepare_image_input 必须返回 PublishedImage。")
        if not self._is_ready_file(publication.published_path):
            raise FileNotFoundError("Provider 返回的公开暂存文件不存在或为空。")
        if not publication.public_url.startswith(("http://", "https://")):
            raise ValueError("Provider 返回的公开图片 URL 不是有效的 http(s) 地址。")
        return publication

    @staticmethod
    def _render_run_id(plan: StoryboardPlan, target: Path) -> str:
        payload = f"{target}|{plan.title}|{plan.base_seed}".encode("utf-8")
        return f"render-{hashlib.sha256(payload).hexdigest()[:16]}"

    @staticmethod
    def _previous_shot(
        plan: StoryboardPlan,
        shot: StoryboardShot,
    ) -> StoryboardShot | None:
        if shot.previous_shot_id is None:
            return None
        return next(
            (
                candidate
                for candidate in plan.shots
                if candidate.shot_id == shot.previous_shot_id
            ),
            None,
        )

    @staticmethod
    def _has_chain_dependent(plan: StoryboardPlan, shot_id: int) -> bool:
        return any(
            shot.continuity_mode == "same_scene_chain"
            and shot.previous_shot_id == shot_id
            for shot in plan.shots
        )

    @staticmethod
    def _artifact_matches_input(
        artifact: VideoArtifact,
        shot: StoryboardShot,
        fingerprint: str,
    ) -> bool:
        if artifact.input_fingerprint:
            return artifact.input_fingerprint == fingerprint
        return artifact.prompt == shot.video_prompt

    @staticmethod
    def _attach_evidence(
        artifact: VideoArtifact,
        shot: StoryboardShot,
        fingerprint: str,
        diagnostics: list[str],
        *,
        used_fallback: bool,
    ) -> None:
        artifact.prompt = shot.video_prompt
        artifact.duration = shot.duration
        artifact.reference_image_paths = list(shot.reference_image_paths)
        artifact.confirmed_visual_inputs = deepcopy(shot.confirmed_visual_inputs)
        artifact.initial_frame_source_path = shot.initial_frame_source_path
        artifact.initial_frame_path = shot.initial_frame_path
        artifact.initial_frame_url = shot.initial_frame_url
        artifact.previous_shot_id = shot.previous_shot_id
        artifact.continuity_mode = shot.continuity_mode
        artifact.transition_type = shot.transition_type
        artifact.transition_reason = shot.transition_reason
        artifact.inherit_previous_frame = shot.inherit_previous_frame
        artifact.input_fingerprint = fingerprint
        artifact.seed = shot.seed
        if artifact.generated_first_frame_path:
            shot.generated_first_frame_path = artifact.generated_first_frame_path
        else:
            artifact.generated_first_frame_path = shot.generated_first_frame_path
        artifact.continuity_diagnostics = StoryRenderer._unique(
            [*artifact.continuity_diagnostics, *diagnostics]
        )
        artifact.used_unreferenced_fallback = used_fallback

    @staticmethod
    def _is_ready_file(path: str | Path | None) -> bool:
        if not path:
            return False
        candidate = Path(path)
        return candidate.is_file() and candidate.stat().st_size > 0

    @staticmethod
    def _append_error(existing: str, message: str) -> str:
        return f"{existing}；{message}" if existing else message

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result

    @staticmethod
    def _store_artifact(
        plan: StoryboardPlan,
        previous: VideoArtifact | None,
        artifact: VideoArtifact,
    ) -> None:
        if previous is None:
            plan.artifacts.append(artifact)
            return
        plan.artifacts[plan.artifacts.index(previous)] = artifact

    def _recover_submission_journal(
        self,
        plan: StoryboardPlan,
        target: Path,
        run_id: str,
    ) -> None:
        """Recover durable per-shot evidence written before a provider submission."""
        path = target / "render_manifest.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or str(data.get("render_run_id", "")) != run_id:
                return
            raw_artifacts = data.get("artifacts", [])
            if not isinstance(raw_artifacts, list):
                return
            known = {artifact.artifact_id for artifact in plan.artifacts}
            for raw in raw_artifacts:
                artifact = self._artifact_from_journal(raw)
                if artifact is None or artifact.artifact_id in known:
                    continue
                plan.artifacts.append(artifact)
                known.add(artifact.artifact_id)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return

    @staticmethod
    def _artifact_from_journal(raw: object) -> VideoArtifact | None:
        if not isinstance(raw, dict):
            return None
        required = {
            "artifact_id",
            "shot_id",
            "provider",
            "model",
            "status",
            "local_path",
            "remote_url",
            "duration",
            "prompt",
            "created_at",
        }
        if required - set(raw):
            return None
        payload = {
            key: value
            for key, value in raw.items()
            if key in VideoArtifact.__dataclass_fields__
        }
        payload["remote_url"] = sanitize_remote_url(str(payload.get("remote_url", "")))
        references: list[VisualReference] = []
        for item in payload.get("confirmed_visual_inputs", []) or []:
            if not isinstance(item, dict):
                continue
            reference_payload = {
                key: value
                for key, value in item.items()
                if key in VisualReference.__dataclass_fields__
            }
            try:
                references.append(VisualReference(**reference_payload))
            except TypeError:
                continue
        payload["confirmed_visual_inputs"] = references
        try:
            artifact = VideoArtifact(**payload)
        except (TypeError, ValueError):
            return None
        if artifact.status not in {
            "succeeded",
            "pending",
            "submission_uncertain",
            "failed",
        }:
            return None
        return artifact

    def _submission_intent_artifact(
        self,
        shot: StoryboardShot,
        attempt: int,
    ) -> VideoArtifact:
        operation_id = f"submit-intent-{uuid4().hex}"
        return VideoArtifact(
            artifact_id=f"intent_shot_{shot.shot_id}_{attempt}_{uuid4().hex[:12]}",
            shot_id=shot.shot_id,
            provider=str(getattr(self.provider, "provider_name", "unknown")),
            model=str(getattr(self.provider, "endpoint", type(self.provider).__name__)),
            status="submission_uncertain",
            local_path="",
            remote_url="",
            duration=shot.duration,
            prompt=shot.video_prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
            request_id=operation_id,
            attempt=attempt,
            error_message=(
                "视频提交已经开始，但尚未取得可持久化的 Provider 任务 ID。"
                "如果进程此时中断，系统会阻止自动重提，请先到 Provider 后台核对。"
            ),
        )

    def _failed_artifact(
        self,
        shot: StoryboardShot,
        attempt: int,
        exc: Exception,
        *,
        request_id: str | None = None,
    ) -> VideoArtifact:
        return VideoArtifact(
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
            request_id=request_id,
            attempt=attempt,
            error_message=str(exc),
        )

    def _uncertain_artifact(
        self,
        shot: StoryboardShot,
        attempt: int,
        exc: Exception,
        *,
        operation_id: str,
    ) -> VideoArtifact:
        return VideoArtifact(
            artifact_id=f"uncertain_shot_{shot.shot_id}_{attempt}",
            shot_id=shot.shot_id,
            provider=str(getattr(self.provider, "provider_name", "unknown")),
            model=str(getattr(self.provider, "endpoint", type(self.provider).__name__)),
            status="submission_uncertain",
            local_path="",
            remote_url="",
            duration=shot.duration,
            prompt=shot.video_prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
            request_id=operation_id,
            attempt=attempt,
            error_message=str(exc),
        )

    def _progress(self, stage: str, fraction: float, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(stage, min(1.0, max(0.0, fraction)), message)

    @staticmethod
    def _write_manifest(manifest: RenderManifest, target: Path) -> None:
        manifest_path = target / "render_manifest.json"
        temporary = target / ".render_manifest.json.tmp"
        try:
            temporary.write_text(
                json.dumps(to_plain_data(manifest), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(manifest_path)
        finally:
            temporary.unlink(missing_ok=True)


def extract_last_frame(
    video_path: str | Path,
    output_path: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ffmpeg_path: str | None = None,
) -> str:
    """Decode the full video and retain the final valid frame at a stable path."""
    source = Path(video_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"无法提取末帧，视频不存在或为空：{source}")
    executable = ffmpeg_path or shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("未找到 ffmpeg，无法提取镜头末帧。")
    target = Path(output_path).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    completed = runner(
        [
            executable,
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-update",
            "1",
            str(target),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        detail = str(getattr(completed, "stderr", ""))[-1000:]
        raise RuntimeError(f"ffmpeg 末帧提取失败：{detail}")
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError("ffmpeg 未生成有效的末帧图片。")
    return str(target)


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
        "\n".join(
            f"file '{path.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
            for path in resolved
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        copy_result = subprocess.run(
            [
                executable,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-an",
                str(target),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
        if copy_result.returncode != 0:
            encoded = subprocess.run(
                [
                    executable,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    str(target),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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
    def is_nonempty_file(candidate_path: str) -> bool:
        if not candidate_path:
            return False
        candidate = Path(candidate_path)
        try:
            return candidate.is_file() and candidate.stat().st_size > 0
        except OSError:
            return False

    has_audio = is_nonempty_file(audio_path)
    has_subtitle = is_nonempty_file(subtitle_path)
    command = [executable, "-y", "-i", video_path]
    audio_index: int | None = None
    subtitle_index: int | None = None
    if has_audio:
        audio_index = 1
        command.extend(["-i", audio_path])
    if has_subtitle:
        subtitle_index = 2 if has_audio else 1
        command.extend(["-i", subtitle_path])

    command.extend(["-map", "0:v:0"])
    if audio_index is not None:
        command.extend(["-map", f"{audio_index}:a:0"])
    else:
        command.extend(["-map", "0:a?"])
    if subtitle_index is not None:
        command.extend(["-map", f"{subtitle_index}:0", "-c:s", "mov_text"])

    command.extend(["-c:v", "copy"])
    if audio_index is not None:
        command.extend(["-c:a", "aac", "-af", "apad"])
        video_duration = probe_media_duration(video_path)
        if video_duration is None:
            raise RuntimeError("无法通过 ffprobe 读取视频时长；为避免短旁白截断成片，已停止合成。")
        command.extend(["-t", f"{video_duration:.3f}"])
    else:
        command.extend(["-c:a", "copy"])
    command.append(str(target))
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
        raise RuntimeError(completed.stderr[-1000:])
    return str(target)


def probe_media_duration(path: str | Path) -> float | None:
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
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
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            return None
        duration = float(completed.stdout.strip())
        return duration if duration > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

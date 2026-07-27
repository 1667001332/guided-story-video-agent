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
    VideoArtifact,
    to_plain_data,
)
from .narration import EdgeNarrationSynthesizer, NarrationUnavailable
from .video_provider import (
    PublishedImage,
    VideoSubmissionUncertainError,
    VideoTaskTerminalError,
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
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        run_id = self._render_run_id(plan, target)
        manifest = RenderManifest(
            status="running",
            output_dir=str(target),
            render_run_id=run_id,
        )
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
            fingerprint = build_input_fingerprint(
                shot,
                provider=provider_name,
                model=model,
            )
            if dependency_error:
                attempt = 1 + sum(
                    item.shot_id == shot.shot_id for item in plan.artifacts
                )
                artifact = self._failed_artifact(
                    shot,
                    attempt,
                    RuntimeError(dependency_error),
                )
                self._attach_evidence(
                    artifact,
                    shot,
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

            reusable = self._find_reusable(plan, shot, fingerprint)
            if reusable:
                self._attach_evidence(
                    reusable,
                    shot,
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
            pending = self._find_pending(plan, shot)
            if pending and not self._artifact_matches_input(
                pending,
                shot,
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
            uncertain = self._find_uncertain(plan, shot)
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
                    shot,
                    attempt,
                    exc,
                    request_id=exc.request_id,
                )
            except VideoSubmissionUncertainError as exc:
                artifact = self._uncertain_artifact(
                    shot,
                    attempt,
                    exc,
                    operation_id=exc.operation_id,
                )
            except Exception as exc:
                artifact = self._failed_artifact(shot, attempt, exc)
            self._attach_evidence(
                artifact,
                shot,
                fingerprint,
                diagnostics,
                used_fallback=used_fallback,
            )
            self._store_artifact(plan, pending, artifact)
            manifest.artifacts.append(artifact)
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
        elif shot.continuity_mode == "new_scene_reference":
            provider_shot.previous_shot_id = None
            shot.initial_frame_source_path = fixed_start_path
            shot.initial_frame_path = fixed_start_path
            shot.initial_frame_url = ""
            provider_shot.initial_frame_source_path = fixed_start_path
            provider_shot.initial_frame_path = shot.initial_frame_path
            provider_shot.initial_frame_url = ""
            if not provider_shot.initial_frame_path:
                used_fallback = True
                diagnostics.append(
                    "新场景没有已确认 start_frame，已显式标记为无参考文本回退；"
                    "人物/地点/道具参考图不会冒充视频首帧"
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
                shot.generated_last_frame_path = str(Path(extracted).resolve())
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
        artifact.reference_image_paths = list(shot.reference_image_paths)
        artifact.confirmed_visual_inputs = deepcopy(shot.confirmed_visual_inputs)
        artifact.initial_frame_source_path = shot.initial_frame_source_path
        artifact.initial_frame_path = shot.initial_frame_path
        artifact.initial_frame_url = shot.initial_frame_url
        artifact.previous_shot_id = shot.previous_shot_id
        artifact.continuity_mode = shot.continuity_mode
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
        pending: VideoArtifact | None,
        artifact: VideoArtifact,
    ) -> None:
        if pending is None:
            plan.artifacts.append(artifact)
            return
        plan.artifacts[plan.artifacts.index(pending)] = artifact

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
    target = Path(output_path).expanduser().resolve()
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
                "-c",
                "copy",
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
    has_audio = bool(audio_path and Path(audio_path).is_file())
    has_subtitle = bool(subtitle_path and Path(subtitle_path).is_file())
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

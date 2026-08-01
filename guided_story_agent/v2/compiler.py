"""Compile MovieIR into a provider execution plan.

``MoviePlanCompiler`` remains as a compatibility facade for callers from
Phase 2, but the real stages are now explicit:

    MoviePlan -> FilmIRBuilder -> FilmIR -> MovieIRBuilder -> MovieIR
        -> VideoJobCompiler -> VideoJob

No stage calls a Provider or rewrites creative decisions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .execution import (
    CompilationOptions,
    CompileDiagnostic,
    CompileError,
    CompileResult,
    ProviderCapabilities,
    VideoJob,
)
from .film_ir_builder import FilmIRBuilder, FilmIRBuildError
from .ir import MovieIR, ShotIR
from .ir_builder import IRBuildError, MovieIRBuilder
from .models import MoviePlan
from .validation import MovieIRValidator


class VideoJobCompiler:
    """Translate an executable, provider-neutral MovieIR into VideoJob."""

    def __init__(self, *, compiler_version: str = "v2-compiler/2") -> None:
        self.compiler_version = compiler_version.strip() or "v2-compiler/2"

    def compile(
        self,
        movie_ir: MovieIR,
        capabilities: ProviderCapabilities,
        options: CompilationOptions | None = None,
    ) -> CompileResult:
        options = options or CompilationOptions(compiler_version=self.compiler_version)
        errors: list[CompileError] = []
        warnings: list[CompileDiagnostic] = []
        if not isinstance(movie_ir, MovieIR):
            errors.append(CompileError("invalid_movie_ir_state", "输入不是 MovieIR。"))
            return _failure(errors, warnings)
        if not isinstance(capabilities, ProviderCapabilities):
            errors.append(CompileError("provider_capability_mismatch", "输入不是 ProviderCapabilities。"))
            return _failure(errors, warnings)
        errors.extend(self._validate_ir(movie_ir))
        validation = MovieIRValidator().validate(movie_ir)
        errors.extend(
            CompileError(issue.code, issue.message, issue.path)
            for issue in validation.errors
        )
        warnings.extend(
            CompileDiagnostic(issue.code, issue.message, issue.path)
            for issue in validation.warnings
        )
        errors.extend(self._validate_capabilities(movie_ir, capabilities, options))
        if errors:
            return _failure(errors, warnings, movie_ir=movie_ir)
        prompt = self._build_prompt(movie_ir)
        if not prompt.strip():
            return _failure(
                [CompileError("prompt_missing", "MovieIR 无法生成执行 Prompt。")],
                warnings,
                movie_ir=movie_ir,
            )
        output_format = options.output_format.strip() or "mp4"
        provider_profile = options.provider_profile.strip() or capabilities.provider_profile.strip()
        metadata = self._metadata(movie_ir, capabilities, options)
        job = VideoJob(
            job_id=f"v2-job-{uuid4().hex}",
            provider_key=capabilities.provider_key,
            provider_prompt=prompt,
            negative_prompt=options.negative_prompt.strip(),
            duration_seconds=float(movie_ir.target_duration_seconds),
            output_format=output_format,
            aspect_ratio=options.aspect_ratio.strip() or movie_ir.aspect_ratio,
            resolution=options.resolution.strip(),
            fps=options.fps,
            references=tuple(str(item) for item in options.references if str(item).strip()),
            character_references=tuple(
                str(item) for item in options.character_references if str(item).strip()
            ),
            continuity_references=tuple(
                str(item) for item in options.continuity_references if str(item).strip()
            ),
            source_movie_plan_id=movie_ir.source_movie_plan_id,
            source_movie_ir_id=movie_ir.ir_id,
            source_film_ir_id=movie_ir.source_film_ir_id,
            compiler_version=options.compiler_version.strip() or self.compiler_version,
            provider_profile=provider_profile,
            execution_units=tuple(self._execution_unit(shot) for shot in movie_ir.shots),
            metadata=metadata,
            created_at=datetime.now(timezone.utc).isoformat(),
            confirmed=True,
        )
        return CompileResult(video_job=job, warnings=tuple(warnings), metadata=metadata)

    @staticmethod
    def _validate_ir(movie_ir: MovieIR) -> list[CompileError]:
        errors: list[CompileError] = []
        if (
            not movie_ir.ir_id.strip()
            or not movie_ir.source_movie_plan_id.strip()
            or not movie_ir.source_film_ir_id.strip()
        ):
            errors.append(CompileError("invalid_movie_ir_state", "MovieIR 缺少稳定来源标识。"))
        if not movie_ir.shots:
            errors.append(CompileError("invalid_movie_ir_state", "MovieIR 必须包含 shots。"))
        orders = [shot.order for shot in movie_ir.shots]
        if orders != list(range(1, len(orders) + 1)):
            errors.append(CompileError("invalid_shot_order", "MovieIR shot order 不连续。", "shots"))
        total = sum(float(shot.duration_seconds) for shot in movie_ir.shots)
        if abs(total - float(movie_ir.target_duration_seconds)) > 1e-6:
            errors.append(CompileError("invalid_total_duration", "MovieIR shot duration 总和不等于目标时长。", "shots"))
        return _unique_compile_errors(errors)

    @staticmethod
    def _validate_capabilities(
        movie_ir: MovieIR,
        capabilities: ProviderCapabilities,
        options: CompilationOptions,
    ) -> list[CompileError]:
        errors: list[CompileError] = []
        duration = float(movie_ir.target_duration_seconds)
        if capabilities.min_duration_seconds is not None and duration < capabilities.min_duration_seconds:
            errors.append(CompileError("duration_out_of_range", "MovieIR 时长低于 Provider 最小时长。", "target_duration_seconds"))
        if capabilities.max_duration_seconds is not None and duration > capabilities.max_duration_seconds:
            errors.append(CompileError("duration_out_of_range", "MovieIR 时长超过 Provider 最大时长；Compiler 不会自动拆分。", "target_duration_seconds"))
        if len(movie_ir.shots) > 1 and not capabilities.supports_multi_scene_prompt:
            errors.append(CompileError("multi_scene_not_supported", "Provider 不支持多 shot Prompt；Compiler 不会合并或拆分。", "shots"))
        if options.references and not capabilities.supports_reference_images:
            errors.append(CompileError("reference_not_supported", "Provider 不支持参考图。", "references"))
        if options.character_references and not capabilities.supports_character_reference:
            errors.append(CompileError("reference_not_supported", "Provider 不支持人物参考。", "character_references"))
        if options.continuity_references and not capabilities.supports_reference_images:
            errors.append(CompileError("reference_not_supported", "Provider 不支持连续性参考。", "continuity_references"))
        if movie_ir.narration_track and not capabilities.supports_audio:
            errors.append(CompileError("audio_not_supported", "Provider 不支持旁白音频。", "narration_track"))
        aspect_ratio = options.aspect_ratio.strip() or movie_ir.aspect_ratio
        if capabilities.supported_aspect_ratios and aspect_ratio not in capabilities.supported_aspect_ratios:
            errors.append(CompileError("unsupported_aspect_ratio", "画幅比例不受 Provider 支持。", "aspect_ratio"))
        if capabilities.supported_resolutions and options.resolution and options.resolution not in capabilities.supported_resolutions:
            errors.append(CompileError("unsupported_resolution", "分辨率不受 Provider 支持。", "resolution"))
        if capabilities.supported_fps and options.fps is not None and options.fps not in capabilities.supported_fps:
            errors.append(CompileError("provider_capability_mismatch", "帧率不受 Provider 支持。", "fps"))
        output_format = options.output_format.strip() or "mp4"
        if capabilities.output_formats and output_format not in capabilities.output_formats:
            errors.append(CompileError("provider_capability_mismatch", "输出格式不受 Provider 支持。", "output_format"))
        return _unique_compile_errors(errors)

    @staticmethod
    def _execution_unit(shot: ShotIR) -> dict[str, Any]:
        return {
            "shot_id": shot.shot_id,
            "scene_id": shot.scene_id,
            "order": shot.order,
            "duration_seconds": shot.duration_seconds,
            "prompt_instruction": {
                "purpose": shot.purpose,
                "visible_action": shot.visible_action,
                "subject": shot.subject,
                "camera": shot.camera,
                "motion": shot.motion,
                "lighting": shot.lighting,
                "composition": shot.composition,
            },
            "references": list(shot.references),
            "character_identity_anchors": list(shot.character_identity_anchors),
            "continuity_anchors": list(shot.continuity_anchors),
            "acceptance_criteria": list(shot.acceptance_criteria),
        }

    @staticmethod
    def _metadata(
        movie_ir: MovieIR,
        capabilities: ProviderCapabilities,
        options: CompilationOptions,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": "movie_ir",
            "source_movie_plan_id": movie_ir.source_movie_plan_id,
            "source_film_ir_id": movie_ir.source_film_ir_id,
            "source_movie_ir_id": movie_ir.ir_id,
            "scene_ids": sorted({shot.scene_id for shot in movie_ir.shots}),
            "shot_ids": [shot.shot_id for shot in movie_ir.shots],
            "shot_count": len(movie_ir.shots),
            "character_anchor_ids": [item.character_id for item in movie_ir.character_anchors],
            "character_ids": [item.character_id for item in movie_ir.character_anchors],
            "continuity_anchor_ids": [item.anchor_id for item in movie_ir.continuity_anchors],
            "continuity_scene_ids": list(
                dict.fromkeys(
                    shot.scene_id for shot in movie_ir.shots if shot.continuity_anchors
                )
            ),
            "provider_key": capabilities.provider_key,
            "provider_name": capabilities.provider_name,
            "provider_profile": options.provider_profile.strip() or capabilities.provider_profile,
            "compiler_version": options.compiler_version.strip(),
        }
        metadata.update(dict(options.metadata))
        return metadata

    def _build_prompt(self, movie_ir: MovieIR) -> str:
        lines = [
            f"影片：{movie_ir.title}",
            f"视觉风格：{movie_ir.visual_style}",
            f"总时长：{movie_ir.target_duration_seconds:g} 秒",
            "按以下顺序执行 shot，不增加、不删除、不重排：",
        ]
        for shot in movie_ir.shots:
            lines.extend(
                [
                    f"Shot {shot.order} / {shot.shot_id}（{shot.duration_seconds:g} 秒）：",
                    f"目的：{shot.purpose}；可见动作：{shot.visible_action}；主体：{shot.subject}",
                    f"地点：{shot.location}；情绪：{shot.emotion}",
                    f"摄影：{shot.camera}；运动：{shot.motion}；光线：{shot.lighting}；构图：{shot.composition}",
                    f"人物：{'、'.join(shot.characters)}；道具：{'、'.join(shot.props)}",
                    f"连续性锚点：{'；'.join(shot.continuity_anchors)}",
                    f"必须看见：{'；'.join(shot.required_visual_evidence)}",
                    f"转场：{shot.transition_in} → {shot.transition_out}",
                ]
            )
        return "\n".join(line for line in lines if line.strip())


class MoviePlanCompiler:
    """Compatibility facade: MoviePlan → FilmIR → MovieIR → VideoJob."""

    def __init__(self, *, compiler_version: str = "v2-compiler/2") -> None:
        self.compiler_version = compiler_version

    def compile(
        self,
        movie_plan: MoviePlan,
        capabilities: ProviderCapabilities,
        options: CompilationOptions | None = None,
    ) -> CompileResult:
        film_result = FilmIRBuilder().build(movie_plan)
        if not film_result.ok or film_result.film_ir is None:
            return CompileResult(
                video_job=None,
                errors=tuple(_film_ir_error(item) for item in film_result.errors),
                warnings=tuple(
                    CompileDiagnostic(item.code, item.message, item.path)
                    for item in film_result.diagnostics
                ),
                metadata={"source": "movie_plan", "compatibility_facade": True},
            )
        ir_result = MovieIRBuilder().build(film_result.film_ir)
        if not ir_result.ok or ir_result.movie_ir is None:
            return CompileResult(
                video_job=None,
                errors=tuple(_ir_error(item) for item in ir_result.errors),
                warnings=tuple(
                    CompileDiagnostic(item.code, item.message, item.path)
                    for item in ir_result.diagnostics
                ),
                metadata={
                    "source": "movie_plan",
                    "source_film_ir_id": film_result.film_ir.ir_id,
                    "compatibility_facade": True,
                },
            )
        return VideoJobCompiler(compiler_version=self.compiler_version).compile(
            ir_result.movie_ir,
            capabilities,
            options,
        )


def compile_movie_plan_to_video_job(
    movie_plan: MoviePlan,
    capabilities: ProviderCapabilities,
    options: CompilationOptions | None = None,
) -> CompileResult:
    return MoviePlanCompiler().compile(movie_plan, capabilities, options)


def _ir_error(error: IRBuildError) -> CompileError:
    return CompileError(error.code, error.message, error.path)


def _film_ir_error(error: FilmIRBuildError) -> CompileError:
    return CompileError(error.code, error.message, error.path)


def _failure(
    errors: list[CompileError],
    warnings: list[CompileDiagnostic],
    *,
    movie_ir: MovieIR | None = None,
) -> CompileResult:
    return CompileResult(
        video_job=None,
        errors=tuple(_unique_compile_errors(errors)),
        warnings=tuple(warnings),
        metadata={
            "source": "movie_ir",
            "source_film_ir_id": movie_ir.source_film_ir_id if movie_ir else "",
            "source_movie_ir_id": movie_ir.ir_id if movie_ir else "",
        },
    )


def _unique_compile_errors(errors: list[CompileError]) -> list[CompileError]:
    seen: set[tuple[str, str, str]] = set()
    result: list[CompileError] = []
    for error in errors:
        key = (error.code, error.path, error.message)
        if key not in seen:
            seen.add(key)
            result.append(error)
    return result

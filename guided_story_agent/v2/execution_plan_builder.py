"""Deterministic MovieIR -> ExecutionBundle lowering."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import warnings
from .artifact_contracts import ExpectedArtifact, ReferenceInput, RetryPolicy, TimeoutPolicy
from .compiler import VideoJobCompiler
from .execution import CompilationOptions, ProviderCapabilities, VideoJob
from .execution_bundle import ExecutionBundle
from .execution_fingerprint import (
    execution_plan_fingerprint,
    execution_unit_fingerprint,
    video_job_fingerprint,
)
from .execution_plan import (
    ArtifactPolicy,
    DependencyEdge,
    ExecutionPlan,
    ExecutionUnit,
    ReferenceFrameStrategy,
    RuntimePolicy,
)
from .execution_plan_validation import validate_execution_bundle
from .ir import MovieIR, ShotIR
from .validation import MovieIRValidator
from .provider_contracts import CapabilitySnapshot, ProviderAssignment
from .fingerprint import content_fingerprint


@dataclass(frozen=True, slots=True)
class ExecutionPlanCompileError:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionPlanCompileDiagnostic:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionPlanCompileResult:
    bundle: ExecutionBundle | None
    errors: tuple[ExecutionPlanCompileError, ...] = ()
    warnings: tuple[ExecutionPlanCompileDiagnostic, ...] = ()

    @property
    def success(self) -> bool:
        return self.bundle is not None and not self.errors

    @property
    def execution_plan(self) -> ExecutionPlan | None:
        return self.bundle.execution_plan if self.bundle is not None else None

    @property
    def video_jobs(self) -> tuple[VideoJob, ...]:
        return self.bundle.video_jobs if self.bundle is not None else ()


class ExecutionPlanCompiler:
    """Lower a MovieIR into static units without entering Provider Runtime."""

    def __init__(self, *, compiler_version: str = "execution-plan-compiler/1") -> None:
        self.compiler_version = compiler_version.strip() or "execution-plan-compiler/1"

    def compile(
        self,
        movie_ir: MovieIR,
        capabilities: ProviderCapabilities,
        options: CompilationOptions | None = None,
    ) -> ExecutionPlanCompileResult:
        errors: list[ExecutionPlanCompileError] = []
        warnings: list[ExecutionPlanCompileDiagnostic] = []
        if not isinstance(movie_ir, MovieIR):
            return _failure("invalid_movie_ir_state", "ExecutionPlanCompiler 只接受 MovieIR。")
        if not isinstance(capabilities, ProviderCapabilities):
            return _failure("provider_capability_mismatch", "ExecutionPlanCompiler 只接受静态 ProviderCapabilities。")
        if not movie_ir.source_movie_plan_id or not movie_ir.source_movie_plan_fingerprint or not movie_ir.source_movie_plan_lineage_token:
            errors.append(ExecutionPlanCompileError("unknown_movie_plan_lineage", "MovieIR 缺少完整 MoviePlan provenance。", "source_movie_plan"))
        if not movie_ir.source_film_ir_id or not movie_ir.source_film_ir_fingerprint:
            errors.append(ExecutionPlanCompileError("unknown_film_ir_lineage", "MovieIR 缺少完整 FilmIR provenance。", "source_film_ir"))
        if not movie_ir.ir_id or not movie_ir.shots:
            errors.append(ExecutionPlanCompileError("invalid_movie_ir_state", "MovieIR 必须包含稳定 ID 和 shots。"))
        movie_validation = MovieIRValidator().validate(movie_ir)
        errors.extend(
            ExecutionPlanCompileError(issue.code, issue.message, issue.path)
            for issue in movie_validation.errors
        )
        if errors:
            return ExecutionPlanCompileResult(None, tuple(_unique(errors)), tuple(warnings))

        options = options or CompilationOptions(compiler_version=self.compiler_version)
        snapshot = CapabilitySnapshot.from_capabilities(capabilities)
        assignment = ProviderAssignment(
            assignment_id=_stable_id("assignment", capabilities.provider_key, capabilities.provider_profile),
            provider_key=capabilities.provider_key,
            provider_profile=capabilities.provider_profile,
            capability_snapshot_id=snapshot.snapshot_id,
            fallback_provider_keys=(),
            selection_reason="静态 capability snapshot 与当前 MovieIR lowering 兼容",
        )
        movie_ir_fingerprint = content_fingerprint(movie_ir.to_dict())
        jobs: list[VideoJob] = []
        units: list[ExecutionUnit] = []
        edges: list[DependencyEdge] = []
        previous_unit: ExecutionUnit | None = None
        for index, shot in enumerate(sorted(movie_ir.shots, key=lambda item: item.order)):
            shot_ir = _single_shot_movie_ir(movie_ir, shot)
            shot_options = replace(
                options,
                compiler_version=options.compiler_version.strip() or self.compiler_version,
                references=tuple(dict.fromkeys((*options.references, *shot.references))),
            )
            result = VideoJobCompiler(compiler_version=self.compiler_version).compile(
                shot_ir,
                capabilities,
                shot_options,
            )
            if not result.success or result.video_job is None:
                errors.extend(
                    ExecutionPlanCompileError(item.code, item.message, f"shots[{index}].{item.path}".rstrip("."))
                    for item in result.errors
                )
                continue
            job = replace(
                result.video_job,
                job_id=_stable_id("execution-job", shot.shot_id, result.video_job.video_job_fingerprint),
                source_movie_plan_id=movie_ir.source_movie_plan_id,
                source_movie_ir_id=movie_ir.ir_id,
                source_film_ir_id=movie_ir.source_film_ir_id,
                source_movie_plan_version=movie_ir.source_movie_plan_version,
                source_movie_plan_fingerprint=movie_ir.source_movie_plan_fingerprint,
                source_movie_plan_lineage_token=movie_ir.source_movie_plan_lineage_token,
                source_film_ir_fingerprint=movie_ir.source_film_ir_fingerprint,
                source_movie_ir_fingerprint=movie_ir_fingerprint,
                schema_version="v2-video-job/1",
            )
            job = replace(job, video_job_fingerprint=video_job_fingerprint(job))
            jobs.append(job)
            unit_id = _stable_id("execution-unit", shot.shot_id)
            depends_on: tuple[str, ...] = ()
            references: list[ReferenceInput] = [
                ReferenceInput(
                    reference_id=f"{unit_id}-input-{ref_index + 1}",
                    kind="declared_movie_ir_reference",
                    source_shot_id=shot.shot_id,
                    metadata={"reference_key": reference},
                )
                for ref_index, reference in enumerate(shot.references)
            ]
            unit_edges: list[DependencyEdge] = []
            if previous_unit is not None:
                depends_on = (previous_unit.execution_unit_id,)
                unit_edges.append(DependencyEdge(previous_unit.execution_unit_id, unit_id, "serial"))
                if previous_unit.source_scene_ids == (shot.scene_id,):
                    references.append(
                        ReferenceInput(
                            reference_id=f"{unit_id}-previous-end-frame",
                            kind="previous_shot_end_frame",
                            source_shot_id=previous_unit.source_shot_ids[-1],
                            source_unit_id=previous_unit.execution_unit_id,
                            required=True,
                        )
                    )
                    unit_edges.append(
                        DependencyEdge(previous_unit.execution_unit_id, unit_id, "reference_frame")
                    )
            unit = ExecutionUnit(
                execution_unit_id=unit_id,
                source_scene_ids=(shot.scene_id,),
                source_shot_ids=(shot.shot_id,),
                video_job_id=job.job_id,
                video_job_fingerprint=job.video_job_fingerprint,
                provider_assignment_id=assignment.assignment_id,
                depends_on=depends_on,
                reference_inputs=tuple(references),
                expected_artifacts=(
                    ExpectedArtifact(
                        artifact_type="video",
                        artifact_key=f"{unit_id}.video",
                        required=True,
                        metadata={"output_format": shot_options.output_format.strip() or "mp4"},
                    ),
                ),
                # Phase 5A consumes this static policy; retries still use the
                # same immutable VideoJob and idempotency key.
                retry_policy=RetryPolicy(
                    max_attempts=3,
                    retryable_error_codes=(
                        "transient_network",
                        "rate_limited",
                        "provider_unavailable",
                        "timeout",
                        "download_interrupted",
                    ),
                ),
                timeout_policy=TimeoutPolicy(),
                metadata={"source_shot_id": shot.shot_id, "source_scene_id": shot.scene_id},
            )
            unit = replace(unit, execution_unit_fingerprint=execution_unit_fingerprint(unit))
            units.append(unit)
            edges.extend(unit_edges)
            previous_unit = unit

        if errors:
            return ExecutionPlanCompileResult(None, tuple(_unique(errors)), tuple(warnings))
        plan = ExecutionPlan(
            schema_version="execution-plan/1",
            execution_plan_id=f"execution-plan-{sha256(movie_ir.ir_id.encode('utf-8')).hexdigest()[:20]}-{len(units)}",
            execution_plan_version=1,
            execution_plan_fingerprint="",
            source_movie_plan_id=movie_ir.source_movie_plan_id,
            source_movie_plan_version=movie_ir.source_movie_plan_version,
            source_movie_plan_fingerprint=movie_ir.source_movie_plan_fingerprint,
            source_movie_plan_lineage_token=movie_ir.source_movie_plan_lineage_token,
            source_film_ir_id=movie_ir.source_film_ir_id,
            source_film_ir_fingerprint=movie_ir.source_film_ir_fingerprint,
            source_movie_ir_id=movie_ir.ir_id,
            source_movie_ir_fingerprint=movie_ir_fingerprint,
            execution_units=tuple(units),
            dependency_graph=tuple(edges),
            provider_assignments=(assignment,),
            runtime_policy=RuntimePolicy(scheduling_strategy="serial", max_parallel_units=1, fail_fast=True),
            reference_frame_strategy=ReferenceFrameStrategy(),
            artifact_policy=ArtifactPolicy(
                expected_artifact_types=("video",),
                output_format=options.output_format.strip() or "mp4",
            ),
            capability_snapshot=snapshot,
            metadata={
                "source": "movie_ir",
                "compiler_version": self.compiler_version,
                "unit_strategy": "one_shot_one_video_job",
                "shot_count": len(units),
            },
        )
        plan = replace(plan, execution_plan_fingerprint=execution_plan_fingerprint(plan))
        bundle = ExecutionBundle(execution_plan=plan, video_jobs=tuple(jobs))
        validation = validate_execution_bundle(bundle)
        if not validation.valid:
            return ExecutionPlanCompileResult(
                None,
                tuple(
                    _unique(
                        errors
                        + [
                            ExecutionPlanCompileError(item.code, item.message, item.path)
                            for item in validation.diagnostics
                        ]
                    )
                ),
                tuple(warnings),
            )
        return ExecutionPlanCompileResult(bundle, (), tuple(warnings))


def compile_movie_ir_to_execution_bundle(
    movie_ir: MovieIR,
    capabilities: ProviderCapabilities,
    options: CompilationOptions | None = None,
) -> ExecutionPlanCompileResult:
    warnings.warn(
        "compile_movie_ir_to_execution_bundle() is deprecated; use ExecutionPlanCompiler().compile()",
        DeprecationWarning,
        stacklevel=2,
    )
    return ExecutionPlanCompiler().compile(movie_ir, capabilities, options)


def _single_shot_movie_ir(movie_ir: MovieIR, shot: ShotIR) -> MovieIR:
    shot_id = shot.shot_id
    lowered_shot = replace(shot, order=1)
    timeline = tuple(item for item in movie_ir.timeline if item.shot_id == shot_id)
    if not timeline:
        timeline = (replace(movie_ir.timeline[shot.order - 1], shot_id=shot_id, order=1, start_seconds=0.0),)
    filtered_continuity = tuple(
        item for item in movie_ir.continuity_anchors if shot_id in item.applies_to_shots
    )
    filtered_acceptance = tuple(
        item for item in movie_ir.acceptance_criteria if item.target_id in {shot_id, movie_ir.source_movie_plan_id}
    )
    return replace(
        movie_ir,
        target_duration_seconds=float(shot.duration_seconds),
        timeline=(replace(timeline[0], order=1, start_seconds=0.0, duration_seconds=float(shot.duration_seconds)),),
        shots=(lowered_shot,),
        continuity_anchors=filtered_continuity,
        narration_track=tuple(item for item in movie_ir.narration_track if item.shot_id == shot_id),
        subtitle_track=tuple(item for item in movie_ir.subtitle_track if item.shot_id == shot_id),
        music_cues=tuple(item for item in movie_ir.music_cues if item.target_id in {shot_id, movie_ir.source_movie_plan_id}),
        transition_cues=(),
        acceptance_criteria=filtered_acceptance,
        metadata={**movie_ir.metadata, "execution_single_shot": shot_id},
    )


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "|".join(str(item) for item in parts)
    return f"{prefix}-{sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _failure(code: str, message: str) -> ExecutionPlanCompileResult:
    return ExecutionPlanCompileResult(None, (ExecutionPlanCompileError(code, message),))


def _unique(errors: list[ExecutionPlanCompileError]) -> list[ExecutionPlanCompileError]:
    seen: set[tuple[str, str, str]] = set()
    result: list[ExecutionPlanCompileError] = []
    for error in errors:
        key = (error.code, error.path, error.message)
        if key not in seen:
            seen.add(key)
            result.append(error)
    return result


__all__ = [
    "ExecutionPlanCompileDiagnostic",
    "ExecutionPlanCompileError",
    "ExecutionPlanCompileResult",
    "ExecutionPlanCompiler",
    "compile_movie_ir_to_execution_bundle",
]

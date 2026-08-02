"""Immutable provider-neutral ExecutionPlan contracts.

ExecutionPlan is a static compiler artifact.  It contains scheduling and
provenance data only; it never contains runtime status or a provider request
response.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .artifact_contracts import ExpectedArtifact, ReferenceInput, RetryPolicy, TimeoutPolicy
from .provider_contracts import CapabilitySnapshot, ProviderAssignment


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    raw = dict(value or {})
    return MappingProxyType({str(key): freeze_value(child) for key, child in raw.items()})


def freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    return value


def plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): plain_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return plain_value(value.to_dict())
    return value


@dataclass(frozen=True, slots=True)
class DependencyEdge:
    from_unit_id: str
    to_unit_id: str
    dependency_type: str

    def __post_init__(self) -> None:
        if not self.from_unit_id.strip() or not self.to_unit_id.strip():
            raise ValueError("DependencyEdge unit IDs are required")
        if not self.dependency_type.strip():
            raise ValueError("DependencyEdge.dependency_type is required")

    def to_dict(self) -> dict[str, str]:
        return {
            "from_unit_id": self.from_unit_id,
            "to_unit_id": self.to_unit_id,
            "dependency_type": self.dependency_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DependencyEdge":
        return cls(
            from_unit_id=str(data.get("from_unit_id", "")),
            to_unit_id=str(data.get("to_unit_id", "")),
            dependency_type=str(data.get("dependency_type", "")),
        )


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    scheduling_strategy: str = "serial"
    max_parallel_units: int = 1
    fail_fast: bool = True
    max_parallelism: int | None = None
    retry_budget: int | None = None

    def __post_init__(self) -> None:
        if not self.scheduling_strategy.strip():
            raise ValueError("RuntimePolicy.scheduling_strategy is required")
        if isinstance(self.max_parallel_units, bool) or self.max_parallel_units < 1:
            raise ValueError("RuntimePolicy.max_parallel_units must be positive")
        if self.max_parallelism is None:
            object.__setattr__(self, "max_parallelism", self.max_parallel_units)
        if isinstance(self.max_parallelism, bool) or self.max_parallelism < 1:
            raise ValueError("RuntimePolicy.max_parallelism must be positive")
        if self.retry_budget is not None and (isinstance(self.retry_budget, bool) or self.retry_budget < 0):
            raise ValueError("RuntimePolicy.retry_budget cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduling_strategy": self.scheduling_strategy,
            "max_parallel_units": self.max_parallel_units,
            "fail_fast": bool(self.fail_fast),
            "max_parallelism": self.max_parallelism,
            "retry_budget": self.retry_budget,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimePolicy":
        return cls(
            scheduling_strategy=str(data.get("scheduling_strategy", "serial")),
            max_parallel_units=int(data.get("max_parallel_units", 1)),
            fail_fast=bool(data.get("fail_fast", True)),
            max_parallelism=(None if data.get("max_parallelism") is None else int(data.get("max_parallelism"))),
            retry_budget=(None if data.get("retry_budget") is None else int(data.get("retry_budget"))),
        )


@dataclass(frozen=True, slots=True)
class ReferenceFrameStrategy:
    mode: str = "explicit_previous_shot_end_frame"
    same_scene_requires_previous_end_frame: bool = True
    missing_reference_behavior: str = "fail_closed"

    def __post_init__(self) -> None:
        if not self.mode.strip() or not self.missing_reference_behavior.strip():
            raise ValueError("ReferenceFrameStrategy values are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "same_scene_requires_previous_end_frame": bool(self.same_scene_requires_previous_end_frame),
            "missing_reference_behavior": self.missing_reference_behavior,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReferenceFrameStrategy":
        return cls(
            mode=str(data.get("mode", "explicit_previous_shot_end_frame")),
            same_scene_requires_previous_end_frame=bool(
                data.get("same_scene_requires_previous_end_frame", True)
            ),
            missing_reference_behavior=str(data.get("missing_reference_behavior", "fail_closed")),
        )


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    expected_artifact_types: tuple[str, ...] = ("video",)
    output_format: str = "mp4"
    require_manifest: bool = True
    allow_partial: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_artifact_types", tuple(str(item) for item in self.expected_artifact_types))
        if not self.expected_artifact_types:
            raise ValueError("ArtifactPolicy.expected_artifact_types cannot be empty")
        if not self.output_format.strip():
            raise ValueError("ArtifactPolicy.output_format is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_artifact_types": list(self.expected_artifact_types),
            "output_format": self.output_format,
            "require_manifest": bool(self.require_manifest),
            "allow_partial": bool(self.allow_partial),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactPolicy":
        return cls(
            expected_artifact_types=tuple(str(item) for item in data.get("expected_artifact_types", ["video"])),
            output_format=str(data.get("output_format", "mp4")),
            require_manifest=bool(data.get("require_manifest", True)),
            allow_partial=bool(data.get("allow_partial", False)),
        )


@dataclass(frozen=True, slots=True)
class ExecutionUnit:
    execution_unit_id: str
    source_scene_ids: tuple[str, ...]
    source_shot_ids: tuple[str, ...]
    video_job_id: str
    video_job_fingerprint: str
    provider_assignment_id: str
    depends_on: tuple[str, ...] = ()
    reference_inputs: tuple[ReferenceInput, ...] = ()
    expected_artifacts: tuple[ExpectedArtifact, ...] = ()
    retry_policy: RetryPolicy = RetryPolicy()
    timeout_policy: TimeoutPolicy = TimeoutPolicy()
    execution_unit_fingerprint: str = ""
    metadata: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        for field_name in (
            "execution_unit_id",
            "video_job_id",
            "video_job_fingerprint",
            "provider_assignment_id",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"ExecutionUnit.{field_name} is required")
        object.__setattr__(self, "source_scene_ids", tuple(str(item) for item in self.source_scene_ids))
        object.__setattr__(self, "source_shot_ids", tuple(str(item) for item in self.source_shot_ids))
        object.__setattr__(self, "depends_on", tuple(str(item) for item in self.depends_on))
        object.__setattr__(self, "reference_inputs", tuple(self.reference_inputs))
        object.__setattr__(self, "expected_artifacts", tuple(self.expected_artifacts))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_unit_id": self.execution_unit_id,
            "source_scene_ids": list(self.source_scene_ids),
            "source_shot_ids": list(self.source_shot_ids),
            "video_job_id": self.video_job_id,
            "video_job_fingerprint": self.video_job_fingerprint,
            "provider_assignment_id": self.provider_assignment_id,
            "depends_on": list(self.depends_on),
            "reference_inputs": [item.to_dict() for item in self.reference_inputs],
            "expected_artifacts": [item.to_dict() for item in self.expected_artifacts],
            "retry_policy": self.retry_policy.to_dict(),
            "timeout_policy": self.timeout_policy.to_dict(),
            "execution_unit_fingerprint": self.execution_unit_fingerprint,
            "metadata": plain_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionUnit":
        return cls(
            execution_unit_id=str(data.get("execution_unit_id", "")),
            source_scene_ids=tuple(str(item) for item in data.get("source_scene_ids", [])),
            source_shot_ids=tuple(str(item) for item in data.get("source_shot_ids", [])),
            video_job_id=str(data.get("video_job_id", "")),
            video_job_fingerprint=str(data.get("video_job_fingerprint", "")),
            provider_assignment_id=str(data.get("provider_assignment_id", "")),
            depends_on=tuple(str(item) for item in data.get("depends_on", [])),
            reference_inputs=tuple(
                ReferenceInput(
                    reference_id=str(item.get("reference_id", "")),
                    kind=str(item.get("kind", "")),
                    source_shot_id=str(item.get("source_shot_id", "")),
                    source_unit_id=str(item.get("source_unit_id", "")),
                    required=bool(item.get("required", True)),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in data.get("reference_inputs", [])
            ),
            expected_artifacts=tuple(
                ExpectedArtifact(
                    artifact_type=str(item.get("artifact_type", "")),
                    artifact_key=str(item.get("artifact_key", "")),
                    required=bool(item.get("required", True)),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in data.get("expected_artifacts", [])
            ),
            retry_policy=RetryPolicy(
                max_attempts=int(data.get("retry_policy", {}).get("max_attempts", 1)),
                backoff_seconds=float(data.get("retry_policy", {}).get("backoff_seconds", 0.0)),
                retryable_error_codes=tuple(
                    str(item) for item in data.get("retry_policy", {}).get("retryable_error_codes", [])
                ),
            ),
            timeout_policy=TimeoutPolicy(
                timeout_seconds=float(data.get("timeout_policy", {}).get("timeout_seconds", 900.0)),
                poll_interval_seconds=float(data.get("timeout_policy", {}).get("poll_interval_seconds", 5.0)),
                submit_timeout_seconds=(
                    None
                    if data.get("timeout_policy", {}).get("submit_timeout_seconds") is None
                    else float(data.get("timeout_policy", {}).get("submit_timeout_seconds"))
                ),
                poll_timeout_seconds=(
                    None
                    if data.get("timeout_policy", {}).get("poll_timeout_seconds") is None
                    else float(data.get("timeout_policy", {}).get("poll_timeout_seconds"))
                ),
                unit_total_timeout_seconds=(
                    None
                    if data.get("timeout_policy", {}).get("unit_total_timeout_seconds") is None
                    else float(data.get("timeout_policy", {}).get("unit_total_timeout_seconds"))
                ),
            ),
            execution_unit_fingerprint=str(data.get("execution_unit_fingerprint", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    schema_version: str
    execution_plan_id: str
    execution_plan_version: int
    execution_plan_fingerprint: str
    source_movie_plan_id: str
    source_movie_plan_version: int
    source_movie_plan_fingerprint: str
    source_movie_plan_lineage_token: str
    source_film_ir_id: str
    source_film_ir_fingerprint: str
    source_movie_ir_id: str
    source_movie_ir_fingerprint: str
    execution_units: tuple[ExecutionUnit, ...]
    dependency_graph: tuple[DependencyEdge, ...]
    provider_assignments: tuple[ProviderAssignment, ...]
    runtime_policy: RuntimePolicy
    reference_frame_strategy: ReferenceFrameStrategy
    artifact_policy: ArtifactPolicy
    capability_snapshot: CapabilitySnapshot
    metadata: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if isinstance(self.execution_plan_version, bool) or self.execution_plan_version < 1:
            raise ValueError("ExecutionPlan.execution_plan_version must be positive")
        for field_name in ("execution_plan_id",):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"ExecutionPlan.{field_name} is required")
        object.__setattr__(self, "execution_units", tuple(self.execution_units))
        object.__setattr__(self, "dependency_graph", tuple(self.dependency_graph))
        object.__setattr__(self, "provider_assignments", tuple(self.provider_assignments))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_plan_id": self.execution_plan_id,
            "execution_plan_version": self.execution_plan_version,
            "execution_plan_fingerprint": self.execution_plan_fingerprint,
            "source_movie_plan_id": self.source_movie_plan_id,
            "source_movie_plan_version": self.source_movie_plan_version,
            "source_movie_plan_fingerprint": self.source_movie_plan_fingerprint,
            "source_movie_plan_lineage_token": self.source_movie_plan_lineage_token,
            "source_film_ir_id": self.source_film_ir_id,
            "source_film_ir_fingerprint": self.source_film_ir_fingerprint,
            "source_movie_ir_id": self.source_movie_ir_id,
            "source_movie_ir_fingerprint": self.source_movie_ir_fingerprint,
            "execution_units": [item.to_dict() for item in self.execution_units],
            "dependency_graph": [item.to_dict() for item in self.dependency_graph],
            "provider_assignments": [item.to_dict() for item in self.provider_assignments],
            "runtime_policy": self.runtime_policy.to_dict(),
            "reference_frame_strategy": self.reference_frame_strategy.to_dict(),
            "artifact_policy": self.artifact_policy.to_dict(),
            "capability_snapshot": self.capability_snapshot.to_dict(),
            "metadata": plain_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionPlan":
        if not isinstance(data, Mapping):
            raise ValueError("execution_plan must be an object")
        return cls(
            schema_version=str(data.get("schema_version", "execution-plan/1")),
            execution_plan_id=str(data.get("execution_plan_id", "")),
            execution_plan_version=int(data.get("execution_plan_version", 0)),
            execution_plan_fingerprint=str(data.get("execution_plan_fingerprint", "")),
            source_movie_plan_id=str(data.get("source_movie_plan_id", "")),
            source_movie_plan_version=int(data.get("source_movie_plan_version", 0)),
            source_movie_plan_fingerprint=str(data.get("source_movie_plan_fingerprint", "")),
            source_movie_plan_lineage_token=str(data.get("source_movie_plan_lineage_token", "")),
            source_film_ir_id=str(data.get("source_film_ir_id", "")),
            source_film_ir_fingerprint=str(data.get("source_film_ir_fingerprint", "")),
            source_movie_ir_id=str(data.get("source_movie_ir_id", "")),
            source_movie_ir_fingerprint=str(data.get("source_movie_ir_fingerprint", "")),
            execution_units=tuple(ExecutionUnit.from_dict(item) for item in data.get("execution_units", [])),
            dependency_graph=tuple(DependencyEdge.from_dict(item) for item in data.get("dependency_graph", [])),
            provider_assignments=tuple(
                ProviderAssignment.from_dict(item) for item in data.get("provider_assignments", [])
            ),
            runtime_policy=RuntimePolicy.from_dict(data.get("runtime_policy", {})),
            reference_frame_strategy=ReferenceFrameStrategy.from_dict(
                data.get("reference_frame_strategy", {})
            ),
            artifact_policy=ArtifactPolicy.from_dict(data.get("artifact_policy", {})),
            capability_snapshot=CapabilitySnapshot.from_dict(data.get("capability_snapshot", {})),
            metadata=dict(data.get("metadata", {})),
        )


__all__ = [
    "ArtifactPolicy",
    "DependencyEdge",
    "ExecutionPlan",
    "ExecutionUnit",
    "ReferenceFrameStrategy",
    "RuntimePolicy",
    "plain_value",
]

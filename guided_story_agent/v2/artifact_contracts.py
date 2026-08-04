"""Static artifact and retry contracts for ExecutionPlan."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .fake_artifact_verifier import ArtifactRecord, ArtifactVerificationResult


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ReferenceInput:
    reference_id: str
    kind: str
    source_shot_id: str = ""
    source_unit_id: str = ""
    required: bool = True
    metadata: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not self.reference_id.strip() or not self.kind.strip():
            raise ValueError("ReferenceInput.reference_id and kind are required")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "kind": self.kind,
            "source_shot_id": self.source_shot_id,
            "source_unit_id": self.source_unit_id,
            "required": bool(self.required),
            "metadata": _plain(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExpectedArtifact:
    artifact_type: str
    artifact_key: str
    required: bool = True
    metadata: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not self.artifact_type.strip() or not self.artifact_key.strip():
            raise ValueError("ExpectedArtifact.artifact_type and artifact_key are required")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_key": self.artifact_key,
            "required": bool(self.required),
            "metadata": _plain(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 1
    backoff_seconds: float = 0.0
    retryable_error_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or self.max_attempts < 1:
            raise ValueError("RetryPolicy.max_attempts must be positive")
        if self.backoff_seconds < 0:
            raise ValueError("RetryPolicy.backoff_seconds cannot be negative")
        object.__setattr__(self, "retryable_error_codes", tuple(str(item) for item in self.retryable_error_codes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "backoff_seconds": self.backoff_seconds,
            "retryable_error_codes": list(self.retryable_error_codes),
        }


@dataclass(frozen=True, slots=True)
class TimeoutPolicy:
    timeout_seconds: float = 900.0
    poll_interval_seconds: float = 5.0
    submit_timeout_seconds: float | None = None
    poll_timeout_seconds: float | None = None
    unit_total_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.poll_interval_seconds <= 0:
            raise ValueError("TimeoutPolicy values must be positive")
        for name in ("submit_timeout_seconds", "poll_timeout_seconds", "unit_total_timeout_seconds"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"TimeoutPolicy.{name} must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "submit_timeout_seconds": self.submit_timeout_seconds,
            "poll_timeout_seconds": self.poll_timeout_seconds,
            "unit_total_timeout_seconds": self.unit_total_timeout_seconds,
        }

    @property
    def unit_total_timeout(self) -> float:
        return self.unit_total_timeout_seconds or self.timeout_seconds

    @property
    def submit_timeout(self) -> float | None:
        return self.submit_timeout_seconds

    @property
    def poll_timeout(self) -> float | None:
        return self.poll_timeout_seconds


__all__ = [
    "ArtifactRecord",
    "ArtifactVerificationResult",
    "ExpectedArtifact",
    "ReferenceInput",
    "RetryPolicy",
    "TimeoutPolicy",
]

"""Serializable provider-neutral request/result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .provider_sanitization import sanitize_response


class ProviderJobStatus(str, Enum):
    CREATED = "CREATED"
    ACCEPTED = "ACCEPTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_value(cls, value: object) -> "ProviderJobStatus":
        text = str(getattr(value, "value", value)).strip().upper()
        aliases = {"COMPLETED": cls.SUCCEEDED, "PROCESSING": cls.RUNNING, "CANCELED": cls.CANCELLED}
        return aliases.get(text, cls(text) if text in cls._value2member_map_ else cls.UNKNOWN)


@dataclass(frozen=True, slots=True)
class SanitizedOutputLocator:
    locator: str
    expires_at: str | None = None
    digest: str | None = None

    def __post_init__(self) -> None:
        if not self.locator.strip():
            raise ValueError("SanitizedOutputLocator.locator is required")
        sanitized = sanitize_response(self.locator)
        object.__setattr__(self, "locator", str(sanitized))

    def to_dict(self) -> dict[str, str | None]:
        return {"locator": self.locator, "expires_at": self.expires_at, "digest": self.digest}


@dataclass(frozen=True, slots=True)
class DownloadDestination:
    """Runtime-owned download target.

    Provider adapters receive explicit paths and never choose the final file
    name from a remote response.
    """

    temporary_path: str
    final_path: str

    def __post_init__(self) -> None:
        if not self.temporary_path or not self.final_path:
            raise ValueError("DownloadDestination requires temporary_path and final_path")
        if self.temporary_path == self.final_path:
            raise ValueError("temporary_path and final_path must differ")

    def to_dict(self) -> dict[str, str]:
        return {"temporary_path": self.temporary_path, "final_path": self.final_path}


@dataclass(frozen=True, slots=True)
class ProviderSubmitResult:
    provider_job: Any
    accepted: bool
    initial_status: Any
    retry_after_seconds: float | None = None
    sanitized_response: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sanitized_response", sanitize_response(dict(self.sanitized_response)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_job": self.provider_job.to_dict(),
            "accepted": self.accepted,
            "initial_status": getattr(self.initial_status, "value", self.initial_status),
            "retry_after_seconds": self.retry_after_seconds,
            "sanitized_response": dict(self.sanitized_response),
        }


@dataclass(frozen=True, slots=True)
class ProviderPollResult:
    provider_job: Any
    status: Any
    progress: float | None = None
    retry_after_seconds: float | None = None
    output_locator: SanitizedOutputLocator | None = None
    sanitized_response: Mapping[str, object] = field(default_factory=dict)
    # Phase 5A compatibility: callers may display a short non-secret message.
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "sanitized_response", sanitize_response(dict(self.sanitized_response)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_job": self.provider_job.to_dict(),
            "status": getattr(self.status, "value", self.status),
            "progress": self.progress,
            "retry_after_seconds": self.retry_after_seconds,
            "output_locator": self.output_locator.to_dict() if self.output_locator else None,
            "sanitized_response": dict(self.sanitized_response),
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ProviderCancelResult:
    supported: bool
    accepted: bool
    final_status: Any | None = None
    sanitized_response: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sanitized_response", sanitize_response(dict(self.sanitized_response)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "accepted": self.accepted,
            "final_status": getattr(self.final_status, "value", self.final_status),
            "sanitized_response": dict(self.sanitized_response),
        }


@dataclass(frozen=True, slots=True)
class ProviderDownloadResult:
    temporary_path: str
    final_candidate_path: str | None
    size_bytes: int
    sha256: str | None
    media_type: str | None
    completed: bool
    resumable: bool
    bytes_downloaded: int
    sanitized_response_metadata: Mapping[str, object] = field(default_factory=dict)
    artifact_type: str = "video"
    provider_job: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sanitized_response_metadata", sanitize_response(dict(self.sanitized_response_metadata)))

    @property
    def storage_path(self) -> str:
        return self.final_candidate_path or self.temporary_path

    @property
    def partial(self) -> bool:
        return not self.completed

    def to_dict(self) -> dict[str, Any]:
        return {
            "temporary_path": self.temporary_path,
            "final_candidate_path": self.final_candidate_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "completed": self.completed,
            "resumable": self.resumable,
            "bytes_downloaded": self.bytes_downloaded,
            "sanitized_response_metadata": dict(self.sanitized_response_metadata),
            "artifact_type": self.artifact_type,
        }


@dataclass(frozen=True, slots=True)
class ProviderVerificationResult:
    valid: bool
    verification_code: str
    message: str
    size_bytes: int
    sha256: str
    media_type: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_response(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "verification_code": self.verification_code,
            "message": self.message,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "DownloadDestination",
    "ProviderCancelResult",
    "ProviderDownloadResult",
    "ProviderPollResult",
    "ProviderSubmitResult",
    "ProviderVerificationResult",
    "ProviderJobStatus",
    "SanitizedOutputLocator",
]

"""Provider capability snapshots for runtime plugin contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


_VOLATILE_METADATA_KEYS = frozenset(
    {"endpoint", "base_url", "queue_length", "current_queue", "balance", "health", "health_status", "rate_limit", "api_key", "authorization"}
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class DurationRange:
    min_seconds: float
    max_seconds: float

    def __post_init__(self) -> None:
        if self.min_seconds < 0 or self.max_seconds < self.min_seconds:
            raise ValueError("DurationRange must have 0 <= min_seconds <= max_seconds")

    def to_dict(self) -> dict[str, float]:
        return {"min_seconds": float(self.min_seconds), "max_seconds": float(self.max_seconds)}


@dataclass(frozen=True, slots=True)
class ProviderCapabilities(Mapping[str, object]):
    """Stable semantic capability model.

    Volatile deployment data such as endpoint, queue depth, balance and health
    is intentionally not represented and therefore cannot affect the
    capability fingerprint.
    """

    schema_version: str = "provider-capabilities/1"
    provider_key: str = ""
    provider_profile: str = ""
    supported_duration_ranges: tuple[DurationRange, ...] = ()
    supported_aspect_ratios: tuple[str, ...] = ()
    supported_resolutions: tuple[str, ...] = ()
    supported_fps: tuple[int, ...] = ()
    supports_text_to_video: bool = True
    supports_image_to_video: bool = False
    supports_first_frame: bool = False
    supports_last_frame: bool = False
    supports_reference_images: bool = False
    supports_negative_prompt: bool = False
    supports_audio: bool = False
    supports_cancel: bool = False
    supports_idempotency: bool = True
    supports_resume_download: bool = False
    supports_task_lookup_by_idempotency_key: bool = False
    max_reference_images: int = 0
    max_prompt_length: int | None = None
    max_concurrent_jobs: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    capability_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.provider_key.strip():
            raise ValueError("ProviderCapabilities.provider_key is required")
        if any(fps <= 0 for fps in self.supported_fps):
            raise ValueError("supported_fps must contain positive integers")
        if self.max_reference_images < 0:
            raise ValueError("max_reference_images cannot be negative")
        frozen = MappingProxyType(
            {
                str(key): _plain(value)
                for key, value in self.metadata.items()
                if str(key).strip().lower().replace("-", "_") not in _VOLATILE_METADATA_KEYS
            }
        )
        object.__setattr__(self, "metadata", frozen)
        computed = self._compute_fingerprint()
        if self.capability_fingerprint and self.capability_fingerprint != computed:
            raise ValueError("capability_fingerprint does not match capability semantics")
        object.__setattr__(self, "capability_fingerprint", computed)

    def _semantic_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider_key": self.provider_key,
            "provider_profile": self.provider_profile,
            "supported_duration_ranges": [item.to_dict() for item in self.supported_duration_ranges],
            "supported_aspect_ratios": list(self.supported_aspect_ratios),
            "supported_resolutions": list(self.supported_resolutions),
            "supported_fps": list(self.supported_fps),
            "supports_text_to_video": self.supports_text_to_video,
            "supports_image_to_video": self.supports_image_to_video,
            "supports_first_frame": self.supports_first_frame,
            "supports_last_frame": self.supports_last_frame,
            "supports_reference_images": self.supports_reference_images,
            "supports_negative_prompt": self.supports_negative_prompt,
            "supports_audio": self.supports_audio,
            "supports_cancel": self.supports_cancel,
            "supports_idempotency": self.supports_idempotency,
            "supports_resume_download": self.supports_resume_download,
            "supports_task_lookup_by_idempotency_key": self.supports_task_lookup_by_idempotency_key,
            "max_reference_images": self.max_reference_images,
            "max_prompt_length": self.max_prompt_length,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "metadata": _plain(self.metadata),
        }

    def _compute_fingerprint(self) -> str:
        canonical = json.dumps(self._semantic_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**self._semantic_dict(), "capability_fingerprint": self.capability_fingerprint}

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: object = None) -> object:
        return self.to_dict().get(key, default)

    def to_legacy_dict(self) -> dict[str, object]:
        ranges = self.supported_duration_ranges
        return {
            "min_duration_seconds": min((item.min_seconds for item in ranges), default=None),
            "max_duration_seconds": max((item.max_seconds for item in ranges), default=None),
            # The legacy compiler flag describes adapter acceptance rather
            # than requiring an explicit finite duration table.
            "supports_long_video": True,
            "supports_multi_scene_prompt": True,
            "supports_reference_images": self.supports_reference_images,
            "supports_character_reference": self.supports_reference_images,
            "supports_audio": self.supports_audio,
            "supports_subtitles": False,
            "supported_aspect_ratios": tuple(self.supported_aspect_ratios),
            "supported_resolutions": tuple(self.supported_resolutions),
            "supported_fps": tuple(float(item) for item in self.supported_fps),
            "output_formats": ("mp4",),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderCapabilities":
        ranges = tuple(
            item if isinstance(item, DurationRange) else DurationRange(float(item["min_seconds"]), float(item["max_seconds"]))
            for item in data.get("supported_duration_ranges", ())
        )
        values = dict(data)
        values["supported_duration_ranges"] = ranges
        values["supported_aspect_ratios"] = tuple(str(item) for item in data.get("supported_aspect_ratios", ()))
        values["supported_resolutions"] = tuple(str(item) for item in data.get("supported_resolutions", ()))
        values["supported_fps"] = tuple(int(item) for item in data.get("supported_fps", ()))
        values["metadata"] = dict(data.get("metadata", {}))
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})


def capability_snapshot_diagnostics(snapshot: Any, current: ProviderCapabilities, *, provider_profile: str = "") -> tuple[str, ...]:
    """Compare a plan snapshot with an assembled runtime capability.

    Snapshots created by Phase 4G/5A contain legacy compiler fields, so those
    fields are compared through the adapter's stable legacy projection.  New
    snapshots use the explicit capability fingerprint.
    """

    diagnostics: list[str] = []
    expected_key = str(getattr(snapshot, "provider_key", ""))
    if expected_key != current.provider_key:
        diagnostics.append("provider_capability_drift")
        diagnostics.append("provider_not_registered" if not current.provider_key else "provider_key_mismatch")
    expected_profile = str(provider_profile or getattr(snapshot, "provider_profile", ""))
    if expected_profile != current.provider_profile:
        diagnostics.append("provider_profile_mismatch")
    values = getattr(snapshot, "capabilities", {})
    expected_schema = values.get("schema_version") if isinstance(values, Mapping) else None
    if expected_schema and expected_schema != current.schema_version:
        diagnostics.append("provider_capability_drift")
    expected_fingerprint = values.get("capability_fingerprint") if isinstance(values, Mapping) else None
    if expected_fingerprint:
        if str(expected_fingerprint) != current.capability_fingerprint:
            diagnostics.append("provider_capability_drift")
    elif isinstance(values, Mapping):
        legacy = current.to_legacy_dict()
        comparable = {key: value for key, value in values.items() if key in legacy}
        current_values = {key: legacy.get(key) for key in comparable}
        if _plain(comparable) != _plain(current_values):
            diagnostics.append("provider_capability_drift")
    return tuple(dict.fromkeys(diagnostics))


__all__ = ["DurationRange", "ProviderCapabilities", "capability_snapshot_diagnostics"]

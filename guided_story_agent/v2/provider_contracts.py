"""Provider-neutral static assignment and capability contracts.

These records describe what a future runtime may use.  They intentionally do
not contain credentials, endpoints, request payloads, task IDs, or responses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping


_FORBIDDEN_KEYS = {
    "api_key",
    "authorization",
    "bearer",
    "endpoint",
    "http_body",
    "http_payload",
    "payload",
    "provider_task_id",
    "remote_response",
    "secret",
    "task_id",
}


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
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    return value


def _ensure_provider_neutral(value: Any, path: str = "provider_contract") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower().replace("-", "_").replace(" ", "_")
            if lowered in _FORBIDDEN_KEYS:
                raise ValueError(f"Provider contract contains forbidden field: {path}.{key}")
            _ensure_provider_neutral(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _ensure_provider_neutral(child, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class CapabilitySnapshot:
    snapshot_id: str
    provider_key: str
    provider_profile: str = ""
    capabilities: Mapping[str, object] = MappingProxyType({})
    schema_version: str = "capability-snapshot/1"

    def __post_init__(self) -> None:
        if not str(self.snapshot_id).strip():
            raise ValueError("CapabilitySnapshot.snapshot_id is required")
        if not str(self.provider_key).strip():
            raise ValueError("CapabilitySnapshot.provider_key is required")
        frozen = _freeze(self.capabilities)
        if not isinstance(frozen, Mapping):
            raise TypeError("CapabilitySnapshot.capabilities must be a mapping")
        object.__setattr__(self, "capabilities", frozen)
        _ensure_provider_neutral(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "provider_key": self.provider_key,
            "provider_profile": self.provider_profile,
            "capabilities": _plain(self.capabilities),
            "schema_version": self.schema_version,
        }

    @property
    def capability_fingerprint(self) -> str:
        value = self.capabilities.get("capability_fingerprint") if isinstance(self.capabilities, Mapping) else None
        return str(value or self.snapshot_id)

    @property
    def capability_schema_version(self) -> str:
        value = self.capabilities.get("schema_version") if isinstance(self.capabilities, Mapping) else None
        return str(value or self.schema_version)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilitySnapshot":
        if not isinstance(data, Mapping):
            raise ValueError("capability_snapshot must be an object")
        return cls(
            snapshot_id=str(data.get("snapshot_id", "")),
            provider_key=str(data.get("provider_key", "")),
            provider_profile=str(data.get("provider_profile", "")),
            capabilities=dict(data.get("capabilities", {})),
            schema_version=str(data.get("schema_version", "capability-snapshot/1")),
        )

    @classmethod
    def from_capabilities(cls, capabilities: Any) -> "CapabilitySnapshot":
        if hasattr(capabilities, "to_dict") and callable(capabilities.to_dict):
            raw = dict(capabilities.to_dict())
        else:
            raw = asdict(capabilities)
        values = {
            key: value
            for key, value in raw.items()
            if key not in {"provider_key", "provider_profile", "provider_name"}
        }
        canonical = json.dumps(_plain(values), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        fingerprint = str(getattr(capabilities, "capability_fingerprint", ""))
        snapshot_id = "capability-" + (fingerprint[:20] if fingerprint else hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20])
        return cls(
            snapshot_id=snapshot_id,
            provider_key=str(capabilities.provider_key),
            provider_profile=str(capabilities.provider_profile),
            capabilities=values,
        )


@dataclass(frozen=True, slots=True)
class ProviderAssignment:
    assignment_id: str
    provider_key: str
    provider_profile: str
    capability_snapshot_id: str
    fallback_provider_keys: tuple[str, ...] = ()
    selection_reason: str = ""

    def __post_init__(self) -> None:
        for field_name in ("assignment_id", "provider_key", "capability_snapshot_id"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"ProviderAssignment.{field_name} is required")
        object.__setattr__(self, "fallback_provider_keys", tuple(str(item) for item in self.fallback_provider_keys))
        _ensure_provider_neutral(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "provider_key": self.provider_key,
            "provider_profile": self.provider_profile,
            "capability_snapshot_id": self.capability_snapshot_id,
            "fallback_provider_keys": list(self.fallback_provider_keys),
            "selection_reason": self.selection_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderAssignment":
        return cls(
            assignment_id=str(data.get("assignment_id", "")),
            provider_key=str(data.get("provider_key", "")),
            provider_profile=str(data.get("provider_profile", "")),
            capability_snapshot_id=str(data.get("capability_snapshot_id", "")),
            fallback_provider_keys=tuple(str(item) for item in data.get("fallback_provider_keys", [])),
            selection_reason=str(data.get("selection_reason", "")),
        )


__all__ = ["CapabilitySnapshot", "ProviderAssignment"]

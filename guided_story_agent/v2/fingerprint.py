"""Immutable content fingerprints for MoviePlan lineage.

The fingerprint is deliberately computed from creative plan content only.  It
is not a session checksum: runtime state, diagnostics, provider data, cache
entries and timestamps are excluded before canonical JSON serialization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported only for type checkers
    from .models import MoviePlan


_EXCLUDED_KEYS = {
    "plan_id",
    "ir_id",
    "job_id",
    "artifact_id",
    "source_movie_plan_id",
    "source_story_plan_id",
    "source_director_plan_id",
    "source_film_ir_id",
    "source_movie_ir_id",
    "revision",
    "confirmed",
    "version",
    "movie_plan_version",
    "fingerprint",
    "movie_plan_fingerprint",
    "lineage_token",
    "movie_plan_lineage_token",
    "metadata",
    "diagnostics",
    "cache",
    "cached_at",
    "created_at",
    "updated_at",
    "runtime",
    "provider",
    "provider_key",
    "provider_profile",
    "provider_payload",
    "request_payload",
    "video_payload",
    "api",
    "api_key",
    "payload",
    "api_payload",
    "http_payload",
    "endpoint",
    "task",
    "task_id",
    "artifact",
    "artifacts",
}


def _canonical_value(value: Any, *, key: str = "") -> Any:
    if key.lower() in _EXCLUDED_KEYS:
        return None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        items = sorted(((str(item), child) for item, child in value.items()), key=lambda pair: pair[0])
        for child_key, child in items:
            if child_key.lower() in _EXCLUDED_KEYS:
                continue
            result[child_key] = _canonical_value(child, key=child_key)
        return result
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonical_value(item) for item in value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _canonical_value(value.value)
    return value


def canonicalize_movie_plan(movie_plan: "MoviePlan | Mapping[str, Any]") -> dict[str, Any]:
    """Return deterministic creative content suitable for hashing."""

    if isinstance(movie_plan, Mapping):
        raw = dict(movie_plan)
    else:
        from .models import as_plain_data

        raw = as_plain_data(movie_plan)
    if not isinstance(raw, dict):
        raise TypeError("MoviePlan must serialize to a JSON object")
    value = _canonical_value(raw)
    assert isinstance(value, dict)
    return value


class CanonicalSerializer:
    """Stable UTF-8 JSON serializer used by the content fingerprint."""

    def serialize(self, value: Any) -> str:
        return json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def serialize_movie_plan(self, movie_plan: "MoviePlan | Mapping[str, Any]") -> str:
        return self.serialize(canonicalize_movie_plan(movie_plan))


@dataclass(frozen=True, slots=True)
class ContentFingerprint:
    value: str
    algorithm: str = "sha256"
    canonical_json: str = ""

    def __str__(self) -> str:
        return self.value

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, item: Any) -> Any:
        return self.value[item]

    def startswith(self, prefix: str) -> bool:
        return self.value.startswith(prefix)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ContentFingerprint):
            return (self.algorithm, self.value) == (other.algorithm, other.value)
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    @property
    def fingerprint(self) -> str:
        return self.value


class FingerprintBuilder:
    """Build SHA-256 fingerprints without including mutable session state."""

    def __init__(self, serializer: CanonicalSerializer | None = None) -> None:
        self.serializer = serializer or CanonicalSerializer()

    def build(self, movie_plan: "MoviePlan | Mapping[str, Any]") -> ContentFingerprint:
        canonical_json = self.serializer.serialize_movie_plan(movie_plan)
        digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return ContentFingerprint(digest, "sha256", canonical_json)


def movie_plan_fingerprint(movie_plan: "MoviePlan | Mapping[str, Any]") -> str:
    return FingerprintBuilder().build(movie_plan).value


def content_fingerprint(value: Any) -> str:
    """Hash a non-Provider IR payload while ignoring identity/runtime fields."""

    serializer = CanonicalSerializer()
    canonical_json = serializer.serialize(value)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def movie_plan_lineage_token(plan_id: str, version: int, fingerprint: str) -> str:
    payload = f"{str(plan_id).strip()}:{int(version)}:{str(fingerprint).strip()}"
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"mp-lineage-v{int(version)}-{suffix}"


def ensure_movie_plan_provenance(
    movie_plan: "MoviePlan",
    *,
    version: int | None = None,
) -> "MoviePlan":
    """Return a copy with deterministic version/fingerprint/token fields."""

    from dataclasses import replace

    fingerprint = movie_plan_fingerprint(movie_plan)
    current_version = int(getattr(movie_plan, "movie_plan_version", 0) or 0)
    resolved_version = int(version if version is not None else (current_version or 1))
    token = movie_plan_lineage_token(movie_plan.plan_id, resolved_version, fingerprint)
    return replace(
        movie_plan,
        movie_plan_version=resolved_version,
        movie_plan_fingerprint=fingerprint,
        movie_plan_lineage_token=token,
    )


__all__ = [
    "CanonicalSerializer",
    "ContentFingerprint",
    "FingerprintBuilder",
    "canonicalize_movie_plan",
    "content_fingerprint",
    "ensure_movie_plan_provenance",
    "movie_plan_fingerprint",
    "movie_plan_lineage_token",
]

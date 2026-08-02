"""Verified, non-media Fake Artifact records for Phase 5A."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .execution_state import ProviderJob
from .provider_runtime import ProviderDownloadResult


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_id: str
    artifact_type: str
    media_type: str
    storage_path: str
    size_bytes: int
    sha256: str
    verification_status: str
    execution_plan_id: str
    execution_plan_fingerprint: str
    execution_unit_id: str
    video_job_id: str
    video_job_fingerprint: str
    provider_job_id: str
    provider_key: str
    movie_plan_version: int
    movie_plan_fingerprint: str
    movie_plan_lineage_token: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "media_type": self.media_type,
            "storage_path": self.storage_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "verification_status": self.verification_status,
            "execution_plan_id": self.execution_plan_id,
            "execution_plan_fingerprint": self.execution_plan_fingerprint,
            "execution_unit_id": self.execution_unit_id,
            "video_job_id": self.video_job_id,
            "video_job_fingerprint": self.video_job_fingerprint,
            "provider_job_id": self.provider_job_id,
            "provider_key": self.provider_key,
            "movie_plan_version": self.movie_plan_version,
            "movie_plan_fingerprint": self.movie_plan_fingerprint,
            "movie_plan_lineage_token": self.movie_plan_lineage_token,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRecord":
        return cls(
            artifact_id=str(data.get("artifact_id", "")),
            artifact_type=str(data.get("artifact_type", "")),
            media_type=str(data.get("media_type", "")),
            storage_path=str(data.get("storage_path", "")),
            size_bytes=int(data.get("size_bytes", 0)),
            sha256=str(data.get("sha256", "")),
            verification_status=str(data.get("verification_status", "")),
            execution_plan_id=str(data.get("execution_plan_id", "")),
            execution_plan_fingerprint=str(data.get("execution_plan_fingerprint", "")),
            execution_unit_id=str(data.get("execution_unit_id", "")),
            video_job_id=str(data.get("video_job_id", "")),
            video_job_fingerprint=str(data.get("video_job_fingerprint", "")),
            provider_job_id=str(data.get("provider_job_id", "")),
            provider_key=str(data.get("provider_key", "")),
            movie_plan_version=int(data.get("movie_plan_version", 0)),
            movie_plan_fingerprint=str(data.get("movie_plan_fingerprint", "")),
            movie_plan_lineage_token=str(data.get("movie_plan_lineage_token", "")),
        )


@dataclass(frozen=True, slots=True)
class ArtifactVerificationResult:
    valid: bool
    artifact: ArtifactRecord | None = None
    errors: tuple[str, ...] = ()


class FakeArtifactVerifier:
    ALLOWED_ARTIFACT_TYPES = {"video", "fake_video", "preview", "provider_response"}

    def verify(
        self,
        artifact: ArtifactRecord | ProviderDownloadResult,
        *,
        expected_artifact_type: str | None = None,
        provider_job: ProviderJob | None = None,
        expected_provenance: Mapping[str, Any] | None = None,
    ) -> ArtifactVerificationResult:
        if isinstance(artifact, ProviderDownloadResult):
            errors = []
            path = Path(artifact.storage_path)
            if not path.exists() or not path.is_file():
                errors.append("artifact file does not exist")
            if path.exists() and path.stat().st_size <= 0:
                errors.append("artifact file is empty")
            if expected_artifact_type and artifact.artifact_type != expected_artifact_type:
                errors.append("artifact type does not match expected contract")
            if errors:
                return ArtifactVerificationResult(False, None, tuple(errors))
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if artifact.sha256 and actual_hash != artifact.sha256:
                errors.append("artifact SHA256 does not match download result")
            if artifact.size_bytes and path.stat().st_size != artifact.size_bytes:
                errors.append("artifact size does not match download result")
            return ArtifactVerificationResult(not errors, None, tuple(errors))

        errors: list[str] = []
        path = Path(artifact.storage_path)
        if artifact.artifact_type not in self.ALLOWED_ARTIFACT_TYPES:
            errors.append("artifact type is not declared by the fake contract")
        if not path.exists() or not path.is_file():
            errors.append("artifact file does not exist")
        else:
            if path.stat().st_size <= 0:
                errors.append("artifact file is empty")
            if path.stat().st_size != artifact.size_bytes:
                errors.append("artifact size mismatch")
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
                errors.append("artifact SHA256 mismatch")
        if artifact.verification_status != "verified":
            errors.append("artifact is not marked verified")
        if provider_job is not None:
            for field in ("provider_job_id", "provider_key", "video_job_id", "video_job_fingerprint"):
                expected = getattr(provider_job, field, None)
                if expected is not None and getattr(artifact, field, None) != expected:
                    errors.append(f"artifact provenance mismatch: {field}")
        for field, expected in (expected_provenance or {}).items():
            if expected is not None and getattr(artifact, field, None) != expected:
                errors.append(f"artifact provenance mismatch: {field}")
        return ArtifactVerificationResult(not errors, artifact if not errors else None, tuple(errors))


__all__ = ["ArtifactRecord", "ArtifactVerificationResult", "FakeArtifactVerifier"]

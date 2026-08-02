"""Deterministic offline ProviderRuntime adapter.

The Fake adapter exercises runtime semantics without knowing about
ExecutionUnitState, EventStore, Session, or compiler objects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .execution_state import ProviderJob
from .provider_capabilities import ProviderCapabilities
from .provider_errors import ProviderErrorCategory, ProviderRuntimeError
from .provider_results import (
    DownloadDestination,
    ProviderCancelResult,
    ProviderDownloadResult,
    ProviderJobStatus,
    ProviderPollResult,
    ProviderSubmitResult,
    ProviderVerificationResult,
    SanitizedOutputLocator,
)
from .provider_runtime import Clock, ProviderRequestContext, SystemClock


@dataclass
class FakeProviderScenario:
    name: str = "success"
    mode: str | None = None
    submit_failures: int = 0
    poll_failures: int = 0
    queued_polls: int = 0
    running_polls: int = 0
    never_complete: bool = False
    download_interrupted: bool = False
    corrupted_artifact: bool = False
    cancel_supported: bool = True
    submission_uncertain: bool = False

    def __post_init__(self) -> None:
        self.name = (self.mode or self.name or "success").strip().lower()
        if self.name == "retryable_submit_failure" and self.submit_failures == 0:
            self.submit_failures = 1
        if self.name == "retryable_poll_failure" and self.poll_failures == 0:
            self.poll_failures = 1
        if self.name == "queued_then_success" and self.queued_polls == 0:
            self.queued_polls = 1
        if self.name == "running_then_success" and self.running_polls == 0:
            self.running_polls = 1
        if self.name == "never_complete":
            self.never_complete = True
        if self.name == "download_interrupted":
            self.download_interrupted = True
        if self.name == "corrupted_artifact":
            self.corrupted_artifact = True
        if self.name == "submission_uncertain":
            self.submission_uncertain = True
        if self.name == "cancel_unsupported":
            self.cancel_supported = False


def _context_value(context: ProviderRequestContext | Mapping[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(context, Mapping):
        return context.get(key, default)
    return getattr(context, key, default)


class FakeProviderRuntime:
    provider_key = "fake"

    def __init__(
        self,
        scenario: FakeProviderScenario | str | None = None,
        *,
        provider_key: str = "fake",
        provider_profile: str = "",
        clock: Clock | None = None,
    ) -> None:
        self.provider_key = provider_key
        self.provider_profile = provider_profile
        self.scenario = scenario if isinstance(scenario, FakeProviderScenario) else FakeProviderScenario(str(scenario or "success"))
        self.clock = clock or SystemClock()
        self._jobs_by_idempotency: dict[str, ProviderJob] = {}
        self._jobs_by_provider_id: dict[str, ProviderJob] = {}
        self._submit_counts: dict[str, int] = {}
        self._poll_counts: dict[str, int] = {}
        self._download_counts: dict[str, int] = {}
        self._cancel_counts: dict[str, int] = {}

    @property
    def submit_count(self) -> int:
        return sum(self._submit_counts.values())

    @property
    def poll_count(self) -> int:
        return sum(self._poll_counts.values())

    @property
    def submit_counts(self) -> Mapping[str, int]:
        return dict(self._submit_counts)

    @property
    def poll_counts(self) -> Mapping[str, int]:
        return dict(self._poll_counts)

    @property
    def real_network_calls(self) -> int:
        return 0

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_key=self.provider_key,
            provider_profile=self.provider_profile,
            supports_text_to_video=True,
            supports_image_to_video=True,
            supports_first_frame=True,
            supports_reference_images=True,
            supports_negative_prompt=True,
            supports_audio=True,
            supports_cancel=bool(self.scenario.cancel_supported),
            supports_idempotency=True,
            supports_resume_download=True,
            supports_task_lookup_by_idempotency_key=True,
            max_reference_images=8,
            metadata={"offline": True, "adapter": "fake"},
        )

    def submit(self, video_job: Any, context: ProviderRequestContext | Mapping[str, Any]) -> ProviderSubmitResult:
        key = str(_context_value(context, "idempotency_key", ""))
        if not key:
            raise ProviderRuntimeError(
                "fake submit requires idempotency_key",
                category=ProviderErrorCategory.INVALID_REQUEST,
                provider_key=self.provider_key,
                provider_code="missing_idempotency_key",
                retryable=False,
            )
        existing = self._jobs_by_idempotency.get(key)
        if existing is not None:
            return ProviderSubmitResult(existing, True, existing.normalized_status, sanitized_response={"idempotent_replay": True})
        count = self._submit_counts.get(key, 0) + 1
        self._submit_counts[key] = count
        if self.scenario.name == "non_retryable_failure":
            raise ProviderRuntimeError(
                "fake provider rejected request",
                category=ProviderErrorCategory.INVALID_REQUEST,
                provider_key=self.provider_key,
                provider_code="fake_invalid_request",
            )
        if count <= self.scenario.submit_failures:
            raise ProviderRuntimeError(
                "fake transient submit failure",
                category=ProviderErrorCategory.TRANSIENT_NETWORK,
                provider_key=self.provider_key,
                provider_code="fake_submit_transient",
                retryable=True,
            )
        provider_job_id = f"provider-job-{uuid4().hex[:20]}"
        remote_job_id = f"fake-task-{uuid4().hex[:20]}"
        job = ProviderJob(
            provider_job_id=provider_job_id,
            provider_key=self.provider_key,
            provider_profile=self.provider_profile,
            remote_job_id=remote_job_id,
            request_id=str(_context_value(context, "request_id", f"fake-request-{uuid4().hex[:16]}")),
            idempotency_key=key,
            status=ProviderJobStatus.QUEUED,
            source_execution_run_id=str(_context_value(context, "execution_run_id")),
            source_execution_plan_id=str(_context_value(context, "execution_plan_id")),
            source_execution_plan_fingerprint=str(_context_value(context, "execution_plan_fingerprint")),
            source_execution_unit_id=str(_context_value(context, "execution_unit_id")),
            source_video_job_id=str(getattr(video_job, "job_id", _context_value(context, "video_job_id"))),
            source_video_job_fingerprint=str(getattr(video_job, "video_job_fingerprint", _context_value(context, "video_job_fingerprint"))),
            source_movie_plan_version=int(_context_value(context, "source_movie_plan_version", 0) or 0),
            source_movie_plan_fingerprint=str(_context_value(context, "source_movie_plan_fingerprint")),
            source_movie_plan_lineage_token=str(_context_value(context, "source_movie_plan_lineage_token")),
            submitted_at=self.clock.now().isoformat(),
            sanitized_provider_metadata={"poll_count": 0, "scenario": self.scenario.name},
        )
        self._jobs_by_idempotency[key] = job
        self._jobs_by_provider_id[job.provider_job_id] = job
        if self.scenario.submission_uncertain:
            raise ProviderRuntimeError(
                "fake provider accepted submission but response was lost",
                category=ProviderErrorCategory.SUBMISSION_UNCERTAIN,
                provider_key=self.provider_key,
                provider_code="submission_response_lost",
                submission_may_have_been_accepted=True,
                sanitized_details={"provider_job_id_known": False},
            )
        return ProviderSubmitResult(job, True, ProviderJobStatus.QUEUED, sanitized_response={"status": "queued"})

    def _remember(self, job: ProviderJob, **changes: Any) -> ProviderJob:
        updated = ProviderJob(**{**job.to_dict(), **changes})
        self._jobs_by_provider_id[updated.provider_job_id] = updated
        self._jobs_by_idempotency[updated.idempotency_key] = updated
        return updated

    def poll(self, provider_job: ProviderJob, context: ProviderRequestContext | Mapping[str, Any] | None = None) -> ProviderPollResult:
        del context
        job = self._jobs_by_provider_id.get(provider_job.provider_job_id, provider_job)
        count = self._poll_counts.get(job.provider_job_id, 0) + 1
        self._poll_counts[job.provider_job_id] = count
        if count <= self.scenario.poll_failures:
            raise ProviderRuntimeError(
                "fake transient poll failure",
                category=ProviderErrorCategory.PROVIDER_UNAVAILABLE,
                provider_key=self.provider_key,
                provider_code="poll_transient",
                retryable=True,
            )
        if self.scenario.name == "non_retryable_failure":
            raise ProviderRuntimeError(
                "fake provider failed while polling",
                category=ProviderErrorCategory.POLICY_REJECTED,
                provider_key=self.provider_key,
                provider_code="poll_rejected",
            )
        if self.scenario.never_complete:
            status = ProviderJobStatus.RUNNING
        elif count <= self.scenario.queued_polls:
            status = ProviderJobStatus.QUEUED
        elif count <= self.scenario.queued_polls + self.scenario.running_polls:
            status = ProviderJobStatus.RUNNING
        else:
            status = ProviderJobStatus.SUCCEEDED
        metadata = dict(job.sanitized_provider_metadata or {})
        metadata["poll_count"] = count
        if status is ProviderJobStatus.SUCCEEDED:
            metadata["output_locator"] = SanitizedOutputLocator(f"fake://artifact/{job.remote_job_id}").to_dict()
        updated = self._remember(job, status=status, last_polled_at=self.clock.now().isoformat(), sanitized_provider_metadata=metadata)
        return ProviderPollResult(updated, status, sanitized_response={"status": status.value.lower()})

    def cancel(self, provider_job: ProviderJob, context: ProviderRequestContext | Mapping[str, Any] | None = None) -> ProviderCancelResult:
        del context
        self._cancel_counts[provider_job.provider_job_id] = self._cancel_counts.get(provider_job.provider_job_id, 0) + 1
        if not self.scenario.cancel_supported:
            return ProviderCancelResult(False, False, sanitized_response={"supported": False})
        self._remember(provider_job, status=ProviderJobStatus.CANCELLED)
        return ProviderCancelResult(True, True, ProviderJobStatus.CANCELLED, {"status": "cancelled"})

    def download(
        self,
        provider_job: ProviderJob,
        destination: DownloadDestination | str,
        context: ProviderRequestContext | Mapping[str, Any] | None = None,
    ) -> ProviderDownloadResult:
        del context
        job = self._jobs_by_provider_id.get(provider_job.provider_job_id, provider_job)
        count = self._download_counts.get(job.provider_job_id, 0) + 1
        self._download_counts[job.provider_job_id] = count
        if isinstance(destination, DownloadDestination):
            target = Path(destination.final_path)
            partial = Path(destination.temporary_path)
        else:
            root = Path(destination)
            root.mkdir(parents=True, exist_ok=True)
            target = root / "fake-video.bin"
            partial = target.with_name(target.name + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "FAKE_PROVIDER_ARTIFACT\n"
            f"provider_job={job.provider_job_id}\n"
            f"video_job={job.source_video_job_id}\n"
            f"video_job_fingerprint={job.source_video_job_fingerprint}\n"
        ).encode("utf-8")
        if self.scenario.download_interrupted and count == 1:
            partial.write_bytes(content[: max(1, len(content) // 2)])
            raise ProviderRuntimeError(
                "fake download interrupted",
                category=ProviderErrorCategory.DOWNLOAD_INTERRUPTED,
                provider_key=self.provider_key,
                provider_code="download_interrupted",
                retryable=True,
            )
        partial.write_bytes(content)
        partial.replace(target)
        digest = hashlib.sha256(content).hexdigest()
        if self.scenario.corrupted_artifact:
            digest = "0" * 64
        return ProviderDownloadResult(
            temporary_path=str(partial),
            final_candidate_path=str(target),
            size_bytes=len(content),
            sha256=digest,
            media_type="application/octet-stream",
            completed=True,
            resumable=True,
            bytes_downloaded=len(content),
            sanitized_response_metadata={"artifact": "fake-video.bin"},
            artifact_type="video",
            provider_job=job,
        )

    def verify(
        self,
        provider_job: ProviderJob,
        download_result: ProviderDownloadResult,
        context: ProviderRequestContext | Mapping[str, Any] | None = None,
    ) -> ProviderVerificationResult:
        del context
        path = Path(download_result.storage_path)
        if not path.exists() or path.stat().st_size == 0:
            return ProviderVerificationResult(False, "missing_or_empty", "fake artifact is missing or empty", 0, "", download_result.media_type or "")
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        valid = actual == (download_result.sha256 or actual) and provider_job.provider_job_id == download_result.provider_job.provider_job_id if download_result.provider_job else actual == (download_result.sha256 or actual)
        return ProviderVerificationResult(
            valid,
            "ok" if valid else "sha256_mismatch",
            "fake artifact verified" if valid else "fake artifact hash mismatch",
            len(data),
            actual,
            download_result.media_type or "application/octet-stream",
        )


__all__ = ["FakeProviderRuntime", "FakeProviderScenario"]

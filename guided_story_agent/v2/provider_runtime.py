"""Provider Runtime plugin protocol and request context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Protocol, runtime_checkable

from .execution_state import ProviderJob
from .provider_capabilities import ProviderCapabilities
from .provider_errors import ProviderRuntimeError
from .provider_results import (
    DownloadDestination,
    ProviderCancelResult,
    ProviderDownloadResult,
    ProviderPollResult,
    ProviderSubmitResult,
    ProviderVerificationResult,
)
from .provider_sanitization import sanitize_response


class Clock(Protocol):
    def now(self) -> datetime: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            import time

            time.sleep(seconds)


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = (start or datetime(2026, 1, 1, tzinfo=timezone.utc)).astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("FakeClock cannot move backwards")
        self._now += timedelta(seconds=seconds)


@dataclass(frozen=True, slots=True)
class ProviderRequestContext:
    request_id: str
    execution_run_id: str
    execution_unit_id: str
    idempotency_key: str
    attempt: int
    execution_plan_id: str
    execution_plan_fingerprint: str
    video_job_id: str
    video_job_fingerprint: str
    source_movie_plan_version: int
    source_movie_plan_fingerprint: str
    source_movie_plan_lineage_token: str
    submit_timeout_seconds: float
    poll_timeout_seconds: float
    download_timeout_seconds: float
    trace_id: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_response(dict(self.metadata)))

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "execution_run_id": self.execution_run_id,
            "execution_unit_id": self.execution_unit_id,
            "idempotency_key": self.idempotency_key,
            "attempt": self.attempt,
            "execution_plan_id": self.execution_plan_id,
            "execution_plan_fingerprint": self.execution_plan_fingerprint,
            "video_job_id": self.video_job_id,
            "video_job_fingerprint": self.video_job_fingerprint,
            "source_movie_plan_version": self.source_movie_plan_version,
            "source_movie_plan_fingerprint": self.source_movie_plan_fingerprint,
            "source_movie_plan_lineage_token": self.source_movie_plan_lineage_token,
            "submit_timeout_seconds": self.submit_timeout_seconds,
            "poll_timeout_seconds": self.poll_timeout_seconds,
            "download_timeout_seconds": self.download_timeout_seconds,
            "trace_id": self.trace_id,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class ProviderRuntime(Protocol):
    @property
    def provider_key(self) -> str: ...

    def capabilities(self) -> ProviderCapabilities: ...

    def submit(self, video_job: Any, context: ProviderRequestContext) -> ProviderSubmitResult: ...

    def poll(self, provider_job: ProviderJob, context: ProviderRequestContext) -> ProviderPollResult: ...

    def cancel(self, provider_job: ProviderJob, context: ProviderRequestContext) -> ProviderCancelResult: ...

    def download(
        self,
        provider_job: ProviderJob,
        destination: DownloadDestination,
        context: ProviderRequestContext,
    ) -> ProviderDownloadResult: ...

    def verify(
        self,
        provider_job: ProviderJob,
        download_result: ProviderDownloadResult,
        context: ProviderRequestContext,
    ) -> ProviderVerificationResult: ...


__all__ = [
    "Clock",
    "DownloadDestination",
    "FakeClock",
    "ProviderCapabilities",
    "ProviderCancelResult",
    "ProviderDownloadResult",
    "ProviderJob",
    "ProviderPollResult",
    "ProviderRequestContext",
    "ProviderRuntime",
    "ProviderRuntimeError",
    "ProviderSubmitResult",
    "ProviderVerificationResult",
    "SystemClock",
]

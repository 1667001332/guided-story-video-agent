"""Structured, fail-closed Provider failure routing.

The protocol is deliberately independent from a concrete Provider.  It turns
the already-sanitized :class:`ProviderRuntimeError` into durable facts and a
small set of actions.  It never mutates MoviePlan, FilmIR, MovieIR, VideoJob,
or ExecutionBundle objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from .provider_errors import ProviderErrorCategory, ProviderRuntimeError, normalize_error_category
from .provider_sanitization import sanitize_response, sanitize_text


class FailureAction(str, Enum):
    RETRY = "RETRY"
    STOP_AND_WARN = "STOP_AND_WARN"
    REQUEST_REVISION = "REQUEST_REVISION"
    ABORT = "ABORT"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


_TRANSIENT_CATEGORIES = frozenset(
    {
        ProviderErrorCategory.TRANSIENT_NETWORK,
        ProviderErrorCategory.RATE_LIMITED,
        ProviderErrorCategory.PROVIDER_UNAVAILABLE,
        ProviderErrorCategory.POLL_TIMEOUT,
        ProviderErrorCategory.DOWNLOAD_TIMEOUT,
        ProviderErrorCategory.DOWNLOAD_INTERRUPTED,
    }
)
_UNCERTAIN_CATEGORIES = frozenset(
    {
        ProviderErrorCategory.SUBMISSION_UNCERTAIN,
        ProviderErrorCategory.SUBMIT_TIMEOUT,
    }
)
_REVISION_CATEGORIES = frozenset(
    {
        ProviderErrorCategory.POLICY_REJECTED,
        ProviderErrorCategory.UNSUPPORTED_CAPABILITY,
    }
)


@dataclass(frozen=True, slots=True)
class ProviderFailureReport:
    failure_id: str
    execution_run_id: str
    execution_unit_id: str
    provider_job_id: str | None
    category: ProviderErrorCategory
    message: str
    retryable: bool
    submission_may_have_been_accepted: bool
    retry_after_seconds: float | None
    source_movie_plan_id: str
    source_movie_plan_fingerprint: str
    source_video_job_fingerprint: str
    sanitized_details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", normalize_error_category(self.category))
        object.__setattr__(self, "message", sanitize_text(str(self.message)))
        object.__setattr__(self, "sanitized_details", _freeze(sanitize_response(self.sanitized_details)))
        if not str(self.failure_id).strip():
            raise ValueError("ProviderFailureReport.failure_id is required")
        if not str(self.execution_run_id).strip() or not str(self.execution_unit_id).strip():
            raise ValueError("ProviderFailureReport execution identifiers are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "execution_run_id": self.execution_run_id,
            "execution_unit_id": self.execution_unit_id,
            "provider_job_id": self.provider_job_id,
            "category": self.category.value,
            "message": self.message,
            "retryable": bool(self.retryable),
            "submission_may_have_been_accepted": bool(self.submission_may_have_been_accepted),
            "retry_after_seconds": self.retry_after_seconds,
            "source_movie_plan_id": self.source_movie_plan_id,
            "source_movie_plan_fingerprint": self.source_movie_plan_fingerprint,
            "source_video_job_fingerprint": self.source_video_job_fingerprint,
            "sanitized_details": _plain(self.sanitized_details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderFailureReport":
        return cls(
            failure_id=str(data.get("failure_id", "")),
            execution_run_id=str(data.get("execution_run_id", "")),
            execution_unit_id=str(data.get("execution_unit_id", "")),
            provider_job_id=data.get("provider_job_id"),
            category=normalize_error_category(str(data.get("category", ProviderErrorCategory.UNKNOWN.value))),
            message=str(data.get("message", "")),
            retryable=bool(data.get("retryable", False)),
            submission_may_have_been_accepted=bool(
                data.get("submission_may_have_been_accepted", False)
            ),
            retry_after_seconds=data.get("retry_after_seconds"),
            source_movie_plan_id=str(data.get("source_movie_plan_id", "")),
            source_movie_plan_fingerprint=str(data.get("source_movie_plan_fingerprint", "")),
            source_video_job_fingerprint=str(data.get("source_video_job_fingerprint", "")),
            sanitized_details=dict(data.get("sanitized_details", {})),
        )


@dataclass(frozen=True, slots=True)
class FailureResolution:
    action: FailureAction
    reason: str
    retry_allowed: bool = False
    create_revision_request: bool = False
    block_execution: bool = False
    invalidate_current_artifacts: bool = False
    revision_request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", FailureAction(self.action))
        object.__setattr__(self, "reason", sanitize_text(str(self.reason)))
        if self.action is FailureAction.RETRY and not self.retry_allowed:
            raise ValueError("RETRY resolution must set retry_allowed=True")
        if self.create_revision_request and not self.revision_request_id:
            raise ValueError("revision_request_id is required for a revision resolution")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "retry_allowed": bool(self.retry_allowed),
            "create_revision_request": bool(self.create_revision_request),
            "block_execution": bool(self.block_execution),
            "invalidate_current_artifacts": bool(self.invalidate_current_artifacts),
            "revision_request_id": self.revision_request_id,
        }


_FORBIDDEN_REVISION_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "bearer",
        "token",
        "secret",
        "password",
        "cookie",
        "signature",
        "endpoint",
        "payload",
        "provider_payload",
        "request_payload",
        "response_body",
        "http_payload",
    }
)


def _safe_revision_context(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_REVISION_KEYS or normalized.endswith("_api_key"):
                continue
            result[str(key)] = _safe_revision_context(child)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_revision_context(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


@dataclass(frozen=True, slots=True)
class ProviderRevisionRequest:
    """Safe Director-facing request produced by a policy/capability failure."""

    request_id: str
    source_movie_plan_id: str
    source_movie_plan_fingerprint: str
    source_video_job_fingerprint: str
    failure_category: ProviderErrorCategory
    affected_execution_unit_id: str
    sanitized_context: Mapping[str, Any] = field(default_factory=dict)
    allowed_revision_scope: tuple[str, ...] = (
        "creative_content_within_existing_brief",
        "media_requirements_within_available_capabilities",
        "affected_execution_unit_only",
    )
    forbidden_changes: tuple[str, ...] = (
        "provider_api_payload",
        "credentials_or_authorization",
        "endpoint_or_private_provider_fields",
        "automatic_apply",
        "unrelated_movie_plan_changes",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_category", normalize_error_category(self.failure_category))
        object.__setattr__(self, "sanitized_context", _freeze(_safe_revision_context(self.sanitized_context)))
        if not str(self.request_id).strip():
            raise ValueError("ProviderRevisionRequest.request_id is required")
        if self.failure_category not in _REVISION_CATEGORIES:
            raise ValueError("ProviderRevisionRequest requires a policy or capability failure")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_movie_plan_id": self.source_movie_plan_id,
            "source_movie_plan_fingerprint": self.source_movie_plan_fingerprint,
            "source_video_job_fingerprint": self.source_video_job_fingerprint,
            "failure_category": self.failure_category.value,
            "affected_execution_unit_id": self.affected_execution_unit_id,
            "sanitized_context": _plain(self.sanitized_context),
            "allowed_revision_scope": list(self.allowed_revision_scope),
            "forbidden_changes": list(self.forbidden_changes),
        }

    def to_director_request(self) -> dict[str, Any]:
        """Return only safe fields understood by the existing candidate adapter."""

        return {
            "request_id": self.request_id,
            "severity": "hard",
            "target": "provider_compatibility",
            "instruction": "请仅在允许的修订范围内解决 Provider 能力或内容政策不匹配。",
            "preserve": ("保留当前 MoviePlan 的来源与核心故事约束",),
            "avoid": self.forbidden_changes,
            "rationale": f"Provider failure category: {self.failure_category.value}",
            **self.to_dict(),
        }


# The shorter name makes the type easy to discover without colliding with the
# existing creative-analysis RevisionRequestBuilder.
RevisionRequest = ProviderRevisionRequest


class ExecutionRecompilePort(Protocol):
    """Port used after an explicit revision apply.

    Implementations receive a freshly compiled immutable ExecutionBundle.  The
    failure protocol never invokes this port automatically.
    """

    def apply_recompiled_bundle(
        self,
        execution_run_id: str,
        revised_bundle: Any,
        *,
        revision_request_id: str,
    ) -> Any: ...


class FailureProtocol:
    """Classify errors and produce deterministic, auditable resolutions."""

    def __init__(self, *, id_factory: Callable[[str], str] | None = None) -> None:
        self._id_factory = id_factory or (lambda prefix: f"{prefix}-{uuid4().hex}")

    def build_report(
        self,
        error: ProviderRuntimeError,
        *,
        execution_run_id: str,
        execution_unit_id: str,
        provider_job_id: str | None,
        source_movie_plan_id: str,
        source_movie_plan_fingerprint: str,
        source_video_job_fingerprint: str,
        failure_id: str | None = None,
    ) -> ProviderFailureReport:
        if not isinstance(error, ProviderRuntimeError):
            raise TypeError("FailureProtocol.build_report requires ProviderRuntimeError")
        return ProviderFailureReport(
            failure_id=failure_id or self._id_factory("failure"),
            execution_run_id=execution_run_id,
            execution_unit_id=execution_unit_id,
            provider_job_id=provider_job_id,
            category=error.category,
            message=error.message,
            retryable=error.retryable,
            submission_may_have_been_accepted=error.submission_may_have_been_accepted,
            retry_after_seconds=error.retry_after_seconds,
            source_movie_plan_id=source_movie_plan_id,
            source_movie_plan_fingerprint=source_movie_plan_fingerprint,
            source_video_job_fingerprint=source_video_job_fingerprint,
            sanitized_details=error.sanitized_details,
        )

    def resolve(
        self,
        report: ProviderFailureReport,
        *,
        retry_count: int = 0,
        max_attempts: int = 1,
        retry_budget_remaining: bool = True,
        retryable_error_codes: Sequence[str] = (),
        error_code: str | None = None,
    ) -> FailureResolution:
        category = normalize_error_category(report.category)
        # Acceptance uncertainty always wins over retryability.  In particular,
        # SUBMIT_TIMEOUT is never a normal POST retry.
        if category in _UNCERTAIN_CATEGORIES or report.submission_may_have_been_accepted:
            return FailureResolution(
                FailureAction.STOP_AND_WARN,
                "Provider submission acceptance is uncertain; manual reconciliation is required.",
                block_execution=True,
            )
        if category in _REVISION_CATEGORIES:
            request_id = self._id_factory("revision-request")
            return FailureResolution(
                FailureAction.REQUEST_REVISION,
                "Provider policy or capability mismatch requires an explicit Director revision.",
                create_revision_request=True,
                block_execution=True,
                invalidate_current_artifacts=True,
                revision_request_id=request_id,
            )
        if category in _TRANSIENT_CATEGORIES and report.retryable:
            allowed_codes = {str(item).lower() for item in retryable_error_codes}
            code = str(error_code or "").lower()
            code_allowed = (
                not allowed_codes
                or category.value.lower() in allowed_codes
                or code in allowed_codes
                or (
                    "timeout" in allowed_codes
                    and category
                    in {
                        ProviderErrorCategory.POLL_TIMEOUT,
                        ProviderErrorCategory.DOWNLOAD_TIMEOUT,
                    }
                )
            )
            budget = retry_count < max(0, int(max_attempts) - 1) and retry_budget_remaining
            if code_allowed and budget:
                return FailureResolution(
                    FailureAction.RETRY,
                    "Safe retry of the current Provider Job is allowed by the retry policy.",
                    retry_allowed=True,
                )
            return FailureResolution(
                FailureAction.ABORT,
                "Retry policy or retry budget is exhausted; the execution is aborted.",
                block_execution=True,
            )
        return FailureResolution(
            FailureAction.ABORT,
            "The Provider failure is not safely recoverable; retain diagnostics and abort.",
            block_execution=True,
        )

    def create_revision_request(
        self,
        report: ProviderFailureReport,
        *,
        request_id: str | None = None,
    ) -> ProviderRevisionRequest:
        return ProviderRevisionRequest(
            request_id=request_id or self._id_factory("revision-request"),
            source_movie_plan_id=report.source_movie_plan_id,
            source_movie_plan_fingerprint=report.source_movie_plan_fingerprint,
            source_video_job_fingerprint=report.source_video_job_fingerprint,
            failure_category=report.category,
            affected_execution_unit_id=report.execution_unit_id,
            sanitized_context=report.sanitized_details,
        )

    def event_payload(
        self,
        report: ProviderFailureReport,
        resolution: FailureResolution,
        *,
        retry_count: int,
        revision_request_id: str | None = None,
        actor: str = "FailureProtocol",
    ) -> dict[str, Any]:
        return {
            "failure_id": report.failure_id,
            "category": report.category.value,
            "action": resolution.action.value,
            "execution_run_id": report.execution_run_id,
            "execution_unit_id": report.execution_unit_id,
            "provider_job_id": report.provider_job_id,
            "source_movie_plan_id": report.source_movie_plan_id,
            "source_movie_plan_fingerprint": report.source_movie_plan_fingerprint,
            "source_video_job_fingerprint": report.source_video_job_fingerprint,
            "retry_count": int(retry_count),
            "revision_request_id": revision_request_id or resolution.revision_request_id,
            "reason_summary": resolution.reason,
            "actor": actor,
        }

    def classify(self, error: ProviderRuntimeError, **kwargs: Any) -> tuple[ProviderFailureReport, FailureResolution]:
        """Convenience API for adapters and offline tests."""

        report_kwargs = {
            key: kwargs.pop(key)
            for key in (
                "execution_run_id",
                "execution_unit_id",
                "provider_job_id",
                "source_movie_plan_id",
                "source_movie_plan_fingerprint",
                "source_video_job_fingerprint",
            )
        }
        report = self.build_report(error, **report_kwargs)
        return report, self.resolve(report, error_code=error.code, **kwargs)


__all__ = [
    "ExecutionRecompilePort",
    "FailureAction",
    "FailureProtocol",
    "FailureResolution",
    "ProviderFailureReport",
    "ProviderRevisionRequest",
    "RevisionRequest",
]

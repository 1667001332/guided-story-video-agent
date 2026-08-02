"""Provider-neutral error categories and centralized normalization."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from .provider_sanitization import sanitize_response


class ProviderErrorCategory(str, Enum):
    TRANSIENT_NETWORK = "TRANSIENT_NETWORK"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    SUBMIT_TIMEOUT = "SUBMIT_TIMEOUT"
    POLL_TIMEOUT = "POLL_TIMEOUT"
    DOWNLOAD_TIMEOUT = "DOWNLOAD_TIMEOUT"
    DOWNLOAD_INTERRUPTED = "DOWNLOAD_INTERRUPTED"
    INVALID_REQUEST = "INVALID_REQUEST"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    POLICY_REJECTED = "POLICY_REJECTED"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_EXPIRED = "TASK_EXPIRED"
    CANCEL_UNSUPPORTED = "CANCEL_UNSUPPORTED"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    SUBMISSION_UNCERTAIN = "SUBMISSION_UNCERTAIN"
    UNKNOWN = "UNKNOWN"


_CATEGORY_ALIASES = {
    "transient_network": ProviderErrorCategory.TRANSIENT_NETWORK,
    "provider_unavailable": ProviderErrorCategory.PROVIDER_UNAVAILABLE,
    "submit_timeout": ProviderErrorCategory.SUBMIT_TIMEOUT,
    "poll_timeout": ProviderErrorCategory.POLL_TIMEOUT,
    "download_timeout": ProviderErrorCategory.DOWNLOAD_TIMEOUT,
    "download_interrupted": ProviderErrorCategory.DOWNLOAD_INTERRUPTED,
    "invalid_request": ProviderErrorCategory.INVALID_REQUEST,
    "unsupported_capability": ProviderErrorCategory.UNSUPPORTED_CAPABILITY,
    "authentication_failed": ProviderErrorCategory.AUTHENTICATION_FAILED,
    "authorization_failed": ProviderErrorCategory.AUTHORIZATION_FAILED,
    "policy_rejected": ProviderErrorCategory.POLICY_REJECTED,
    "task_not_found": ProviderErrorCategory.TASK_NOT_FOUND,
    "task_expired": ProviderErrorCategory.TASK_EXPIRED,
    "cancel_unsupported": ProviderErrorCategory.CANCEL_UNSUPPORTED,
    "malformed_response": ProviderErrorCategory.MALFORMED_RESPONSE,
    "submission_uncertain": ProviderErrorCategory.SUBMISSION_UNCERTAIN,
    "rate_limited": ProviderErrorCategory.RATE_LIMITED,
    "unknown": ProviderErrorCategory.UNKNOWN,
}


def normalize_error_category(value: ProviderErrorCategory | str) -> ProviderErrorCategory:
    if isinstance(value, ProviderErrorCategory):
        return value
    text = str(value).strip()
    return _CATEGORY_ALIASES.get(text.lower(), ProviderErrorCategory(text.upper()) if text.upper() in ProviderErrorCategory.__members__ else ProviderErrorCategory.UNKNOWN)


def default_retryable(category: ProviderErrorCategory) -> bool:
    return category in {
        ProviderErrorCategory.TRANSIENT_NETWORK,
        ProviderErrorCategory.RATE_LIMITED,
        ProviderErrorCategory.PROVIDER_UNAVAILABLE,
        ProviderErrorCategory.SUBMIT_TIMEOUT,
        ProviderErrorCategory.POLL_TIMEOUT,
        ProviderErrorCategory.DOWNLOAD_TIMEOUT,
        ProviderErrorCategory.DOWNLOAD_INTERRUPTED,
    }


class ProviderRuntimeError(RuntimeError):
    """Sanitized, serializable provider failure.

    ``provider_accepted`` and ``metadata`` remain accepted as Phase 5A
    aliases.  New code should use ``submission_may_have_been_accepted`` and
    ``sanitized_details``.
    """

    def __init__(
        self,
        message: str,
        *,
        category: ProviderErrorCategory | str,
        provider_key: str = "",
        provider_code: str | None = None,
        code: str | None = None,
        retryable: bool | None = None,
        submission_may_have_been_accepted: bool = False,
        provider_accepted: bool | None = None,
        retry_after_seconds: float | None = None,
        sanitized_details: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.category = normalize_error_category(category)
        self.provider_key = str(provider_key or "")
        self.provider_code = str(provider_code or code or self.category.value)
        self.message = str(message)
        self.retryable = default_retryable(self.category) if retryable is None else bool(retryable)
        self.submission_may_have_been_accepted = bool(
            submission_may_have_been_accepted if provider_accepted is None else provider_accepted
        )
        self.retry_after_seconds = retry_after_seconds
        self.sanitized_details = sanitize_response(dict(sanitized_details or metadata or {}))
        super().__init__(self.message)

    @property
    def code(self) -> str:
        return self.provider_code

    @property
    def provider_accepted(self) -> bool:
        return self.submission_may_have_been_accepted

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.sanitized_details

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "provider_key": self.provider_key,
            "provider_code": self.provider_code,
            "message": self.message,
            "retryable": self.retryable,
            "submission_may_have_been_accepted": self.submission_may_have_been_accepted,
            "retry_after_seconds": self.retry_after_seconds,
            "sanitized_details": dict(self.sanitized_details),
        }


__all__ = [
    "ProviderErrorCategory",
    "ProviderRuntimeError",
    "default_retryable",
    "normalize_error_category",
]

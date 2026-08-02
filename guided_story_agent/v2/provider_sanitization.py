"""Provider response sanitization for durable runtime records.

Only sanitized provider metadata may cross the adapter boundary into runtime
events, checkpoints, Session state, or ProviderJob records.  This module is
deliberately independent from any HTTP client or concrete provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any


REDACTED = "[REDACTED]"
SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "cookie",
        "set-cookie",
        "set_cookie",
        "signature",
        "signed_url",
        "credential",
        "password",
    }
)


def _normalized_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _sanitize_url(value: str) -> str:
    try:
        split = urlsplit(value)
    except ValueError:
        return value
    if not split.query:
        return value
    query = []
    for key, item in parse_qsl(split.query, keep_blank_values=True):
        query.append((key, REDACTED if _normalized_key(key) in SENSITIVE_KEYS else item))
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), split.fragment))


def sanitize_value(value: Any, *, _key: str | None = None) -> Any:
    """Recursively redact secrets while preserving response shape."""

    if _key is not None and _normalized_key(_key) in SENSITIVE_KEYS:
        return REDACTED
    if isinstance(value, Mapping):
        return {str(key): sanitize_value(child, _key=str(key)) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_value(item) for item in value)
    if isinstance(value, str) and ("://" in value or value.startswith("?")):
        return _sanitize_url(value)
    return deepcopy(value)


def sanitize_response(value: Any) -> Any:
    return sanitize_value(value)


def sanitize_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): sanitize_value(value, _key=str(key)) for key, value in headers.items()}


def contains_sensitive_value(value: Any, secrets: tuple[str, ...]) -> bool:
    """Test helper used by leak-prevention tests."""

    if isinstance(value, Mapping):
        return any(contains_sensitive_value(key, secrets) or contains_sensitive_value(child, secrets) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_sensitive_value(item, secrets) for item in value)
    return any(secret and secret in str(value) for secret in secrets)


__all__ = [
    "REDACTED",
    "SENSITIVE_KEYS",
    "contains_sensitive_value",
    "sanitize_headers",
    "sanitize_response",
    "sanitize_value",
]

"""Injectable HTTP transport boundary used by the Mock adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str] = field(default_factory=dict)
    json_data: Mapping[str, Any] | None = None
    content: bytes = b""


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse: ...


__all__ = ["HttpResponse", "HttpTransport"]

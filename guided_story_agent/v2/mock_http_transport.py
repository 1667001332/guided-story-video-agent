"""Scriptable, never-networking HTTP transport for contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .http_transport import HttpResponse


@dataclass(frozen=True, slots=True)
class MockHttpRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    json_body: Mapping[str, Any] | None
    timeout: float | None


class MockHttpTransport:
    def __init__(
        self,
        responses: list[HttpResponse | BaseException] | tuple[HttpResponse | BaseException, ...] | None = None,
        *,
        handler: Callable[[MockHttpRequest], HttpResponse | BaseException] | None = None,
    ) -> None:
        self._responses = list(responses or ())
        self._handler = handler
        self.requests: list[MockHttpRequest] = []
        self.real_network_calls = 0

    @property
    def request_count(self) -> int:
        return len(self.requests)

    def enqueue(self, response: HttpResponse | BaseException) -> None:
        self._responses.append(response)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        request = MockHttpRequest(str(method).upper(), str(url), dict(headers or {}), dict(json_body) if json_body else None, timeout)
        self.requests.append(request)
        response: HttpResponse | BaseException
        if self._handler is not None:
            response = self._handler(request)
        elif self._responses:
            response = self._responses.pop(0)
        else:
            raise RuntimeError("MockHttpTransport has no scripted response")
        if isinstance(response, BaseException):
            raise response
        return response


__all__ = ["MockHttpRequest", "MockHttpTransport"]

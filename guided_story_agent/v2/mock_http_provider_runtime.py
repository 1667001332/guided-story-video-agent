"""Mock HTTP ProviderRuntime used exclusively for offline contract tests."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .execution_state import ProviderJob
from .http_transport import HttpResponse, HttpTransport
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
from .provider_runtime import ProviderRequestContext
from .provider_sanitization import sanitize_headers, sanitize_response


def _status(value: object) -> ProviderJobStatus:
    return ProviderJobStatus.from_value(value)


def _json(response: HttpResponse, *, operation: str, provider_key: str) -> Mapping[str, Any]:
    if not isinstance(response.json_data, Mapping):
        raise ProviderRuntimeError(
            f"mock-http {operation} response is malformed",
            category=ProviderErrorCategory.MALFORMED_RESPONSE,
            provider_key=provider_key,
            provider_code="malformed_json",
            submission_may_have_been_accepted=operation == "submit",
            sanitized_details={"status_code": response.status_code},
        )
    return response.json_data


def _status_error(response: HttpResponse, *, operation: str, provider_key: str) -> ProviderRuntimeError | None:
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        return ProviderRuntimeError(
            "mock-http rate limited",
            category=ProviderErrorCategory.RATE_LIMITED,
            provider_key=provider_key,
            provider_code="rate_limited",
            retryable=True,
            retry_after_seconds=float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None,
            sanitized_details={"status_code": response.status_code},
        )
    if response.status_code == 503:
        body = response.json_data if isinstance(response.json_data, Mapping) else {}
        accepted = body.get("accepted")
        return ProviderRuntimeError(
            "mock-http provider unavailable",
            category=ProviderErrorCategory.PROVIDER_UNAVAILABLE,
            provider_key=provider_key,
            provider_code="provider_unavailable",
            retryable=accepted is False,
            submission_may_have_been_accepted=operation == "submit" and accepted is not False,
            sanitized_details={"status_code": response.status_code, "accepted": accepted},
        )
    if response.status_code in {401}:
        return ProviderRuntimeError(
            "mock-http authentication failed",
            category=ProviderErrorCategory.AUTHENTICATION_FAILED,
            provider_key=provider_key,
            provider_code="authentication_failed",
            sanitized_details={"status_code": response.status_code},
        )
    if response.status_code in {403}:
        return ProviderRuntimeError(
            "mock-http authorization failed",
            category=ProviderErrorCategory.AUTHORIZATION_FAILED,
            provider_key=provider_key,
            provider_code="authorization_failed",
            sanitized_details={"status_code": response.status_code},
        )
    if response.status_code == 404:
        return ProviderRuntimeError(
            "mock-http task not found",
            category=ProviderErrorCategory.TASK_NOT_FOUND,
            provider_key=provider_key,
            provider_code="task_not_found",
            sanitized_details={"status_code": response.status_code},
        )
    if response.status_code in {405, 501} and operation == "cancel":
        return ProviderRuntimeError(
            "mock-http cancel unsupported",
            category=ProviderErrorCategory.CANCEL_UNSUPPORTED,
            provider_key=provider_key,
            provider_code="cancel_unsupported",
            sanitized_details={"status_code": response.status_code},
        )
    if response.status_code >= 400:
        return ProviderRuntimeError(
            f"mock-http {operation} rejected",
            category=ProviderErrorCategory.INVALID_REQUEST if operation == "submit" else ProviderErrorCategory.UNKNOWN,
            provider_key=provider_key,
            provider_code="http_error",
            submission_may_have_been_accepted=operation == "submit",
            sanitized_details={"status_code": response.status_code},
        )
    return None


class MockHttpProviderRuntime:
    provider_key = "mock-http"

    def __init__(
        self,
        transport: HttpTransport,
        *,
        base_url: str = "mock://provider",
        provider_profile: str = "mock-http",
        authorization: str = "",
    ) -> None:
        self.transport = transport
        self.base_url = base_url.rstrip("/")
        self.provider_profile = provider_profile
        self._authorization = authorization
        self._jobs_by_idempotency: dict[str, ProviderJob] = {}

    @property
    def real_network_calls(self) -> int:
        return int(getattr(self.transport, "real_network_calls", 0))

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_key=self.provider_key,
            provider_profile=self.provider_profile,
            supports_text_to_video=True,
            supports_image_to_video=True,
            supports_first_frame=True,
            supports_last_frame=True,
            supports_reference_images=True,
            supports_negative_prompt=True,
            supports_audio=True,
            supports_cancel=True,
            supports_idempotency=True,
            supports_resume_download=True,
            supports_task_lookup_by_idempotency_key=True,
            max_reference_images=16,
            metadata={"offline": True, "adapter": "mock-http"},
        )

    def _headers(self, context: ProviderRequestContext) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "X-Request-ID": context.request_id}
        if self._authorization:
            headers["Authorization"] = self._authorization
        return headers

    def submit(self, video_job: Any, context: ProviderRequestContext) -> ProviderSubmitResult:
        existing = self._jobs_by_idempotency.get(context.idempotency_key)
        if existing is not None:
            return ProviderSubmitResult(existing, True, existing.normalized_status, sanitized_response={"idempotent_replay": True})
        try:
            response = self.transport.request(
                "POST",
                f"{self.base_url}/tasks",
                headers=self._headers(context),
                json_body={
                    "idempotency_key": context.idempotency_key,
                    "video_job_id": context.video_job_id,
                    "video_job_fingerprint": context.video_job_fingerprint,
                    "prompt": getattr(video_job, "provider_prompt", getattr(video_job, "prompt", "")),
                    "duration_seconds": getattr(video_job, "duration_seconds", 0),
                    "aspect_ratio": getattr(video_job, "aspect_ratio", ""),
                },
                timeout=context.submit_timeout_seconds,
            )
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise ProviderRuntimeError(
                "mock-http submit response disconnected",
                category=ProviderErrorCategory.SUBMISSION_UNCERTAIN,
                provider_key=self.provider_key,
                provider_code="submit_response_disconnected",
                submission_may_have_been_accepted=True,
                sanitized_details={"exception_type": type(exc).__name__},
            ) from exc
        error = _status_error(response, operation="submit", provider_key=self.provider_key)
        if error:
            raise error
        body = _json(response, operation="submit", provider_key=self.provider_key)
        remote_job_id = body.get("task_id") or body.get("video_id") or body.get("job_id") or body.get("operation_id")
        if not remote_job_id:
            raise ProviderRuntimeError(
                "mock-http submit response lacks remote task id",
                category=ProviderErrorCategory.MALFORMED_RESPONSE,
                provider_key=self.provider_key,
                provider_code="missing_remote_job_id",
                submission_may_have_been_accepted=True,
                sanitized_details={"response": sanitize_response(body)},
            )
        initial = _status(body.get("status", "accepted"))
        job = ProviderJob(
            provider_job_id=f"mock-provider-job-{remote_job_id}",
            provider_key=self.provider_key,
            provider_profile=self.provider_profile,
            remote_job_id=str(remote_job_id),
            request_id=context.request_id,
            idempotency_key=context.idempotency_key,
            status=initial,
            source_execution_run_id=context.execution_run_id,
            source_execution_plan_id=context.execution_plan_id,
            source_execution_plan_fingerprint=context.execution_plan_fingerprint,
            source_execution_unit_id=context.execution_unit_id,
            source_video_job_id=context.video_job_id,
            source_video_job_fingerprint=context.video_job_fingerprint,
            source_movie_plan_version=context.source_movie_plan_version,
            source_movie_plan_fingerprint=context.source_movie_plan_fingerprint,
            source_movie_plan_lineage_token=context.source_movie_plan_lineage_token,
            sanitized_provider_metadata={"response_schema": "mock-http/submit"},
        )
        self._jobs_by_idempotency[context.idempotency_key] = job
        return ProviderSubmitResult(job, bool(body.get("accepted", True)), initial, sanitized_response=sanitize_response(body))

    def poll(self, provider_job: ProviderJob, context: ProviderRequestContext) -> ProviderPollResult:
        try:
            response = self.transport.request(
                "GET",
                f"{self.base_url}/tasks/{provider_job.remote_job_id}",
                headers=self._headers(context),
                timeout=context.poll_timeout_seconds,
            )
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise ProviderRuntimeError(
                "mock-http poll transport failed",
                category=ProviderErrorCategory.TRANSIENT_NETWORK,
                provider_key=self.provider_key,
                provider_code="poll_transport_failed",
                retryable=True,
                sanitized_details={"exception_type": type(exc).__name__},
            ) from exc
        error = _status_error(response, operation="poll", provider_key=self.provider_key)
        if error:
            raise error
        body = _json(response, operation="poll", provider_key=self.provider_key)
        status = _status(body.get("status", "unknown"))
        metadata = dict(provider_job.sanitized_provider_metadata or {})
        locator = None
        output_url = body.get("output_url") or body.get("output_locator")
        if output_url:
            locator = SanitizedOutputLocator(str(output_url), body.get("expires_at"))
            metadata["output_locator"] = locator.to_dict()
        updated = ProviderJob(**{**provider_job.to_dict(), "status": status, "last_polled_at": datetime.now(timezone.utc).isoformat(), "sanitized_provider_metadata": metadata})
        self._jobs_by_idempotency[updated.idempotency_key] = updated
        return ProviderPollResult(
            updated,
            status,
            progress=float(body["progress"]) if body.get("progress") is not None else None,
            retry_after_seconds=float(body["retry_after_seconds"]) if body.get("retry_after_seconds") is not None else None,
            output_locator=locator,
            sanitized_response=sanitize_response(body),
        )

    def cancel(self, provider_job: ProviderJob, context: ProviderRequestContext) -> ProviderCancelResult:
        try:
            response = self.transport.request(
                "DELETE",
                f"{self.base_url}/tasks/{provider_job.remote_job_id}",
                headers=self._headers(context),
                timeout=context.poll_timeout_seconds,
            )
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise ProviderRuntimeError(
                "mock-http cancel transport failed",
                category=ProviderErrorCategory.TRANSIENT_NETWORK,
                provider_key=self.provider_key,
                provider_code="cancel_transport_failed",
                retryable=True,
                sanitized_details={"exception_type": type(exc).__name__},
            ) from exc
        if response.status_code in {405, 501}:
            return ProviderCancelResult(False, False, sanitized_response={"status_code": response.status_code})
        error = _status_error(response, operation="cancel", provider_key=self.provider_key)
        if error:
            if error.category is ProviderErrorCategory.CANCEL_UNSUPPORTED:
                return ProviderCancelResult(False, False, sanitized_response=error.sanitized_details)
            raise error
        body = _json(response, operation="cancel", provider_key=self.provider_key)
        if body.get("supported") is False:
            return ProviderCancelResult(False, False, sanitized_response=sanitize_response(body))
        accepted = bool(body.get("accepted", response.status_code in {200, 202, 204}))
        final = _status(body["status"]) if body.get("status") else (ProviderJobStatus.CANCELLED if accepted else None)
        return ProviderCancelResult(True, accepted, final, sanitize_response(body))

    @staticmethod
    def _validate_destination(destination: DownloadDestination) -> None:
        temp = Path(destination.temporary_path)
        final = Path(destination.final_path)
        if any(part in {"..", "."} for part in (*temp.parts, *final.parts)):
            raise ProviderRuntimeError(
                "download destination path traversal rejected",
                category=ProviderErrorCategory.INVALID_REQUEST,
                provider_key="mock-http",
                provider_code="unsafe_download_path",
            )
        if temp.parent != final.parent:
            raise ProviderRuntimeError(
                "temporary and final download paths must share a directory",
                category=ProviderErrorCategory.INVALID_REQUEST,
                provider_key="mock-http",
                provider_code="unsafe_download_path",
            )

    def download(self, provider_job: ProviderJob, destination: DownloadDestination, context: ProviderRequestContext) -> ProviderDownloadResult:
        self._validate_destination(destination)
        metadata = dict(provider_job.sanitized_provider_metadata or {})
        raw_locator = metadata.get("output_locator")
        locator = raw_locator.get("locator") if isinstance(raw_locator, Mapping) else raw_locator
        if not locator:
            raise ProviderRuntimeError(
                "mock-http output locator is missing",
                category=ProviderErrorCategory.MALFORMED_RESPONSE,
                provider_key=self.provider_key,
                provider_code="missing_output_locator",
            )
        try:
            response = self.transport.request(
                "GET",
                str(locator),
                headers={"Accept": "application/octet-stream", **({"Authorization": self._authorization} if self._authorization else {})},
                timeout=context.download_timeout_seconds,
            )
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise ProviderRuntimeError(
                "mock-http download interrupted",
                category=ProviderErrorCategory.DOWNLOAD_INTERRUPTED,
                provider_key=self.provider_key,
                provider_code="download_transport_failed",
                retryable=True,
                sanitized_details={"exception_type": type(exc).__name__},
            ) from exc
        if response.status_code >= 400:
            raise ProviderRuntimeError(
                "mock-http download rejected",
                category=ProviderErrorCategory.DOWNLOAD_INTERRUPTED,
                provider_key=self.provider_key,
                provider_code="download_http_error",
                retryable=True,
                sanitized_details={"status_code": response.status_code},
            )
        content = bytes(response.content or b"")
        Path(destination.temporary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(destination.temporary_path).write_bytes(content)
        Path(destination.temporary_path).replace(destination.final_path)
        digest = hashlib.sha256(content).hexdigest()
        return ProviderDownloadResult(
            temporary_path=destination.temporary_path,
            final_candidate_path=destination.final_path,
            size_bytes=len(content),
            sha256=digest,
            media_type=response.headers.get("Content-Type", "application/octet-stream"),
            completed=True,
            resumable=True,
            bytes_downloaded=len(content),
            sanitized_response_metadata={"headers": sanitize_headers(response.headers)},
            artifact_type="video",
            provider_job=provider_job,
        )

    def verify(self, provider_job: ProviderJob, download_result: ProviderDownloadResult, context: ProviderRequestContext) -> ProviderVerificationResult:
        del context
        path = Path(download_result.storage_path)
        if not path.exists() or not path.is_file():
            return ProviderVerificationResult(False, "missing_file", "download file is missing", 0, "", download_result.media_type or "")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        valid = bool(content) and digest == (download_result.sha256 or digest)
        return ProviderVerificationResult(
            valid,
            "ok" if valid else "sha256_mismatch",
            "mock download verified" if valid else "mock download hash mismatch",
            len(content),
            digest,
            download_result.media_type or "application/octet-stream",
            {"provider_job_id": provider_job.provider_job_id},
        )


__all__ = ["MockHttpProviderRuntime"]

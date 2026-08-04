from __future__ import annotations


def test_provider_job_serialization_contains_sanitized_common_fields(provider_and_transport, request_context, video_job) -> None:
    provider, _ = provider_and_transport
    job = provider.submit(video_job, request_context).provider_job
    data = job.to_dict()
    assert data["remote_job_id"]
    assert data["source_execution_plan_fingerprint"] == request_context.execution_plan_fingerprint
    assert "provider_metadata" not in data
    assert "sanitized_provider_metadata" in data

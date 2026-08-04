from __future__ import annotations

from guided_story_agent.v2 import ProviderJobStatus


def test_submit_returns_provenance_and_stable_remote_handle(provider_and_transport, request_context, video_job) -> None:
    provider, _ = provider_and_transport
    result = provider.submit(video_job, request_context)
    assert result.accepted
    assert result.provider_job.remote_job_id
    assert result.provider_job.idempotency_key == request_context.idempotency_key
    assert result.provider_job.source_video_job_fingerprint == request_context.video_job_fingerprint
    assert result.initial_status in set(ProviderJobStatus)

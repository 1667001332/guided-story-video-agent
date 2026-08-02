from __future__ import annotations

from guided_story_agent.v2 import ProviderJobStatus


def test_poll_returns_only_standard_status(provider_and_transport, request_context, video_job) -> None:
    provider, _ = provider_and_transport
    job = provider.submit(video_job, request_context).provider_job
    result = provider.poll(job, request_context)
    assert result.status in set(ProviderJobStatus)
    assert result.provider_job.remote_job_id == job.remote_job_id

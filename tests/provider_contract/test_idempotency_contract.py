from __future__ import annotations


def test_same_key_is_idempotent(provider_and_transport, request_context, video_job) -> None:
    provider, transport = provider_and_transport
    first = provider.submit(video_job, request_context).provider_job
    second = provider.submit(video_job, request_context).provider_job
    assert first.remote_job_id == second.remote_job_id
    if transport is not None:
        assert transport.request_count == 1

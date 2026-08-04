from __future__ import annotations

from guided_story_agent.v2 import DownloadDestination


def test_verify_returns_structured_result(provider_and_transport, request_context, video_job, artifact_dir) -> None:
    provider, _ = provider_and_transport
    job = provider.submit(video_job, request_context).provider_job
    if provider.provider_key == "mock-http":
        provider.poll(job, request_context)
        provider.poll(job, request_context)
        job = provider.poll(job, request_context).provider_job
    result = provider.download(job, DownloadDestination(str(artifact_dir / "a.part"), str(artifact_dir / "a.bin")), request_context)
    verification = provider.verify(job, result, request_context)
    assert verification.valid
    assert verification.sha256

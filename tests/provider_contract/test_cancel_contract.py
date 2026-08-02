from __future__ import annotations

from guided_story_agent.v2 import FakeProviderRuntime, FakeProviderScenario


def test_cancel_unsupported_is_explicit(request_context, video_job) -> None:
    provider = FakeProviderRuntime(FakeProviderScenario("cancel_unsupported"))
    job = provider.submit(video_job, request_context).provider_job
    result = provider.cancel(job, request_context)
    assert result.supported is False
    assert result.accepted is False

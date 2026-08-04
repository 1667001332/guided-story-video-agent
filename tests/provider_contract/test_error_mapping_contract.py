from __future__ import annotations

import pytest

from guided_story_agent.v2 import FakeProviderRuntime, FakeProviderScenario, ProviderErrorCategory, ProviderRuntimeError


def test_submission_uncertain_is_not_a_normal_retry(request_context, video_job) -> None:
    provider = FakeProviderRuntime(FakeProviderScenario("submission_uncertain"))
    with pytest.raises(ProviderRuntimeError) as error:
        provider.submit(video_job, request_context)
    assert error.value.category is ProviderErrorCategory.SUBMISSION_UNCERTAIN
    assert error.value.submission_may_have_been_accepted
    assert not error.value.retryable

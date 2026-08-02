from __future__ import annotations

from guided_story_agent.v2 import ProviderErrorCategory, ProviderRuntimeError


def test_error_aliases_are_normalized_and_sanitized() -> None:
    error = ProviderRuntimeError(
        "safe message",
        category="submission_uncertain",
        provider_accepted=True,
        metadata={"authorization": "TEST_PROVIDER_SECRET_123"},
    )
    assert error.category is ProviderErrorCategory.SUBMISSION_UNCERTAIN
    assert error.submission_may_have_been_accepted
    assert "TEST_PROVIDER_SECRET_123" not in repr(error.to_dict())

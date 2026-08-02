from __future__ import annotations

from guided_story_agent.v2.provider_sanitization import sanitize_response


def test_sanitization_handles_nested_values_and_signed_query() -> None:
    result = sanitize_response({"headers": {"set-cookie": "TEST_PROVIDER_SECRET_123"}, "url": "https://mock/a?signature=TEST_PROVIDER_SECRET_123&x=1"})
    assert "TEST_PROVIDER_SECRET_123" not in repr(result)
    assert "x=1" in result["url"]

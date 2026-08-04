from __future__ import annotations

from guided_story_agent.v2.provider_sanitization import REDACTED, sanitize_response


def test_recursive_response_sanitization_redacts_secrets() -> None:
    value = sanitize_response({"Authorization": "Bearer TEST_PROVIDER_SECRET_123", "nested": [{"signed_url": "https://mock.test/a?signature=TEST_PROVIDER_SECRET_123&x=1"}]})
    text = repr(value)
    assert "TEST_PROVIDER_SECRET_123" not in text
    assert REDACTED in text

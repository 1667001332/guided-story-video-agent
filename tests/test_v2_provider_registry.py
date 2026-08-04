from __future__ import annotations

import pytest

from guided_story_agent.v2 import FakeProviderRuntime, ProviderRegistrationError, ProviderRuntimeRegistry


def test_registry_registers_by_runtime_and_rejects_duplicate() -> None:
    registry = ProviderRuntimeRegistry()
    runtime = FakeProviderRuntime()
    registry.register(runtime)
    assert registry.contains("fake")
    assert registry.list_provider_keys() == ("fake",)
    with pytest.raises(ProviderRegistrationError):
        registry.register(runtime)


def test_missing_provider_fails_closed() -> None:
    with pytest.raises(Exception):
        ProviderRuntimeRegistry().get("missing")

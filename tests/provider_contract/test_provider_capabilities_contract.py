from __future__ import annotations

from guided_story_agent.v2 import ProviderRuntime


def test_both_adapters_expose_standard_capabilities(provider_and_transport) -> None:
    provider, _ = provider_and_transport
    assert isinstance(provider, ProviderRuntime)
    capabilities = provider.capabilities()
    assert capabilities.provider_key == provider.provider_key
    assert capabilities.capability_fingerprint
    assert "endpoint" not in capabilities.to_dict()

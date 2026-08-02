from __future__ import annotations

import pytest

from guided_story_agent.v2 import DurationRange, ExecutionRuntime, FakeProviderRuntime, ProviderRuntimeRegistry, RuntimeProviderCapabilities
from test_v2_execution_bundle import _bundle


def test_capability_fingerprint_ignores_volatile_fields() -> None:
    first = RuntimeProviderCapabilities(
        provider_key="mock-http",
        supported_duration_ranges=(DurationRange(1, 10),),
        metadata={"profile": "stable"},
    )
    second = RuntimeProviderCapabilities(
        provider_key="mock-http",
        supported_duration_ranges=(DurationRange(1, 10),),
        metadata={"profile": "stable"},
    )
    assert first.capability_fingerprint == second.capability_fingerprint
    assert "endpoint" not in first.to_dict()


def test_capability_semantic_change_changes_fingerprint() -> None:
    first = RuntimeProviderCapabilities(provider_key="fake", supports_cancel=False)
    second = RuntimeProviderCapabilities(provider_key="fake", supports_cancel=True)
    assert first.capability_fingerprint != second.capability_fingerprint


def test_runtime_blocks_provider_profile_drift() -> None:
    runtime = ExecutionRuntime(
        _bundle(),
        provider_registry=ProviderRuntimeRegistry({"fake": FakeProviderRuntime(provider_profile="drift")}),
    )
    with pytest.raises(RuntimeError, match="provider_profile_mismatch"):
        runtime.create_run()

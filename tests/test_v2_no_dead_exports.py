from __future__ import annotations

import guided_story_agent.models as legacy_models
import guided_story_agent.v2 as v2
import guided_story_agent.v2.execution as execution
import guided_story_agent.v2.provider_capabilities as capabilities
import guided_story_agent.v2.provider_errors as provider_errors


REMOVED_OBJECTS = {
    "ReadinessReport",
    "Artifact",
    "ProviderErrorMapper",
    "legacy_capability_fingerprint",
}


def test_confirmed_dead_candidates_are_absent_from_source_modules_and_facade() -> None:
    assert not hasattr(legacy_models, "ReadinessReport")
    assert not hasattr(execution, "Artifact")
    assert not hasattr(provider_errors, "ProviderErrorMapper")
    assert not hasattr(capabilities, "legacy_capability_fingerprint")
    assert REMOVED_OBJECTS.isdisjoint(set(v2.__all__))


def test_implementation_support_is_not_declared_as_main_public_facade() -> None:
    internal_names = {
        "InMemoryExecutionEventStore",
        "JsonExecutionEventStore",
        "InMemoryRuntimeStateStore",
        "JsonRuntimeStateStore",
        "ExecutionCheckpoint",
        "FakeProviderRuntime",
        "MockHttpProviderRuntime",
        "MockHttpTransport",
        "DependencyResolver",
        "RuntimeTransitionService",
        "CreativeGraph",
        "MoviePlanVersionRecord",
    }
    assert internal_names.isdisjoint(set(v2.__all__))


def test_deprecated_wrappers_are_compatibility_only() -> None:
    wrappers = {
        "build_movie_ir",
        "compile_movie_ir_to_execution_bundle",
        "compile_movie_plan_to_video_job",
    }
    assert wrappers.isdisjoint(set(v2.__all__))
    assert all(hasattr(v2, name) for name in wrappers)

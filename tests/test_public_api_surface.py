from __future__ import annotations

import importlib

import guided_story_agent.v2 as v2


CORE_EXPORTS = {
    "MoviePlan",
    "FilmIR",
    "MovieIR",
    "VideoJob",
    "ExecutionPlan",
    "ExecutionBundle",
    "ExecutionPlanCompiler",
    "ExecutionRuntime",
    "ProviderRuntime",
    "ProviderJob",
    "ProviderCapabilities",
    "ProviderRuntimeRegistry",
    "validate_movie_plan",
    "validate_video_job",
    "validate_execution_plan",
    "validate_execution_bundle",
}


def test_v2_facade_declares_only_valid_names_and_keeps_core_boundary() -> None:
    assert len(v2.__all__) < 292
    assert len(v2.__all__) == len(set(v2.__all__))
    assert CORE_EXPORTS <= set(v2.__all__)
    assert all(hasattr(v2, name) for name in v2.__all__)


def test_packaged_entrypoints_and_offline_support_still_import() -> None:
    for module_name in (
        "guided_story_agent.cli",
        "guided_story_agent.web_app",
        "guided_story_agent.batch_test",
        "guided_story_agent.selfplay",
        "guided_story_agent.v2.fake_provider_runtime",
        "guided_story_agent.v2.mock_http_provider_runtime",
    ):
        assert importlib.import_module(module_name)

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from guided_story_agent.agent import RuleBasedStoryAgent
from guided_story_agent.cli import run_interactive
from guided_story_agent.models import CreativeBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import (
    CompilationOptions,
    MoviePlanCompiler,
    ProviderCapabilities,
    RuleBasedDirectorAgent,
    as_plain_data,
)
from test_v2_contracts import make_plan


def test_movie_plan_compiles_to_provider_only_video_job() -> None:
    plan = make_plan()
    result = MoviePlanCompiler().compile(
        plan,
        ProviderCapabilities(
            "long-video",
            provider_profile="test",
            supports_long_video=True,
            supports_multi_scene_prompt=True,
            supports_audio=True,
        ),
        CompilationOptions(
            aspect_ratio="16:9",
            resolution="1920x1080",
            fps=24,
            negative_prompt="avoid identity drift",
        ),
    )

    assert result.success
    assert result.video_job is not None
    assert result.video_job.duration_seconds == plan.timing_plan.declared_total_seconds
    assert result.video_job.provider_prompt.strip()
    assert result.video_job.source_movie_plan_id == plan.plan_id
    assert result.video_job.source_film_ir_id
    assert result.video_job.metadata["scene_ids"] == ["s1", "s2"]
    assert result.video_job.metadata["continuity_scene_ids"] == ["s1", "s2"]
    assert "character_ids" in result.video_job.metadata


def test_compiler_does_not_mutate_movie_plan() -> None:
    plan = make_plan()
    before = as_plain_data(deepcopy(plan))

    result = MoviePlanCompiler().compile(plan, ProviderCapabilities("fake"))

    assert result.success
    assert as_plain_data(plan) == before


def test_provider_fields_never_flow_back_into_movie_plan() -> None:
    plan = make_plan()
    result = MoviePlanCompiler().compile(plan, ProviderCapabilities("fake"))

    assert result.success
    payload = as_plain_data(plan)
    assert not any(key in payload for key in ("provider", "api", "task_id", "payload", "endpoint"))
    assert result.video_job is not None
    assert result.video_job.provider_key == "fake"


def test_compiler_rejects_long_video_without_splitting() -> None:
    plan = make_plan()
    result = MoviePlanCompiler().compile(
        plan,
        ProviderCapabilities("short", max_duration_seconds=9, supports_long_video=False),
    )

    assert result.video_job is None
    assert any(item.code == "duration_out_of_range" for item in result.errors)


def test_compiler_accepts_long_video_when_capability_allows_it() -> None:
    result = MoviePlanCompiler().compile(
        make_plan(),
        ProviderCapabilities("long", max_duration_seconds=120, supports_long_video=True),
    )

    assert result.success


def test_compiler_rejects_multi_scene_provider_without_compressing() -> None:
    result = MoviePlanCompiler().compile(
        make_plan(),
        ProviderCapabilities("single", supports_multi_scene_prompt=False),
    )

    assert result.video_job is None
    assert any(item.code == "multi_scene_not_supported" for item in result.errors)


def test_compiler_rejects_missing_director_field_without_repair() -> None:
    plan = make_plan()
    first = replace(plan.script.scenes[0], goal="")
    broken_script = replace(plan.script, scenes=(first, *plan.script.scenes[1:]))
    broken = replace(plan, script=broken_script)

    result = MoviePlanCompiler().compile(broken, ProviderCapabilities("fake"))

    assert result.video_job is None
    assert any(item.code == "missing_required_field" for item in result.errors)
    assert broken.script.scenes[0].goal == ""


def test_session_compiles_confirmed_movie_plan_and_round_trips() -> None:
    output = Path("outputs") / f"_v2_compiler_test_{uuid4().hex[:10]}"
    session = GuidedStorySession(
        CreativeBrief(target_seconds=30),
        RuleBasedStoryAgent(),
        director_agent=RuleBasedDirectorAgent(),
        v2_enabled=True,
    )
    session.generate_movie_plan("雨夜车站")
    session.confirm_movie_plan()
    assert session.build_movie_ir_from_confirmed_movie_plan() is not None
    result = session.compile_confirmed_movie_plan(
        ProviderCapabilities("offline", supports_long_video=True, supports_audio=True)
    )

    assert result.success
    assert session.v2_video_job is not None
    assert session.story is None
    assert session.script is None
    assert session.storyboard is None
    assert session.video_job is None
    assert session.stage.value == "video_job_compiled"

    try:
        path = session.save(output / "session.json")
        loaded = GuidedStorySession.load(path)
        assert loaded.v2_video_job is not None
        assert loaded.v2_video_job.source_movie_plan_id == session.confirmed_movie_plan.plan_id
        assert loaded.stage.value == "video_job_compiled"
    finally:
        rmtree(output, ignore_errors=True)


def test_v2_cli_compile_smoke_does_not_render() -> None:
    output = Path("outputs") / f"_v2_cli_compile_test_{uuid4().hex[:8]}"
    inputs = iter(["雨夜车站", "/confirm-plan", "/build-film-ir", "/build-ir", "/compile", "/render", "/quit"])
    messages: list[str] = []
    try:
        session = run_interactive(
            target_seconds=30,
            output_dir=output,
            v2=True,
            director_agent=RuleBasedDirectorAgent(),
            input_fn=lambda prompt: next(inputs),
            output_fn=messages.append,
        )
        assert session.movie_ir is not None
        assert session.film_ir is not None
        assert session.v2_video_job is not None
        assert session.v2_video_job.source_film_ir_id == session.film_ir.ir_id
        assert not (output / "video").exists()
        assert "V2 render is not connected in Phase 3A" in "\n".join(messages)
        saved = (output / "session.json").read_text(encoding="utf-8")
        assert '"v2_video_job"' in saved
        assert '"provider_job"' not in saved
        assert '"story": null' in saved
    finally:
        rmtree(output, ignore_errors=True)

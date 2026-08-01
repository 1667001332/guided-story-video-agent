from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest

from guided_story_agent.cli import run_interactive
from guided_story_agent.models import CreativeBrief
from guided_story_agent.agent import RuleBasedStoryAgent
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import (
    DirectorGenerationError,
    OpenAIDirectorAgent,
    RuleBasedDirectorAgent,
    as_plain_data,
    validate_movie_plan,
)
from guided_story_agent.v2.openai_director import movie_plan_from_data
from guided_story_agent.v2.models import CreativeBrief as V2CreativeBrief


def _brief() -> V2CreativeBrief:
    return V2CreativeBrief(30, "short film", "cinematic", "general audience")


def test_openai_director_parses_one_complete_movie_plan() -> None:
    fixture = RuleBasedDirectorAgent().create_movie_plan(_brief(), "雨夜车站")
    agent = OpenAIDirectorAgent(
        client=None,
        model="fixture",
        completion_fn=lambda payload: as_plain_data(fixture),
    )

    plan = agent.create_movie_plan(_brief(), "雨夜车站")

    assert plan.story.title == fixture.story.title
    assert plan.script.scenes[0].scene_id == "scene-1"
    assert plan.film_beats[0].shot_ids == ("shot-1",)
    assert validate_movie_plan(plan, _brief()).valid


def test_director_rejects_provider_fields_in_movie_plan() -> None:
    fixture = as_plain_data(RuleBasedDirectorAgent().create_movie_plan(_brief(), "雨夜车站"))
    fixture["provider_payload"] = {"model": "agnes"}

    with pytest.raises(DirectorGenerationError, match="Provider/API"):
        movie_plan_from_data(fixture)


def test_director_rejects_incomplete_json_instead_of_repairing() -> None:
    fixture = as_plain_data(RuleBasedDirectorAgent().create_movie_plan(_brief(), "雨夜车站"))
    del fixture["timing_plan"]["entries"]

    with pytest.raises(DirectorGenerationError):
        movie_plan_from_data(fixture)


def test_v2_session_saves_confirmed_movie_plan_without_legacy_story() -> None:
    output = Path("outputs") / f"_v2_director_test_{uuid4().hex[:10]}"
    try:
        session = GuidedStorySession(
            CreativeBrief(target_seconds=30),
            director_agent=RuleBasedDirectorAgent(),
            v2_enabled=True,
        )
        session.generate_movie_plan("雨夜车站")
        session.confirm_movie_plan()
        path = session.save(output / "session.json")

        loaded = GuidedStorySession.load(path)

        assert loaded.confirmed_movie_plan is not None
        assert loaded.confirmed_movie_plan.confirmed is True
        assert loaded.story is None
        assert loaded.script is None
        assert loaded.stage.value == "movie_plan_confirmed"
    finally:
        rmtree(output, ignore_errors=True)


def test_v2_cli_path_is_opt_in_and_does_not_render() -> None:
    output = Path("outputs") / f"_v2_cli_test_{uuid4().hex[:10]}"
    inputs = iter(["雨夜车站", "/confirm-plan", "/quit"])
    try:
        session = run_interactive(
            target_seconds=30,
            output_dir=output,
            v2=True,
            director_agent=RuleBasedDirectorAgent(),
            input_fn=lambda prompt: next(inputs),
            output_fn=lambda message: None,
        )
        assert session.confirmed_movie_plan is not None
        assert not (output / "video").exists()
    finally:
        rmtree(output, ignore_errors=True)


def test_v2_is_opt_in_and_legacy_session_remains_unchanged() -> None:
    session = GuidedStorySession(CreativeBrief(target_seconds=30), RuleBasedStoryAgent())

    with pytest.raises(RuntimeError, match="V2 DirectorAgent 未启用"):
        session.generate_movie_plan("雨夜车站")

    assert session.movie_plan is None
    assert session.story is None

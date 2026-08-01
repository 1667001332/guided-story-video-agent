from __future__ import annotations

from dataclasses import replace

import pytest

from guided_story_agent.v2 import (
    DirectorPlan,
    DirectorGenerationError,
    RuleBasedDirectorAgent,
    StoryPlan,
    as_plain_data,
)
from guided_story_agent.v2.models import CreativeBrief
from guided_story_agent.v2.openai_director import movie_plan_from_data


def _plan():
    return RuleBasedDirectorAgent().create_movie_plan(
        CreativeBrief(30, "short film", "cinematic", "general audience"),
        "雨夜车站",
    )


def test_rule_based_director_returns_explicit_story_and_director_layers() -> None:
    plan = _plan()

    assert isinstance(plan.story_plan, StoryPlan)
    assert isinstance(plan.director_plan, DirectorPlan)
    assert plan.story_plan.story_beats
    assert plan.director_plan.visual_motif_strategy == plan.visual_style


def test_legacy_movie_plan_json_migrates_nested_layers() -> None:
    payload = as_plain_data(_plan())
    payload.pop("story_plan")
    payload.pop("director_plan")

    restored = movie_plan_from_data(payload)

    assert restored.story_plan is not None
    assert restored.director_plan is not None
    assert restored.story_plan.title == restored.story.title
    assert restored.director_plan.visual_motif_strategy == restored.visual_style


def test_nested_plan_round_trip_is_typed() -> None:
    restored = movie_plan_from_data(as_plain_data(_plan()))

    assert restored.story_plan is not None
    assert restored.story_plan.characters
    assert isinstance(restored.director_plan, DirectorPlan)


def test_nested_provider_fields_are_rejected() -> None:
    payload = as_plain_data(_plan())
    payload["director_plan"]["provider_payload"] = {"model": "fake"}

    with pytest.raises(DirectorGenerationError, match="unsupported fields"):
        movie_plan_from_data(payload)


def test_movie_plan_is_immutable_when_layers_are_supplied() -> None:
    plan = _plan()
    updated = replace(plan, director_plan=DirectorPlan(ending_tone="quiet"))

    assert plan.director_plan != updated.director_plan
    assert plan.story_plan == updated.story_plan


def test_validator_rejects_prompt_stuffing_in_director_layer() -> None:
    from guided_story_agent.v2 import validate_movie_plan

    plan = replace(
        _plan(),
        director_plan=DirectorPlan(visual_motif_strategy="masterpiece, best quality"),
    )

    report = validate_movie_plan(
        plan,
        CreativeBrief(30, "short film", "cinematic", "general audience"),
    )

    assert not report.valid
    assert any("masterpiece" in item for item in report.errors)

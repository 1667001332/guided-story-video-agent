from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest

from guided_story_agent.agent import RuleBasedStoryAgent
from guided_story_agent.models import CreativeBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import (
    FilmIRBuilder,
    MovieIR,
    MovieIRBuilder,
    RuleBasedDirectorAgent,
)
from test_v2_contracts import make_plan


def _build_ir(plan):
    film_result = FilmIRBuilder().build(plan)
    assert film_result.ok and film_result.film_ir is not None
    result = MovieIRBuilder().build(film_result.film_ir)
    return result


def test_movie_ir_dict_roundtrip() -> None:
    result = _build_ir(make_plan())

    assert result.ok and result.movie_ir is not None
    restored = MovieIR.from_dict(result.movie_ir.to_dict())
    assert restored == result.movie_ir


def test_movie_ir_rejects_provider_specific_fields() -> None:
    result = _build_ir(make_plan())
    assert result.movie_ir is not None
    payload = result.movie_ir.to_dict()
    payload["provider_task_id"] = "remote-1"

    with pytest.raises(ValueError, match="Provider/API"):
        MovieIR.from_dict(payload)


def test_builder_generates_provider_neutral_shots_and_timeline() -> None:
    result = _build_ir(make_plan())

    assert result.ok and result.movie_ir is not None
    ir = result.movie_ir
    assert [item.order for item in ir.timeline] == [1, 2]
    assert [item.shot_id for item in ir.shots] == ["shot-1", "shot-2"]
    assert sum(item.duration_seconds for item in ir.shots) == ir.target_duration_seconds
    assert ir.shots[0].required_visual_evidence
    assert ir.shots[0].character_identity_anchors == ("c1",)
    assert ir.source_film_ir_id
    assert not any("provider" in key.lower() for key in ir.to_dict())


def test_movie_ir_builder_rejects_movie_plan_input() -> None:
    result = MovieIRBuilder().build(make_plan())

    assert not result.ok
    assert any(item.code == "invalid_film_ir_state" for item in result.errors)


def test_builder_fails_closed_without_shot_level_plan() -> None:
    result = FilmIRBuilder().build(replace(make_plan(), shot_plan=()))

    assert not result.ok
    assert result.film_ir is None
    assert any(item.code == "missing_shot_level_plan" for item in result.errors)
    assert any("DirectorAgent must regenerate" in item.message for item in result.errors)


def test_builder_rejects_duration_mismatch_without_repair() -> None:
    plan = make_plan()
    broken_shot = replace(plan.shot_plan[0], duration_seconds=8.0)
    broken = replace(plan, shot_plan=(broken_shot, plan.shot_plan[1]))

    film_result = FilmIRBuilder().build(broken)
    result = film_result

    assert not result.ok
    assert any(item.code == "invalid_total_duration" for item in result.errors)
    assert broken.shot_plan[0].duration_seconds == 8.0


def test_builder_requires_stable_shot_order() -> None:
    plan = make_plan()
    broken = replace(plan, shot_plan=(replace(plan.shot_plan[1], order=3), plan.shot_plan[0]))

    result = FilmIRBuilder().build(broken)

    assert not result.ok
    assert any(item.code == "invalid_shot_order" for item in result.errors)


def test_continuity_anchor_can_span_multiple_shots() -> None:
    plan = make_plan()
    shared = replace(plan.shot_plan[1], continuity_anchors=("雨幕方向一致",))
    first = replace(plan.shot_plan[0], continuity_anchors=("雨幕方向一致",))
    result = _build_ir(replace(plan, shot_plan=(first, shared)))

    assert result.ok and result.movie_ir is not None
    anchor = next(item for item in result.movie_ir.continuity_anchors if item.description == "雨幕方向一致")
    assert anchor.applies_to_shots == ("shot-1", "shot-2")


def test_acceptance_criteria_attach_to_shots_and_movie() -> None:
    result = _build_ir(make_plan())

    assert result.ok and result.movie_ir is not None
    assert any(item.criterion_type == "shot" for item in result.movie_ir.acceptance_criteria)
    assert any(item.criterion_type == "movie" for item in result.movie_ir.acceptance_criteria)


def test_session_persists_movie_ir_without_legacy_story() -> None:
    output = Path("outputs") / f"_v2_ir_test_{uuid4().hex[:10]}"
    try:
        session = GuidedStorySession(
            CreativeBrief(target_seconds=30),
            RuleBasedStoryAgent(),
            director_agent=RuleBasedDirectorAgent(),
            v2_enabled=True,
        )
        session.generate_movie_plan("雨夜车站")
        session.confirm_movie_plan()
        assert session.build_film_ir_from_confirmed_movie_plan() is not None
        ir = session.build_movie_ir_from_film_ir()

        assert ir is not None
        assert session.stage.value == "movie_ir_built"
        assert session.story is None
        assert session.script is None
        path = session.save(output / "session.json")
        loaded = GuidedStorySession.load(path)
        assert loaded.movie_ir is not None
        assert loaded.movie_ir.source_movie_plan_id == session.confirmed_movie_plan.plan_id
        assert loaded.v2_video_job is None
    finally:
        rmtree(output, ignore_errors=True)

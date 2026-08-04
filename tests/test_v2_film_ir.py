from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest

from guided_story_agent.cli import run_interactive
from guided_story_agent.v2 import RuleBasedDirectorAgent
from guided_story_agent.models import CreativeBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import FilmIR, FilmIRBuilder, MovieIRBuilder
from test_v2_contracts import make_plan


def test_film_ir_round_trip_and_source_identity() -> None:
    result = FilmIRBuilder().build(make_plan())

    assert result.ok and result.film_ir is not None
    film_ir = result.film_ir
    restored = FilmIR.from_dict(film_ir.to_dict())

    assert restored == film_ir
    assert film_ir.source_movie_plan_id == "plan-1"
    assert [item.shot_ids for item in film_ir.beats] == [("shot-1",), ("shot-2",)]


def test_film_ir_rejects_provider_and_prompt_fields() -> None:
    result = FilmIRBuilder().build(make_plan())
    assert result.film_ir is not None
    payload = result.film_ir.to_dict()
    payload["provider_task_id"] = "remote-1"

    with pytest.raises(ValueError, match="Provider/API"):
        FilmIR.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dramatic_purpose", ""),
        ("viewer_state_before", ""),
        ("viewer_state_after", ""),
        ("required_evidence", ()),
        ("acceptance_criteria", ()),
    ),
)
def test_film_ir_builder_requires_cinematic_contract(field: str, value) -> None:
    plan = make_plan()
    broken = replace(plan.film_beats[0], **{field: value})
    result = FilmIRBuilder().build(replace(plan, film_beats=(broken, *plan.film_beats[1:])))

    assert not result.ok
    assert result.film_ir is None


def test_film_ir_builder_fails_closed_without_beats() -> None:
    result = FilmIRBuilder().build(replace(make_plan(), film_beats=()))

    assert not result.ok
    assert any(item.code == "missing_film_level_beats" for item in result.errors)
    assert any("DirectorAgent must regenerate" in item.message for item in result.errors)


def test_film_ir_beats_trace_to_movie_plan_scene_and_shot() -> None:
    result = FilmIRBuilder().build(make_plan())

    assert result.film_ir is not None
    for beat in result.film_ir.beats:
        assert beat.scene_id in {"s1", "s2"}
        assert beat.shot_ids
        assert set(beat.shot_ids).issubset({"shot-1", "shot-2"})


def test_movie_ir_lowering_carries_film_evidence_without_reading_movie_plan() -> None:
    film_result = FilmIRBuilder().build(make_plan())
    assert film_result.film_ir is not None
    ir_result = MovieIRBuilder().build(film_result.film_ir)

    assert ir_result.ok and ir_result.movie_ir is not None
    ir = ir_result.movie_ir
    assert ir.source_film_ir_id == film_result.film_ir.ir_id
    assert "陌生时间清晰可见" in ir.shots[0].required_visual_evidence
    assert any(item.criterion_type == "film-beat" for item in ir.acceptance_criteria)
    assert not any("provider_prompt" in key for key in ir.to_dict())


def test_session_rejects_movie_ir_before_film_ir_and_persists_full_chain() -> None:
    session = GuidedStorySession(
        CreativeBrief(target_seconds=30),
        director_agent=RuleBasedDirectorAgent(),
        v2_enabled=True,
    )
    session.generate_movie_plan("雨夜车站")
    session.confirm_movie_plan()

    with pytest.raises(RuntimeError, match="Build FilmIR first with /build-film-ir"):
        session.build_movie_ir_from_film_ir()
    film_ir = session.build_film_ir_from_confirmed_movie_plan()
    assert film_ir is not None
    movie_ir = session.build_movie_ir_from_film_ir()
    assert movie_ir is not None
    assert session.stage.value == "movie_ir_built"

    output = Path("outputs") / f"_v2_film_ir_session_{uuid4().hex[:8]}"
    try:
        path = session.save(output / "session.json")
        loaded = GuidedStorySession.load(path)
        assert loaded.film_ir is not None
        assert loaded.movie_ir is not None
        assert loaded.movie_ir.source_film_ir_id == loaded.film_ir.ir_id
        assert loaded.story is None
        assert loaded.storyboard is None
    finally:
        rmtree(output, ignore_errors=True)


def test_v2_cli_film_ir_flow_stays_offline() -> None:
    output = Path("outputs") / f"_v2_film_ir_cli_{uuid4().hex[:8]}"
    inputs = iter(
        ["雨夜车站", "/confirm-plan", "/build-film-ir", "/build-ir", "/compile", "/render", "/quit"]
    )
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
        assert session.confirmed_movie_plan is not None
        assert session.film_ir is not None
        assert session.movie_ir is not None
        assert session.v2_video_job is not None
        assert session.v2_video_job.source_film_ir_id == session.film_ir.ir_id
        assert session.v2_video_job.source_movie_ir_id == session.movie_ir.ir_id
        assert session.story is None
        assert session.storyboard is None
        assert session.stage.value == "video_job_compiled"
        assert "V2 render is not connected" in "\n".join(messages)
        assert not (output / "video").exists()
    finally:
        rmtree(output, ignore_errors=True)

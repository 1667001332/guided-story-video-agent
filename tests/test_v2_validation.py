from __future__ import annotations

from dataclasses import replace

from guided_story_agent.v2 import FilmIRBuilder, FilmIRValidator, MovieIRBuilder, MovieIRValidator
from test_v2_contracts import make_plan


def _irs():
    plan = make_plan()
    film_result = FilmIRBuilder().build(plan)
    assert film_result.film_ir is not None
    movie_result = MovieIRBuilder().build(film_result.film_ir)
    assert movie_result.movie_ir is not None
    return film_result.film_ir, movie_result.movie_ir


def test_film_ir_validator_rejects_missing_beat_and_viewer_state() -> None:
    film_ir, _ = _irs()
    beat = replace(film_ir.beats[0], viewer_state_before="")
    broken = replace(film_ir, beats=(beat, *film_ir.beats[1:]))

    result = FilmIRValidator().validate(broken)

    assert not result.ok
    assert any(
        item.code == "missing_film_decision"
        and item.path == "beats[0].viewer_state_before"
        for item in result.issues
    )


def test_film_ir_validator_rejects_missing_beats() -> None:
    film_ir, _ = _irs()

    result = FilmIRValidator().validate(replace(film_ir, beats=()))

    assert not result.ok
    assert any(item.code == "missing_film_beats" for item in result.issues)


def test_film_ir_validator_rejects_provider_metadata() -> None:
    film_ir, _ = _irs()
    broken = replace(film_ir, metadata={"provider_task_id": "remote-1"})

    result = FilmIRValidator().validate(broken)

    assert not result.ok
    assert any(item.code == "provider_field_in_ir" for item in result.issues)


def test_film_ir_validator_rejects_prompt_stuffing() -> None:
    film_ir, _ = _irs()
    result = FilmIRValidator().validate(
        replace(film_ir, visual_style="masterpiece ultra realistic")
    )

    assert not result.ok
    assert any(item.code == "prompt_leakage" for item in result.issues)


def test_movie_ir_validator_rejects_missing_film_source() -> None:
    _, movie_ir = _irs()
    result = MovieIRValidator().validate(replace(movie_ir, source_film_ir_id=""))

    assert not result.ok
    assert any(item.code == "missing_source_film_ir_id" for item in result.issues)


def test_movie_ir_validator_rejects_prompt_stuffing() -> None:
    _, movie_ir = _irs()
    shot = replace(movie_ir.shots[0], visible_action="masterpiece best quality action")
    broken = replace(movie_ir, shots=(shot, *movie_ir.shots[1:]))

    result = MovieIRValidator().validate(broken)

    assert not result.ok
    assert any(item.code == "prompt_leakage" for item in result.issues)


def test_movie_ir_validator_rejects_overlapping_timeline() -> None:
    _, movie_ir = _irs()
    entry = replace(movie_ir.timeline[1], start_seconds=1.0)
    broken = replace(movie_ir, timeline=(movie_ir.timeline[0], entry))

    result = MovieIRValidator().validate(broken)

    assert not result.ok
    assert any(item.code == "invalid_timeline_continuity" for item in result.issues)


def test_movie_ir_validator_rejects_broken_timeline() -> None:
    _, movie_ir = _irs()
    entry = replace(movie_ir.timeline[1], start_seconds=movie_ir.timeline[0].duration_seconds + 2.0)
    broken = replace(movie_ir, timeline=(movie_ir.timeline[0], entry))

    result = MovieIRValidator().validate(broken)

    assert not result.ok
    assert any(item.code == "invalid_timeline_continuity" for item in result.issues)

from __future__ import annotations

from dataclasses import replace

from guided_story_agent.v2 import (
    AudienceUnderstandingDiagnosticsPass,
    EmotionalContinuityPass,
    FilmIRBuilder,
    PacingDiagnosticsPass,
    creative_pass_pipeline,
)
from guided_story_agent.v2.models import as_plain_data
from test_v2_contracts import make_plan


def _film_ir():
    result = FilmIRBuilder().build(make_plan())
    assert result.film_ir is not None
    return result.film_ir


def test_emotional_continuity_gap_returns_diagnostic() -> None:
    film_ir = _film_ir()
    second = replace(
        film_ir.beats[1],
        viewer_state_before="观众突然失去上下文",
    )
    broken = replace(film_ir, beats=(film_ir.beats[0], second))

    result = EmotionalContinuityPass().run(broken)

    assert result.ok
    assert any(item.code == "emotional_continuity_gap" for item in result.diagnostics)


def test_audience_understanding_missing_is_hard_error() -> None:
    film_ir = _film_ir()
    broken = replace(
        film_ir,
        beats=(replace(film_ir.beats[0], required_audience_understanding=""), *film_ir.beats[1:]),
    )

    result = AudienceUnderstandingDiagnosticsPass().run(broken)

    assert not result.ok
    assert result.ir is None
    assert any(item.code == "missing_audience_understanding" for item in result.diagnostics)


def test_creative_pipeline_is_pure_and_does_not_mutate_movie_plan() -> None:
    plan = make_plan()
    before = as_plain_data(plan)
    film_ir = _film_ir()

    result = creative_pass_pipeline().run(film_ir)

    assert result.ok
    assert as_plain_data(plan) == before
    assert result.ir == film_ir


def test_pacing_diagnostics_does_not_reallocate_duration() -> None:
    film_ir = _film_ir()
    result = PacingDiagnosticsPass().run(film_ir)

    assert result.ok
    assert result.ir == film_ir

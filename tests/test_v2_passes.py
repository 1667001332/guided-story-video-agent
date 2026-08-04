from __future__ import annotations

from dataclasses import replace

from guided_story_agent.v2 import (
    EnsureAudienceUnderstandingPass,
    FilmIRBuilder,
    MovieIRBuilder,
    NormalizeFilmBeatOrderPass,
    NormalizeShotTimelinePass,
    PassPipeline,
    PromptLeakageDiagnosticsPass,
)
from test_v2_contracts import make_plan


def _irs():
    film_result = FilmIRBuilder().build(make_plan())
    assert film_result.film_ir is not None
    movie_result = MovieIRBuilder().build(film_result.film_ir)
    assert movie_result.movie_ir is not None
    return film_result.film_ir, movie_result.movie_ir


def test_film_beat_order_pass_normalizes_order_and_returns_diagnostic() -> None:
    film_ir, _ = _irs()
    first = replace(film_ir.beats[0], order=2)
    second = replace(film_ir.beats[1], order=1)
    broken = replace(film_ir, beats=(first, second))

    result = NormalizeFilmBeatOrderPass().run(broken)

    assert result.ok and result.ir is not None
    assert [item.order for item in result.ir.beats] == [1, 2]
    assert any(item.code == "beat_order_normalized" for item in result.diagnostics)


def test_movie_timeline_pass_normalizes_gap() -> None:
    _, movie_ir = _irs()
    broken = replace(
        movie_ir,
        timeline=(movie_ir.timeline[0], replace(movie_ir.timeline[1], start_seconds=99.0)),
    )

    result = NormalizeShotTimelinePass().run(broken)

    assert result.ok and result.ir is not None
    assert result.ir.timeline[1].start_seconds == result.ir.timeline[0].duration_seconds
    assert any(item.code == "shot_timeline_normalized" for item in result.diagnostics)


def test_prompt_leakage_pass_fails_closed() -> None:
    _, movie_ir = _irs()
    shot = replace(movie_ir.shots[0], visible_action="masterpiece best quality")
    broken = replace(movie_ir, shots=(shot, *movie_ir.shots[1:]))

    result = PromptLeakageDiagnosticsPass().run(broken)

    assert not result.ok
    assert result.ir is None
    assert any(item.code == "prompt_leakage" for item in result.diagnostics)


def test_pass_pipeline_returns_diagnostics_and_fails_closed() -> None:
    film_ir, _ = _irs()
    broken = replace(
        film_ir,
        beats=(replace(film_ir.beats[0], required_audience_understanding=""), *film_ir.beats[1:]),
    )

    result = PassPipeline((EnsureAudienceUnderstandingPass(),)).run(broken)

    assert not result.ok
    assert result.ir is None
    assert any(item.code == "missing_audience_understanding" for item in result.diagnostics)

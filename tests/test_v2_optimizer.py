from __future__ import annotations

from dataclasses import replace

from guided_story_agent.v2 import FilmIRBuilder, MovieIRBuilder
from guided_story_agent.v2.optimizer import (
    DetectRedundantShotOptimizer,
    DetectTimelineBudgetRiskOptimizer,
    FilmIROptimizer,
    MergeAdjacentCompatibleShotsOptimizer,
    MovieIROptimizer,
)
from test_v2_contracts import make_plan


def _irs():
    film_result = FilmIRBuilder().build(make_plan())
    assert film_result.film_ir is not None
    movie_result = MovieIRBuilder().build(film_result.film_ir)
    assert movie_result.movie_ir is not None
    return film_result.film_ir, movie_result.movie_ir


def test_adjacent_compatible_shots_are_merge_candidates_without_mutation() -> None:
    _, movie_ir = _irs()
    left, right = movie_ir.shots
    compatible_right = replace(
        right,
        scene_id=left.scene_id,
        camera=left.camera,
        motion=left.motion,
        lighting=left.lighting,
        composition=left.composition,
    )
    candidate_ir = replace(movie_ir, shots=(left, compatible_right))

    result = MergeAdjacentCompatibleShotsOptimizer().optimize(candidate_ir)

    assert result.ok
    assert result.after_ir == candidate_ir
    assert any(item.code == "merge_candidate" for item in result.transformations)


def test_redundant_shot_is_identified() -> None:
    _, movie_ir = _irs()
    left, right = movie_ir.shots
    candidate_ir = replace(
        movie_ir,
        shots=(
            left,
            replace(
                right,
                scene_id=left.scene_id,
                visible_action=left.visible_action,
                subject=left.subject,
            ),
        ),
    )

    result = DetectRedundantShotOptimizer().optimize(candidate_ir)

    assert result.ok
    assert any(item.code == "redundant_shot_candidate" for item in result.diagnostics)


def test_timeline_budget_risk_is_identified() -> None:
    _, movie_ir = _irs()
    first, second = movie_ir.shots
    candidate_ir = replace(movie_ir, shots=(replace(first, duration_seconds=0.1), second))

    result = DetectTimelineBudgetRiskOptimizer().optimize(candidate_ir)

    assert result.ok
    assert any(item.code == "timeline_budget_risk" for item in result.diagnostics)


def test_optimizer_result_preserves_before_after_and_records_transformations() -> None:
    film_ir, movie_ir = _irs()

    film_result = FilmIROptimizer().optimize(film_ir)
    movie_result = MovieIROptimizer().optimize(movie_ir)

    assert film_result.before_ir == film_ir
    assert film_result.after_ir == film_ir
    assert movie_result.before_ir == movie_ir
    assert movie_result.after_ir == movie_ir
    assert isinstance(movie_result.transformations, tuple)


def test_film_optimizer_blocks_missing_audience_contract() -> None:
    film_ir, _ = _irs()
    broken = replace(
        film_ir,
        beats=(replace(film_ir.beats[0], required_audience_understanding=""), *film_ir.beats[1:]),
    )

    result = FilmIROptimizer().optimize(broken)

    assert not result.ok
    assert result.after_ir is None
    assert any(item.code == "missing_audience_understanding" for item in result.errors)

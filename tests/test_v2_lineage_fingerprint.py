from __future__ import annotations

from dataclasses import replace

from guided_story_agent.v2 import (
    FilmIRBuilder,
    MovieIRBuilder,
    ProviderCapabilities,
    SourceLineageGuard,
    VideoJobCompiler,
)
from test_v2_contracts import make_plan


def _compiled():
    plan = make_plan()
    film_result = FilmIRBuilder().build(plan)
    assert film_result.film_ir is not None
    movie_result = MovieIRBuilder().build(film_result.film_ir)
    assert movie_result.movie_ir is not None
    job_result = VideoJobCompiler().compile(
        movie_result.movie_ir,
        ProviderCapabilities(
            "offline", supports_long_video=True, supports_multi_scene_prompt=True
        ),
    )
    assert job_result.video_job is not None
    return plan, film_result.film_ir, movie_result.movie_ir, job_result.video_job


def test_all_runtime_boundaries_carry_plan_fingerprint() -> None:
    plan, film_ir, movie_ir, job = _compiled()
    assert film_ir.source_movie_plan_version == plan.movie_plan_version
    assert film_ir.source_movie_plan_fingerprint == plan.movie_plan_fingerprint
    assert movie_ir.source_movie_plan_fingerprint == plan.movie_plan_fingerprint
    assert job.source_movie_plan_fingerprint == plan.movie_plan_fingerprint
    assert movie_ir.source_film_ir_fingerprint
    assert job.source_movie_ir_fingerprint


def test_fingerprint_mismatch_is_stale_and_missing_fingerprint_is_unknown() -> None:
    plan, film_ir, _, _ = _compiled()
    guard = SourceLineageGuard()
    mismatch = guard.check_film_ir(
        replace(film_ir, source_movie_plan_fingerprint="0" * 64),
        current_movie_plan_id=plan.plan_id,
        current_movie_plan_version=plan.movie_plan_version,
        current_movie_plan_fingerprint=plan.movie_plan_fingerprint,
        current_movie_plan_lineage_token=plan.movie_plan_lineage_token,
        current_film_ir_id=film_ir.ir_id,
    )
    assert mismatch.status == "stale"
    assert any("fingerprint" in item.code for item in mismatch.diagnostics)

    unknown = guard.check_film_ir(
        replace(film_ir, source_movie_plan_fingerprint=""),
        current_movie_plan_id=plan.plan_id,
        current_movie_plan_version=plan.movie_plan_version,
        current_movie_plan_fingerprint=plan.movie_plan_fingerprint,
        current_movie_plan_lineage_token=plan.movie_plan_lineage_token,
        current_film_ir_id=film_ir.ir_id,
    )
    assert unknown.status == "unknown"

from __future__ import annotations

from dataclasses import replace

from guided_story_agent.v2 import (
    ExecutionPlanCompiler,
    FilmIRBuilder,
    MovieIRBuilder,
    ProviderCapabilities,
    validate_execution_bundle,
)
from test_v2_contracts import make_plan


def _bundle():
    film = FilmIRBuilder().build(make_plan()).film_ir
    assert film is not None
    movie = MovieIRBuilder().build(film).movie_ir
    assert movie is not None
    result = ExecutionPlanCompiler().compile(
        movie,
        ProviderCapabilities(
            "fake",
            supports_reference_images=True,
            supports_character_reference=True,
            supports_audio=True,
            supports_long_video=True,
        ),
    )
    assert result.bundle is not None
    return result.bundle


def test_bundle_round_trip_and_fingerprint_cover_job_collection() -> None:
    bundle = _bundle()
    restored = type(bundle).from_dict(bundle.to_dict())

    assert restored == bundle
    assert validate_execution_bundle(restored).valid
    assert set(restored.video_job_map) == {job.job_id for job in restored.video_jobs}


def test_bundle_rejects_dangling_job_and_fingerprint_mismatch() -> None:
    bundle = _bundle()
    unit = replace(bundle.execution_plan.execution_units[0], video_job_id="missing-job")
    plan = replace(bundle.execution_plan, execution_units=(unit,))
    invalid = replace(bundle, execution_plan=plan, bundle_fingerprint="wrong")
    result = validate_execution_bundle(invalid)

    codes = {item.code for item in result.diagnostics}
    assert "missing_video_job" in codes
    assert "bundle_fingerprint_mismatch" in codes


def test_bundle_rejects_duplicate_and_unused_jobs() -> None:
    bundle = _bundle()
    duplicate = replace(bundle, video_jobs=(bundle.video_jobs[0], bundle.video_jobs[0]))
    result = validate_execution_bundle(duplicate)

    codes = {item.code for item in result.diagnostics}
    assert "duplicate_video_job_id" in codes

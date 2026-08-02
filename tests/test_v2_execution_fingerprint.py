from __future__ import annotations

from dataclasses import replace

from guided_story_agent.v2 import (
    ExecutionPlanCompiler,
    FilmIRBuilder,
    MovieIRBuilder,
    ProviderCapabilities,
    video_job_fingerprint,
)
from guided_story_agent.v2.execution_fingerprint import execution_plan_fingerprint
from test_v2_contracts import make_plan


def _movie_ir():
    film = FilmIRBuilder().build(make_plan()).film_ir
    assert film is not None
    movie = MovieIRBuilder().build(film).movie_ir
    assert movie is not None
    return movie


def test_video_job_fingerprint_ignores_id_and_created_at_but_covers_prompt() -> None:
    result = ExecutionPlanCompiler().compile(
        _movie_ir(),
        ProviderCapabilities("fake", supports_reference_images=True, supports_audio=True, supports_long_video=True),
    )
    assert result.bundle is not None
    job = result.bundle.video_jobs[0]
    changed_identity = replace(job, job_id="different", created_at="2099-01-01T00:00:00Z")
    changed_prompt = replace(job, provider_prompt=job.provider_prompt + " changed", video_job_fingerprint="")

    assert video_job_fingerprint(job) == video_job_fingerprint(changed_identity)
    assert video_job_fingerprint(job) != video_job_fingerprint(changed_prompt)


def test_plan_fingerprint_changes_for_provider_policy_and_dag() -> None:
    result = ExecutionPlanCompiler().compile(
        _movie_ir(),
        ProviderCapabilities("fake", supports_reference_images=True, supports_audio=True, supports_long_video=True),
    )
    assert result.bundle is not None
    plan = result.bundle.execution_plan
    assert execution_plan_fingerprint(plan) == plan.execution_plan_fingerprint
    assert execution_plan_fingerprint(replace(plan, execution_plan_id="other")) == plan.execution_plan_fingerprint
    assert execution_plan_fingerprint(
        replace(plan, runtime_policy=replace(plan.runtime_policy, fail_fast=False))
    ) != plan.execution_plan_fingerprint

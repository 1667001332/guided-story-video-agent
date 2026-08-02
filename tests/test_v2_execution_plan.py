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


def _movie_ir():
    film = FilmIRBuilder().build(make_plan())
    assert film.film_ir is not None
    result = MovieIRBuilder().build(film.film_ir)
    assert result.movie_ir is not None
    return result.movie_ir


def _capabilities():
    return ProviderCapabilities(
        "offline",
        provider_profile="offline",
        supports_long_video=True,
        supports_multi_scene_prompt=True,
        supports_reference_images=True,
        supports_character_reference=True,
        supports_audio=True,
    )


def test_execution_plan_compiler_lowers_movie_ir_without_provider_runtime() -> None:
    result = ExecutionPlanCompiler().compile(_movie_ir(), _capabilities())

    assert result.success
    assert result.bundle is not None
    assert len(result.bundle.execution_plan.execution_units) == 2
    assert len(result.bundle.video_jobs) == 2
    assert result.bundle.execution_plan.provider_assignments[0].provider_key == "offline"
    assert all(unit.video_job_id for unit in result.bundle.execution_plan.execution_units)
    assert all(unit.video_job_fingerprint for unit in result.bundle.execution_plan.execution_units)
    assert validate_execution_bundle(result.bundle).valid


def test_execution_plan_is_immutable_and_does_not_mutate_movie_ir() -> None:
    movie_ir = _movie_ir()
    before = movie_ir.to_dict()
    result = ExecutionPlanCompiler().compile(movie_ir, _capabilities())

    assert result.success and result.bundle is not None
    assert movie_ir.to_dict() == before
    try:
        result.bundle.execution_plan.metadata["changed"] = True
    except TypeError:
        pass
    else:  # pragma: no cover - mappingproxy is expected
        raise AssertionError("ExecutionPlan metadata must be read-only")


def test_same_movie_ir_and_capabilities_have_same_plan_and_bundle_fingerprints() -> None:
    movie_ir = _movie_ir()
    first = ExecutionPlanCompiler().compile(movie_ir, _capabilities())
    second = ExecutionPlanCompiler().compile(movie_ir, _capabilities())

    assert first.success and second.success
    assert first.bundle is not None and second.bundle is not None
    assert first.bundle.execution_plan.execution_plan_fingerprint == second.bundle.execution_plan.execution_plan_fingerprint
    assert first.bundle.bundle_fingerprint == second.bundle.bundle_fingerprint


def test_same_scene_lowering_adds_explicit_reference_frame_edge() -> None:
    movie_ir = _movie_ir()
    second = replace(movie_ir.shots[1], scene_id=movie_ir.shots[0].scene_id)
    changed = replace(movie_ir, shots=(movie_ir.shots[0], second))
    result = ExecutionPlanCompiler().compile(changed, _capabilities())

    assert result.success and result.bundle is not None
    edges = result.bundle.execution_plan.dependency_graph
    assert any(edge.dependency_type == "reference_frame" for edge in edges)
    assert result.bundle.execution_plan.execution_units[1].reference_inputs[-1].source_unit_id

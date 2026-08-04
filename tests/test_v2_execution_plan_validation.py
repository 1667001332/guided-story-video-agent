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
        ProviderCapabilities("fake", supports_reference_images=True, supports_audio=True, supports_long_video=True),
    )
    assert result.bundle is not None
    return result.bundle


def test_validation_rejects_self_reference_cycle_and_dependency_mismatch() -> None:
    bundle = _bundle()
    first = bundle.execution_plan.execution_units[0]
    broken_unit = replace(first, depends_on=("missing-dependency",), execution_unit_fingerprint="")
    broken_plan = replace(
        bundle.execution_plan,
        execution_units=(broken_unit, *bundle.execution_plan.execution_units[1:]),
        dependency_graph=(*bundle.execution_plan.dependency_graph, type(bundle.execution_plan.dependency_graph[0])(first.execution_unit_id, first.execution_unit_id, "serial")),
    )
    result = validate_execution_bundle(replace(bundle, execution_plan=broken_plan, bundle_fingerprint="bad"))
    codes = {item.code for item in result.diagnostics}
    assert "invalid_self_dependency" in codes
    assert "dependency_graph_mismatch" in codes
    assert "invalid_dependency_cycle" in codes


def test_validation_recursively_rejects_provider_boundary_fields() -> None:
    bundle = _bundle()
    plan = replace(bundle.execution_plan, metadata={"nested": {"secrets": {"api_key": "do-not-store"}}})
    result = validate_execution_bundle(replace(bundle, execution_plan=plan, bundle_fingerprint="bad"))

    assert any(item.code == "provider_boundary_violation" for item in result.diagnostics)


def test_validation_reports_unknown_lineage_for_missing_provenance() -> None:
    bundle = _bundle()
    plan = replace(bundle.execution_plan, source_movie_plan_fingerprint="", execution_plan_fingerprint="bad")
    result = validate_execution_bundle(replace(bundle, execution_plan=plan, bundle_fingerprint="bad"))

    assert result.status == "unknown"

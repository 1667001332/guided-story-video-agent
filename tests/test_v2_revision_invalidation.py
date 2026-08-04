from __future__ import annotations

from guided_story_agent.models import CreativeBrief as LegacyBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import (
    ApplyRevisionCommand,
    RevisionCandidateFactory,
    RevisionDiffBuilder,
    RevisionGuard,
    RuleBasedDirectorAgent,
    invalidate_downstream_after_movie_plan_change,
)


def _session() -> GuidedStorySession:
    session = GuidedStorySession(
        LegacyBrief(target_seconds=30),
        director_agent=RuleBasedDirectorAgent(),
        v2_enabled=True,
    )
    session.generate_movie_plan("雨夜车站")
    session.confirm_movie_plan()
    return session


def test_invalidation_clears_active_v2_outputs_and_records_stale_artifacts() -> None:
    session = _session()
    session.film_ir = object()
    session.movie_ir = object()
    session.v2_video_job = object()
    session.creative_analysis_results = [{"analysis": "old"}]
    session.creative_optimizer_result = {"result": "old"}
    session.creative_revision_requests = [{"request_id": "old"}]
    session.revision_candidates = [{"candidate_id": "old"}]
    session.revision_decisions = [{"decision": "accept"}]
    session.guarded_revision_results = [{"result": "old"}]

    result = invalidate_downstream_after_movie_plan_change(
        session,
        "test movie plan change",
        source_movie_plan_id=session.current_movie_plan_id,
    )

    assert result.succeeded is True
    assert "film_ir" in result.invalidated
    assert "movie_ir" in result.invalidated
    assert "v2_video_job" in result.invalidated
    assert session.film_ir is None
    assert session.movie_ir is None
    assert session.v2_video_job is None
    assert session.creative_analysis_results == []
    assert session.creative_optimizer_result is None
    assert session.creative_revision_requests == []
    assert session.revision_candidates == []
    assert session.revision_decisions == []
    assert session.guarded_revision_results == []
    assert {item["artifact_type"] for item in session.stale_artifacts} >= {
        "film_ir",
        "movie_ir",
        "v2_video_job",
    }


def test_explicit_apply_invalidates_built_downstream_state() -> None:
    session = _session()
    assert session.build_film_ir_from_confirmed_movie_plan() is not None
    assert session.build_movie_ir_from_film_ir() is not None
    assert session.compile_confirmed_movie_plan().success is True
    session.creative_analysis_results = [{"analysis": "old"}]
    session.creative_optimizer_result = {"result": "old"}
    plan = session.confirmed_movie_plan
    candidate = RevisionCandidateFactory().create_noop_candidate(plan)
    decision = RevisionGuard().evaluate(plan, candidate, RevisionDiffBuilder().build_diff(plan, candidate))
    session.revision_candidates = [candidate.to_dict()]
    session.revision_decisions = [decision.to_dict()]

    result = session.apply_revision(
        ApplyRevisionCommand(
            "apply-built",
            candidate.candidate_id,
            plan.plan_id,
            "invalidate old downstream",
            "test-user",
        )
    )

    assert result.succeeded is True
    assert session.stage.value == "movie_plan_revised"
    assert session.film_ir is None
    assert session.movie_ir is None
    assert session.v2_video_job is None
    assert session.current_film_ir_id is None
    assert session.current_movie_ir_id is None
    assert session.current_video_job_id is None
    assert session.source_lineage_diagnostics == []
    assert session.stale_lineage_diagnostics == []
    assert session.revision_apply_history

from __future__ import annotations

from dataclasses import replace

from guided_story_agent.models import CreativeBrief as LegacyBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import (
    ApplyRevisionCommand,
    RevisionCandidate,
    RevisionDiffBuilder,
    RevisionGuard,
    RevisionRollbackService,
    RollbackRevisionCommand,
    RuleBasedDirectorAgent,
)


def _applied_session() -> tuple[GuidedStorySession, str]:
    session = GuidedStorySession(
        LegacyBrief(target_seconds=30),
        director_agent=RuleBasedDirectorAgent(),
        v2_enabled=True,
    )
    session.generate_movie_plan("雨夜车站")
    session.confirm_movie_plan()
    original = session.confirmed_movie_plan
    revised = replace(original, plan_id="revised-plan", revision=2)
    candidate = RevisionCandidate(
        "candidate-revised",
        original.plan_id,
        revised_movie_plan=revised,
        candidate_type="fake_targeted_patch",
        created_by="test",
    )
    decision = RevisionGuard().evaluate(
        original,
        candidate,
        RevisionDiffBuilder().build_diff(original, candidate),
    )
    assert decision.decision == "accept"
    session.revision_candidates = [candidate.to_dict()]
    session.revision_decisions = [decision.to_dict()]
    session.apply_revision(
        ApplyRevisionCommand(
            "apply-revised",
            candidate.candidate_id,
            original.plan_id,
            "apply revised plan",
            "test-user",
        )
    )
    return session, original.plan_id


def test_rollback_restores_history_and_invalidates_downstream() -> None:
    session, original_id = _applied_session()
    session.film_ir = object()
    session.movie_ir = object()
    session.v2_video_job = object()

    result = session.rollback_revision(
        RollbackRevisionCommand(
            "rollback-1",
            original_id,
            "restore original plan",
            "test-user",
        )
    )

    assert result.succeeded is True
    assert result.rolled_back is True
    assert result.restored_movie_plan_id == original_id
    assert session.current_movie_plan_id == original_id
    assert session.previous_movie_plan_id == "revised-plan"
    assert session.stage.value == "movie_plan_rolled_back"
    assert session.film_ir is None
    assert session.movie_ir is None
    assert session.v2_video_job is None
    assert session.current_film_ir_id is None
    assert session.current_movie_ir_id is None
    assert session.current_video_job_id is None
    assert session.source_lineage_diagnostics == []
    assert session.stale_lineage_diagnostics == []
    assert session.revision_rollback_history


def test_rollback_missing_target_is_rejected() -> None:
    session, _ = _applied_session()
    result = session.rollback_revision(
        RollbackRevisionCommand(
            "rollback-missing",
            "does-not-exist",
            "missing target",
            "test-user",
        )
    )

    assert result.succeeded is False
    assert result.stop_reason == "target_not_found"


def test_rollback_service_requires_revalidation_brief() -> None:
    from tests.test_v2_contracts import make_plan

    issues = RevisionRollbackService().validate_restored_plan(make_plan())
    assert issues[0]["code"] == "missing_validation_brief"

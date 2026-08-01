from __future__ import annotations

from dataclasses import replace

from tests.test_v2_contracts import make_plan

from guided_story_agent.v2 import (
    ApplyRevisionCommand,
    CreativeBrief,
    RevisionApplyService,
    RevisionCandidate,
    RevisionCandidateFactory,
    RevisionDiffBuilder,
    RevisionGuard,
)


def _accepted_candidate():
    plan = make_plan()
    candidate = RevisionCandidate(
        "candidate-apply",
        plan.plan_id,
        revised_movie_plan=replace(plan, plan_id="plan-applied", revision=2),
        candidate_type="fake_targeted_patch",
        created_by="test",
    )
    decision = RevisionGuard().evaluate(
        plan,
        candidate,
        RevisionDiffBuilder().build_diff(plan, candidate),
    )
    assert decision.decision == "accept"
    return plan, candidate, decision


def _command(plan, candidate_id: str) -> ApplyRevisionCommand:
    return ApplyRevisionCommand(
        command_id="apply-command-1",
        candidate_id=candidate_id,
        source_movie_plan_id=plan.plan_id,
        apply_reason="test explicit apply",
        confirmed_by="test-user",
    )


def test_accepted_candidate_can_apply_without_provider() -> None:
    plan, candidate, decision = _accepted_candidate()

    result = RevisionApplyService(CreativeBrief(10, "short film", "cinematic", "adult")).apply_revision(
        plan,
        candidate,
        decision,
        _command(plan, candidate.candidate_id),
    )

    assert result.applied is True
    assert result.new_movie_plan_id == "plan-applied"
    assert result.stop_reason == "applied"


def test_accept_with_warning_can_apply() -> None:
    plan, candidate, decision = _accepted_candidate()
    warning = replace(decision, decision="accept_with_warning")

    result = RevisionApplyService(CreativeBrief(10, "short film", "cinematic", "adult")).apply_revision(
        plan,
        candidate,
        warning,
        _command(plan, candidate.candidate_id),
    )

    assert result.succeeded is True


def test_rejected_pending_and_missing_plan_candidates_cannot_apply() -> None:
    plan = make_plan()
    factory = RevisionCandidateFactory()
    command = _command(plan, "candidate")
    service = RevisionApplyService(CreativeBrief(10, "short film", "cinematic", "adult"))

    rejected = factory.create_noop_candidate(plan)
    reject_decision = replace(
        RevisionGuard().evaluate(plan, rejected, RevisionDiffBuilder().build_diff(plan, rejected)),
        decision="reject",
    )
    assert service.apply_revision(plan, rejected, reject_decision, command).stop_reason == "candidate_not_accepted"

    pending = factory.create_pending_director_candidate(plan)
    pending_decision = RevisionGuard().evaluate(plan, pending, RevisionDiffBuilder().build_diff(plan, pending))
    assert service.apply_revision(plan, pending, pending_decision, command).stop_reason == "candidate_not_accepted"

    missing = replace(rejected, revised_movie_plan=None)
    accepted = replace(reject_decision, decision="accept", accepted_candidate_id=missing.candidate_id)
    result = service.apply_revision(plan, missing, accepted, replace(command, candidate_id=missing.candidate_id))
    assert result.stop_reason == "candidate_has_no_movie_plan"


def test_source_mismatch_and_hard_revalidation_fail_closed() -> None:
    plan, candidate, decision = _accepted_candidate()
    service = RevisionApplyService(CreativeBrief(10, "short film", "cinematic", "adult"))
    mismatched = replace(_command(plan, candidate.candidate_id), source_movie_plan_id="other-plan")
    assert service.apply_revision(plan, candidate, decision, mismatched).stop_reason == "source_movie_plan_mismatch"

    invalid = replace(candidate, revised_movie_plan=replace(candidate.revised_movie_plan, visual_style=""))
    result = service.apply_revision(plan, invalid, decision, _command(plan, invalid.candidate_id))
    assert result.stop_reason == "revalidation_failed"
    assert result.revalidation_issues

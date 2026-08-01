from __future__ import annotations

from dataclasses import replace

from tests.test_v2_contracts import make_plan

from guided_story_agent.v2 import (
    CreativeRevisionRequest,
    RevisionCandidate,
    RevisionCandidateFactory,
    RevisionDiffBuilder,
    RevisionGuard,
    RevisionGuardPolicy,
)


def _request(
    *,
    target: str = "character_arc",
    preserve: tuple[str, ...] = (),
    avoid: tuple[str, ...] = (),
    source_codes: tuple[str, ...] = (),
) -> CreativeRevisionRequest:
    return CreativeRevisionRequest(
        "request-plan-1",
        "warning",
        target,
        "请导演重新审视现有创作决策。",
        preserve=preserve,
        avoid=avoid,
        source_suggestion_codes=source_codes,
    )


def test_pending_candidate_returns_pending_director() -> None:
    plan = make_plan()
    candidate = RevisionCandidateFactory().create_pending_director_candidate(plan)
    diff = RevisionDiffBuilder().build_diff(plan, candidate)

    decision = RevisionGuard().evaluate(plan, candidate, diff)

    assert decision.decision == "pending_director"
    assert decision.accepted_candidate_id is None


def test_provider_leakage_is_rejected() -> None:
    plan = make_plan()
    candidate = RevisionCandidateFactory().create_policy_violation_candidate(plan)
    diff = RevisionDiffBuilder().build_diff(plan, candidate)

    decision = RevisionGuard().evaluate(plan, candidate, diff)

    assert decision.decision == "reject"
    assert decision.rejected_candidate_id == candidate.candidate_id
    assert decision.diagnostics[0]["code"] == "provider_leakage"


def test_prompt_stuffing_is_rejected() -> None:
    plan = make_plan()
    candidate = RevisionCandidate(
        "candidate-prompt",
        plan.plan_id,
        revised_movie_plan=plan,
        candidate_type="fake_policy_violation",
        created_by="test",
        metadata={"unsafe_terms": ["best quality"]},
    )
    diff = RevisionDiffBuilder().build_diff(plan, candidate)

    decision = RevisionGuard().evaluate(plan, candidate, diff)

    assert decision.decision == "reject"
    assert decision.diagnostics[0]["code"] == "prompt_stuffing"


def test_preserve_violation_is_rejected() -> None:
    plan = make_plan()
    candidate = RevisionCandidate(
        "candidate-preserve",
        plan.plan_id,
        revised_movie_plan=replace(plan, story_plan=replace(plan.story_plan, characters=())),
        candidate_type="fake_targeted_patch",
        created_by="test",
    )
    request = _request(preserve=("保留现有主要人物",))
    diff = RevisionDiffBuilder().build_diff(plan, candidate, (request,))

    decision = RevisionGuard().evaluate(plan, candidate, diff, requests=(request,))

    assert decision.decision == "reject"
    assert decision.diagnostics[0]["code"] == "main_character_removed"


def test_avoid_violation_is_rejected() -> None:
    plan = make_plan()
    new_character = replace(plan.story_plan.characters[0], character_id="new", name="新角色")
    candidate = RevisionCandidate(
        "candidate-avoid",
        plan.plan_id,
        revised_movie_plan=replace(
            plan,
            story_plan=replace(plan.story_plan, characters=plan.story_plan.characters + (new_character,)),
        ),
        candidate_type="fake_targeted_patch",
        created_by="test",
    )
    request = _request(avoid=("不新增主要人物",))
    diff = RevisionDiffBuilder().build_diff(plan, candidate, (request,))

    decision = RevisionGuard().evaluate(plan, candidate, diff, requests=(request,))

    assert decision.decision == "reject"
    assert decision.diagnostics[0]["code"] == "avoid_violation"


def test_hard_validation_issue_is_rejected() -> None:
    plan = make_plan()
    candidate = RevisionCandidateFactory().create_noop_candidate(plan)
    diff = RevisionDiffBuilder().build_diff(plan, candidate)

    decision = RevisionGuard().evaluate(
        plan,
        candidate,
        diff,
        validation_issues=({"code": "invalid_plan", "severity": "error"},),
    )

    assert decision.decision == "reject"
    assert decision.diagnostics[0]["code"] == "hard_validation_issue"


def test_missing_target_response_is_rejected_by_default_policy() -> None:
    plan = make_plan()
    candidate = RevisionCandidateFactory().create_noop_candidate(plan)
    request = _request(target="character_arc", source_codes=("character_arc_flat",))
    diff = RevisionDiffBuilder().build_diff(plan, candidate, (request,))

    decision = RevisionGuard().evaluate(plan, candidate, diff, requests=(request,))

    assert decision.decision == "reject"
    assert decision.diagnostics[0]["code"] == "target_not_addressed"


def test_targeted_candidate_can_be_accepted_without_mutating_original() -> None:
    plan = make_plan()
    before = plan
    candidate = RevisionCandidate(
        "candidate-targeted",
        plan.plan_id,
        revised_movie_plan=replace(
            plan,
            story_plan=replace(
                plan.story_plan,
                story_beats=plan.story_plan.story_beats + ("承担选择后果",),
            ),
        ),
        candidate_type="fake_targeted_patch",
        created_by="test",
    )
    request = _request(target="character_arc", source_codes=("character_arc_flat",))
    diff = RevisionDiffBuilder().build_diff(plan, candidate, (request,))

    decision = RevisionGuard().evaluate(plan, candidate, diff, requests=(request,))

    assert decision.decision in {"accept", "accept_with_warning"}
    assert decision.accepted_candidate_id == candidate.candidate_id
    assert plan == before


def test_resolution_and_conflict_changes_are_blocked_by_default_policy() -> None:
    plan = make_plan()
    candidate = RevisionCandidate(
        "candidate-core-change",
        plan.plan_id,
        revised_movie_plan=replace(
            plan,
            story_plan=replace(plan.story_plan, conflict="changed", resolution="changed"),
        ),
        candidate_type="fake_targeted_patch",
        created_by="test",
    )
    diff = RevisionDiffBuilder().build_diff(plan, candidate)

    decision = RevisionGuard().evaluate(plan, candidate, diff)

    assert decision.decision == "reject"
    assert decision.diagnostics[0]["code"] == "core_conflict_change"


def test_policy_can_explicitly_allow_conflict_and_resolution_changes() -> None:
    plan = make_plan()
    candidate = RevisionCandidate(
        "candidate-authorized",
        plan.plan_id,
        revised_movie_plan=replace(
            plan,
            story_plan=replace(plan.story_plan, conflict="changed", resolution="changed"),
        ),
        candidate_type="fake_targeted_patch",
        created_by="test",
        metadata={"addressed_targets": ["climax_conflict"]},
    )
    request = _request(target="climax_conflict", source_codes=("show_cost_in_resolution",))
    diff = RevisionDiffBuilder().build_diff(plan, candidate, (request,))
    policy = RevisionGuardPolicy(allow_core_conflict_change=True, allow_resolution_change=True)

    decision = RevisionGuard(policy=policy).evaluate(plan, candidate, diff, requests=(request,))

    assert decision.decision in {"accept", "accept_with_warning"}

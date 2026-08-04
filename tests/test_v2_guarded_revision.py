from __future__ import annotations

from dataclasses import replace

from tests.test_v2_contracts import make_plan

from guided_story_agent.v2 import (
    CreativeBrief,
    CreativeRevisionRequest,
    DirectorAgentRevisionAdapter,
    DirectorRevisionContext,
    RevisionCandidateFactory,
    RuleBasedDirectorRevisionAdapter,
    run_director_revision_guarded,
)


def _request() -> CreativeRevisionRequest:
    return CreativeRevisionRequest(
        "request-plan-1",
        "warning",
        "climax_conflict",
        "请强化已有高潮中的冲突代价。",
        preserve=("保留现有主要人物",),
        avoid=("不新增主要人物",),
    )


def test_pending_candidate_runs_diff_and_guard() -> None:
    plan = make_plan()
    requests = (_request(),)
    context = DirectorRevisionContext.from_requests(plan, requests)

    result = run_director_revision_guarded(
        plan,
        requests,
        RuleBasedDirectorRevisionAdapter(),
        context,
        brief=CreativeBrief(10, "short film", "cinematic", "adult"),
    )

    assert result.decision is not None
    assert result.decision.decision == "pending_director"
    assert result.diff is not None
    assert result.succeeded is False


def test_noop_candidate_is_not_implicitly_accepted_when_target_is_missing() -> None:
    plan = make_plan()
    requests = (_request(),)
    context = DirectorRevisionContext.from_requests(plan, requests)
    result = run_director_revision_guarded(
        plan,
        requests,
        RuleBasedDirectorRevisionAdapter("noop_candidate"),
        context,
        brief=CreativeBrief(10, "short film", "cinematic", "adult"),
    )

    assert result.decision is not None
    assert result.decision.decision == "reject"
    assert result.candidate is not None
    assert plan == make_plan()


def test_policy_violation_candidate_is_rejected_before_any_apply() -> None:
    plan = make_plan()
    context = DirectorRevisionContext.from_requests(plan, (_request(),))
    result = run_director_revision_guarded(
        plan,
        (_request(),),
        RuleBasedDirectorRevisionAdapter("policy_violation_candidate"),
        context,
        brief=CreativeBrief(10, "short film", "cinematic", "adult"),
    )

    assert result.decision is not None
    assert result.decision.decision == "reject"
    assert result.decision.diagnostics[0]["code"] in {"provider_leakage", "prompt_stuffing"}
    assert result.candidate is not None
    assert result.candidate.revised_movie_plan is not plan


def test_valid_external_candidate_can_be_accepted_but_original_is_unchanged() -> None:
    class FakeDirector:
        def revise_movie_plan(self, brief, plan, feedback):
            del brief, feedback
            return replace(
                plan,
                director_plan=replace(
                    plan.director_plan,
                    climax_emphasis="让已有冲突付出可见代价",
                ),
            )

    plan = make_plan()
    request = _request()
    context = DirectorRevisionContext.from_requests(plan, (request,))
    result = run_director_revision_guarded(
        plan,
        (request,),
        DirectorAgentRevisionAdapter(
            FakeDirector(),
            brief=CreativeBrief(10, "short film", "cinematic", "adult"),
        ),
        context,
        brief=CreativeBrief(10, "short film", "cinematic", "adult"),
    )

    assert result.decision is not None
    assert result.decision.decision in {"accept", "accept_with_warning"}
    assert plan == make_plan()
    assert result.to_dict()["decision"]["decision"] == result.decision.decision


def test_hard_validation_issue_is_fail_closed() -> None:
    plan = make_plan()
    context = DirectorRevisionContext.from_requests(plan)
    candidate_adapter = RuleBasedDirectorRevisionAdapter("noop_candidate")
    result = run_director_revision_guarded(
        plan,
        (),
        candidate_adapter,
        context,
        validation_issues=({"code": "invalid_plan", "severity": "error"},),
        brief=CreativeBrief(10, "short film", "cinematic", "adult"),
    )

    assert result.decision is not None
    assert result.decision.decision == "reject"


def test_factory_pending_candidate_keeps_provider_state_out_of_revision_result() -> None:
    plan = make_plan()
    candidate = RevisionCandidateFactory().create_pending_director_candidate(plan)
    assert candidate.to_dict()["revised_movie_plan"] is None

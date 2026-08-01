from __future__ import annotations

from dataclasses import replace

from tests.test_v2_contracts import make_plan

from guided_story_agent.v2 import (
    CreativeBrief,
    CreativeRevisionRequest,
    DirectorAgentRevisionAdapter,
    DirectorRevisionContext,
    OpenAIDirectorRevisionAdapter,
    RevisionCandidate,
    RuleBasedDirectorRevisionAdapter,
    build_director_revision_prompt,
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


def test_rule_based_default_returns_pending_director_candidate() -> None:
    plan = make_plan()
    requests = (_request(),)
    context = DirectorRevisionContext.from_requests(plan, requests)

    result = RuleBasedDirectorRevisionAdapter().revise(plan, requests, context)

    assert result.status == "pending_director"
    assert result.candidate is not None
    assert result.candidate.revised_movie_plan is None
    assert result.candidate.candidate_type == "external_director_candidate"


def test_rule_based_noop_is_candidate_only_and_does_not_replace_plan() -> None:
    plan = make_plan()
    context = DirectorRevisionContext.from_requests(plan, (_request(),))

    result = RuleBasedDirectorRevisionAdapter("noop_candidate").revise(
        plan, (_request(),), context
    )

    assert result.candidate is not None
    assert result.candidate.revised_movie_plan == plan
    assert plan == make_plan()


def test_policy_violation_requires_explicit_fixture_mode() -> None:
    plan = make_plan()
    context = DirectorRevisionContext.from_requests(plan, (_request(),))

    result = RuleBasedDirectorRevisionAdapter("policy_violation_candidate").revise(
        plan, (_request(),), context
    )

    assert result.status == "candidate_created"
    assert result.candidate is not None
    assert result.candidate.candidate_type == "fake_policy_violation"


def test_context_and_adapter_result_are_serializable() -> None:
    plan = make_plan()
    request = _request()
    context = DirectorRevisionContext.from_requests(plan, (request,))
    result = RuleBasedDirectorRevisionAdapter().revise(plan, (request,), context)

    assert context.to_dict()["request_ids"] == [request.request_id]
    assert result.to_dict()["candidate"]["candidate_id"]


def test_prompt_packs_constraints_without_provider_payload_or_path() -> None:
    plan = make_plan()
    request = _request()
    context = DirectorRevisionContext.from_requests(plan, (request,))

    prompt = build_director_revision_prompt(plan, (request,), context)

    assert "保留现有主要人物" in prompt
    assert "不新增主要人物" in prompt
    assert "candidate-only" in prompt
    assert "provider_payload" not in prompt.lower()
    assert "api_key" not in prompt.lower()
    assert "c:\\" not in prompt.lower()


def test_director_agent_adapter_wraps_external_plan_without_session_or_provider() -> None:
    class FakeDirector:
        def __init__(self) -> None:
            self.calls = 0

        def revise_movie_plan(self, brief, plan, feedback):
            del brief
            self.calls += 1
            assert "candidate-only" in feedback
            return replace(
                plan,
                director_plan=replace(
                    plan.director_plan,
                    climax_emphasis="让已有冲突付出可见代价",
                ),
            )

    director = FakeDirector()
    plan = make_plan()
    request = _request()
    context = DirectorRevisionContext.from_requests(plan, (request,))
    result = DirectorAgentRevisionAdapter(
        director,
        brief=CreativeBrief(10, "short film", "cinematic", "adult"),
    ).revise(plan, (request,), context)

    assert director.calls == 1
    assert result.status == "candidate_created"
    assert result.candidate is not None
    assert result.candidate.candidate_type == "external_director_candidate"
    assert result.candidate.revised_movie_plan != plan


def test_openai_named_adapter_is_the_same_candidate_boundary() -> None:
    class FakeDirector:
        def revise_movie_plan(self, brief, plan, feedback):
            del brief, feedback
            return plan

    plan = make_plan()
    context = DirectorRevisionContext.from_requests(plan)
    result = OpenAIDirectorRevisionAdapter(FakeDirector()).revise(plan, (), context)

    assert result.adapter_name == "openai_director_revision"
    assert isinstance(result.candidate, RevisionCandidate)


def test_adapter_protects_original_from_in_place_director_mutation() -> None:
    class MutatingDirector:
        def revise_movie_plan(self, brief, plan, feedback):
            del brief, feedback
            object.__setattr__(plan, "visual_style", "mutated only in the candidate input")
            return plan

    plan = make_plan()
    original_style = plan.visual_style
    context = DirectorRevisionContext.from_requests(plan)
    result = DirectorAgentRevisionAdapter(MutatingDirector()).revise(plan, (), context)

    assert plan.visual_style == original_style
    assert result.candidate is not None
    assert result.candidate.revised_movie_plan is not plan

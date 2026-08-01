from __future__ import annotations

import json

from tests.test_v2_creative_optimizer import _result
from tests.test_v2_contracts import make_plan

from guided_story_agent.v2 import (
    CreativeOptimizer,
    CreativeRevisionRequest,
    RevisionPolicy,
    RevisionRequestBuilder,
    RuleBasedDirectorRevisionLoop,
)


def test_revision_request_builder_combines_related_creative_suggestions() -> None:
    plan = make_plan()
    optimizer_result = CreativeOptimizer().optimize(
        plan,
        (
            _result(
                "conflict_progression",
                "conflict_not_escalating",
                "resolution_without_cost",
                "climax_resolves_no_conflict",
            ),
        ),
    )
    result = RevisionRequestBuilder().build(plan, optimizer_result)

    assert result.stop_reason == "pending_director_revision"
    assert len(result.requests) == 1
    request = result.requests[0]
    assert request.requires_director
    assert not request.auto_apply_allowed
    assert "高潮" in request.instruction
    assert "不新增主要人物" in request.avoid
    assert json.loads(json.dumps(result.to_dict(), ensure_ascii=False))["requests"]


def test_deferred_suggestion_does_not_create_director_request() -> None:
    plan = make_plan()
    optimizer_result = CreativeOptimizer().optimize(
        plan,
        (_result("emotion_flow", "ending_tone_missing"),),
    )
    result = RevisionRequestBuilder().build(plan, optimizer_result)

    assert result.requests == ()
    assert result.deferred_suggestions
    assert result.stop_reason == "policy_deferred"


def test_rule_based_loop_records_request_without_fabricating_plan() -> None:
    request = CreativeRevisionRequest(
        "revision-plan-1-1",
        "warning",
        "climax_conflict",
        "请导演强化已有高潮。",
        preserve=("保留现有主要人物",),
        avoid=("不新增主要人物",),
    )
    plan = make_plan()
    result = RuleBasedDirectorRevisionLoop().run(
        plan,
        creative_revision_requests=(request,),
    )

    assert not result.accepted
    assert result.revised_movie_plan is None
    assert result.stop_reason == "pending_director_revision"
    assert result.revision_history[1]["status"] == "pending_director"


def test_revision_builder_fail_closed_on_validation_error() -> None:
    result = RevisionRequestBuilder().build(
        make_plan(),
        CreativeOptimizer().optimize(make_plan(), ()),
        validation_issues=({"code": "missing_conflict", "severity": "error"},),
    )

    assert not result.succeeded
    assert result.stop_reason == "hard_validation_error"
    assert result.requests == ()


def test_revision_policy_classifies_hard_warning_and_deferred() -> None:
    plan = make_plan()
    optimizer_result = CreativeOptimizer().optimize(
        plan,
        (
            _result("conflict_progression", "conflict_missing"),
            _result("emotion_flow", "ending_tone_missing"),
        ),
    )
    policy = RevisionPolicy()
    assert {policy.classify(item) for item in optimizer_result.suggestions} == {"hard", "deferred"}

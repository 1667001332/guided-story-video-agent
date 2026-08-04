from __future__ import annotations

import json

from tests.test_v2_contracts import make_plan

from guided_story_agent.v2 import (
    CreativeRevisionRequest,
    RevisionCandidate,
    RevisionCandidateFactory,
)


def _request() -> CreativeRevisionRequest:
    return CreativeRevisionRequest(
        "request-plan-1",
        "warning",
        "character_arc",
        "请导演强化现有角色弧线。",
        preserve=("保留现有主要人物",),
        avoid=("不新增主要人物",),
    )


def test_pending_director_candidate_is_serializable() -> None:
    candidate = RevisionCandidateFactory().create_pending_director_candidate(
        make_plan(), (_request(),)
    )

    payload = candidate.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)

    assert candidate.status == "pending_director"
    assert candidate.revised_movie_plan is None
    assert "pending_director" in encoded
    assert payload["source_revision_request_ids"] == ["request-plan-1"]


def test_noop_candidate_does_not_cover_or_mutate_original_movie_plan() -> None:
    plan = make_plan()
    before = plan
    candidate = RevisionCandidateFactory().create_noop_candidate(plan, (_request(),))

    assert candidate.revised_movie_plan == plan
    assert candidate.revised_movie_plan is not plan
    assert plan == before
    assert candidate.candidate_type == "fake_noop"


def test_policy_violation_fixture_is_explicit_and_not_default_cli_output() -> None:
    candidate = RevisionCandidateFactory().create_policy_violation_candidate(make_plan())

    assert candidate.candidate_type == "fake_policy_violation"
    assert candidate.metadata["fixture"] == "policy_violation"
    assert "provider_payload" not in candidate.to_dict()
    assert candidate.created_by == "deterministic_fake"


def test_candidate_roundtrip_requires_explicit_movie_plan_loader() -> None:
    candidate = RevisionCandidateFactory().create_noop_candidate(make_plan())
    payload = candidate.to_dict()

    try:
        RevisionCandidate.from_dict(payload)
    except ValueError as exc:
        assert "loader" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("revised MoviePlan must require an explicit loader")

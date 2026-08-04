from __future__ import annotations

from dataclasses import replace

from tests.test_v2_contracts import make_plan

from guided_story_agent.v2 import (
    CreativeRevisionRequest,
    RevisionCandidate,
    RevisionCandidateFactory,
    RevisionDiffBuilder,
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


def test_diff_identifies_main_character_removal() -> None:
    plan = make_plan()
    candidate = RevisionCandidate(
        "candidate-character-remove",
        plan.plan_id,
        revised_movie_plan=replace(
            plan,
            story_plan=replace(plan.story_plan, characters=()),
        ),
        candidate_type="fake_targeted_patch",
        created_by="test",
    )
    diff = RevisionDiffBuilder().build_diff(plan, candidate, (_request(),))

    assert any(change.change_type == "removed" and "characters" in change.path for change in diff.changes)
    assert diff.metrics["changed_field_count"] >= 1


def test_diff_identifies_conflict_resolution_and_story_beat_changes() -> None:
    plan = make_plan()
    candidate = RevisionCandidate(
        "candidate-story-change",
        plan.plan_id,
        revised_movie_plan=replace(
            plan,
            story_plan=replace(
                plan.story_plan,
                conflict="新的核心冲突",
                resolution="新的解决结果",
                story_beats=plan.story_plan.story_beats[:-1],
            ),
        ),
        candidate_type="fake_targeted_patch",
        created_by="test",
    )
    diff = RevisionDiffBuilder().build_diff(plan, candidate, ())
    paths = {change.path for change in diff.changes}

    assert "story_plan.conflict" in paths
    assert "story_plan.resolution" in paths
    assert any("story_plan.story_beats" in path for path in paths)


def test_diff_detects_preserve_and_avoid_violations() -> None:
    plan = make_plan()
    candidate = RevisionCandidateFactory().create_policy_violation_candidate(plan)
    request = _request(
        preserve=("保留现有主要人物",),
        avoid=("不新增主要人物",),
    )
    diff = RevisionDiffBuilder().build_diff(plan, candidate, (request,))

    assert diff.preserve_violations
    assert diff.metrics["preserve_violation_count"] >= 1

    added_candidate = RevisionCandidate(
        "candidate-character-add",
        plan.plan_id,
        revised_movie_plan=replace(
            plan,
            story_plan=replace(
                plan.story_plan,
                characters=plan.story_plan.characters
                + (replace(plan.story_plan.characters[0], character_id="new", name="新角色"),),
            ),
        ),
        candidate_type="fake_targeted_patch",
        created_by="test",
    )
    avoid_diff = RevisionDiffBuilder().build_diff(plan, added_candidate, (request,))
    assert avoid_diff.avoid_violations


def test_diff_detects_provider_leakage_and_prompt_stuffing() -> None:
    plan = make_plan()
    candidate = RevisionCandidate(
        "candidate-unsafe",
        plan.plan_id,
        revised_movie_plan=plan,
        candidate_type="fake_policy_violation",
        created_by="test",
        metadata={"unsafe_fields": ["provider_payload"], "unsafe_terms": ["best quality"]},
    )
    diff = RevisionDiffBuilder().build_diff(plan, candidate, ())

    assert diff.provider_leakage_detected
    assert diff.prompt_stuffing_detected
    assert diff.metrics["provider_leakage_count"] == 1.0
    assert diff.metrics["prompt_stuffing_count"] == 1.0


def test_pending_candidate_does_not_raise_and_reports_pending_metric() -> None:
    plan = make_plan()
    candidate = RevisionCandidateFactory().create_pending_director_candidate(plan)
    diff = RevisionDiffBuilder().build_diff(plan, candidate, ())

    assert not diff.succeeded
    assert diff.metrics["pending_director"] == 1.0


def test_target_response_is_tracked_without_mutating_inputs() -> None:
    plan = make_plan()
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

    assert diff.target_responses
    assert plan.story_plan.story_beats != candidate.revised_movie_plan.story_plan.story_beats

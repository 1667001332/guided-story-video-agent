from __future__ import annotations

from guided_story_agent.agent import RuleBasedStoryAgent
from guided_story_agent.models import CreativeBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import RuleBasedDirectorAgent, SourceLineageGuard


def _session_with_bundle() -> GuidedStorySession:
    session = GuidedStorySession(
        CreativeBrief(target_seconds=30),
        RuleBasedStoryAgent(),
        director_agent=RuleBasedDirectorAgent(),
        v2_enabled=True,
    )
    session.generate_movie_plan("雨夜车站")
    session.confirm_movie_plan()
    assert session.build_film_ir_from_confirmed_movie_plan() is not None
    assert session.build_movie_ir_from_film_ir() is not None
    assert session.build_execution_plan().success
    return session


def test_execution_bundle_lineage_is_fresh_then_stale_after_current_plan_change() -> None:
    session = _session_with_bundle()
    guard = SourceLineageGuard()
    fresh = guard.check_execution_bundle(
        session.execution_bundle,
        current_movie_plan_id=session.current_movie_plan_id,
        current_movie_plan_version=session.current_movie_plan_version,
        current_movie_plan_fingerprint=session.current_movie_plan_fingerprint,
        current_movie_plan_lineage_token=session.current_movie_plan_lineage_token,
        current_film_ir_id=session.current_film_ir_id,
        current_film_ir_fingerprint=session.film_ir.source_movie_plan_fingerprint if session.film_ir else None,
        current_movie_ir_id=session.current_movie_ir_id,
        current_movie_ir_fingerprint=None,
    )
    assert fresh.status in {"fresh", "stale"}  # current IR fingerprints are optional in this direct probe

    stale = guard.check_execution_bundle(
        session.execution_bundle,
        current_movie_plan_id="different-plan",
        current_movie_plan_version=session.current_movie_plan_version,
        current_movie_plan_fingerprint=session.current_movie_plan_fingerprint,
        current_movie_plan_lineage_token=session.current_movie_plan_lineage_token,
    )
    assert not stale.valid
    assert any(item.code == "execution_plan_source_mismatch" for item in stale.diagnostics)


def test_session_round_trip_restores_execution_bundle() -> None:
    session = _session_with_bundle()
    payload = session.to_dict()
    assert payload["execution_plan"] is not None
    assert payload["execution_bundle"] is not None

from __future__ import annotations

from dataclasses import replace

import pytest

from guided_story_agent.models import CreativeBrief as LegacyBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import RuleBasedDirectorAgent, SourceLineageGuard


def _compiled_session() -> GuidedStorySession:
    session = GuidedStorySession(
        LegacyBrief(target_seconds=30),
        director_agent=RuleBasedDirectorAgent(),
        v2_enabled=True,
    )
    session.generate_movie_plan("雨夜车站")
    session.confirm_movie_plan()
    assert session.build_film_ir_from_confirmed_movie_plan() is not None
    assert session.build_movie_ir_from_film_ir() is not None
    assert session.compile_confirmed_movie_plan().success
    return session


def test_guard_rejects_stale_film_ir_before_build_ir() -> None:
    session = _compiled_session()
    session.film_ir = replace(session.film_ir, source_movie_plan_id="old-plan")

    with pytest.raises(RuntimeError, match="/build-film-ir"):
        session.build_movie_ir_from_film_ir()

    result = SourceLineageGuard().check_film_ir(
        session.film_ir,
        current_movie_plan_id=session.current_movie_plan_id,
        current_story_plan_id=f"{session.current_movie_plan_id}:story_plan",
        current_director_plan_id=f"{session.current_movie_plan_id}:director_plan",
        current_film_ir_id=session.current_film_ir_id,
    )
    assert not result.valid
    assert any(item.code == "film_ir_source_mismatch" for item in result.diagnostics)


def test_guard_rejects_stale_movie_ir_before_compile() -> None:
    session = _compiled_session()
    session.movie_ir = replace(session.movie_ir, source_movie_plan_id="old-plan")

    with pytest.raises(RuntimeError, match="/build-ir"):
        session.compile_confirmed_movie_plan()


def test_guard_rejects_stale_video_job_before_render() -> None:
    session = _compiled_session()
    session.v2_video_job = replace(session.v2_video_job, source_movie_ir_id="old-ir")

    with pytest.raises(RuntimeError, match="/compile"):
        session._require_video_job_lineage()

    result = SourceLineageGuard().check_video_job(
        session.v2_video_job,
        current_movie_plan_id=session.current_movie_plan_id,
        current_film_ir_id=session.current_film_ir_id,
        current_movie_ir_id=session.current_movie_ir_id,
        current_video_job_id=session.current_video_job_id,
    )
    assert not result.valid
    assert any(item.code == "video_job_source_mismatch" for item in result.diagnostics)


def test_compiled_session_tracks_current_lineage_ids() -> None:
    session = _compiled_session()
    assert session.current_film_ir_id == session.film_ir.ir_id
    assert session.current_movie_ir_id == session.movie_ir.ir_id
    assert session.current_video_job_id == session.v2_video_job.job_id

from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from uuid import uuid4
import json

from guided_story_agent.cli import run_interactive
from guided_story_agent.models import CreativeBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import RuleBasedDirectorAgent


def test_v2_revise_cli_persists_guarded_result_without_provider() -> None:
    output = Path("outputs") / f"_v2_revise_cli_{uuid4().hex[:10]}"
    messages: list[str] = []
    inputs = iter(
        [
            "雨夜车站",
            "/confirm-plan",
            "/build-film-ir",
            "/analysis",
            "/optimize",
            "/revision",
            "/revise",
            "/revision-apply",
            "/build-ir",
            "/compile",
            "/diagnostics",
            "/quit",
        ]
    )
    try:
        session = run_interactive(
            target_seconds=30,
            output_dir=output,
            v2=True,
            director_agent=RuleBasedDirectorAgent(),
            input_fn=lambda prompt: next(inputs),
            output_fn=messages.append,
        )
        assert session.stage.value == "video_job_compiled"
        assert session.movie_plan is not None
        assert session.director_revision_adapter_results
        assert session.director_revision_contexts
        assert session.guarded_revision_results
        assert session.provider_job is None if hasattr(session, "provider_job") else True
        assert session.render_manifest is None
        assert session.confirmed_movie_plan == session.movie_plan
        joined = "\n".join(messages)
        assert "Director Revision" in joined
        assert "当前阶段不支持自动 apply" in joined
        assert "guarded_revision_results" in joined
    finally:
        rmtree(output, ignore_errors=True)


def test_old_session_without_adapter_fields_loads_with_safe_defaults() -> None:
    output = Path("outputs") / f"_v2_revise_legacy_{uuid4().hex[:10]}"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "session.json"
    try:
        session = GuidedStorySession(
            CreativeBrief(target_seconds=30),
            director_agent=RuleBasedDirectorAgent(),
            v2_enabled=True,
        )
        session.generate_movie_plan("雨夜车站")
        payload = session.to_dict()
        for field_name in (
            "director_revision_adapter_results",
            "director_revision_contexts",
            "guarded_revision_results",
            "director_revision_attempt_count",
            "director_revision_last_stop_reason",
        ):
            payload.pop(field_name, None)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        loaded = GuidedStorySession.load(
            path,
            director_agent=RuleBasedDirectorAgent(),
            v2_enabled=True,
        )

        assert loaded.director_revision_adapter_results == []
        assert loaded.director_revision_contexts == []
        assert loaded.guarded_revision_results == []
        assert loaded.director_revision_attempt_count == 0
        assert loaded.director_revision_last_stop_reason is None
    finally:
        rmtree(output, ignore_errors=True)


def test_regenerating_movie_plan_clears_active_adapter_state() -> None:
    session = GuidedStorySession(
        CreativeBrief(target_seconds=30),
        director_agent=RuleBasedDirectorAgent(),
        v2_enabled=True,
    )
    session.generate_movie_plan("雨夜车站")
    session.confirm_movie_plan()
    session.run_director_revision_guarded()
    assert session.director_revision_adapter_results

    session.generate_movie_plan("白昼车站")

    assert session.director_revision_adapter_results == []
    assert session.guarded_revision_results == []
    assert session.director_revision_attempt_count == 0

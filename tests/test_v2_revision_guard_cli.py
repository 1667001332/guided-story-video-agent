from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

from guided_story_agent.agent import RuleBasedStoryAgent
from guided_story_agent.cli import run_interactive
from guided_story_agent.models import CreativeBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import RuleBasedDirectorAgent


def test_old_session_without_guard_fields_loads_with_defaults() -> None:
    output = Path("outputs") / f"_v2_4d1_legacy_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=True)
    try:
        session = GuidedStorySession(CreativeBrief(target_seconds=30), RuleBasedStoryAgent())
        payload = session.to_dict()
        for field in (
            "revision_candidates",
            "revision_diffs",
            "revision_decisions",
            "revision_guard_diagnostics",
            "revision_active_candidate_id",
            "revision_accepted_movie_plan_id",
            "revision_rollback_movie_plan_id",
        ):
            payload.pop(field, None)
        path = output / "legacy.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        loaded = GuidedStorySession.load(path)

        assert loaded.revision_candidates == []
        assert loaded.revision_diffs == []
        assert loaded.revision_decisions == []
        assert loaded.revision_guard_diagnostics == []
        assert loaded.revision_active_candidate_id is None
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_session_guard_without_candidate_is_pending_and_serializable() -> None:
    session = GuidedStorySession(
        CreativeBrief(target_seconds=30),
        RuleBasedStoryAgent(),
        director_agent=RuleBasedDirectorAgent(),
        v2_enabled=True,
    )
    session.generate_movie_plan("雨夜车站")
    session.confirm_movie_plan()

    decision = session.run_revision_guard()
    payload = session.to_dict()

    assert decision["decision"] == "pending_director"
    assert session.revision_candidates == []
    assert len(session.revision_diffs) == 1
    assert len(session.revision_decisions) == 1
    assert payload["revision_active_candidate_id"] is None
    json.dumps(payload, ensure_ascii=False)


def test_offline_cli_guard_path_reaches_compiled_stage_without_mp4() -> None:
    output = Path("outputs") / f"_v2_4d1_cli_{uuid4().hex[:8]}"
    inputs = iter(
        [
            "雨夜车站",
            "/confirm-plan",
            "/build-film-ir",
            "/analysis",
            "/optimize",
            "/revision",
            "/revision-guard",
            "/build-ir",
            "/compile",
            "/diagnostics",
            "/render",
            "/quit",
        ]
    )
    messages: list[str] = []
    try:
        session = run_interactive(
            target_seconds=30,
            output_dir=output,
            v2=True,
            director_agent=RuleBasedDirectorAgent(),
            input_fn=lambda prompt: next(inputs),
            output_fn=messages.append,
        )
        joined = "\n".join(messages)

        assert session.stage.value == "video_job_compiled"
        assert session.revision_decisions[-1]["decision"] == "pending_director"
        assert session.v2_video_job is not None
        assert session.render_manifest is None
        assert not (output / "video").exists()
        assert "RevisionGuard：decision=pending_director" in joined
        assert "revision_candidates: 0" in joined
        assert "revision_diffs: 1" in joined
        assert "revision_decisions: 1" in joined
        assert "provider_job" not in joined.lower()
    finally:
        shutil.rmtree(output, ignore_errors=True)

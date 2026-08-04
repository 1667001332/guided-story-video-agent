from __future__ import annotations

import json
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from guided_story_agent.cli import run_interactive
from guided_story_agent.models import CreativeBrief as LegacyBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import RuleBasedDirectorAgent


def test_revision_apply_cli_requires_explicit_confirmation_and_does_not_auto_apply() -> None:
    output = Path("outputs") / f"_v2_revision_apply_cli_{uuid4().hex[:10]}"
    messages: list[str] = []
    inputs = iter(
        [
            "雨夜车站",
            "/confirm-plan",
            "/revise",
            "/revision-apply",
            "/diagnostics",
            "/revision-rollback",
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
        assert session.stage.value == "movie_plan_confirmed"
        assert session.movie_plan_version_history == []
        joined = "\n".join(messages)
        assert "没有 accepted candidate" in joined
        assert "没有可回滚的 MoviePlan 版本" in joined
        assert "movie_plan_version_history: 0" in joined
        assert session.render_manifest is None
    finally:
        rmtree(output, ignore_errors=True)


def test_old_session_without_phase_4d3_fields_loads_with_safe_defaults() -> None:
    session = GuidedStorySession(
        LegacyBrief(target_seconds=30),
        director_agent=RuleBasedDirectorAgent(),
        v2_enabled=True,
    )
    session.generate_movie_plan("雨夜车站")
    payload = session.to_dict()
    for field_name in (
        "movie_plan_version_history",
        "revision_apply_history",
        "revision_rollback_history",
        "revision_apply_results",
        "revision_rollback_results",
        "current_movie_plan_id",
        "previous_movie_plan_id",
        "stale_artifacts",
    ):
        payload.pop(field_name, None)

    output = Path("outputs") / f"_v2_legacy_session_{uuid4().hex[:10]}"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "session.json"
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        loaded = GuidedStorySession.load(
            path,
            director_agent=RuleBasedDirectorAgent(),
            v2_enabled=True,
        )

        assert loaded.current_movie_plan_id == loaded.movie_plan.plan_id
        assert loaded.movie_plan_version_history == []
        assert loaded.revision_apply_history == []
        assert loaded.revision_rollback_history == []
        assert loaded.revision_apply_results == []
        assert loaded.revision_rollback_results == []
        assert loaded.stale_artifacts == []
    finally:
        rmtree(output, ignore_errors=True)

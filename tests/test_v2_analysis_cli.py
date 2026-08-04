from __future__ import annotations

import json
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from guided_story_agent.cli import run_interactive
from guided_story_agent.models import CreativeBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2.openai_director import RuleBasedDirectorAgent


def test_session_persists_creative_analysis_fields() -> None:
    output = Path("outputs") / f"_v2_analysis_session_{uuid4().hex[:8]}"
    try:
        session = GuidedStorySession(
            CreativeBrief(target_seconds=30),
            director_agent=RuleBasedDirectorAgent(),
            v2_enabled=True,
        )
        session.generate_movie_plan("雨夜车站")
        session.confirm_movie_plan()
        session.build_film_ir_from_confirmed_movie_plan()
        session.run_creative_analysis()
        payload = session.to_dict()

        assert payload["creative_analysis_results"]
        assert payload["creative_analysis_diagnostics"] is not None
        assert payload["creative_analysis_artifacts"]
        assert payload["creative_analysis_metrics"]
        assert json.loads(json.dumps(payload, ensure_ascii=False))["movie_plan"]
        assert session.movie_ir is None
        assert session.v2_video_job is None
    finally:
        rmtree(output, ignore_errors=True)


def test_old_session_without_analysis_fields_loads_with_defaults() -> None:
    output = Path("outputs") / f"_v2_analysis_legacy_{uuid4().hex[:8]}"
    try:
        session = GuidedStorySession(
            CreativeBrief(target_seconds=30),
            director_agent=RuleBasedDirectorAgent(),
            v2_enabled=True,
        )
        payload = session.to_dict()
        for key in (
            "creative_analysis_results",
            "creative_analysis_diagnostics",
            "creative_analysis_artifacts",
            "creative_analysis_metrics",
        ):
            payload.pop(key, None)
        path = output / "legacy.json"
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        loaded = GuidedStorySession.load(path, director_agent=RuleBasedDirectorAgent(), v2_enabled=True)

        assert loaded.creative_analysis_results == []
        assert loaded.creative_analysis_diagnostics == []
        assert loaded.creative_analysis_artifacts == []
        assert loaded.creative_analysis_metrics == {}
    finally:
        rmtree(output, ignore_errors=True)


def test_analysis_command_runs_offline_and_diagnostics_exposes_summary() -> None:
    output = Path("outputs") / f"_v2_analysis_cli_{uuid4().hex[:8]}"
    messages: list[str] = []
    inputs = iter(
        [
            "雨夜车站",
            "/confirm-plan",
            "/build-film-ir",
            "/analysis",
            "/build-ir",
            "/compile",
            "/diagnostics",
            "/render",
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

        joined = "\n".join(messages)
        assert session.stage.value == "video_job_compiled"
        assert session.creative_analysis_results
        assert session.creative_analysis_artifacts
        assert "Creative Analysis" in joined
        assert "creative_analysis_metrics" in joined
        assert "Provider execution belongs" in joined
        assert session.v2_video_job is not None
    finally:
        rmtree(output, ignore_errors=True)

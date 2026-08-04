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


def test_old_session_without_phase_4c_fields_loads_with_defaults() -> None:
    output = Path("outputs") / f"_v2_4c_legacy_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=True)
    try:
        session = GuidedStorySession(CreativeBrief(target_seconds=30), RuleBasedStoryAgent())
        payload = session.to_dict()
        for field in (
            "creative_optimizer_result",
            "creative_optimizer_suggestions",
            "creative_optimizer_candidates",
            "creative_optimizer_diagnostics",
            "creative_revision_requests",
            "creative_revision_request_history",
            "creative_revision_stop_reason",
        ):
            payload.pop(field, None)
        path = output / "legacy.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        loaded = GuidedStorySession.load(path)

        assert loaded.creative_optimizer_result is None
        assert loaded.creative_optimizer_suggestions == []
        assert loaded.creative_revision_requests == []
        assert loaded.creative_revision_stop_reason is None
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_offline_cli_runs_optimizer_and_revision_without_provider() -> None:
    output = Path("outputs") / f"_v2_4c_cli_{uuid4().hex[:8]}"
    inputs = iter(
        [
            "雨夜车站",
            "/confirm-plan",
            "/build-film-ir",
            "/analysis",
            "/optimize",
            "/revision",
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
        assert session.film_ir is not None
        assert session.movie_ir is not None
        assert session.v2_video_job is not None
        assert session.creative_optimizer_result is not None
        assert session.creative_revision_request_history
        assert session.v2_video_job is not None
        assert session.render_manifest is None
        assert "Creative Optimizer 已完成" in joined
        assert "Director Revision Requests" in joined
        assert "creative_optimizer_suggestions:" in joined
        assert "provider_job" not in joined.lower()
        loaded = GuidedStorySession.load(output / "session.json")
        assert loaded.creative_optimizer_result == session.creative_optimizer_result
        assert loaded.creative_revision_requests == session.creative_revision_requests
        assert loaded.creative_revision_request_history == session.creative_revision_request_history
    finally:
        shutil.rmtree(output, ignore_errors=True)

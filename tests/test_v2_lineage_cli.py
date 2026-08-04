from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from guided_story_agent.cli import run_interactive
from guided_story_agent.v2 import RuleBasedDirectorAgent


def test_cli_diagnostics_prints_lineage_and_current_ids() -> None:
    output = Path("outputs") / f"_v2_lineage_cli_{uuid4().hex[:10]}"
    messages: list[str] = []
    inputs = iter(
        [
            "雨夜车站",
            "/confirm-plan",
            "/build-film-ir",
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
        assert "FilmIR: fresh" in joined
        assert "MovieIR: fresh" in joined
        assert "VideoJob: fresh" in joined
        assert "current_film_ir_id:" in joined
        assert "current_movie_ir_id:" in joined
        assert "current_video_job_id:" in joined
        assert "source_lineage_diagnostics: 0" in joined
        assert "V2 render is not connected" in joined
        assert session.provider_job is None if hasattr(session, "provider_job") else True
        assert session.render_manifest is None
    finally:
        rmtree(output, ignore_errors=True)

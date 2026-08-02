from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from guided_story_agent.cli import run_interactive
from guided_story_agent.v2 import RuleBasedDirectorAgent


def test_execution_plan_cli_is_offline_and_keeps_provider_outputs_null() -> None:
    output = Path("outputs") / f"_v2_execution_cli_{uuid4().hex[:8]}"
    messages: list[str] = []
    inputs = iter(
        [
            "雨夜车站",
            "/confirm-plan",
            "/build-film-ir",
            "/build-ir",
            "/build-execution-plan",
            "/show-execution-plan",
            "/validate-execution-plan",
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
            input_fn=lambda _: next(inputs),
            output_fn=messages.append,
        )
        joined = "\n".join(messages)
        assert session.execution_plan is not None
        assert session.execution_bundle is not None
        assert session.validate_current_execution_plan().valid
        assert "ExecutionBundle 已生成" in joined
        assert "ExecutionBundle validation: status=fresh" in joined
        assert "provider_job: null" in joined
        assert "artifact: null" in joined
        assert "未调用 Provider" in joined
        assert not (output / "video").exists()
        saved = (output / "session.json").read_text(encoding="utf-8")
        assert '"execution_plan"' in saved
        assert '"execution_bundle"' in saved
    finally:
        rmtree(output, ignore_errors=True)

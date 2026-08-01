from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
from uuid import uuid4

from guided_story_agent.agent import RuleBasedStoryAgent
from guided_story_agent.cli import run_interactive
from guided_story_agent.models import CreativeBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import (
    Diagnostic,
    RuleBasedDirectorAgent,
    RuleBasedDirectorRevisionLoop,
    ValidationIssue,
)
from test_v2_contracts import make_plan


def test_hard_error_revision_loop_fails_closed_without_fabricating_content() -> None:
    plan = make_plan()
    issue = ValidationIssue(
        "missing_viewer_state",
        "viewer state is missing",
        "beats[0].viewer_state_before",
        "error",
    )

    result = RuleBasedDirectorRevisionLoop().run(
        plan,
        validation_issues=(issue,),
    )

    assert not result.accepted
    assert result.revised_movie_plan is None
    assert result.stop_reason == "hard_error_requires_director_revision"
    assert result.revision_history[0]["status"] == "rejected"


def test_suggestions_are_recorded_but_do_not_block_or_change_plan() -> None:
    plan = make_plan()
    diagnostic = Diagnostic(
        "weak_climax_candidate",
        "review the ending",
        "beats[-1]",
        "warning",
    )

    result = RuleBasedDirectorRevisionLoop().run(
        plan,
        creative_diagnostics=(diagnostic,),
    )

    assert result.accepted
    assert result.revised_movie_plan == plan
    assert result.stop_reason == "suggestions_recorded_no_revision"


def test_missing_film_beats_are_rejected_without_local_repair() -> None:
    result = RuleBasedDirectorRevisionLoop().run(replace(make_plan(), film_beats=()))

    assert not result.accepted
    assert result.revised_movie_plan is None
    assert result.revision_history[0]["request"]["validation_issues"][0]["code"] == "missing_film_beats"


def test_revision_history_is_json_serializable() -> None:
    result = RuleBasedDirectorRevisionLoop().run(make_plan())

    encoded = json.dumps(result.revision_history, ensure_ascii=False)

    assert "no_revision_required" in encoded


def test_session_persists_creative_optimizer_and_revision_fields() -> None:
    output = Path("outputs") / f"_v2_3c5_revision_{uuid4().hex[:8]}"
    session = GuidedStorySession(
        CreativeBrief(target_seconds=30),
        RuleBasedStoryAgent(),
        director_agent=RuleBasedDirectorAgent(),
        v2_enabled=True,
    )
    try:
        session.generate_movie_plan("雨夜车站")
        session.confirm_movie_plan()
        assert session.build_film_ir_from_confirmed_movie_plan() is not None
        assert session.creative_pass_diagnostics is not None
        assert session.film_ir_optimizer_diagnostics is not None
        assert session.director_revision_history
        assert session.build_movie_ir_from_film_ir() is not None
        assert session.movie_ir_optimizer_diagnostics is not None

        path = session.save(output / "session.json")
        loaded = GuidedStorySession.load(path)
        assert loaded.creative_pass_diagnostics == session.creative_pass_diagnostics
        assert loaded.film_ir_optimizer_diagnostics == session.film_ir_optimizer_diagnostics
        assert loaded.movie_ir_optimizer_diagnostics == session.movie_ir_optimizer_diagnostics
        assert loaded.director_revision_history == session.director_revision_history

        legacy_payload = session.to_dict()
        for field in (
            "creative_pass_diagnostics",
            "film_ir_optimizer_diagnostics",
            "movie_ir_optimizer_diagnostics",
            "director_revision_history",
            "director_revision_stop_reason",
        ):
            legacy_payload.pop(field, None)
        legacy_path = output / "legacy-session.json"
        legacy_path.write_text(
            json.dumps(legacy_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        legacy_loaded = GuidedStorySession.load(legacy_path)
        assert legacy_loaded.creative_pass_diagnostics == []
        assert legacy_loaded.film_ir_optimizer_diagnostics == []
        assert legacy_loaded.movie_ir_optimizer_diagnostics == []
        assert legacy_loaded.director_revision_history == []
        assert legacy_loaded.director_revision_stop_reason is None
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_v2_cli_diagnostics_command_exposes_new_layers() -> None:
    output = Path("outputs") / f"_v2_3c5_cli_{uuid4().hex[:8]}"
    inputs = iter(
        [
            "雨夜车站",
            "/confirm-plan",
            "/build-film-ir",
            "/build-ir",
            "/diagnostics",
            "/quit",
        ]
    )
    messages: list[str] = []
    try:
        run_interactive(
            target_seconds=30,
            output_dir=output,
            v2=True,
            director_agent=RuleBasedDirectorAgent(),
            input_fn=lambda prompt: next(inputs),
            output_fn=messages.append,
        )
        joined = "\n".join(messages)
        assert "creative_pass_diagnostics:" in joined
        assert "film_ir_optimizer_diagnostics:" in joined
        assert "movie_ir_optimizer_diagnostics:" in joined
        assert "director_revision_history:" in joined
    finally:
        shutil.rmtree(output, ignore_errors=True)

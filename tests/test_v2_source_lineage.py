from __future__ import annotations

import json
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from guided_story_agent.models import CreativeBrief as LegacyBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import (
    CompilationOptions,
    FilmIRBuilder,
    MovieIRBuilder,
    ProviderCapabilities,
    RuleBasedDirectorAgent,
    VideoJobCompiler,
)
from test_v2_contracts import make_plan


def test_all_v2_artifacts_record_source_lineage() -> None:
    plan = make_plan()
    film_result = FilmIRBuilder().build(plan)
    assert film_result.ok and film_result.film_ir is not None
    film_ir = film_result.film_ir

    assert film_ir.source_movie_plan_id == plan.plan_id
    assert film_ir.source_story_plan_id == f"{plan.plan_id}:story_plan"
    assert film_ir.source_director_plan_id == f"{plan.plan_id}:director_plan"

    movie_result = MovieIRBuilder().build(film_ir)
    assert movie_result.ok and movie_result.movie_ir is not None
    movie_ir = movie_result.movie_ir
    assert movie_ir.source_movie_plan_id == plan.plan_id
    assert movie_ir.source_film_ir_id == film_ir.ir_id

    compile_result = VideoJobCompiler().compile(
        movie_ir,
        ProviderCapabilities(
            "offline",
            supports_long_video=True,
            supports_multi_scene_prompt=True,
            supports_audio=True,
        ),
        CompilationOptions(),
    )
    assert compile_result.success and compile_result.video_job is not None
    job = compile_result.video_job
    assert job.source_movie_plan_id == plan.plan_id
    assert job.source_film_ir_id == film_ir.ir_id
    assert job.source_movie_ir_id == movie_ir.ir_id


def test_old_ir_payloads_with_missing_lineage_load_as_unknown() -> None:
    plan = make_plan()
    film_result = FilmIRBuilder().build(plan)
    assert film_result.film_ir is not None
    film_payload = film_result.film_ir.to_dict()
    film_payload.pop("source_movie_plan_id")
    film_payload.pop("source_story_plan_id")
    film_payload.pop("source_director_plan_id")

    from guided_story_agent.v2 import FilmIR, MovieIR

    restored_film = FilmIR.from_dict(film_payload)
    assert restored_film.source_movie_plan_id == ""
    movie_result = MovieIRBuilder().build(film_result.film_ir)
    assert movie_result.movie_ir is not None
    movie_payload = movie_result.movie_ir.to_dict()
    movie_payload.pop("source_movie_plan_id")
    movie_payload.pop("source_film_ir_id")
    restored_movie = MovieIR.from_dict(movie_payload)
    assert restored_movie.source_movie_plan_id == ""
    assert restored_movie.source_film_ir_id == ""


def test_old_session_without_source_lineage_loads_as_stale_not_crashed() -> None:
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
    payload = session.to_dict()
    for key in (
        "source_movie_plan_id",
        "source_story_plan_id",
        "source_director_plan_id",
        "source_film_ir_id",
        "source_movie_ir_id",
    ):
        payload["film_ir"].pop(key, None)
        payload["movie_ir"].pop(key, None)
        payload["v2_video_job"].pop(key, None)
    for key in (
        "current_film_ir_id",
        "current_movie_ir_id",
        "current_video_job_id",
        "source_lineage_diagnostics",
        "stale_lineage_diagnostics",
    ):
        payload.pop(key, None)

    output = Path("outputs") / f"_v2_old_lineage_{uuid4().hex[:10]}"
    try:
        output.mkdir(parents=True, exist_ok=True)
        path = output / "session.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        loaded = GuidedStorySession.load(path, v2_enabled=True)
        assert loaded.film_ir is not None
        assert loaded.movie_ir is not None
        assert loaded.v2_video_job is not None
        assert loaded.stale_lineage_diagnostics
    finally:
        rmtree(output, ignore_errors=True)

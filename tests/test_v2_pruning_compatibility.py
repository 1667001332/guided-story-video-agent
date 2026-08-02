from __future__ import annotations

import json
import warnings
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from guided_story_agent.agent import RuleBasedStoryAgent
from guided_story_agent.models import CreativeBrief
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.v2 import (
    ExecutionBundle,
    ExecutionPlanCompiler,
    FilmIRBuilder,
    MovieIRBuilder,
    ProviderCapabilities,
    build_movie_ir,
    compile_movie_ir_to_execution_bundle,
    compile_movie_plan_to_video_job,
)
from test_v2_contracts import make_plan


def _compiled_bundle():
    film = FilmIRBuilder().build(make_plan()).film_ir
    assert film is not None
    movie = MovieIRBuilder().build(film).movie_ir
    assert movie is not None
    result = ExecutionPlanCompiler().compile(
        movie,
        ProviderCapabilities(
            "fake",
            supports_reference_images=True,
            supports_character_reference=True,
            supports_audio=True,
            supports_long_video=True,
        ),
    )
    assert result.bundle is not None
    return result.bundle


def test_execution_bundle_json_contract_remains_unchanged() -> None:
    bundle = _compiled_bundle()
    restored = ExecutionBundle.from_dict(bundle.to_dict())

    assert restored == bundle
    assert restored.bundle_fingerprint == bundle.bundle_fingerprint


def test_deprecated_wrappers_remain_thin_compatibility_entries() -> None:
    plan = make_plan()
    film = FilmIRBuilder().build(plan).film_ir
    assert film is not None
    movie = MovieIRBuilder().build(film).movie_ir
    assert movie is not None
    capabilities = ProviderCapabilities(
        "fake",
        supports_reference_images=True,
        supports_character_reference=True,
        supports_audio=True,
        supports_long_video=True,
        supports_multi_scene_prompt=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        assert build_movie_ir(film).movie_ir is not None
        assert compile_movie_ir_to_execution_bundle(movie, capabilities).bundle is not None
        assert compile_movie_plan_to_video_job(plan, capabilities).video_job is not None

    assert len([item for item in caught if issubclass(item.category, DeprecationWarning)]) == 3


def test_older_session_payload_without_new_execution_fields_still_loads() -> None:
    session = GuidedStorySession(CreativeBrief(target_seconds=30), RuleBasedStoryAgent())
    source = session.to_dict()
    for key in (
        "execution_bundle",
        "execution_plan",
        "execution_run",
        "provider_jobs",
        "runtime_artifacts",
        "current_execution_bundle_fingerprint",
        "current_execution_plan_fingerprint",
        "current_execution_plan_id",
        "current_execution_run_id",
        "latest_execution_checkpoint_id",
    ):
        source.pop(key, None)
    source["schema_version"] = 5
    output_dir = Path("outputs") / f"_p1_legacy_{uuid4().hex[:10]}"
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        path = output_dir / "old-session.json"
        path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

        loaded = GuidedStorySession.load(path, agent=RuleBasedStoryAgent())

        assert loaded.execution_bundle is None
        assert loaded.execution_plan is None
        assert loaded.execution_run is None
    finally:
        rmtree(output_dir, ignore_errors=True)

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from tests.test_v2_contracts import make_plan

from guided_story_agent.v2 import (
    AudienceKnowledgeAnalysis,
    CharacterArcAnalysis,
    ConflictProgressionAnalysis,
    EmotionFlowAnalysis,
    FilmIRBuilder,
    PlanLayerConsistencyAnalysis,
    StoryCharacterGoal,
    creative_analysis_pipeline,
)


def _film_inputs():
    plan = make_plan()
    story_plan = replace(
        plan.story_plan,
        conflict="一个必须完成的选择",
        stakes="错过最后一班车",
        resolution="主角承担选择的后果",
        character_goals=(StoryCharacterGoal("c1", "赶上列车", "时间不足", "完成选择"),),
    )
    director_plan = replace(
        plan.director_plan,
        audience_knowledge="观众先知道时间异常，再理解主角的选择",
        ending_tone="克制而带有余韵",
        emotional_intention="不安逐步上升到决绝",
        withholding_strategy="先隐藏时间异常的原因",
    )
    plan = replace(plan, story_plan=story_plan, director_plan=director_plan, confirmed=True)
    film_ir = FilmIRBuilder().build(plan).film_ir
    assert film_ir is not None
    return plan, film_ir


def test_emotion_flow_generates_curve_artifact_and_metrics() -> None:
    plan, film_ir = _film_inputs()

    result = EmotionFlowAnalysis().analyze(plan, film_ir)

    assert result.succeeded
    assert result.artifacts[0].artifact_type == "emotion_curve"
    assert result.metrics["emotion_point_count"] == 2.0


def test_emotion_flow_detects_climax_that_is_not_emotional_peak() -> None:
    plan, film_ir = _film_inputs()
    assert film_ir is not None
    lowered_beats = tuple(
        replace(beat, tension_level=0.9 if index == 0 else 0.2)
        for index, beat in enumerate(film_ir.beats)
    )
    film_ir = replace(
        film_ir,
        beats=lowered_beats,
        emotion_curve=tuple(
            replace(point, intensity=0.9 if index == 0 else 0.2)
            for index, point in enumerate(film_ir.emotion_curve)
        ),
    )

    result = EmotionFlowAnalysis().analyze(plan, film_ir)

    assert any(item.code == "climax_not_emotional_peak" for item in result.diagnostics)


def test_audience_knowledge_generates_graph_and_reveal_diagnostic() -> None:
    plan, film_ir = _film_inputs()
    beats = tuple(replace(beat, dramatic_purpose="reveal", narrative_function="reveal") for beat in film_ir.beats)
    film_ir = replace(film_ir, beats=beats)

    result = AudienceKnowledgeAnalysis().analyze(plan, film_ir)

    assert result.artifacts[0].artifact_type == "audience_knowledge_graph"
    assert any(item.code == "reveal_without_setup" for item in result.diagnostics)


def test_conflict_progression_reports_missing_or_non_escalating_conflict() -> None:
    plan, film_ir = _film_inputs()
    plan = replace(plan, story_plan=replace(plan.story_plan, conflict="", stakes=""))
    result = ConflictProgressionAnalysis().analyze(plan, film_ir)

    codes = {item.code for item in result.diagnostics}
    assert "conflict_missing" in codes
    assert "stakes_missing" in codes
    assert result.artifacts[0].artifact_type == "conflict_graph"


def test_character_arc_reports_missing_protagonist_goal() -> None:
    plan, film_ir = _film_inputs()
    plan = replace(plan, story_plan=replace(plan.story_plan, character_goals=()))

    result = CharacterArcAnalysis().analyze(plan, film_ir)

    assert any(item.code == "protagonist_goal_missing" for item in result.diagnostics)
    assert result.artifacts[0].artifact_type == "character_arc_graph"


def test_plan_layer_consistency_detects_legacy_drift() -> None:
    plan, film_ir = _film_inputs()
    plan = replace(plan, story_plan=replace(plan.story_plan, title="different title"))

    result = PlanLayerConsistencyAnalysis().analyze(plan, film_ir)

    assert any(item.code == "story_plan_legacy_conflict" for item in result.diagnostics)


def test_analysis_pipeline_is_read_only_and_does_not_need_provider() -> None:
    plan, film_ir = _film_inputs()
    plan_before = deepcopy(plan)
    film_before = film_ir.to_dict()

    results = creative_analysis_pipeline().run(plan, film_ir)

    assert len(results) == 5
    assert plan == plan_before
    assert film_ir.to_dict() == film_before
    assert all("provider_payload" not in artifact.to_dict() for result in results for artifact in result.artifacts)

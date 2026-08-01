from __future__ import annotations

from copy import deepcopy

from tests.test_v2_contracts import make_plan

from guided_story_agent.v2 import (
    CreativeAnalysisDiagnostic,
    CreativeAnalysisResult,
    CreativeOptimizer,
    EmotionOptimizer,
    OptimizationSuggestion,
    TransformationCandidate,
)


def _result(analysis_type: str, *codes: str) -> CreativeAnalysisResult:
    diagnostics = tuple(
        CreativeAnalysisDiagnostic(code, f"diagnostic: {code}", f"{analysis_type}.{code}")
        for code in codes
    )
    return CreativeAnalysisResult(
        analysis_type,
        "plan-1",
        "plan-1:story_plan",
        "plan-1:director_plan",
        None,
        diagnostics,
    )


def test_creative_optimizer_maps_analysis_without_mutating_movie_plan() -> None:
    plan = make_plan()
    before = deepcopy(plan)
    result = CreativeOptimizer().optimize(
        plan,
        (
            _result(
                "emotion_flow",
                "climax_not_emotional_peak",
                "ending_tone_missing",
            ),
            _result("conflict_progression", "conflict_not_escalating"),
        ),
    )

    assert plan == before
    assert {item.code for item in result.suggestions} == {
        "strengthen_climax_emotional_peak",
        "clarify_ending_tone",
        "escalate_conflict_before_climax",
    }
    assert result.suggestions[0].source_diagnostic_codes
    assert result.transformation_candidates
    assert all(not item.executable for item in result.transformation_candidates)
    assert result.to_dict()["source_movie_plan_id"] == plan.plan_id


def test_domain_optimizer_can_run_independently() -> None:
    result = EmotionOptimizer().optimize(
        make_plan(),
        (_result("emotion_flow", "emotion_curve_missing"),),
    )

    assert [item.code for item in result.suggestions] == ["clarify_emotion_curve"]
    assert result.suggestions[0].severity == "deferred"


def test_optimizer_models_reject_executable_transformations() -> None:
    suggestion = OptimizationSuggestion(
        "clarify_core_conflict",
        "明确已有冲突",
        "hard",
        reason="analysis evidence",
    )
    assert "provider_payload" not in suggestion.to_dict()
    candidate = TransformationCandidate(
        "candidate_clarify_core_conflict",
        "future candidate",
        "hard",
    )
    assert candidate.to_dict()["executable"] is False

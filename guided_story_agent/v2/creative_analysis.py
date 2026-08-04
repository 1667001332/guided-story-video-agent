"""Read-only creative analysis for StoryPlan, DirectorPlan, and FilmIR.

This module deliberately sits beside (not inside) the compiler.  Analyses
produce immutable diagnostics, metrics, and graph artifacts.  They never repair
or rewrite MoviePlan, FilmIR, MovieIR, prompts, or Provider requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .creative_graph import (
    CreativeGraph,
    CreativeGraphEdge,
    CreativeGraphNode,
    _ensure_safe,
)
from .film_ir import FilmBeat, FilmIR
from .models import MoviePlan


@dataclass(frozen=True, slots=True)
class CreativeAnalysisDiagnostic:
    code: str
    message: str
    path: str = ""
    severity: str = "warning"
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class CreativeAnalysisArtifact:
    artifact_type: str
    nodes: tuple[dict[str, object], ...] = ()
    edges: tuple[dict[str, object], ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_safe(
            {"nodes": self.nodes, "edges": self.edges, "metadata": self.metadata},
            f"artifact[{self.artifact_type}]",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": self.artifact_type,
            "nodes": [dict(node) for node in self.nodes],
            "edges": [dict(edge) for edge in self.edges],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_graph(
        cls,
        graph: CreativeGraph,
        *,
        metadata: dict[str, object] | None = None,
    ) -> "CreativeAnalysisArtifact":
        graph_data = graph.to_dict()
        graph_metadata = dict(graph_data["metadata"])
        if metadata:
            graph_metadata.update(metadata)
        return cls(
            artifact_type=graph.graph_type,
            nodes=tuple(graph_data["nodes"]),
            edges=tuple(graph_data["edges"]),
            metadata=graph_metadata,
        )


@dataclass(frozen=True, slots=True)
class CreativeAnalysisResult:
    analysis_type: str
    source_movie_plan_id: str
    source_story_plan_id: str | None
    source_director_plan_id: str | None
    source_film_ir_id: str | None
    diagnostics: tuple[CreativeAnalysisDiagnostic, ...] = ()
    artifacts: tuple[CreativeAnalysisArtifact, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    succeeded: bool = True

    def __post_init__(self) -> None:
        _ensure_safe(
            {
                "diagnostics": [item.to_dict() for item in self.diagnostics],
                "artifacts": [item.to_dict() for item in self.artifacts],
                "metrics": self.metrics,
            },
            f"analysis[{self.analysis_type}]",
        )

    @property
    def errors(self) -> tuple[CreativeAnalysisDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "error")

    def to_dict(self) -> dict[str, object]:
        return {
            "analysis_type": self.analysis_type,
            "source_movie_plan_id": self.source_movie_plan_id,
            "source_story_plan_id": self.source_story_plan_id,
            "source_director_plan_id": self.source_director_plan_id,
            "source_film_ir_id": self.source_film_ir_id,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "metrics": {str(key): float(value) for key, value in self.metrics.items()},
            "succeeded": self.succeeded,
        }


class CreativeAnalysis(Protocol):
    name: str

    def analyze(
        self,
        movie_plan: MoviePlan,
        film_ir: FilmIR | None,
    ) -> CreativeAnalysisResult: ...


def _source_ids(
    movie_plan: MoviePlan,
    film_ir: FilmIR | None,
) -> tuple[str, str | None, str | None, str | None]:
    story_plan = getattr(movie_plan, "story_plan", None)
    director_plan = getattr(movie_plan, "director_plan", None)
    return (
        movie_plan.plan_id,
        f"{movie_plan.plan_id}:story_plan" if story_plan is not None else None,
        f"{movie_plan.plan_id}:director_plan" if director_plan is not None else None,
        film_ir.ir_id if film_ir is not None else None,
    )


def _result(
    name: str,
    movie_plan: MoviePlan,
    film_ir: FilmIR | None,
    diagnostics: Sequence[CreativeAnalysisDiagnostic] = (),
    artifacts: Sequence[CreativeAnalysisArtifact] = (),
    metrics: dict[str, float] | None = None,
    *,
    succeeded: bool = True,
) -> CreativeAnalysisResult:
    source_ids = _source_ids(movie_plan, film_ir)
    return CreativeAnalysisResult(
        analysis_type=name,
        source_movie_plan_id=source_ids[0],
        source_story_plan_id=source_ids[1],
        source_director_plan_id=source_ids[2],
        source_film_ir_id=source_ids[3],
        diagnostics=tuple(diagnostics),
        artifacts=tuple(artifacts),
        metrics={key: float(value) for key, value in (metrics or {}).items()},
        succeeded=succeeded,
    )


def _beats(movie_plan: MoviePlan, film_ir: FilmIR | None) -> tuple[FilmBeat | Any, ...]:
    if film_ir is not None and film_ir.beats:
        return tuple(film_ir.beats)
    return tuple(movie_plan.film_beats)


def _graph_artifact(
    graph_type: str,
    nodes: Sequence[CreativeGraphNode],
    edges: Sequence[CreativeGraphEdge],
    movie_plan: MoviePlan,
    film_ir: FilmIR | None,
) -> CreativeAnalysisArtifact:
    graph = CreativeGraph(
        graph_type=graph_type,
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "source_movie_plan_id": movie_plan.plan_id,
            "source_film_ir_id": film_ir.ir_id if film_ir else None,
        },
    )
    return CreativeAnalysisArtifact.from_graph(graph)


def _purpose(beat: Any) -> str:
    return f"{beat.dramatic_purpose} {beat.narrative_function}".lower()


def _is_setup(beat: Any) -> bool:
    return any(token in _purpose(beat) for token in ("setup", "establish", "intro"))


def _is_reveal(beat: Any) -> bool:
    return any(token in _purpose(beat) for token in ("reveal", "disclose", "turn"))


def _is_payoff(beat: Any) -> bool:
    return any(token in _purpose(beat) for token in ("payoff", "resolution", "climax"))


class EmotionFlowAnalysis:
    name = "emotion_flow"

    def analyze(self, movie_plan: MoviePlan, film_ir: FilmIR | None) -> CreativeAnalysisResult:
        director_plan = movie_plan.director_plan
        beats = _beats(movie_plan, film_ir)
        points = tuple(film_ir.emotion_curve) if film_ir is not None else tuple(movie_plan.emotion_curve)
        diagnostics: list[CreativeAnalysisDiagnostic] = []
        if not points:
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "emotion_curve_missing",
                    "没有可分析的情绪曲线。",
                    "emotion_curve",
                    evidence=tuple(beat.beat_id for beat in beats),
                )
            )
        if not director_plan or not director_plan.ending_tone.strip():
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "ending_tone_missing",
                    "DirectorPlan 没有声明 ending_tone。",
                    "director_plan.ending_tone",
                )
            )
        if not director_plan or not director_plan.emotional_intention.strip():
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "emotional_intention_unlinked",
                    "DirectorPlan emotional_intention 没有可链接的情绪曲线意图。",
                    "director_plan.emotional_intention",
                )
            )

        nodes = [
            CreativeGraphNode(
                id=f"emotion-{index + 1}",
                type="emotion_point",
                label=str(point.label),
                data={
                    "beat_id": point.beat_id,
                    "intensity": float(point.intensity),
                },
            )
            for index, point in enumerate(points)
        ]
        edges = [
            CreativeGraphEdge(
                source=nodes[index].id,
                target=nodes[index + 1].id,
                type="emotion_sequence",
            )
            for index in range(max(0, len(nodes) - 1))
        ]
        peak_intensity = max((float(point.intensity) for point in points), default=0.0)
        peak_index = max(range(len(points)), key=lambda index: float(points[index].intensity), default=-1)
        climax_ids = {
            beat.beat_id
            for beat in beats
            if _is_payoff(beat) or "climax" in _purpose(beat)
        }
        peak_id = points[peak_index].beat_id if peak_index >= 0 else ""
        climax_alignment = 1.0 if peak_id and peak_id in climax_ids else 0.0
        if points and climax_ids and climax_alignment == 0.0:
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "climax_not_emotional_peak",
                    "高潮 beat 不是情绪曲线峰值。",
                    "emotion_curve",
                    evidence=(peak_id, *sorted(climax_ids)),
                )
            )
        ending_alignment = (
            1.0
            if director_plan
            and director_plan.ending_tone.strip()
            and beats
            and _is_payoff(beats[-1])
            else 0.0
        )
        artifact = _graph_artifact("emotion_curve", nodes, edges, movie_plan, film_ir)
        return _result(
            self.name,
            movie_plan,
            film_ir,
            diagnostics,
            (artifact,),
            {
                "emotion_point_count": float(len(points)),
                "peak_intensity": peak_intensity,
                "climax_alignment_score": climax_alignment,
                "ending_tone_alignment_score": ending_alignment,
            },
        )


class AudienceKnowledgeAnalysis:
    name = "audience_knowledge"

    def analyze(self, movie_plan: MoviePlan, film_ir: FilmIR | None) -> CreativeAnalysisResult:
        director_plan = movie_plan.director_plan
        beats = _beats(movie_plan, film_ir)
        diagnostics: list[CreativeAnalysisDiagnostic] = []
        if not director_plan or not director_plan.audience_knowledge.strip():
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "audience_knowledge_missing",
                    "DirectorPlan 没有声明 audience_knowledge。",
                    "director_plan.audience_knowledge",
                )
            )
        setup_indices = [index for index, beat in enumerate(beats) if _is_setup(beat)]
        reveal_indices = [index for index, beat in enumerate(beats) if _is_reveal(beat)]
        payoff_indices = [index for index, beat in enumerate(beats) if _is_payoff(beat)]
        for index, beat in enumerate(beats):
            if not beat.viewer_state_before.strip() or not beat.viewer_state_after.strip():
                diagnostics.append(
                    CreativeAnalysisDiagnostic(
                        "viewer_state_missing",
                        "beat 缺少完整的观众状态转换。",
                        f"beats[{index}]",
                        evidence=(beat.beat_id,),
                    )
                )
            if _is_reveal(beat) and not any(item < index for item in setup_indices):
                diagnostics.append(
                    CreativeAnalysisDiagnostic(
                        "reveal_without_setup",
                        "揭示没有可追溯的前置建立。",
                        f"beats[{index}]",
                        evidence=(beat.beat_id,),
                    )
                )
        if setup_indices and not any(index > setup_indices[0] for index in payoff_indices):
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "setup_without_payoff",
                    "故事建立后没有可识别的 payoff 或 resolution。",
                    "beats",
                    evidence=(beats[setup_indices[0]].beat_id,),
                )
            )
        if reveal_indices and reveal_indices[0] == len(beats) - 1:
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "audience_knows_too_late",
                    "关键揭示发生在结尾，观众缺少反应和理解时间。",
                    f"beats[{reveal_indices[0]}]",
                )
            )
        withheld_count = 1.0 if director_plan and director_plan.withholding_strategy.strip() else 0.0
        nodes = [
            CreativeGraphNode(
                id=f"audience-{index + 1}",
                type="viewer_state",
                label=beat.viewer_state_after or beat.viewer_state_before,
                data={
                    "beat_id": beat.beat_id,
                    "before": beat.viewer_state_before,
                    "after": beat.viewer_state_after,
                    "understanding": beat.required_audience_understanding,
                },
            )
            for index, beat in enumerate(beats)
        ]
        edges = [
            CreativeGraphEdge(
                source=nodes[index].id,
                target=nodes[index + 1].id,
                type="knowledge_transition",
            )
            for index in range(max(0, len(nodes) - 1))
        ]
        confusion_count = sum(
            1
            for item in diagnostics
            if item.code in {"viewer_state_missing", "reveal_without_setup"}
        )
        artifact = _graph_artifact("audience_knowledge_graph", nodes, edges, movie_plan, film_ir)
        return _result(
            self.name,
            movie_plan,
            film_ir,
            diagnostics,
            (artifact,),
            {
                "reveal_count": float(len(reveal_indices)),
                "withheld_count": withheld_count,
                "unexplained_key_event_count": float(confusion_count),
                "audience_confusion_risk": float(confusion_count / max(1, len(beats))),
            },
        )


class ConflictProgressionAnalysis:
    name = "conflict_progression"

    def analyze(self, movie_plan: MoviePlan, film_ir: FilmIR | None) -> CreativeAnalysisResult:
        story_plan = movie_plan.story_plan
        beats = _beats(movie_plan, film_ir)
        conflict = story_plan.conflict.strip() if story_plan else ""
        stakes = story_plan.stakes.strip() if story_plan else ""
        resolution = story_plan.resolution.strip() if story_plan else ""
        diagnostics: list[CreativeAnalysisDiagnostic] = []
        if not conflict:
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "conflict_missing",
                    "StoryPlan 没有声明核心冲突。",
                    "story_plan.conflict",
                )
            )
        if not stakes:
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "stakes_missing",
                    "StoryPlan 没有声明 stakes。",
                    "story_plan.stakes",
                )
            )
        tension = [float(getattr(beat, "tension_level", 0.0)) for beat in beats]
        escalation_steps = sum(right > left for left, right in zip(tension, tension[1:]))
        setup_indices = [index for index, beat in enumerate(beats) if _is_setup(beat)]
        payoff_indices = [index for index, beat in enumerate(beats) if _is_payoff(beat)]
        if len(beats) >= 2 and escalation_steps == 0:
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "conflict_not_escalating",
                    "冲突 beat 之间没有可观察的张力升级。",
                    "beats",
                    evidence=tuple(beat.beat_id for beat in beats),
                )
            )
        if payoff_indices and not conflict:
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "climax_resolves_no_conflict",
                    "存在 payoff/resolution，但没有可解析的核心冲突。",
                    f"beats[{payoff_indices[-1]}]",
                )
            )
        if resolution and payoff_indices and not any(beat.required_evidence for beat in beats):
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "resolution_without_cost",
                    "有结局声明，但没有可见的代价或证据。",
                    "story_plan.resolution",
                )
            )
        nodes = [
            CreativeGraphNode(
                id=f"conflict-{index + 1}",
                type="conflict_beat",
                label=beat.dramatic_purpose or beat.narrative_function,
                data={
                    "beat_id": beat.beat_id,
                    "tension_level": float(getattr(beat, "tension_level", 0.0)),
                    "dramatic_purpose": beat.dramatic_purpose,
                    "narrative_function": beat.narrative_function,
                },
            )
            for index, beat in enumerate(beats)
        ]
        edges = [
            CreativeGraphEdge(
                source=nodes[index].id,
                target=nodes[index + 1].id,
                type="conflict_progression",
                data={"tension_delta": tension[index + 1] - tension[index]},
            )
            for index in range(max(0, len(nodes) - 1))
        ]
        conflict_setup_score = 1.0 if conflict and setup_indices else 0.0
        escalation_score = min(1.0, escalation_steps / max(1, len(beats) - 1))
        climax_resolution_score = 1.0 if conflict and payoff_indices else 0.0
        cost_visibility_score = 1.0 if payoff_indices and any(beat.required_evidence for beat in beats) else 0.0
        artifact = _graph_artifact("conflict_graph", nodes, edges, movie_plan, film_ir)
        return _result(
            self.name,
            movie_plan,
            film_ir,
            diagnostics,
            (artifact,),
            {
                "conflict_setup_score": conflict_setup_score,
                "escalation_score": escalation_score,
                "climax_resolution_score": climax_resolution_score,
                "cost_visibility_score": cost_visibility_score,
            },
        )


def _character_state_ids(beats: Sequence[Any]) -> dict[str, list[str]]:
    states: dict[str, list[str]] = {}
    for beat in beats:
        for raw in beat.character_emotional_state:
            character_id, _, state = raw.partition(":")
            character_id = character_id.strip()
            state = state.strip()
            if character_id:
                states.setdefault(character_id, []).append(state)
    return states


class CharacterArcAnalysis:
    name = "character_arc"

    def analyze(self, movie_plan: MoviePlan, film_ir: FilmIR | None) -> CreativeAnalysisResult:
        story_plan = movie_plan.story_plan
        beats = _beats(movie_plan, film_ir)
        characters = tuple(story_plan.characters) if story_plan else tuple(movie_plan.character_sheet.characters)
        goals = {item.character_id: item for item in (story_plan.character_goals if story_plan else ())}
        states = _character_state_ids(beats)
        character_ids = [item.character_id for item in characters]
        for character_id in states:
            if character_id not in character_ids:
                character_ids.append(character_id)
        protagonist = next(
            (item.character_id for item in characters if item.role.lower() == "protagonist"),
            character_ids[0] if character_ids else "protagonist",
        )
        diagnostics: list[CreativeAnalysisDiagnostic] = []
        protagonist_goal = goals.get(protagonist)
        if protagonist_goal is None or not protagonist_goal.goal.strip():
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "protagonist_goal_missing",
                    "主要角色没有明确的 character goal。",
                    f"story_plan.character_goals[{protagonist}]",
                    evidence=(protagonist,),
                )
            )
        if protagonist and len(set(states.get(protagonist, ()))) <= 1:
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "character_arc_flat",
                    "主要角色缺少可观察的状态变化。",
                    f"characters[{protagonist}]",
                    evidence=tuple(states.get(protagonist, ())),
                )
            )
        choice_indices = [
            index
            for index, beat in enumerate(beats)
            if any(token in _purpose(beat) for token in ("choice", "decision", "turn", "payoff", "resolution"))
        ]
        if not choice_indices:
            diagnostics.append(
                CreativeAnalysisDiagnostic(
                    "irreversible_choice_missing",
                    "没有可识别的不可逆选择或决定。",
                    "beats",
                    evidence=(protagonist,),
                )
            )
        if beats and protagonist:
            first_seen = next(
                (
                    index
                    for index, beat in enumerate(beats)
                    if any(
                        raw.partition(":")[0].strip() == protagonist
                        for raw in beat.character_emotional_state
                    )
                ),
                None,
            )
            last_seen = next(
                (
                    index
                    for index in range(len(beats) - 1, -1, -1)
                    if any(
                        raw.partition(":")[0].strip() == protagonist
                        for raw in beats[index].character_emotional_state
                    )
                ),
                None,
            )
            if first_seen is not None and last_seen is not None and last_seen < len(beats) - 1:
                diagnostics.append(
                    CreativeAnalysisDiagnostic(
                        "character_disappears_after_setup",
                        "主要角色在建立后没有持续出现在后续 beat。",
                        f"beats[{last_seen}]",
                        evidence=(protagonist,),
                    )
                )
            if protagonist_goal and last_seen is not None and last_seen == len(beats) - 1 and not protagonist_goal.outcome.strip():
                diagnostics.append(
                    CreativeAnalysisDiagnostic(
                        "character_goal_not_resolved",
                        "主要角色有目标，但没有声明目标结果。",
                        f"story_plan.character_goals[{protagonist}]",
                        evidence=(protagonist_goal.goal,),
                    )
                )
        nodes: list[CreativeGraphNode] = []
        for character_id in character_ids:
            goal = goals.get(character_id)
            nodes.append(
                CreativeGraphNode(
                    id=f"character-{character_id}",
                    type="character",
                    label=next((item.name for item in characters if item.character_id == character_id), character_id),
                    data={
                        "character_id": character_id,
                        "goal": goal.goal if goal else "",
                        "states": tuple(states.get(character_id, ())),
                    },
                )
            )
        beat_nodes = [
            CreativeGraphNode(
                id=f"character-beat-{index + 1}",
                type="character_beat",
                label=beat.beat_id,
                data={"beat_id": beat.beat_id, "character_states": tuple(beat.character_emotional_state)},
            )
            for index, beat in enumerate(beats)
        ]
        nodes.extend(beat_nodes)
        edges: list[CreativeGraphEdge] = []
        for character_id in character_ids:
            node_id = f"character-{character_id}"
            for index, beat in enumerate(beats):
                if any(raw.startswith(f"{character_id}:") for raw in beat.character_emotional_state):
                    edges.append(CreativeGraphEdge(node_id, f"character-beat-{index + 1}", "character_participation"))
        goal_coverage = (
            sum(bool(goals.get(character_id) and goals[character_id].goal.strip()) for character_id in character_ids)
            / max(1, len(character_ids))
        )
        arc_completion = 1.0 if protagonist_goal and protagonist_goal.outcome.strip() and len(set(states.get(protagonist, ()))) > 1 else 0.0
        artifact = _graph_artifact("character_arc_graph", nodes, edges, movie_plan, film_ir)
        return _result(
            self.name,
            movie_plan,
            film_ir,
            diagnostics,
            (artifact,),
            {
                "character_goal_coverage": float(goal_coverage),
                "irreversible_choice_count": float(len(choice_indices)),
                "arc_completion_score": float(arc_completion),
            },
        )


class PlanLayerConsistencyAnalysis:
    name = "plan_layer_consistency"

    def analyze(self, movie_plan: MoviePlan, film_ir: FilmIR | None) -> CreativeAnalysisResult:
        story_plan = movie_plan.story_plan
        director_plan = movie_plan.director_plan
        diagnostics: list[CreativeAnalysisDiagnostic] = []
        conflicts = 0
        if story_plan and story_plan.title and movie_plan.story.title and story_plan.title != movie_plan.story.title:
            conflicts += 1
            diagnostics.append(CreativeAnalysisDiagnostic("story_plan_legacy_conflict", "StoryPlan title 与 legacy Story.title 不一致。", "story_plan.title", evidence=(story_plan.title, movie_plan.story.title)))
        if story_plan and story_plan.logline and movie_plan.story.logline and story_plan.logline != movie_plan.story.logline:
            conflicts += 1
            diagnostics.append(CreativeAnalysisDiagnostic("story_plan_legacy_conflict", "StoryPlan logline 与 legacy Story.logline 不一致。", "story_plan.logline", evidence=(story_plan.logline, movie_plan.story.logline)))
        legacy_characters = {item.character_id for item in movie_plan.character_sheet.characters}
        layered_characters = {item.character_id for item in story_plan.characters} if story_plan else set()
        if legacy_characters != layered_characters:
            conflicts += 1
            diagnostics.append(CreativeAnalysisDiagnostic("character_projection_mismatch", "StoryPlan.characters 与 legacy CharacterSheet 不一致。", "story_plan.characters", evidence=tuple(sorted(legacy_characters ^ layered_characters))))
        legacy_beats = tuple(scene.goal.strip() for scene in movie_plan.script.scenes)
        layered_beats = tuple(story_plan.story_beats) if story_plan else ()
        if layered_beats and legacy_beats != layered_beats:
            conflicts += 1
            diagnostics.append(CreativeAnalysisDiagnostic("film_beat_projection_mismatch", "StoryPlan.story_beats 与 legacy Script scene goals 不一致。", "story_plan.story_beats", evidence=(*legacy_beats, *layered_beats)))
        if director_plan and director_plan.visual_motif_strategy and movie_plan.visual_style and director_plan.visual_motif_strategy != movie_plan.visual_style:
            conflicts += 1
            diagnostics.append(CreativeAnalysisDiagnostic("director_plan_visual_style_mismatch", "DirectorPlan visual motif 与 legacy visual_style 不一致。", "director_plan.visual_motif_strategy", evidence=(director_plan.visual_motif_strategy, movie_plan.visual_style)))
        if movie_plan.film_beats and film_ir is not None:
            plan_ids = {beat.beat_id for beat in movie_plan.film_beats}
            ir_ids = {beat.beat_id for beat in film_ir.beats}
            if plan_ids != ir_ids:
                conflicts += 1
                diagnostics.append(CreativeAnalysisDiagnostic("film_beat_projection_mismatch", "FilmIR beats 与 MoviePlan.film_beats 不一致。", "film_ir.beats", evidence=tuple(sorted(plan_ids ^ ir_ids))))
        nodes = (
            CreativeGraphNode("legacy-story", "legacy_layer", "legacy story", {"title": movie_plan.story.title}),
            CreativeGraphNode("story-plan", "story_layer", "StoryPlan", {"title": story_plan.title if story_plan else ""}),
            CreativeGraphNode("legacy-director", "legacy_layer", "legacy director", {"visual_style": movie_plan.visual_style}),
            CreativeGraphNode("director-plan", "director_layer", "DirectorPlan", {"visual_motif_strategy": director_plan.visual_motif_strategy if director_plan else ""}),
        )
        edges = (
            CreativeGraphEdge("story-plan", "legacy-story", "compatibility_projection"),
            CreativeGraphEdge("director-plan", "legacy-director", "compatibility_projection"),
        )
        artifact = _graph_artifact("plan_layer_consistency_graph", nodes, edges, movie_plan, film_ir)
        return _result(
            self.name,
            movie_plan,
            film_ir,
            diagnostics,
            (artifact,),
            {
                "story_plan_legacy_alignment": 1.0 if not conflicts else 0.0,
                "director_plan_legacy_alignment": 1.0 if not conflicts else 0.0,
                "missing_projection_count": float(not story_plan) + float(not director_plan),
                "conflict_count": float(conflicts),
            },
        )


class CreativeAnalysisPipeline:
    """Run read-only analyses in order and stop on hard analysis errors."""

    def __init__(self, analyses: Sequence[CreativeAnalysis]) -> None:
        self.analyses = tuple(analyses)

    def run(
        self,
        movie_plan: MoviePlan,
        film_ir: FilmIR | None = None,
    ) -> tuple[CreativeAnalysisResult, ...]:
        if not isinstance(movie_plan, MoviePlan):
            return (
                CreativeAnalysisResult(
                    "creative_analysis_pipeline",
                    "",
                    None,
                    None,
                    None,
                    (
                        CreativeAnalysisDiagnostic(
                            "invalid_movie_plan",
                            "Creative Analysis 只接受 MoviePlan。",
                            "movie_plan",
                            "error",
                        ),
                    ),
                    succeeded=False,
                ),
            )
        results: list[CreativeAnalysisResult] = []
        for analysis in self.analyses:
            try:
                result = analysis.analyze(movie_plan, film_ir)
            except Exception as exc:  # pragma: no cover - defensive boundary
                result = _result(
                    getattr(analysis, "name", analysis.__class__.__name__),
                    movie_plan,
                    film_ir,
                    (
                        CreativeAnalysisDiagnostic(
                            "creative_analysis_exception",
                            f"Creative Analysis 执行失败：{exc}",
                            getattr(analysis, "name", "analysis"),
                            "error",
                        ),
                    ),
                    succeeded=False,
                )
            results.append(result)
            if not result.succeeded or result.errors:
                break
        return tuple(results)

    def run_aggregate(
        self,
        movie_plan: MoviePlan,
        film_ir: FilmIR | None = None,
    ) -> CreativeAnalysisResult:
        results = self.run(movie_plan, film_ir)
        diagnostics = tuple(item for result in results for item in result.diagnostics)
        artifacts = tuple(item for result in results for item in result.artifacts)
        metrics = {
            f"{result.analysis_type}.{key}": value
            for result in results
            for key, value in result.metrics.items()
        }
        first = results[0] if results else _result("creative_analysis_pipeline", movie_plan, film_ir)
        return CreativeAnalysisResult(
            "creative_analysis_pipeline",
            first.source_movie_plan_id,
            first.source_story_plan_id,
            first.source_director_plan_id,
            first.source_film_ir_id,
            diagnostics,
            artifacts,
            metrics,
            bool(results) and all(result.succeeded for result in results),
        )


def creative_analysis_pipeline(
    analyses: Sequence[CreativeAnalysis] | None = None,
) -> CreativeAnalysisPipeline:
    return CreativeAnalysisPipeline(
        analyses
        or (
            EmotionFlowAnalysis(),
            AudienceKnowledgeAnalysis(),
            ConflictProgressionAnalysis(),
            CharacterArcAnalysis(),
            PlanLayerConsistencyAnalysis(),
        )
    )


__all__ = [
    "CreativeAnalysisDiagnostic",
    "CreativeAnalysisArtifact",
    "CreativeAnalysisResult",
    "CreativeAnalysis",
    "CreativeAnalysisPipeline",
    "EmotionFlowAnalysis",
    "AudienceKnowledgeAnalysis",
    "ConflictProgressionAnalysis",
    "CharacterArcAnalysis",
    "PlanLayerConsistencyAnalysis",
    "creative_analysis_pipeline",
]

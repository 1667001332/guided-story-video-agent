from __future__ import annotations

import json

import pytest

from guided_story_agent.v2.creative_graph import (
    CharacterArcGraph,
    ConflictGraph,
    CreativeGraph,
    CreativeGraphEdge,
    CreativeGraphNode,
    EmotionGraph,
)


def test_creative_graph_nodes_edges_and_graph_are_json_serializable() -> None:
    nodes = (
        CreativeGraphNode("a", "emotion", "start", {"intensity": 0.2}),
        CreativeGraphNode("b", "emotion", "end", {"intensity": 0.8}),
    )
    edges = (CreativeGraphEdge("a", "b", "sequence"),)
    graph = CreativeGraph("emotion_graph", nodes, edges, {"source": "test"})

    payload = graph.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    restored = CreativeGraph.from_dict(payload)
    assert restored == graph


def test_semantic_graph_markers_use_the_same_stable_contract() -> None:
    graph = EmotionGraph(
        "emotion_graph",
        (CreativeGraphNode("a", "emotion", "start"),),
    )
    conflict = ConflictGraph(
        "conflict_graph",
        (CreativeGraphNode("a", "conflict", "setup"),),
    )
    character = CharacterArcGraph(
        "character_arc_graph",
        (CreativeGraphNode("a", "character", "hero"),),
    )

    assert graph.graph_type == "emotion_graph"
    assert conflict.graph_type == "conflict_graph"
    assert character.graph_type == "character_arc_graph"


@pytest.mark.parametrize("field", ["provider_payload", "endpoint", "model", "task"])
def test_creative_graph_rejects_provider_fields(field: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        CreativeGraphNode("a", "test", "bad", {field: "not allowed"})

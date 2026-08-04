"""Provider-neutral creative graph contracts.

Graphs are deliberately small, immutable domain values.  They describe
relationships between story, director, audience, emotion, conflict, and
character decisions; they never contain Provider/API/runtime payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_FORBIDDEN_KEYS = {
    "provider",
    "provider_key",
    "provider_name",
    "provider_profile",
    "api",
    "api_key",
    "payload",
    "provider_payload",
    "request_payload",
    "video_payload",
    "api_payload",
    "http_payload",
    "endpoint",
    "model",
    "task",
    "task_id",
    "request_id",
    "video_id",
    "submit",
    "poll",
    "download",
}
_FORBIDDEN_TERMS = (
    "masterpiece",
    "best quality",
    "ultra realistic",
    "ultra-realistic",
    "cinematic masterpiece",
    "8k",
    "award winning",
    "photorealistic masterpiece",
)


def _ensure_safe(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"CreativeGraph contains forbidden field: {path}.{key_text}")
            _ensure_safe(child, f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _ensure_safe(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for term in _FORBIDDEN_TERMS:
            if term in lowered:
                raise ValueError(f"CreativeGraph contains prompt stuffing at {path}: {term}")


def _plain(value: Any) -> Any:
    if isinstance(value, CreativeGraphNode):
        return value.to_dict()
    if isinstance(value, CreativeGraphEdge):
        return value.to_dict()
    if isinstance(value, CreativeGraph):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CreativeGraphNode:
    id: str
    type: str
    label: str
    data: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("CreativeGraphNode.id is required")
        if not self.type.strip():
            raise ValueError("CreativeGraphNode.type is required")
        _ensure_safe(self.data, f"node[{self.id}].data")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "data": _plain(self.data),
        }


@dataclass(frozen=True, slots=True)
class CreativeGraphEdge:
    source: str
    target: str
    type: str
    data: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("CreativeGraphEdge source and target are required")
        if not self.type.strip():
            raise ValueError("CreativeGraphEdge.type is required")
        _ensure_safe(self.data, f"edge[{self.source}->{self.target}].data")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "data": _plain(self.data),
        }


@dataclass(frozen=True, slots=True)
class CreativeGraph:
    graph_type: str
    nodes: tuple[CreativeGraphNode, ...] = ()
    edges: tuple[CreativeGraphEdge, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.graph_type.strip():
            raise ValueError("CreativeGraph.graph_type is required")
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("CreativeGraph node ids must be unique")
        known = set(node_ids)
        if any(edge.source not in known or edge.target not in known for edge in self.edges):
            raise ValueError("CreativeGraph edges must reference existing nodes")
        _ensure_safe(self.metadata, f"graph[{self.graph_type}].metadata")

    def to_dict(self) -> dict[str, object]:
        return {
            "graph_type": self.graph_type,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CreativeGraph":
        if not isinstance(data, dict):
            raise ValueError("CreativeGraph must be a JSON object")
        _ensure_safe(data, "creative_graph")
        allowed = {"graph_type", "nodes", "edges", "metadata"}
        unexpected = sorted(set(data) - allowed)
        if unexpected:
            raise ValueError("CreativeGraph contains unsupported fields: " + ", ".join(unexpected))
        raw_nodes = data.get("nodes", [])
        raw_edges = data.get("edges", [])
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValueError("CreativeGraph nodes and edges must be arrays")
        if any(not isinstance(item, dict) for item in (*raw_nodes, *raw_edges)):
            raise ValueError("CreativeGraph nodes and edges must contain objects")
        nodes = tuple(
            CreativeGraphNode(
                id=str(item["id"]),
                type=str(item["type"]),
                label=str(item.get("label", "")),
                data=dict(item.get("data", {})),
            )
            for item in raw_nodes
        )
        edges = tuple(
            CreativeGraphEdge(
                source=str(item["source"]),
                target=str(item["target"]),
                type=str(item["type"]),
                data=dict(item.get("data", {})),
            )
            for item in raw_edges
        )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("CreativeGraph metadata must be an object")
        return cls(str(data["graph_type"]), nodes, edges, dict(metadata))


# These names are semantic graph type markers.  Keeping one stable graph
# contract avoids introducing five subtly different graph implementations.
EmotionGraph = CreativeGraph
AudienceKnowledgeGraph = CreativeGraph
ConflictGraph = CreativeGraph
CharacterArcGraph = CreativeGraph
PlanLayerConsistencyGraph = CreativeGraph


__all__ = [
    "CreativeGraphNode",
    "CreativeGraphEdge",
    "CreativeGraph",
    "EmotionGraph",
    "AudienceKnowledgeGraph",
    "ConflictGraph",
    "CharacterArcGraph",
    "PlanLayerConsistencyGraph",
]

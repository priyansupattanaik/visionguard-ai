from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphNode:
    id: str
    kind: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    source: str
    relation: str
    target: str
    evidence_ids: list[str]
    start_seconds: float
    end_seconds: float


class VideoKnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        missing = {edge.source, edge.target} - self.nodes.keys()
        if missing:
            raise ValueError(f"graph edge references missing nodes: {sorted(missing)}")
        if not edge.evidence_ids:
            raise ValueError("graph edges require evidence provenance")
        self.edges.append(edge)

    def related(self, node_id: str, relation: str | None = None) -> list[GraphEdge]:
        return [
            edge for edge in self.edges
            if (edge.source == node_id or edge.target == node_id)
            and (relation is None or edge.relation == relation)
        ]

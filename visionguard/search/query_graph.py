"""LangGraph query orchestration for evidence-first video retrieval."""
from __future__ import annotations

from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .query_planner import DeterministicQueryPlanner


class QueryState(TypedDict, total=False):
    query: str
    detector_labels: list[str]
    top_k: int
    plan: dict
    evidence: list[dict]
    reasoning: str
    verification: dict
    answer: str
    abstained: bool


class VideoQueryGraph:
    """The sole query control path; nodes never manufacture source evidence."""

    def __init__(self, planner: DeterministicQueryPlanner, retrieve_object: Callable, retrieve_event: Callable,
                 retrieve_zone: Callable, retrieve_semantic: Callable, verify: Callable | None = None,
                 retrieve_count: Callable | None = None, retrieve_temporal: Callable | None = None):
        self.planner, self.retrieve_object, self.retrieve_event = planner, retrieve_object, retrieve_event
        self.retrieve_zone, self.retrieve_semantic, self.verify = retrieve_zone, retrieve_semantic, verify
        self.retrieve_count, self.retrieve_temporal = retrieve_count, retrieve_temporal
        graph = StateGraph(QueryState)
        graph.add_node("understand", self._understand)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("reason", self._reason)
        graph.add_node("verify", self._verify)
        graph.add_node("respond", self._respond)
        graph.add_edge(START, "understand")
        graph.add_edge("understand", "retrieve")
        graph.add_edge("retrieve", "reason")
        graph.add_edge("reason", "verify")
        graph.add_edge("verify", "respond")
        graph.add_edge("respond", END)
        self.graph = graph.compile()

    def _understand(self, state: QueryState) -> dict:
        plan = self.planner.plan(state["query"], state.get("detector_labels", []))
        words = set(plan.normalized_query.split())
        if {"verify", "confirm", "confirmed"}.intersection(words):
            plan.intent = "verification_request"
        elif {"zone", "gate", "entrance", "exit", "door", "region"}.intersection(words):
            plan.intent = "zone_search"
        elif plan.intent == "visual_search" and ({"scene", "happening", "describe"}.intersection(words) or plan.unknown_terms):
            plan.intent = "semantic_scene_search"
        if plan.intent == "semantic_scene_search" and plan.unknown_terms and {"find", "show", "locate"}.intersection(words):
            plan.intent = "unsupported_object"
            plan.limitations.append(
                "The requested object is not a runtime detector class. Use an explicit confirm request for bounded visual verification."
            )
        return {"plan": plan.to_dict()}

    def _retrieve(self, state: QueryState) -> dict:
        plan, top_k = state["plan"], state["top_k"]
        if plan["intent"] in {"unsupported_event", "unsupported_object"}:
            evidence = []
        elif plan.get("requires_count"):
            evidence = self.retrieve_count(plan, top_k) if self.retrieve_count else []
        elif plan.get("temporal_relation") != "none":
            evidence = self.retrieve_temporal(plan, top_k) if self.retrieve_temporal else []
        elif plan["intent"] == "object_search":
            evidence = self.retrieve_object(plan, top_k)
        elif plan["intent"] == "event_search":
            evidence = self.retrieve_event(plan, top_k)
        elif plan["intent"] == "zone_search":
            evidence = self.retrieve_zone(plan, top_k)
        else:
            evidence = self.retrieve_semantic(plan, top_k)
        return {"evidence": evidence}

    @staticmethod
    def _reason(state: QueryState) -> dict:
        evidence = state.get("evidence", [])
        return {"reasoning": f"Retrieved {len(evidence)} stored evidence item(s).", "abstained": not bool(evidence)}

    def _verify(self, state: QueryState) -> dict:
        if state["plan"]["intent"] != "verification_request":
            return {"verification": {"state": "not_requested", "confirmed": False}}
        if self.verify is None:
            return {
                "verification": {"state": "not_configured", "confirmed": False},
                "evidence": [],
                "abstained": True,
            }
        result = self.verify(state["query"], state.get("evidence", []))
        if not result.get("confirmed"):
            return {"verification": result, "evidence": [], "abstained": True}
        return {
            "verification": result,
            "evidence": list(result.get("evidence") or state.get("evidence", [])),
            "abstained": False,
        }

    @staticmethod
    def _respond(state: QueryState) -> dict:
        verification = state.get("verification", {})
        verification_required = state.get("plan", {}).get("intent") == "verification_request"
        if state.get("abstained") or (verification_required and not verification.get("confirmed")):
            return {"answer": "Insufficient grounded evidence found for this request."}
        if state.get("plan", {}).get("requires_count"):
            count = int(state["evidence"][0].get("count", 0))
            entities = state["plan"].get("entities", [])
            label = ", ".join(entities) if entities else "supported"
            return {"answer": f"Found {count} distinct tracked {label} object(s)."}
        citations = "; ".join(f"{row.get('peak_ts', row.get('timestamp', 0.0)):.3f}s" for row in state["evidence"])
        states = {row.get("evidence_state") for row in state["evidence"]}
        if states == {"semantic_description"}:
            return {"answer": f"Retrieved unverified semantic description evidence at {citations}."}
        if "verified_claim" in states:
            return {"answer": f"Verified evidence is available at {citations}."}
        return {"answer": f"Grounded evidence is available at {citations}."}

    def invoke(self, query: str, detector_labels: list[str], top_k: int = 4) -> dict:
        return self.graph.invoke({"query": query, "detector_labels": detector_labels, "top_k": top_k})

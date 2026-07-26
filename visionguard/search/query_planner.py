"""Deterministic, detector-aware query planning without an object whitelist."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path


class EvidenceKind(str, Enum):
    OBJECT = "object"
    TRACK = "track"
    EVENT = "event"
    SPEECH = "speech"
    SCENE = "scene"


class TemporalRelation(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    BETWEEN = "between"
    NONE = "none"


@dataclass(slots=True)
class QueryPlan:
    query: str
    normalized_query: str = ""
    intent: str = "visual_search"
    entities: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    unknown_terms: list[str] = field(default_factory=list)
    retrieval_routes: list[str] = field(default_factory=list)
    clarification: str | None = None
    limitations: list[str] = field(default_factory=list)
    temporal_relation: TemporalRelation = TemporalRelation.NONE
    reference_event: str | None = None
    requires_count: bool = False
    required_kinds: list[EvidenceKind] = field(default_factory=list)
    retrieval_limit: int = 20

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["temporal_relation"] = self.temporal_relation.value
        payload["required_kinds"] = [kind.value for kind in self.required_kinds]
        return payload


@lru_cache(maxsize=1)
def load_query_rules() -> dict:
    path = Path(__file__).with_name("query_rules.json")
    return json.loads(path.read_text(encoding="utf-8"))


class DeterministicQueryPlanner:
    """Routes natural-language queries using runtime detector labels and language rules."""

    def __init__(self, retrieval_limit: int = 20):
        if not 1 <= retrieval_limit <= 100:
            raise ValueError("retrieval_limit must be between 1 and 100")
        vocabulary = load_query_rules()
        self.attributes = set(vocabulary["attributes"])
        self.events = set(vocabulary["events"])
        self.temporal_markers = vocabulary["temporal_markers"]
        self.count_words = set(vocabulary["count_words"])
        self.entity_aliases = {
            str(canonical).casefold(): {
                " ".join(re.findall(r"[a-z0-9]+", str(alias).casefold()))
                for alias in aliases
            }
            for canonical, aliases in vocabulary.get("entity_aliases", {}).items()
        }
        self.speech_words = set(vocabulary["speech_words"])
        self.stop_words = set(vocabulary.get("stop_words", []))
        self.temporal_events = set(vocabulary.get("track_events", []))
        self.retrieval_limit = retrieval_limit

    @staticmethod
    def _label_forms(label: str) -> set[str]:
        label = " ".join(re.findall(r"[a-z0-9]+", label.casefold()))
        if not label:
            return set()
        words = label.split()
        last = words[-1]
        if last.endswith("y") and len(last) > 1:
            plural = f"{last[:-1]}ies"
        elif last.endswith(("s", "x", "z", "ch", "sh")):
            plural = f"{last}es"
        else:
            plural = f"{last}s"
        return {label, " ".join([*words[:-1], plural])}

    def resolve_entities(self, query: str, detector_labels=()) -> list[str]:
        normalized = " ".join(re.findall(r"[a-z0-9]+", query.casefold()))
        padded = f" {normalized} "
        resolved = set()
        for label in detector_labels or ():
            canonical = str(label).strip().casefold()
            if canonical and any(f" {form} " in padded for form in self._label_forms(canonical)):
                resolved.add(canonical)
                continue
            aliases = self.entity_aliases.get(canonical, set())
            if aliases and any(f" {alias} " in padded for alias in aliases):
                resolved.add(canonical)

        # Category aliases deliberately resolve only labels that the detector
        # actually supports. They turn a user request for "vehicles" into the
        # concrete stored classes, without claiming the detector saw a generic
        # vehicle class.
        vehicle_aliases = self.entity_aliases.get("vehicle", set())
        if vehicle_aliases and any(f" {alias} " in padded for alias in vehicle_aliases):
            vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
            supported = {str(label).strip().casefold() for label in detector_labels or ()}
            resolved.update(vehicle_classes.intersection(supported))
        return sorted(resolved)

    def plan(self, query: str, detector_labels=()) -> QueryPlan:
        normalized = query.strip().casefold()
        if not normalized:
            raise ValueError("query cannot be empty")
        words = re.findall(r"[a-z0-9]+", normalized)
        temporal = TemporalRelation.NONE
        for marker, relation_name in self.temporal_markers.items():
            if marker in words:
                temporal = TemporalRelation(relation_name)
                break

        entities = self.resolve_entities(normalized, detector_labels)
        attributes = sorted({word for word in words if word in self.attributes})
        events = sorted({word for word in words if word in self.events})
        recognized_words = set(attributes) | set(events) | self.stop_words | self.count_words | self.speech_words
        for name in entities:
            recognized_words.update(name.split())
            for form in self._label_forms(name):
                recognized_words.update(form.split())
            for alias in self.entity_aliases.get(name, set()):
                recognized_words.update(alias.split())
        if set(entities).intersection({"car", "truck", "bus", "motorcycle", "bicycle"}):
            for alias in self.entity_aliases.get("vehicle", set()):
                recognized_words.update(alias.split())
        unknown_terms = sorted({word for word in words if word not in recognized_words})

        routes = []
        if entities:
            routes.append("detector_metadata")
        if events and set(events).intersection(self.temporal_events):
            routes.append("track_events")
        elif events:
            routes.append("visual_semantic")
        if self.speech_words.intersection(words):
            routes.append("speech")
        if unknown_terms and "visual_semantic" not in routes:
            routes.append("visual_semantic")

        clarification = None
        intent = "visual_search"
        if not entities and not events and not attributes and not unknown_terms:
            intent = "ambiguous"
            clarification = "Describe the object, action, color, or event you want to find."
        elif self.count_words.intersection(words):
            intent = "count"
        elif events:
            intent = "event_search"
        elif entities:
            intent = "object_search"

        required = [EvidenceKind.EVENT] if events or temporal != TemporalRelation.NONE else [
            EvidenceKind.OBJECT,
            EvidenceKind.TRACK,
            EvidenceKind.SCENE,
        ]
        if self.speech_words.intersection(words):
            required.append(EvidenceKind.SPEECH)
        return QueryPlan(
            query=query,
            normalized_query=" ".join(words),
            intent=intent,
            entities=entities,
            attributes=attributes,
            events=events,
            unknown_terms=unknown_terms,
            retrieval_routes=routes,
            clarification=clarification,
            temporal_relation=temporal,
            reference_event=events[-1] if events else None,
            requires_count=bool(self.count_words.intersection(words)),
            required_kinds=required,
            retrieval_limit=self.retrieval_limit,
        )

from __future__ import annotations

import re

from visionguard.evidence_api.schemas.domain import EvidenceKind, QueryPlan, TemporalRelation
from visionguard.evidence_api.query_language import load_query_vocabulary


class DeterministicQueryPlanner:
    def __init__(self, retrieval_limit: int = 20):
        vocabulary = load_query_vocabulary()
        self.objects = set(vocabulary["objects"])
        self.attributes = set(vocabulary["attributes"])
        self.events = set(vocabulary["events"])
        self.temporal_markers = vocabulary["temporal_markers"]
        self.count_words = set(vocabulary["count_words"])
        self.speech_words = set(vocabulary["speech_words"])
        self.stop_words = set(vocabulary.get("stop_words", []))
        self.object_aliases = {
            alias.casefold(): [name.casefold() for name in names]
            for alias, names in vocabulary.get("object_aliases", {}).items()
        }
        self.temporal_events = set(vocabulary.get("track_events", []))
        self.retrieval_limit = retrieval_limit

    def resolve_entities(self, query: str, detector_labels=()) -> list[str]:
        normalized = " ".join(re.findall(r"[a-z0-9]+", query.casefold()))
        padded = f" {normalized} "
        resolved = set()
        for alias, names in self.object_aliases.items():
            if f" {alias} " in padded:
                resolved.update(names)
        for label in detector_labels or ():
            canonical = str(label).strip().casefold()
            if canonical and f" {canonical} " in padded:
                resolved.add(canonical)
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
        entities = sorted(
            set(self.resolve_entities(normalized, detector_labels))
            | {word for word in words if word in self.objects}
        )
        attributes = sorted({word for word in words if word in self.attributes})
        events = sorted({word for word in words if word in self.events})
        recognized_words = set(attributes) | set(events) | self.stop_words | self.count_words | self.speech_words
        for alias in self.object_aliases:
            if f" {alias} " in f" {' '.join(words)} ":
                recognized_words.update(alias.split())
        recognized_words.update(name for name in entities if " " not in name)
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
        if not routes and unknown_terms:
            routes.append("visual_semantic")

        clarification = None
        intent = "visual_search"
        if attributes and not entities:
            intent = "ambiguous"
            clarification = f"Which object should be {', '.join(attributes)}? For example: '{attributes[0]} car' or '{attributes[0]} person'."
        elif not entities and not events and not unknown_terms:
            intent = "ambiguous"
            clarification = "Describe the object, action, color, or event you want to find."
        elif self.count_words.intersection(words):
            intent = "count"
        elif events:
            intent = "event_search"
        elif entities:
            intent = "object_search"

        required = [EvidenceKind.EVENT] if events or temporal != TemporalRelation.NONE else [EvidenceKind.OBJECT, EvidenceKind.TRACK, EvidenceKind.SCENE]
        if self.speech_words.intersection(words):
            required.append(EvidenceKind.SPEECH)
        return QueryPlan(
            query=query, normalized_query=" ".join(words), intent=intent,
            entities=entities, attributes=attributes, events=events,
            unknown_terms=unknown_terms, retrieval_routes=routes,
            clarification=clarification,
            temporal_relation=temporal, reference_event=events[-1] if events else None,
            requires_count=bool(self.count_words.intersection(words)),
            required_kinds=required,
            retrieval_limit=self.retrieval_limit,
        )

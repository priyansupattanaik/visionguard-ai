from visionguard.evidence_api.schemas.domain import Citation, GroundedAnswer, QueryPlan


class GroundedResponder:
    FALLBACK = "I cannot verify this from the available evidence."

    def respond(self, plan: QueryPlan, reasoning: dict) -> GroundedAnswer:
        rows = reasoning.get("rows", [])
        if not reasoning.get("supported") or not rows:
            return GroundedAnswer(
                answer=self.FALLBACK, confidence=0, verified=False,
                uncertainty="No retrieved evidence passed verification.",
                reasoning_summary=reasoning.get("summary", ""),
            )
        citations = [Citation(
            evidence_id=row.evidence.id,
            timestamp_start=row.evidence.start_seconds,
            timestamp_end=row.evidence.end_seconds,
            frame_id=row.evidence.frame_id,
            track_ids=row.evidence.track_ids,
            event_ids=row.evidence.event_ids,
        ) for row in rows]
        confidence = min(row.verification_confidence for row in rows)
        return GroundedAnswer(
            answer=reasoning["summary"], confidence=confidence, citations=citations,
            verified=True, reasoning_summary="Answer synthesized exclusively from verified evidence.",
        )

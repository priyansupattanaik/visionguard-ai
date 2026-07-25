from __future__ import annotations

from visionguard.evidence_api.schemas.domain import QueryPlan, VerifiedEvidence


class EvidenceReasoner:
    """Produces a compact evidence synthesis; it cannot access the video or unverified rows."""

    def reason(self, plan: QueryPlan, verified: list[VerifiedEvidence]) -> dict:
        accepted = [row for row in verified if row.accepted]
        accepted.sort(key=lambda row: (row.evidence.start_seconds, row.evidence.id))
        if not accepted:
            return {"supported": False, "summary": "No candidate evidence passed verification.", "rows": []}
        if plan.requires_count:
            unique_tracks = {track for row in accepted for track in row.evidence.track_ids}
            count = len(unique_tracks) if unique_tracks else len(accepted)
            summary = f"Verified evidence supports a count of {count}."
        else:
            summary = " ".join(row.evidence.text for row in accepted if row.evidence.text).strip()
            if not summary:
                summary = "Verified evidence is present, but it has no textual description."
        return {"supported": True, "summary": summary, "rows": accepted}

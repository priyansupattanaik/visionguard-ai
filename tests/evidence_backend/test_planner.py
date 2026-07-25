from visionguard.evidence_api.agents import DeterministicQueryPlanner
from visionguard.evidence_api.schemas.domain import TemporalRelation

def test_planner_extracts_temporal_and_attributes():
    plan=DeterministicQueryPlanner().plan("What happened after the man entered near the red car?")
    assert plan.temporal_relation == TemporalRelation.AFTER
    assert {"man","car"} <= set(plan.entities) and "red" in plan.attributes and "entered" in plan.events

def test_planner_detects_counting():
    assert DeterministicQueryPlanner().plan("How many people entered?").requires_count

"""Schema tests: enum closure, defensive repairs, contract shape."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from frontline.schema import (
    Category,
    Priority,
    TriageDecision,
    safe_default,
)


def _valid(**over):
    base = dict(
        id="m1",
        category=Category.billing,
        priority=Priority.P2,
        summary="A short grounded summary.",
        suggested_action="Escalate to billing",
        needs_human=False,
        confidence=0.8,
    )
    base.update(over)
    return TriageDecision(**base)


def test_contract_fields_exact():
    d = _valid()
    contract = d.to_contract()
    assert set(contract) == {
        "id", "category", "priority", "summary", "suggested_action",
        "needs_human", "confidence",
    }
    assert contract["category"] == "billing"
    assert contract["priority"] == "P2"


def test_summary_truncated_to_240():
    d = _valid(summary="x" * 500)
    assert len(d.summary) <= 240


def test_confidence_clamped():
    assert _valid(confidence=5.0).confidence == 1.0
    assert _valid(confidence=-3.0).confidence == 0.0
    assert _valid(confidence="not a number").confidence == 0.0


def test_invalid_category_raises():
    with pytest.raises(ValidationError):
        TriageDecision(
            id="m1", category="nonsense", priority="P2",
            summary="s", suggested_action="a", needs_human=False, confidence=0.5,
        )


def test_safe_default_is_valid_and_conservative():
    d = safe_default("m9")
    assert d.id == "m9"
    assert d.needs_human is True
    assert d.priority == Priority.P2
    assert d.confidence == 0.0
    # round-trips
    TriageDecision.model_validate(d.model_dump())


def test_priority_rank_and_most_urgent():
    assert Priority.P0.rank < Priority.P3.rank
    assert Priority.most_urgent(Priority.P2, Priority.P0) == Priority.P0

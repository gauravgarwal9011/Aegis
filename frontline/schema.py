"""Pydantic v2 schema for triage decisions + the LangGraph typed state.

The `TriageDecision` model is the HARD OUTPUT CONTRACT. It is deliberately
defensive: length / range problems are *repaired* by field validators (so the
model never raises on cosmetic issues), while the closed enums stay strict (so
an invalid category/priority raises and is caught by the repair path).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

SUMMARY_MAX = 240


class Category(str, Enum):
    """Closed set of triage categories — never invent new ones."""

    billing = "billing"
    technical_issue = "technical_issue"
    account_access = "account_access"
    feature_request = "feature_request"
    complaint = "complaint"
    general_inquiry = "general_inquiry"
    spam_or_abuse = "spam_or_abuse"
    out_of_scope = "out_of_scope"


class Priority(str, Enum):
    """Urgency band.

    P0 = active outage / security or data breach / legal or safety threat /
         payment fraud — immediate human.
    P1 = blocking issue for a paying user, angry churn-risk, or time-sensitive.
    P2 = normal issue, no immediate blocker.
    P3 = low urgency: general questions, minor requests, thanks, spam.
    """

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"

    @property
    def rank(self) -> int:
        """0 == most urgent. Useful for "conservative (higher) priority" logic."""
        return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}[self.value]

    @classmethod
    def most_urgent(cls, a: "Priority", b: "Priority") -> "Priority":
        return a if a.rank <= b.rank else b


class TriageDecision(BaseModel):
    """The per-message structured decision. This IS the output contract."""

    id: str = Field(..., description="Stable id of the input message.")
    category: Category
    priority: Priority
    summary: str = Field(
        ...,
        description="<= 240 chars, factual, grounded ONLY in the message.",
    )
    suggested_action: str = Field(
        ..., description="Short imperative, e.g. 'Escalate to billing team'."
    )
    needs_human: bool
    confidence: float = Field(..., description="0.0-1.0")

    # --- defensive repairs (never raise on cosmetic issues) --------------
    @field_validator("summary", mode="before")
    @classmethod
    def _clip_summary(cls, v: object) -> str:
        s = "" if v is None else str(v)
        s = " ".join(s.split())  # collapse whitespace
        if len(s) > SUMMARY_MAX:
            s = s[: SUMMARY_MAX - 1].rstrip() + "…"
        return s

    @field_validator("suggested_action", mode="before")
    @classmethod
    def _clean_action(cls, v: object) -> str:
        return " ".join(("" if v is None else str(v)).split()) or "Review manually"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: object) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))

    def to_contract(self) -> dict:
        """Serialize to the plain JSON contract (enum values as strings)."""
        return {
            "id": self.id,
            "category": self.category.value,
            "priority": self.priority.value,
            "summary": self.summary,
            "suggested_action": self.suggested_action,
            "needs_human": self.needs_human,
            "confidence": round(self.confidence, 4),
        }


def safe_default(message_id: str, note: str = "safe default") -> TriageDecision:
    """The last-resort decision when everything else fails. Always valid."""
    return TriageDecision(
        id=message_id,
        category=Category.general_inquiry,
        priority=Priority.P2,
        summary=f"Unable to confidently triage this message ({note}).",
        suggested_action="Route to a human agent for manual review",
        needs_human=True,
        confidence=0.0,
    )


class State(TypedDict, total=False):
    """Typed LangGraph state threaded through the triage graph."""

    id: str
    message: str
    lang: Optional[str]
    sanitized_text: str
    is_empty: bool
    is_adversarial: bool
    decision: Optional[TriageDecision]
    attempts: int
    notes: list[str]
    token_usage: dict
    latency_ms: float
    enrich: bool

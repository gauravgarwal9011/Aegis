"""Batch orchestration: run the compiled graph over many messages.

Every message is isolated — one malformed/failed message must never abort the
batch. If the whole graph invocation throws for a message (it shouldn't), we
still emit a safe-default decision for that id.
"""

from __future__ import annotations

from typing import Iterable

from . import tracing
from .graph import build_graph
from .metrics import Aggregate, MessageMetric
from .llm import active_model
from .schema import TriageDecision, safe_default


def _normalize_message(raw: object, idx: int) -> dict:
    """Coerce arbitrary input into {id, text, lang} without ever throwing."""
    if isinstance(raw, dict):
        mid = str(raw.get("id") or f"auto-{idx:03d}")
        text = raw.get("text", raw.get("message", ""))
        lang = raw.get("lang")
    else:
        mid = f"auto-{idx:03d}"
        text = raw
        lang = None
    if text is None:
        text = ""
    return {"id": mid, "text": str(text), "lang": lang}


def triage_one(message: dict, enrich: bool = False) -> tuple[TriageDecision, MessageMetric]:
    """Run one message through the graph. Never raises."""
    app = build_graph()
    mid = message["id"]
    initial = {
        "id": mid,
        "message": message["text"],
        "lang": message.get("lang"),
        "notes": [],
        "enrich": enrich,
    }
    try:
        with tracing.span("triage.message", message_id=mid):
            final = app.invoke(initial)
        decision = final.get("decision") or safe_default(mid, "no decision in state")
        usage = final.get("token_usage", {}) or {}
        metric = MessageMetric(
            id=mid,
            model=active_model(),
            prompt_tokens=int(usage.get("prompt", 0)),
            completion_tokens=int(usage.get("completion", 0)),
            latency_ms=float(final.get("latency_ms", 0.0)),
        )
        return decision, metric
    except Exception as exc:  # absolute backstop — keep the batch alive
        decision = safe_default(mid, f"graph error: {exc!r}")
        return decision, MessageMetric(id=mid, model=active_model())


def run_batch(
    messages: Iterable[object], enrich: bool = False
) -> tuple[list[TriageDecision], Aggregate]:
    """Triage every message; isolate per-message failures."""
    tracing.configure()
    decisions: list[TriageDecision] = []
    agg = Aggregate()
    for idx, raw in enumerate(messages):
        msg = _normalize_message(raw, idx)
        decision, metric = triage_one(msg, enrich=enrich)
        decisions.append(decision)
        agg.add(metric)
    return decisions, agg

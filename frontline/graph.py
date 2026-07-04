"""The LangGraph StateGraph that triages one message.

Nodes (each wrapped in its own Logfire span):
  1. ingest_sanitize  - normalize/trim/limit length; detect empty/garbage;
                        flag obvious injection markers.
  2. classify         - LLM call (structured output) -> candidate decision.
                        Skips the LLM for empty/garbage input (cost win).
  3. validate_repair  - on failure retry classify once, then safe default.
  4. guardrails       - enforce enums, injection handling, groundedness, and
                        uncertainty routing; may override priority/needs_human.
  5. enrich (optional)- conditional: mock order lookup behind a flag.
  6. finalize         - attach metrics, emit final TriageDecision.

The graph is compiled once at import and reused.
"""

from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from typing import Optional

from langgraph.graph import END, START, StateGraph

from . import tracing
from .llm import (
    active_model,
    detect_injection,
    get_base_chat_model,
    rules_classify,
)
from .metrics import estimate_tokens
from .prompt import build_messages
from .schema import (
    Category,
    Priority,
    State,
    TriageDecision,
    safe_default,
)
from .tools import extract_order_id, lookup_order_status

MAX_ATTEMPTS = 2
MAX_LEN = int(os.getenv("MAX_MESSAGE_CHARS", "8000"))


def _confidence_threshold() -> float:
    try:
        return float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    except ValueError:
        return 0.6


@lru_cache(maxsize=1)
def _structured_llm():
    """Build & cache the structured-output model for the active provider."""
    return get_base_chat_model().with_structured_output(TriageDecision)


def reset_llm_cache() -> None:
    """Clear the cached model — used by tests that switch providers."""
    _structured_llm.cache_clear()


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def ingest_sanitize(state: State) -> dict:
    with tracing.span("node.ingest_sanitize", message_id=state.get("id", "")):
        raw = state.get("message", "")
        if not isinstance(raw, str):
            raw = str(raw)
        # Strip control chars, normalize whitespace, cap length.
        cleaned = "".join(ch for ch in raw if ch == "\n" or ch >= " ")
        cleaned = cleaned.replace("\r", " ").strip()
        truncated = cleaned[:MAX_LEN]
        notes = list(state.get("notes", []))
        if len(cleaned) > MAX_LEN:
            notes.append(f"truncated from {len(cleaned)} to {MAX_LEN} chars")

        is_empty = len(truncated.strip()) == 0 or _looks_like_garbage(truncated)
        is_adversarial = detect_injection(truncated)
        if is_adversarial:
            notes.append("injection markers detected — content treated as data")
        if is_empty:
            notes.append("empty/garbage input — LLM skipped")

        return {
            "sanitized_text": truncated,
            "is_empty": is_empty,
            "is_adversarial": is_adversarial,
            "attempts": 0,
            "notes": notes,
        }


def _looks_like_garbage(text: str) -> bool:
    """Emoji-only / punctuation-only / no alphanumeric content -> garbage."""
    t = text.strip()
    if not t:
        return True
    # No letters or digits anywhere (emoji-only, punctuation-only).
    return not any(c.isalnum() for c in t)


def classify(state: State) -> dict:
    msg_id = state.get("id", "")
    text = state.get("sanitized_text", "")
    attempts = state.get("attempts", 0) + 1
    notes = list(state.get("notes", []))

    # Cost optimization: skip the LLM entirely for empty/garbage messages.
    if state.get("is_empty"):
        decision = rules_classify(text, msg_id)  # returns a safe empty-decision
        return {
            "decision": decision,
            "attempts": attempts,
            "notes": notes,
            "token_usage": {"prompt": 0, "completion": 0},
            "latency_ms": 0.0,
        }

    messages = build_messages(text)
    prompt_text = "\n".join(
        m.content if isinstance(m.content, str) else str(m.content) for m in messages
    )
    model = active_model()
    start = time.perf_counter()
    decision: Optional[TriageDecision] = None
    with tracing.span(
        "llm.classify",
        message_id=msg_id,
        provider=os.getenv("LLM_PROVIDER", "mock"),
        model=model,
        attempt=attempts,
    ) as span:
        try:
            decision = _structured_llm().invoke(messages)
            # Never trust the model for the id.
            decision = decision.model_copy(update={"id": msg_id})
        except Exception as exc:  # transient / parse / validation error
            notes.append(f"classify attempt {attempts} failed: {exc!r}")
            decision = None
        latency_ms = (time.perf_counter() - start) * 1000.0
        p_tok = estimate_tokens(prompt_text)
        c_tok = estimate_tokens(decision.summary if decision else "")
        try:
            span.set_attribute("prompt_tokens", p_tok)
            span.set_attribute("completion_tokens", c_tok)
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("ok", decision is not None)
        except Exception:
            pass

    return {
        "decision": decision,
        "attempts": attempts,
        "notes": notes,
        "token_usage": {"prompt": p_tok, "completion": c_tok},
        "latency_ms": latency_ms,
    }


def validate_repair(state: State) -> dict:
    with tracing.span("node.validate_repair", message_id=state.get("id", "")):
        decision = state.get("decision")
        attempts = state.get("attempts", 0)
        notes = list(state.get("notes", []))

        if decision is not None:
            # Re-validate defensively (round-trips through Pydantic).
            try:
                decision = TriageDecision.model_validate(decision.model_dump())
                return {"decision": decision, "notes": notes}
            except Exception as exc:
                notes.append(f"revalidation failed: {exc!r}")
                decision = None

        if decision is None and attempts < MAX_ATTEMPTS:
            # Leave decision None; router sends us back to classify.
            notes.append("retrying classify after failure")
            return {"notes": notes}

        # Out of retries -> safe default (last-resort repair).
        notes.append("all classify attempts failed — using safe default")
        return {"decision": safe_default(state.get("id", ""), "repair fallback"), "notes": notes}


def route_after_validate(state: State) -> str:
    decision = state.get("decision")
    attempts = state.get("attempts", 0)
    if decision is None and attempts < MAX_ATTEMPTS:
        return "retry"
    return "ok"


def guardrails(state: State) -> dict:
    """The reliability core: enums, injection, groundedness, uncertainty."""
    with tracing.span("node.guardrails", message_id=state.get("id", "")):
        decision = state.get("decision") or safe_default(state.get("id", ""))
        notes = list(state.get("notes", []))
        source = state.get("sanitized_text", "")
        threshold = _confidence_threshold()

        category = decision.category
        priority = decision.priority
        summary = decision.summary
        action = decision.suggested_action
        needs_human = decision.needs_human
        confidence = decision.confidence

        # (2) Injection handling: content is data. Force spam_or_abuse + human,
        #     and set a fixed, content-independent priority (P3). This is what
        #     makes "priority unchanged by injected content" provable.
        if state.get("is_adversarial"):
            category = Category.spam_or_abuse
            priority = Priority.P3
            needs_human = True
            action = "Discard / block sender; do not action content"
            notes.append("guardrail: injection -> spam_or_abuse, P3, needs_human")

        # (3) Groundedness: every digit-run in the summary must appear in source.
        summary_digits = re.findall(r"\d+", summary)
        ungrounded = [d for d in summary_digits if d not in source]
        if ungrounded:
            # Rebuild summary from the source (grounded by construction).
            grounded = " ".join(source.split())[:200] or "Message received."
            notes.append(
                f"guardrail: dropped ungrounded digits {ungrounded}; summary regenerated"
            )
            summary = grounded
            confidence = min(confidence, 0.5)

        # (1) Uncertainty -> human, with conservative (higher) priority.
        uncertain = (
            confidence < threshold
            or category == Category.out_of_scope
            or state.get("is_empty")
        )
        if uncertain:
            needs_human = True
            if category not in (Category.spam_or_abuse,) and not state.get(
                "is_adversarial"
            ):
                priority = _bump_priority(priority)
                notes.append(
                    f"guardrail: low-confidence/ambiguous -> needs_human, priority->{priority.value}"
                )

        repaired = TriageDecision(
            id=state.get("id", ""),
            category=category,
            priority=priority,
            summary=summary,
            suggested_action=action,
            needs_human=needs_human,
            confidence=confidence,
        )
        return {"decision": repaired, "notes": notes}


def _bump_priority(p: Priority) -> Priority:
    """Move one band more urgent (P3->P2->P1). P0/P1 stay (already urgent)."""
    return {
        Priority.P3: Priority.P2,
        Priority.P2: Priority.P1,
        Priority.P1: Priority.P1,
        Priority.P0: Priority.P0,
    }[p]


def route_after_guardrails(state: State) -> str:
    if not state.get("enrich"):
        return "finalize"
    if extract_order_id(state.get("sanitized_text", "")):
        return "enrich"
    return "finalize"


def enrich(state: State) -> dict:
    """Optional mock enrichment — never breaks the main path."""
    with tracing.span("node.enrich", message_id=state.get("id", "")):
        notes = list(state.get("notes", []))
        try:
            order_id = extract_order_id(state.get("sanitized_text", ""))
            if order_id:
                info = lookup_order_status(order_id)
                notes.append(f"enrich: order {order_id} status={info['status']}")
        except Exception as exc:
            notes.append(f"enrich failed (ignored): {exc!r}")
        return {"notes": notes}


def finalize(state: State) -> dict:
    with tracing.span("node.finalize", message_id=state.get("id", "")):
        decision = state.get("decision") or safe_default(state.get("id", ""))
        return {"decision": decision}


# --------------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(State)
    g.add_node("ingest_sanitize", ingest_sanitize)
    g.add_node("classify", classify)
    g.add_node("validate_repair", validate_repair)
    g.add_node("guardrails", guardrails)
    g.add_node("enrich", enrich)
    g.add_node("finalize", finalize)

    g.add_edge(START, "ingest_sanitize")
    g.add_edge("ingest_sanitize", "classify")
    g.add_edge("classify", "validate_repair")
    g.add_conditional_edges(
        "validate_repair",
        route_after_validate,
        {"retry": "classify", "ok": "guardrails"},
    )
    g.add_conditional_edges(
        "guardrails",
        route_after_guardrails,
        {"enrich": "enrich", "finalize": "finalize"},
    )
    g.add_edge("enrich", "finalize")
    g.add_edge("finalize", END)
    return g.compile()

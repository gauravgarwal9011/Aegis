"""Mock enrichment tools for the optional `enrich` node.

Deterministic and offline. The enrich node is behind a flag and must never
break the main triage path — callers guard every call.
"""

from __future__ import annotations

import re

_ORDER_RE = re.compile(r"(?:order|#|ord)\s*#?\s*(\d{4,})", re.IGNORECASE)


def extract_order_id(text: str) -> str | None:
    """Return the first order-id-like token in the text, if any."""
    m = _ORDER_RE.search(text or "")
    return m.group(1) if m else None


def lookup_order_status(order_id: str) -> dict:
    """Mock order lookup — deterministic status derived from the id.

    In a real system this hits an orders service. Here it maps the id to one of
    a few statuses so the enrich path is exercised without network access.
    """
    statuses = ["processing", "shipped", "delivered", "refunded", "cancelled"]
    try:
        idx = int(order_id) % len(statuses)
    except (TypeError, ValueError):
        idx = 0
    return {
        "order_id": order_id,
        "status": statuses[idx],
        "found": True,
    }

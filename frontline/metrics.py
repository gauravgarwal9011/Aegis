"""Per-message and aggregate token / cost / latency metrics.

Tokens are *estimated* for the mock provider (and, for simplicity, also for
real providers here — Logfire captures exact provider-reported counts in its
spans). Cost uses a small, configurable price table in USD per 1M tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per 1,000,000 tokens: (prompt_price, completion_price)
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "mock-rules-v1": (0.0, 0.0),
    # Anthropic (current tiers)
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Groq (approximate published per-1M pricing; confirm against groq.com)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}
_DEFAULT_PRICE = (0.50, 1.50)  # conservative fallback for unknown models


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token, min 1 for non-empty text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_price, c_price = PRICE_TABLE.get(model, _DEFAULT_PRICE)
    return (prompt_tokens * p_price + completion_tokens * c_price) / 1_000_000.0


@dataclass
class MessageMetric:
    id: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        return cost_usd(self.model, self.prompt_tokens, self.completion_tokens)


@dataclass
class Aggregate:
    metrics: list[MessageMetric] = field(default_factory=list)

    def add(self, m: MessageMetric) -> None:
        self.metrics.append(m)

    @property
    def count(self) -> int:
        return len(self.metrics)

    @property
    def total_tokens(self) -> int:
        return sum(m.total_tokens for m in self.metrics)

    @property
    def total_cost(self) -> float:
        return sum(m.cost for m in self.metrics)

    @property
    def total_latency_ms(self) -> float:
        return sum(m.latency_ms for m in self.metrics)

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.count if self.count else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.count if self.count else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.count if self.count else 0.0

    def summary(self) -> dict:
        return {
            "messages": self.count,
            "total_tokens": self.total_tokens,
            "avg_tokens": round(self.avg_tokens, 1),
            "total_cost_usd": round(self.total_cost, 6),
            "avg_cost_usd": round(self.avg_cost, 6),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }

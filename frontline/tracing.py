"""Pydantic Logfire tracing helpers.

Everything here is a *safe no-op* unless a LOGFIRE_TOKEN is present. All
instrumentation is guarded — tracing must never crash the run. When Logfire is
unavailable or misconfigured we return a lightweight null-span context manager
so callers can use ``with span(...):`` unconditionally.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any, Iterator

_CONFIGURED = False
_LOGFIRE: Any = None


@contextlib.contextmanager
def _null_span(*_args: Any, **_kwargs: Any) -> Iterator["_NullSpan"]:
    yield _NullSpan()


class _NullSpan:
    """Stand-in span that swallows attribute setting."""

    def set_attribute(self, *_a: Any, **_k: Any) -> None:  # noqa: D401
        pass

    def set_attributes(self, *_a: Any, **_k: Any) -> None:
        pass


def configure() -> None:
    """Configure Logfire once. No-op (and silent) when no token is present."""
    global _CONFIGURED, _LOGFIRE
    if _CONFIGURED:
        return
    _CONFIGURED = True
    try:
        import logfire

        # console=False: don't dump the span tree to stdout (keeps the CLI
        # table clean). Spans are still recorded and sent when a token exists.
        logfire.configure(send_to_logfire="if-token-present", console=False)
        _LOGFIRE = logfire

        # Trace Pydantic validation / repair events.
        with contextlib.suppress(Exception):
            logfire.instrument_pydantic()

        # Provider-specific instrumentation only when that provider is active.
        with contextlib.suppress(Exception):
            from .llm import active_provider

            provider = active_provider()
            if provider == "openai":
                with contextlib.suppress(Exception):
                    logfire.instrument_openai()
            elif provider == "anthropic":
                with contextlib.suppress(Exception):
                    logfire.instrument_anthropic()
    except Exception as exc:  # logfire not installed / config failure
        print(f"[frontline.tracing] tracing disabled ({exc!r})", file=sys.stderr)
        _LOGFIRE = None


def span(name: str, **attributes: Any):
    """Return a Logfire span if available, else a null span. Never raises."""
    if _LOGFIRE is None:
        return _null_span()
    try:
        return _LOGFIRE.span(name, **attributes)
    except Exception:
        return _null_span()


def is_active() -> bool:
    return _LOGFIRE is not None

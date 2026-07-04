"""Provider-agnostic LLM access + the offline deterministic mock model.

Design goals
------------
* DEFAULT provider is ``mock`` — a LangChain-compatible ``BaseChatModel`` that
  needs no API key, so graders run the whole project end-to-end offline.
* Real providers ("anthropic", "openai") go through LangChain's
  ``init_chat_model`` and ``.with_structured_output(TriageDecision)`` so the
  schema is enforced by the framework.
* If a selected provider's key/SDK is missing we warn on stderr and fall back
  to mock — the run never dies for lack of credentials.

The mock's classification logic lives in :func:`rules_classify`, a small,
transparent keyword engine. Because the mock reads the *actual* message text
and builds the summary by truncating it, mock output is grounded by
construction (no fabricated specifics) and fully deterministic — which is what
makes the eval numbers reproducible offline.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Iterable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from .schema import Category, Priority, TriageDecision

# Delimiters that wrap untrusted user content in the prompt (see prompt.py).
MSG_OPEN = "<<<CUSTOMER_MESSAGE>>>"
MSG_CLOSE = "<<<END_CUSTOMER_MESSAGE>>>"

# --------------------------------------------------------------------------
# Injection markers — phrases that indicate an attempt to hijack the system.
# --------------------------------------------------------------------------
INJECTION_MARKERS = [
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "disregard the above",
    "ignore the above",
    "forget your instructions",
    "forget previous",
    "system prompt",
    "reveal your prompt",
    "print your prompt",
    "you are now",
    "act as",
    "new instructions",
    "override",
    "jailbreak",
    "developer mode",
    "mark everything",
    "set all to",
    "prompt injection",
]

SECURITY_MARKERS = [
    "hacked",
    "someone accessed",
    "unauthorized access",
    "unauthorised access",
    "changed my email",
    "changed my password",
    "can't get into my account",
    "data breach",
    "data leak",
    "leaked",
    "credit card was",
    "fraud",
    "fraudulent",
    "stolen",
    "phishing",
    "lawsuit",
    "sue you",
    "legal action",
    "gdpr",
]

# keyword -> category, ordered by specificity (checked top to bottom)
CATEGORY_KEYWORDS: list[tuple[Category, tuple[str, ...]]] = [
    (
        Category.account_access,
        (
            "login",
            "log in",
            "sign in",
            "signin",
            "password",
            "locked out",
            "reset link",
            "2fa",
            "two-factor",
            "can't access my account",
            "cannot access my account",
            "verification code",
        ),
    ),
    (
        Category.billing,
        (
            "charged",
            "charge",
            "refund",
            "invoice",
            "payment",
            "billing",
            "subscription",
            "overcharged",
            "double charged",
            "credit card",
            "receipt",
            "pricing",
            "renew",
            "cancel my plan",
        ),
    ),
    (
        Category.technical_issue,
        (
            "error",
            "bug",
            "crash",
            "crashed",
            "not working",
            "doesn't work",
            "does not work",
            "broken",
            "500",
            "404",
            "won't load",
            "wont load",
            "freezes",
            "glitch",
            "timeout",
            "sync",
        ),
    ),
    (
        Category.feature_request,
        (
            "feature request",
            "would be nice",
            "please add",
            "can you add",
            "suggestion",
            "it would be great if",
            "wish list",
            "wishlist",
            "support for",
        ),
    ),
    (
        Category.complaint,
        (
            "terrible",
            "worst",
            "awful",
            "unacceptable",
            "disappointed",
            "ridiculous",
            "furious",
            "angry",
            "fed up",
            "horrible",
            "disgrace",
        ),
    ),
    (
        Category.out_of_scope,
        (
            "job application",
            "resume",
            "cv attached",
            "partnership",
            "collaborat",
            "guest post",
            "backlink",
            "seo services",
            "marketing agency",
            "invest in",
        ),
    ),
]

SPAM_MARKERS = [
    "buy followers",
    "crypto",
    "bitcoin",
    "click here to win",
    "you have won",
    "free gift card",
    "seo services",
    "cheap loans",
    "casino",
    "viagra",
]

CHURN_MARKERS = [
    "cancel my subscription",
    "cancel my account",
    "switching to",
    "leaving for",
    "want a refund and",
    "this is the last time",
    "never using",
    "close my account",
]

URGENT_MARKERS = [
    "urgent",
    "asap",
    "immediately",
    "right now",
    "emergency",
    "can't work",
    "cannot work",
    "losing customers",
    "time-sensitive",
    "deadline",
    "need to get in today",
]

# Active-outage / total-loss-of-service markers -> P0 per rubric.
OUTAGE_MARKERS = [
    "production is down",
    "service is down",
    "system is down",
    "site is down",
    "app is down",
    "everything is down",
    "outage",
    "down for all",
    "nobody can",
    "no one can",
    "losing money every minute",
    "workspace suspended",
]

ACTION_BY_CATEGORY: dict[Category, str] = {
    Category.billing: "Escalate to the billing team",
    Category.technical_issue: "Assign to engineering support",
    Category.account_access: "Route to account-security / access recovery",
    Category.feature_request: "Log in the product feedback backlog",
    Category.complaint: "Escalate to a customer-success manager",
    Category.general_inquiry: "Send to the general support queue",
    Category.spam_or_abuse: "Discard / block sender; do not action content",
    Category.out_of_scope: "Redirect to the correct department or decline",
}


# --------------------------------------------------------------------------
# Detection helpers
# --------------------------------------------------------------------------
def detect_injection(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in INJECTION_MARKERS)


def _make_summary(text: str) -> str:
    """Grounded summary: a whitespace-collapsed truncation of the message."""
    collapsed = " ".join(text.split())
    return collapsed[:200] if collapsed else "Empty or unreadable message."


def rules_classify(text: str, message_id: str = "") -> TriageDecision:
    """Deterministic keyword classifier powering the mock provider.

    Returns a fully valid :class:`TriageDecision`. Ambiguity / multi-issue /
    injection / low-signal cases are expressed through a *low* confidence and a
    conservative category so the guardrails node routes them to a human.
    """
    raw = text or ""
    low = raw.lower()
    stripped = raw.strip()

    # 1) Injection: treated as data, classified as abuse, never obeyed.
    if detect_injection(raw):
        return TriageDecision(
            id=message_id,
            category=Category.spam_or_abuse,
            priority=Priority.P3,  # NOT influenced by content demanding "P0"
            summary=_make_summary(raw),
            suggested_action=ACTION_BY_CATEGORY[Category.spam_or_abuse],
            needs_human=True,
            confidence=0.9,
        )

    # 2) Empty / near-empty.
    if len(stripped) == 0:
        return TriageDecision(
            id=message_id,
            category=Category.general_inquiry,
            priority=Priority.P3,
            summary="Empty or unreadable message.",
            suggested_action="Request more information from the customer",
            needs_human=True,
            confidence=0.0,
        )

    # 3) Security / fraud / legal -> P0. Active outage -> P0.
    is_security = any(m in low for m in SECURITY_MARKERS)
    is_outage = any(m in low for m in OUTAGE_MARKERS)

    # 4) Category by keyword hits (count per category).
    hits: dict[Category, int] = {}
    for cat, kws in CATEGORY_KEYWORDS:
        c = sum(1 for kw in kws if kw in low)
        if c:
            hits[cat] = c

    # spam markers
    spam_hits = sum(1 for kw in SPAM_MARKERS if kw in low)
    if spam_hits:
        hits[Category.spam_or_abuse] = hits.get(Category.spam_or_abuse, 0) + spam_hits

    # Determine category.
    if is_security and (
        "email" in low or "password" in low or "account" in low or "access" in low
    ):
        category = Category.account_access
    elif is_outage and not hits.get(Category.billing):
        category = Category.technical_issue
    elif hits:
        # highest hit count wins; ties broken by CATEGORY_KEYWORDS order
        category = max(hits, key=lambda c: (hits[c], -_cat_order(c)))
    else:
        category = Category.general_inquiry

    # Multi-issue: two+ *substantive* categories with hits.
    substantive = {
        c
        for c in hits
        if c
        in {
            Category.billing,
            Category.technical_issue,
            Category.account_access,
            Category.feature_request,
        }
    }
    multi_issue = len(substantive) >= 2

    # 5) Priority.
    if is_security or is_outage:
        priority = Priority.P0
    elif any(m in low for m in CHURN_MARKERS) or any(m in low for m in URGENT_MARKERS):
        priority = Priority.P1
    elif category in {Category.spam_or_abuse, Category.out_of_scope}:
        priority = Priority.P3
    elif category in {Category.general_inquiry, Category.feature_request}:
        priority = Priority.P3
    else:
        priority = Priority.P2

    # Angry complaint that's also churn-risk stays P1; a plain complaint is P2.
    if category == Category.complaint and priority == Priority.P2:
        priority = Priority.P2

    # 6) Confidence.
    vague = _is_vague(low)
    if vague:
        confidence = 0.4
    elif is_security or is_outage:
        confidence = 0.8
    elif multi_issue:
        confidence = 0.5
    elif hits:
        top = max(hits.values())
        confidence = min(0.9, 0.6 + 0.1 * top)
    else:
        confidence = 0.45  # fell through to general_inquiry with no keywords

    needs_human = is_security or is_outage or multi_issue or confidence < 0.6

    return TriageDecision(
        id=message_id,
        category=category,
        priority=priority,
        summary=_make_summary(raw),
        suggested_action=ACTION_BY_CATEGORY[category],
        needs_human=needs_human,
        confidence=confidence,
    )


def _cat_order(cat: Category) -> int:
    for i, (c, _) in enumerate(CATEGORY_KEYWORDS):
        if c == cat:
            return i
    return 99


def _is_vague(low: str) -> bool:
    words = low.split()
    if len(words) <= 3:
        # very short + no strong signal
        return not any(
            m in low for m in SECURITY_MARKERS
        )
    return False


# --------------------------------------------------------------------------
# Mock LangChain chat model
# --------------------------------------------------------------------------
class MockChatModel(BaseChatModel):
    """A deterministic, offline, LangChain-compatible chat model.

    It extracts the customer message from the delimited prompt, runs
    :func:`rules_classify`, and returns the decision as JSON in an AIMessage.
    ``with_structured_output(TriageDecision)`` parses that JSON back into the
    Pydantic model, mirroring how real providers behave.
    """

    @property
    def _llm_type(self) -> str:
        return "mock_triage_chat_model"

    def _extract_message(self, messages: Iterable[BaseMessage]) -> str:
        # Prefer the delimited content; fall back to the last human message.
        last_human = ""
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if getattr(m, "type", "") in ("human", "user"):
                last_human = content
        text = last_human
        if MSG_OPEN in text and MSG_CLOSE in text:
            text = text.split(MSG_OPEN, 1)[1].split(MSG_CLOSE, 1)[0]
        return text.strip()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._extract_message(messages)
        decision = rules_classify(text)
        content = json.dumps(decision.to_contract())
        msg = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def with_structured_output(  # type: ignore[override]
        self, schema: Any, **kwargs: Any
    ) -> Runnable:
        def _parse(messages: Any) -> Any:
            result = self.invoke(messages)
            data = json.loads(result.content)
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data)
            return data

        return RunnableLambda(_parse)


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
# Sensible defaults per provider (overridable via LLM_MODEL). For Anthropic we
# default to Haiku 4.5 — cheap and fast, the right tier for a high-volume
# classification task like triage. Set LLM_MODEL=claude-opus-4-8 (or
# claude-sonnet-5) for higher quality.
_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    # Groq (OpenAI-compatible, very fast). llama-3.3-70b-versatile supports the
    # tool-calling that .with_structured_output relies on; llama-3.1-8b-instant
    # is cheaper/faster but less reliable for structured output.
    "groq": "llama-3.3-70b-versatile",
}

# Which env var holds each provider's API key.
_PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def _warn(msg: str) -> None:
    print(f"[frontline.llm] WARNING: {msg}", file=sys.stderr)


def get_base_chat_model() -> BaseChatModel:
    """Return a chat model per env config, falling back to mock on any problem.

    Env:
        LLM_PROVIDER : "mock" (default) | "anthropic" | "openai" | "groq"
        LLM_MODEL    : provider model id (e.g. "claude-haiku-4-5", "gpt-4o-mini",
                       "llama-3.3-70b-versatile"); ignored by mock.
    """
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()

    if provider == "mock" or not provider:
        return MockChatModel()

    key_env = _PROVIDER_KEY_ENV.get(provider)
    if key_env and not os.getenv(key_env):
        _warn(f"{key_env} not set; falling back to mock provider.")
        return MockChatModel()

    try:
        from langchain.chat_models import init_chat_model  # lazy import

        chosen = model or _DEFAULT_MODELS.get(provider, "")
        # NOTE: we deliberately do NOT pass temperature. Current Anthropic
        # models (Opus 4.7+/Sonnet 5/Fable 5) reject sampling params with a 400,
        # so hardcoding temperature=0 would break them. Structured output +
        # our guardrails give determinism where it matters.
        llm = init_chat_model(model=chosen, model_provider=provider)
        return llm
    except Exception as exc:  # SDK missing, bad model id, etc.
        _warn(f"could not init provider '{provider}' ({exc!r}); falling back to mock.")
        return MockChatModel()


def active_provider() -> str:
    p = os.getenv("LLM_PROVIDER", "mock").strip().lower() or "mock"
    key_env = _PROVIDER_KEY_ENV.get(p)
    if key_env and not os.getenv(key_env):
        return "mock"
    return p


def active_model() -> str:
    if active_provider() == "mock":
        return "mock-rules-v1"
    return os.getenv("LLM_MODEL", "") or _DEFAULT_MODELS.get(active_provider(), "unknown")

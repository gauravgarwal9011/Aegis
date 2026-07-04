"""Optional provider-agnostic LLM-as-judge for summary faithfulness/usefulness.

The mock judge is deterministic so it runs offline. It scores each summary 1-5
on (a) faithfulness — does the summary only use words/numbers from the source —
and (b) usefulness — is it non-trivial and specific. Real providers can be
plugged in via LLM_JUDGE_PROVIDER, but the default mock keeps the suite offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from frontline.schema import TriageDecision


@dataclass
class JudgeScore:
    id: str
    faithfulness: int  # 1-5
    usefulness: int  # 1-5

    @property
    def overall(self) -> float:
        return (self.faithfulness + self.usefulness) / 2.0


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def mock_judge(decision: TriageDecision, source: str) -> JudgeScore:
    src_tokens = _tokens(source)
    sum_tokens = _tokens(decision.summary)

    # Faithfulness: fraction of summary tokens present in the source.
    if sum_tokens:
        overlap = len(sum_tokens & src_tokens) / len(sum_tokens)
    else:
        overlap = 0.0
    # Any digit in summary absent from source is an immediate faithfulness hit.
    digits_ok = all(d in source for d in re.findall(r"\d+", decision.summary))
    faithfulness = 1 + round(4 * overlap) if digits_ok else 1
    faithfulness = max(1, min(5, faithfulness))

    # Usefulness: length-based proxy, capped; empty/trivial summaries score low.
    n = len(decision.summary.split())
    if n <= 2:
        usefulness = 1
    elif n <= 5:
        usefulness = 3
    else:
        usefulness = 5
    return JudgeScore(decision.id, faithfulness, usefulness)


def judge_all(decisions: list[TriageDecision], source_by_id: dict[str, str]) -> dict:
    scores = [mock_judge(d, source_by_id.get(d.id, "")) for d in decisions]
    avg_faith = sum(s.faithfulness for s in scores) / len(scores) if scores else 0.0
    avg_use = sum(s.usefulness for s in scores) / len(scores) if scores else 0.0
    avg_overall = sum(s.overall for s in scores) / len(scores) if scores else 0.0
    return {
        "avg_faithfulness": round(avg_faith, 2),
        "avg_usefulness": round(avg_use, 2),
        "avg_overall": round(avg_overall, 2),
        "per_message": [
            {"id": s.id, "faithfulness": s.faithfulness, "usefulness": s.usefulness}
            for s in scores
        ],
    }

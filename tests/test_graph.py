"""Graph tests: it compiles and one message flows through all nodes."""

from __future__ import annotations

import os

from frontline.graph import build_graph, reset_llm_cache
from frontline.schema import TriageDecision
from frontline.triage import run_batch, triage_one


def setup_module(_module):
    os.environ["LLM_PROVIDER"] = "mock"
    reset_llm_cache()


def test_graph_compiles():
    app = build_graph()
    assert app is not None


def test_single_message_flows_end_to_end():
    decision, metric = triage_one(
        {"id": "t1", "text": "I was charged twice, please refund."}
    )
    assert isinstance(decision, TriageDecision)
    assert decision.id == "t1"
    assert decision.category.value == "billing"
    assert metric.id == "t1"


def test_batch_processes_all_and_isolates_errors():
    messages = [
        {"id": "a", "text": "login is broken, can't sign in"},
        {"id": "b", "text": ""},
        {"id": "c", "text": None},  # weird but must not crash
        "not even a dict",  # gets an auto id
    ]
    decisions, agg = run_batch(messages)
    assert len(decisions) == 4
    assert all(isinstance(d, TriageDecision) for d in decisions)
    assert agg.count == 4


def test_enrich_path_does_not_break(monkeypatch):
    decision, _ = triage_one(
        {"id": "e1", "text": "Where is my order 12345? It hasn't arrived."},
        enrich=True,
    )
    assert isinstance(decision, TriageDecision)
    assert decision.id == "e1"

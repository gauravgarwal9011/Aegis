"""Scored eval suite for the Frontline triage system.

Runs against a REAL mock run (offline, deterministic). Real-LLM numbers require
provider API keys — the harness is provider-agnostic and would score whatever
provider is configured via LLM_PROVIDER.
"""

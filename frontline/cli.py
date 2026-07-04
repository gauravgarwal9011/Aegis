"""CLI entrypoint: load dataset -> run graph -> table + results.json + report.html.

    python -m frontline.cli --input data/messages.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console

from .triage import run_batch
from .ui_cli import render_counts, render_metrics, render_table
from .ui_html import write_html


def _load_dotenv() -> None:
    """Load a local .env into os.environ if python-dotenv is installed.

    Optional dependency — the mock path needs no env at all, so a missing
    python-dotenv is a silent no-op.
    """
    try:
        from dotenv import find_dotenv, load_dotenv

        # usecwd=True: find .env relative to where the command was run.
        load_dotenv(find_dotenv(usecwd=True))
    except Exception:
        pass


def _load_messages(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Input file not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    except json.JSONDecodeError as exc:
        print(f"Input file is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if isinstance(data, dict) and "messages" in data:
        data = data["messages"]
    if not isinstance(data, list):
        print("Input JSON must be a list of messages.", file=sys.stderr)
        raise SystemExit(2)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="frontline", description="Triage customer messages.")
    parser.add_argument("--input", "-i", default="data/messages.json", help="Path to messages JSON.")
    parser.add_argument("--out-dir", default="outputs", help="Directory for results.json / report.html.")
    parser.add_argument("--enrich", action="store_true", help="Enable the optional order-lookup enrich node.")
    parser.add_argument("--no-html", action="store_true", help="Skip writing the HTML report.")
    args = parser.parse_args(argv)

    # Load .env (provider, model, API keys) if python-dotenv is available.
    _load_dotenv()

    # Windows consoles default to cp1252 and choke on non-ASCII (accented /
    # non-English) output. Force UTF-8 so the run never dies on rendering.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    console = Console()
    messages = _load_messages(Path(args.input))
    console.print(
        f"[bold]Frontline[/] triaging {len(messages)} messages "
        f"(provider=[cyan]{os.getenv('LLM_PROVIDER', 'mock')}[/])…\n"
    )

    decisions, agg = run_batch(messages, enrich=args.enrich)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps([d.to_contract() for d in decisions], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    render_table(decisions, console)
    render_counts(decisions, console)
    render_metrics(agg, console)
    console.print(f"\nWrote [green]{results_path}[/]")

    if not args.no_html:
        # Surface the eval overall grade in the report if a prior eval run exists.
        eval_score = None
        eval_report = Path("eval") / "report.json"
        if eval_report.exists():
            try:
                eval_score = json.loads(eval_report.read_text(encoding="utf-8")).get(
                    "overall_grade"
                )
            except Exception:
                eval_score = None
        report_path = out_dir / "report.html"
        write_html(str(report_path), decisions, agg=agg, eval_score=eval_score)
        console.print(f"Wrote [green]{report_path}[/]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

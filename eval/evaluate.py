"""Scored eval harness: ground-truth agreement (A) + behavioral evals (B) +
LLM-as-judge (C). Prints a scored table, writes eval/report.json and
eval/report.html. Numbers are computed from a REAL mock run — never fabricated.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from frontline.schema import Priority, TriageDecision
from frontline.triage import run_batch, triage_one

from .cases import BEHAVIORAL_CASES, EvalContext
from .judge import judge_all

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EVAL_DIR = ROOT / "eval"


def _load(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_one_text(text: str) -> TriageDecision:
    decision, _ = triage_one({"id": "probe", "text": text})
    return decision


# --------------------------------------------------------------------------
# A) Ground-truth agreement
# --------------------------------------------------------------------------
def ground_truth_agreement(by_id: dict[str, TriageDecision], labels: list[dict]) -> dict:
    n = len(labels)
    cat_ok = pri_exact = pri_within1 = human_ok = 0
    mismatches = []
    for lbl in labels:
        mid = lbl["id"]
        d = by_id.get(mid)
        if d is None:
            mismatches.append({"id": mid, "field": "missing", "system": "-", "label": "-", "message": lbl.get("text", "")})
            continue
        c_ok = d.category.value == lbl["category"]
        sys_rank = d.priority.rank
        lbl_rank = Priority(lbl["priority"]).rank
        p_exact = d.priority.value == lbl["priority"]
        p_w1 = abs(sys_rank - lbl_rank) <= 1
        h_ok = bool(d.needs_human) == bool(lbl["needs_human"])

        cat_ok += int(c_ok)
        pri_exact += int(p_exact)
        pri_within1 += int(p_w1)
        human_ok += int(h_ok)

        for field, ok, sysv, lblv in (
            ("category", c_ok, d.category.value, lbl["category"]),
            ("priority", p_exact, d.priority.value, lbl["priority"]),
            ("needs_human", h_ok, d.needs_human, lbl["needs_human"]),
        ):
            if not ok:
                mismatches.append({
                    "id": mid, "field": field, "system": sysv, "label": lblv,
                    "message": lbl.get("text", ""),
                })

    # overall = exact matches across the 3 core fields / (3 * N)
    overall = 100.0 * (cat_ok + pri_exact + human_ok) / (3 * n) if n else 0.0
    return {
        "count": n,
        "overall_pct": round(overall, 1),
        "category_exact_pct": round(100.0 * cat_ok / n, 1) if n else 0.0,
        "priority_exact_pct": round(100.0 * pri_exact / n, 1) if n else 0.0,
        "priority_within1_pct": round(100.0 * pri_within1 / n, 1) if n else 0.0,
        "needs_human_pct": round(100.0 * human_ok / n, 1) if n else 0.0,
        "mismatches": mismatches,
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_eval() -> dict:
    messages = _load(DATA / "messages.json")
    labels = _load(DATA / "ground_truth.json")
    source_by_id = {m["id"]: m.get("text", "") for m in messages}

    decisions, agg = run_batch(messages)
    by_id = {d.id: d for d in decisions}

    # A) agreement
    agreement = ground_truth_agreement(by_id, labels)

    # B) behavioral
    ctx = EvalContext(decisions=decisions, source_by_id=source_by_id, run_one=_run_one_text)
    behavioral = [c(ctx) for c in BEHAVIORAL_CASES]
    behavioral_mean = sum(r.score for r in behavioral) / len(behavioral) if behavioral else 0.0

    # C) judge
    judged = judge_all(decisions, source_by_id)

    overall_grade = round((agreement["overall_pct"] + behavioral_mean) / 2.0, 1)

    report = {
        "provider_note": "Computed from a real mock run; real-LLM numbers require API keys.",
        "ground_truth": agreement,
        "behavioral": [
            {"name": r.name, "score": round(r.score, 1), "passed": r.passed, "detail": r.detail}
            for r in behavioral
        ],
        "behavioral_mean": round(behavioral_mean, 1),
        "judge": judged,
        "overall_grade": overall_grade,
        "metrics": agg.summary(),
    }
    return report


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def print_report(report: dict, console: Console | None = None) -> None:
    console = console or Console()

    gt = report["ground_truth"]
    console.print("\n[bold underline]A) Ground-truth agreement[/]")
    gt_table = Table(show_header=True)
    gt_table.add_column("metric")
    gt_table.add_column("score", justify="right")
    for label, key in (
        ("overall", "overall_pct"),
        ("category (exact)", "category_exact_pct"),
        ("priority (exact)", "priority_exact_pct"),
        ("priority (within-1)", "priority_within1_pct"),
        ("needs_human", "needs_human_pct"),
    ):
        gt_table.add_row(label, f"{gt[key]:.1f}%")
    console.print(gt_table)

    if gt["mismatches"]:
        console.print("[bold]Mismatch table (system vs label):[/]")
        mt = Table(show_header=True)
        mt.add_column("id"); mt.add_column("field"); mt.add_column("system")
        mt.add_column("label"); mt.add_column("message", overflow="fold")
        for m in gt["mismatches"]:
            mt.add_row(m["id"], m["field"], str(m["system"]), str(m["label"]), m["message"][:80])
        console.print(mt)
    else:
        console.print("[green]No mismatches.[/]")

    console.print("\n[bold underline]B) Behavioral / reliability evals[/]")
    bt = Table(show_header=True)
    bt.add_column("eval"); bt.add_column("score", justify="right"); bt.add_column("pass")
    bt.add_column("detail", overflow="fold")
    for r in report["behavioral"]:
        mark = "[green]PASS[/]" if r["passed"] else "[red]FAIL[/]"
        bt.add_row(r["name"], f"{r['score']:.1f}", mark, r["detail"])
    console.print(bt)
    console.print(f"behavioral mean: [bold]{report['behavioral_mean']:.1f}[/]")

    j = report["judge"]
    console.print("\n[bold underline]C) LLM-as-judge (mock, offline)[/]")
    console.print(
        f"avg faithfulness: {j['avg_faithfulness']}/5 | "
        f"avg usefulness: {j['avg_usefulness']}/5 | "
        f"avg overall: {j['avg_overall']}/5"
    )

    console.print(
        f"\n[bold]OVERALL GRADE: {report['overall_grade']:.1f}%[/] "
        f"(mean of ground-truth agreement and behavioral score)"
    )
    console.print(f"[dim]{report['provider_note']}[/]")


def write_reports(report: dict) -> None:
    (EVAL_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (EVAL_DIR / "report.html").write_text(_report_html(report), encoding="utf-8")


def _report_html(report: dict) -> str:
    gt = report["ground_truth"]
    beh_rows = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{r['score']:.1f}</td>"
        f"<td class='{'pass' if r['passed'] else 'fail'}'>{'PASS' if r['passed'] else 'FAIL'}</td>"
        f"<td>{html.escape(r['detail'])}</td></tr>"
        for r in report["behavioral"]
    )
    mm_rows = "".join(
        f"<tr><td>{html.escape(m['id'])}</td><td>{html.escape(m['field'])}</td>"
        f"<td>{html.escape(str(m['system']))}</td><td>{html.escape(str(m['label']))}</td>"
        f"<td>{html.escape(m['message'][:120])}</td></tr>"
        for m in gt["mismatches"]
    ) or "<tr><td colspan='5'>No mismatches</td></tr>"
    j = report["judge"]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Frontline Eval Report</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1115;color:#e6e6e6;padding:24px}}
h1{{margin-bottom:0}} .grade{{font-size:40px;font-weight:800;color:#4ade80}}
table{{border-collapse:collapse;width:100%;margin:12px 0;background:#1a1d24;border:1px solid #2a2f3a}}
th,td{{border-bottom:1px solid #2a2f3a;padding:8px 10px;text-align:left;font-size:14px}}
th{{color:#9aa0aa}} .pass{{color:#4ade80;font-weight:700}} .fail{{color:#f87171;font-weight:700}}
.muted{{color:#9aa0aa}}
</style></head><body>
<h1>Frontline Eval Report</h1>
<div class="grade">{report['overall_grade']:.1f}%</div>
<p class="muted">{html.escape(report['provider_note'])}</p>

<h2>A) Ground-truth agreement (n={gt['count']})</h2>
<table><tr><th>metric</th><th>score</th></tr>
<tr><td>overall</td><td>{gt['overall_pct']:.1f}%</td></tr>
<tr><td>category (exact)</td><td>{gt['category_exact_pct']:.1f}%</td></tr>
<tr><td>priority (exact)</td><td>{gt['priority_exact_pct']:.1f}%</td></tr>
<tr><td>priority (within-1)</td><td>{gt['priority_within1_pct']:.1f}%</td></tr>
<tr><td>needs_human</td><td>{gt['needs_human_pct']:.1f}%</td></tr></table>

<h3>Mismatch table (system vs label)</h3>
<table><tr><th>id</th><th>field</th><th>system</th><th>label</th><th>message</th></tr>
{mm_rows}</table>

<h2>B) Behavioral / reliability evals (mean {report['behavioral_mean']:.1f})</h2>
<table><tr><th>eval</th><th>score</th><th>pass</th><th>detail</th></tr>
{beh_rows}</table>

<h2>C) LLM-as-judge (mock, offline)</h2>
<p>avg faithfulness: {j['avg_faithfulness']}/5 &nbsp; avg usefulness: {j['avg_usefulness']}/5 &nbsp;
avg overall: {j['avg_overall']}/5</p>
</body></html>"""


def main() -> int:
    import sys

    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except Exception:
        pass

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
    report = run_eval()
    print_report(report)
    write_reports(report)
    console = Console()
    console.print(f"\nWrote [green]{EVAL_DIR / 'report.json'}[/] and [green]{EVAL_DIR / 'report.html'}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

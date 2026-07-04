"""Rich CLI table renderer for triage decisions + a metrics summary panel."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .metrics import Aggregate
from .schema import Priority, TriageDecision

_PRIORITY_STYLE = {
    "P0": "bold white on red",
    "P1": "bold red",
    "P2": "yellow",
    "P3": "dim",
}


def render_table(
    decisions: list[TriageDecision], console: Console | None = None
) -> None:
    console = console or Console()
    table = Table(title="Frontline Triage Decisions", show_lines=False, expand=True)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("category", no_wrap=True)
    table.add_column("pri", justify="center", no_wrap=True)
    table.add_column("human", justify="center", no_wrap=True)
    table.add_column("conf", justify="right", no_wrap=True)
    table.add_column("summary", overflow="fold")

    for d in decisions:
        pri_style = _PRIORITY_STYLE.get(d.priority.value, "")
        table.add_row(
            d.id,
            d.category.value,
            f"[{pri_style}]{d.priority.value}[/]",
            "[bold red]YES[/]" if d.needs_human else "no",
            f"{d.confidence:.2f}",
            d.summary,
        )
    console.print(table)


def render_metrics(agg: Aggregate, console: Console | None = None) -> None:
    console = console or Console()
    s = agg.summary()
    console.print(
        f"\n[bold]Metrics[/] — messages: {s['messages']} | "
        f"tokens: {s['total_tokens']} (avg {s['avg_tokens']}) | "
        f"cost: ${s['total_cost_usd']:.6f} (avg ${s['avg_cost_usd']:.6f}) | "
        f"latency: {s['total_latency_ms']:.1f}ms total (avg {s['avg_latency_ms']:.1f}ms)"
    )


def render_counts(decisions: list[TriageDecision], console: Console | None = None) -> None:
    console = console or Console()
    by_pri: dict[str, int] = {}
    needs_human = 0
    for d in decisions:
        by_pri[d.priority.value] = by_pri.get(d.priority.value, 0) + 1
        needs_human += 1 if d.needs_human else 0
    parts = [f"{p}: {by_pri.get(p, 0)}" for p in ("P0", "P1", "P2", "P3")]
    console.print(
        f"[bold]Priorities[/] — " + " | ".join(parts) + f"  |  needs_human: {needs_human}"
    )

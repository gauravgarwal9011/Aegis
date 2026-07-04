"""Self-contained HTML report (inline CSS/JS, no external assets)."""

from __future__ import annotations

import html
import json
from typing import Optional

from .metrics import Aggregate
from .schema import TriageDecision

_CSS = """
:root{--bg:#0f1115;--card:#1a1d24;--fg:#e6e6e6;--muted:#9aa0aa;--border:#2a2f3a;}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     background:var(--bg);color:var(--fg);padding:24px;}
h1{margin:0 0 4px}
.sub{color:var(--muted);margin-bottom:20px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;
      padding:12px 16px;min-width:120px}
.card .n{font-size:26px;font-weight:700}
.card .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
input,select{background:var(--card);color:var(--fg);border:1px solid var(--border);
      border-radius:8px;padding:8px 10px;font-size:14px}
.tablewrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px}
table{width:100%;border-collapse:collapse;background:var(--card)}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top}
th{cursor:pointer;user-select:none;font-size:13px;color:var(--muted);position:sticky;top:0;background:var(--card)}
th:hover{color:var(--fg)}
td.summary{max-width:340px;word-break:break-word}
td.action{max-width:220px;word-break:break-word;color:#a5b4fc}
.badge{padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700;white-space:nowrap}
.P0{background:#7f1d1d;color:#fff}.P1{background:#b91c1c;color:#fff}
.P2{background:#a16207;color:#fff}.P3{background:#374151;color:#cbd5e1}
.human-yes{color:#f87171;font-weight:700}.human-no{color:var(--muted)}
.grade{font-size:22px;font-weight:800}
footer{color:var(--muted);margin-top:18px;font-size:12px}
"""

_JS = """
const rows = Array.from(document.querySelectorAll('#tbl tbody tr'));
const q = document.getElementById('q');
const fpri = document.getElementById('fpri');
const fcat = document.getElementById('fcat');
const fhuman = document.getElementById('fhuman');
function apply(){
  const t=(q.value||'').toLowerCase();
  rows.forEach(r=>{
    const okT = r.innerText.toLowerCase().includes(t);
    const okP = !fpri.value || r.dataset.pri===fpri.value;
    const okC = !fcat.value || r.dataset.cat===fcat.value;
    const okH = !fhuman.value || r.dataset.human===fhuman.value;
    r.style.display = (okT&&okP&&okC&&okH)?'':'none';
  });
}
[q,fpri,fcat,fhuman].forEach(e=>e.addEventListener('input',apply));
let sortState={};
document.querySelectorAll('#tbl th').forEach((th,i)=>{
  th.addEventListener('click',()=>{
    const asc = !(sortState[i]==='asc'); sortState={}; sortState[i]=asc?'asc':'desc';
    const body=document.querySelector('#tbl tbody');
    const sorted=rows.slice().sort((a,b)=>{
      const x=a.children[i].innerText, y=b.children[i].innerText;
      const nx=parseFloat(x), ny=parseFloat(y);
      let c; if(!isNaN(nx)&&!isNaN(ny)) c=nx-ny; else c=x.localeCompare(y);
      return asc?c:-c;
    });
    sorted.forEach(r=>body.appendChild(r));
  });
});
"""


def _counts(decisions: list[TriageDecision]) -> tuple[dict, dict, int]:
    by_pri: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    human = 0
    for d in decisions:
        by_pri[d.priority.value] = by_pri.get(d.priority.value, 0) + 1
        by_cat[d.category.value] = by_cat.get(d.category.value, 0) + 1
        human += 1 if d.needs_human else 0
    return by_pri, by_cat, human


def render_html(
    decisions: list[TriageDecision],
    agg: Optional[Aggregate] = None,
    eval_score: Optional[float] = None,
    title: str = "Frontline Triage Report",
) -> str:
    by_pri, by_cat, human = _counts(decisions)
    cats = sorted(by_cat)

    cards = [
        f'<div class="card"><div class="n">{len(decisions)}</div><div class="l">messages</div></div>',
        f'<div class="card"><div class="n">{human}</div><div class="l">needs human</div></div>',
    ]
    for p in ("P0", "P1", "P2", "P3"):
        cards.append(
            f'<div class="card"><div class="n">{by_pri.get(p,0)}</div><div class="l">{p}</div></div>'
        )
    if eval_score is not None:
        cards.append(
            f'<div class="card"><div class="grade">{eval_score:.0f}%</div><div class="l">eval score</div></div>'
        )
    if agg is not None:
        s = agg.summary()
        cards.append(
            f'<div class="card"><div class="n">${s["total_cost_usd"]:.4f}</div><div class="l">total cost</div></div>'
        )
        cards.append(
            f'<div class="card"><div class="n">{s["avg_latency_ms"]:.0f}ms</div><div class="l">avg latency</div></div>'
        )

    cat_options = "".join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in cats)

    body_rows = []
    for d in decisions:
        body_rows.append(
            f'<tr data-pri="{d.priority.value}" data-cat="{html.escape(d.category.value)}" '
            f'data-human="{"yes" if d.needs_human else "no"}">'
            f"<td>{html.escape(d.id)}</td>"
            f"<td>{html.escape(d.category.value)}</td>"
            f'<td><span class="badge {d.priority.value}">{d.priority.value}</span></td>'
            f'<td class="{"human-yes" if d.needs_human else "human-no"}">'
            f'{"YES" if d.needs_human else "no"}</td>'
            f"<td>{d.confidence:.2f}</td>"
            f'<td class="summary">{html.escape(d.summary)}</td>'
            f'<td class="action">{html.escape(d.suggested_action)}</td>'
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<div class="sub">Generated offline from a real triage run. Click headers to sort; use the controls to filter.</div>
<div class="cards">{''.join(cards)}</div>
<div class="controls">
  <input id="q" placeholder="search…" />
  <select id="fpri"><option value="">all priorities</option>
    <option>P0</option><option>P1</option><option>P2</option><option>P3</option></select>
  <select id="fcat"><option value="">all categories</option>{cat_options}</select>
  <select id="fhuman"><option value="">all</option>
    <option value="yes">needs human</option><option value="no">auto</option></select>
</div>
<div class="tablewrap"><table id="tbl"><thead><tr>
  <th>id</th><th>category</th><th>priority</th><th>needs_human</th>
  <th>confidence</th><th>summary</th><th>suggested_action</th>
</tr></thead><tbody>
{''.join(body_rows)}
</tbody></table></div>
<footer>Frontline triage • mock provider produces deterministic offline results •
real-LLM numbers require provider API keys.</footer>
<script>{_JS}</script>
</body></html>"""


def write_html(path: str, decisions, agg=None, eval_score=None, title="Frontline Triage Report") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(decisions, agg=agg, eval_score=eval_score, title=title))

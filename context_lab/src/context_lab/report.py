"""Side-by-side reporting, built only from saved bundles and grade files.

Shape borrowed from the production harness's ``evals/report.py``: rows are
metrics, columns are runs. One configuration per column makes the comparison
you actually want ("what did summarizing cost me?") a single horizontal read.

Three rules this report inherits from the production one, all of them lessons
paid for in earlier experiments:

1. **The gate is a column header, not a footnote.** A run that failed the
   validity gate gets its memory metrics replaced by ``n/a (gate)``. An invalid
   memory number must be impossible to quote by accident.
2. **Grade provenance is part of the measurement.** Graded rows carry a
   ``[code]``/``[judge]``/``[mixed]`` tag naming who produced the number.
3. **Latency is marked confounded.** Arms run sequentially against a shared
   API, so wall-clock differences carry provider load as well as policy effects.
   Token and cost columns are the honest efficiency headline.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from .gate import check_run
from .grading import load_grades
from .runner import load_bundles, load_meta

NA = "n/a"
GATE_NA = "n/a (gate)"


def _fmt_usd(value: float | None) -> str:
    if value is None:
        return NA
    return f"${value:.4f}"


def _fmt_pct(numerator: int, denominator: int) -> str:
    if not denominator:
        return NA
    return f"{100 * numerator / denominator:.0f}% ({numerator}/{denominator})"


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _src_tag(rows: list[dict]) -> str:
    """Label a graded metric with who graded it: code, judge, or both."""
    sources = {row.get("grader") for row in rows if row.get("grader")}
    if not sources:
        return ""
    if len(sources) > 1:
        return " [mixed]"
    return f" [{sources.pop()}]"


def collect_run(run_dir: str | Path, battery_record: dict) -> dict:
    """Reduce one run directory to the numbers the report prints."""
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    bundles = load_bundles(run_dir)
    gate = check_run(run_dir, battery_record)

    answered = [b for b in bundles if not b.get("error")]
    input_tokens = sum(int((b.get("usage") or {}).get("input_tokens") or 0) for b in answered)
    # A provider that reports no cache accounting gives None, which is not the
    # same claim as zero hits. Keep them distinguishable all the way to print.
    cached_values = [(b.get("usage") or {}).get("cached_input_tokens") for b in answered]
    cache_measured = any(v is not None for v in cached_values)
    cached_tokens = sum(int(v) for v in cached_values if v is not None)
    output_tokens = sum(
        int((b.get("usage") or {}).get("output_tokens") or 0) for b in answered
    )
    turn_costs = [b["est_cost_usd"] for b in answered if b.get("est_cost_usd") is not None]
    summary_costs = [
        b["est_summary_cost_usd"]
        for b in bundles
        if b.get("est_summary_cost_usd") is not None
    ]
    latencies = [b["latency_ms"] for b in answered if b.get("latency_ms")]
    compactions = sum(
        (b.get("context_stats") or {}).get("compactions_this_turn") or 0 for b in bundles
    )
    capped = sum(
        len((b.get("context_stats") or {}).get("cap_events") or []) for b in bundles
    )
    final_history = bundles[-1]["context_stats"]["history_tokens_approx"] if bundles else 0

    return {
        "run_dir": run_dir,
        "label": meta["config"]["name"],
        "model": meta["model"],
        "provider": meta["provider"],
        "gate": gate,
        "turns": len(bundles),
        "errors": len(bundles) - len(answered),
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "cache_measured": cache_measured,
        "output_tokens": output_tokens,
        "session_cost": sum(turn_costs) + sum(summary_costs) if turn_costs else None,
        "turn_cost": (sum(turn_costs) + sum(summary_costs)) / len(answered)
        if turn_costs and answered
        else None,
        "summary_cost": sum(summary_costs) if summary_costs else 0.0,
        "median_latency_ms": _median(latencies),
        "compactions": compactions,
        "capped_payloads": capped,
        "final_history_tokens": final_history,
        "grades_code": load_grades(run_dir, "grades_code.jsonl"),
        "grades_judge": load_grades(run_dir, "grades_judge.jsonl"),
    }


def _probe_cell(run: dict, rows: list[dict]) -> str:
    """A probe metric, blanked when the run failed the validity gate."""
    if not rows:
        return NA
    if not run["gate"].valid:
        return GATE_NA
    passed = sum(1 for row in rows if row["grade"] == "pass")
    return _fmt_pct(passed, len(rows))


def build_report(run_dirs: list[str | Path], battery_record: dict) -> str:
    """Render the side-by-side markdown report for a set of run directories."""
    runs = [collect_run(run_dir, battery_record) for run_dir in run_dirs]
    headers = ["metric"] + [run["label"] for run in runs]

    def row(name: str, cells: list[str]) -> list[str]:
        return [name] + cells

    body: list[list[str]] = [
        row("validity gate", ["VALID" if r["gate"].valid else "**INVALID**" for r in runs]),
        row("gate reason", [r["gate"].reason or "-" for r in runs]),
        row("turns", [str(r["turns"]) for r in runs]),
        row("errors", [str(r["errors"]) for r in runs]),
        row("compactions fired", [str(r["compactions"]) for r in runs]),
        row("payloads capped", [str(r["capped_payloads"]) for r in runs]),
        row("input tokens (billed, all calls)", [f"{r['input_tokens']:,}" for r in runs]),
        row("output tokens", [f"{r['output_tokens']:,}" for r in runs]),
        row(
            "cache hit ratio (input)",
            [
                _fmt_pct(r["cached_tokens"], r["input_tokens"])
                if r["cache_measured"] and r["input_tokens"]
                else "not reported"
                for r in runs
            ],
        ),
        row("final history size (approx tokens)", [f"{r['final_history_tokens']:,}" for r in runs]),
        row("est cost, whole session", [_fmt_usd(r["session_cost"]) for r in runs]),
        row("est cost / turn", [_fmt_usd(r["turn_cost"]) for r in runs]),
        row("of which summarization", [_fmt_usd(r["summary_cost"]) for r in runs]),
        row(
            "median latency ms [confounded]",
            [f"{r['median_latency_ms']:.0f}" if r["median_latency_ms"] else NA for r in runs],
        ),
    ]

    # A grading layer gets a row only when it actually produced grades. An
    # empty row labelled "probe accuracy" reads like a measured n/a rather than
    # "nobody graded this", which is the same class of confusion the cache row
    # avoids by distinguishing "not reported" from 0%.
    for key, name in (("grades_code", "code"), ("grades_judge", "judge")):
        graded = [g for r in runs for g in r[key]]
        if not graded:
            continue
        body.append(
            row(
                f"probe accuracy{_src_tag(graded)}",
                [_probe_cell(r, r[key]) if r[key] else "not graded" for r in runs],
            )
        )

    # One sub-row per probe type, so "which memory did it lose?" is visible.
    probe_types = sorted({p["probe_type"] for p in battery_record.get("probes", [])})
    for probe_type in probe_types:
        cells = []
        for run in runs:
            rows = [g for g in run["grades_code"] if g["item_type"] == f"probe:{probe_type}"]
            cells.append(_probe_cell(run, rows))
        body.append(row(f"└ {probe_type} [code]", cells))

    lines = [
        "# context-lab report",
        "",
        f"battery: `{battery_record['session_id']}` "
        f"({len(battery_record['turns'])} turns, {len(battery_record['probes'])} probes)",
        f"model: `{runs[0]['model']}` via `{runs[0]['provider']}`" if runs else "",
        "",
        "Token and cost rows are the efficiency headline. Latency rows are marked",
        "`[confounded]`: arms run sequentially against a shared API, so wall-clock",
        "differences include provider load. Memory rows read `n/a (gate)` when the run",
        "failed the validity gate, because a memory number from a run where compaction",
        "never fired measures an idle code path.",
        "",
        "**Read the cache hit ratio row before you read the cost rows.** Keeping the",
        "full history is cheap only when the repeated prefix is billed at a cache",
        "discount. `not reported` means the provider returned no cache accounting, so",
        "every input token here was priced at the full rate and the cost comparison",
        "below is a no-cache comparison. That is a different experiment from the one",
        "a cache-discounted provider runs, and its cost answer can legitimately flip.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for cells in body:
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_report(
    run_dirs: list[str | Path], battery_record: dict, out_path: str | Path
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_report(run_dirs, battery_record)
    out_path.write_text(text, encoding="utf-8")
    return out_path

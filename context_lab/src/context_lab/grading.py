"""Offline grading: cheap code checks over saved bundles.

Reads ``bundles.jsonl``, never calls the session model, and writes
``grades_code.jsonl``: one row per probe with a pass/fail from regex checks
plus full provenance (who graded, when, under which gate verdict). Re-run it as
often as you like; it is free.

Code checks are the fast, deterministic layer: "does the answer name the planted
Python version", "does it still use the old port after the user changed it",
"does it call print() when the user banned print()". They cannot judge nuance,
which is the LLM judge's job in ``judge.py``. On a battery designed with crisp
planted facts they still catch most memory failures for $0, which is why they
run first.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .gate import check_run
from .runner import load_bundles, load_meta


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_probe_answer(answer: str, checks: dict) -> tuple[str, str]:
    """Apply expected/forbidden regexes. Returns (grade, detail)."""
    if not answer.strip():
        return "fail", "empty answer"
    details = []
    grade = "pass"
    expected = checks.get("expected_any") or []
    if expected and not any(re.search(p, answer, re.IGNORECASE) for p in expected):
        grade = "fail"
        details.append(f"no expected pattern matched: {expected}")
    forbidden = checks.get("forbidden_any") or []
    hits = [p for p in forbidden if re.search(p, answer, re.IGNORECASE)]
    if hits:
        grade = "fail"
        details.append(f"forbidden pattern matched: {hits}")
    return grade, "; ".join(details) or "all checks passed"


def grade_run(run_dir: str | Path, battery_record: dict) -> Path:
    """Grade one run's probes with code checks; write grades_code.jsonl."""
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    bundles = {b["turn_index"]: b for b in load_bundles(run_dir)}
    gate = check_run(run_dir, battery_record)
    rows = []
    for probe in battery_record.get("probes", []):
        bundle = bundles.get(probe["turn_index"])
        answer = (bundle or {}).get("answer") or ""
        grade, detail = check_probe_answer(answer, probe.get("checks") or {})
        rows.append(
            {
                "run_id": (bundle or {}).get("run_id"),
                "session_id": battery_record["session_id"],
                "config": meta["config"]["name"],
                "model": meta["model"],
                "turn_index": probe["turn_index"],
                "item_type": f"probe:{probe['probe_type']}",
                "criterion": probe["rule"],
                "expected_facts": probe.get("expected_facts", []),
                "grade": grade,
                "detail": detail,
                "grader": "code",
                "graded_at": _now_iso(),
                "gate_valid": gate.valid,
                "gate_reason": gate.reason,
            }
        )
    out_path = run_dir / "grades_code.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    npass = sum(1 for r in rows if r["grade"] == "pass")
    print(
        f"{run_dir.name}: code-graded {len(rows)} probes, {npass} pass"
        + ("" if gate.valid else f"  [gate: INVALID: {gate.reason}]")
    )
    return out_path


def load_grades(run_dir: str | Path, name: str) -> list[dict]:
    path = Path(run_dir) / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

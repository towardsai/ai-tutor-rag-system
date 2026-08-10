"""The validity gate: did the run actually test what it claims to test?

A memory probe measures whether facts survive compaction. If compaction never
fired before the probe, the probe measured nothing: the "memory" number would
be a fact about an idle code path, not about the policy. The production harness
enforces this with ``evals.check_triggers``; this is the same gate at teaching
size, reading only saved bundles.

Rules, derived from the run's own config:

- summarizing config: every probe turn must come at or after the first
  compaction event, and at least one compaction event must exist. Otherwise the
  run is INVALID for memory conclusions.
- non-summarizing config (``full_history``, ``capped``): the run must contain
  ZERO compaction events; it is the control arm, and a control that compacted
  would invalidate the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .runner import load_bundles, load_meta


@dataclass
class GateResult:
    run_dir: str
    config: str
    valid: bool
    expects_compaction: bool
    compaction_turns: list[int] = field(default_factory=list)
    probe_turns: list[int] = field(default_factory=list)
    invalid_probe_turns: list[int] = field(default_factory=list)
    errors: list[int] = field(default_factory=list)
    reason: str = ""

    def summary_line(self) -> str:
        verdict = "VALID" if self.valid else "INVALID"
        return (
            f"{Path(self.run_dir).name:28s} config={self.config:22s} "
            f"compactions@{self.compaction_turns} probes@{self.probe_turns} "
            f"-> {verdict}{': ' + self.reason if self.reason else ''}"
        )


def probe_turn_indexes(battery_record: dict) -> list[int]:
    return sorted(probe["turn_index"] for probe in battery_record.get("probes", []))


def check_run(run_dir: str | Path, battery_record: dict) -> GateResult:
    """Gate one run directory against the battery's probe schedule."""
    meta = load_meta(run_dir)
    bundles = load_bundles(run_dir)
    expects = bool(meta["config"].get("summarize"))
    compaction_turns = [
        b["turn_index"]
        for b in bundles
        if (b.get("context_stats") or {}).get("compactions_this_turn")
    ]
    probes = probe_turn_indexes(battery_record)
    errors = [b["turn_index"] for b in bundles if b.get("error")]
    result = GateResult(
        run_dir=str(run_dir),
        config=meta["config"]["name"],
        valid=True,
        expects_compaction=expects,
        compaction_turns=compaction_turns,
        probe_turns=probes,
        errors=errors,
    )
    if errors:
        result.valid = False
        result.reason = f"API errors at turns {errors}"
        return result
    if expects:
        first = min(compaction_turns) if compaction_turns else None
        bad = [t for t in probes if first is None or t < first]
        if bad:
            result.valid = False
            result.invalid_probe_turns = bad
            result.reason = (
                "compaction never fired"
                if first is None
                else f"probes {bad} ran before the first compaction (turn {first})"
            )
    elif compaction_turns:
        result.valid = False
        result.reason = (
            f"control arm compacted at turns {compaction_turns} (expected none)"
        )
    return result


def check_runs(
    run_dirs: list[str | Path], battery_record: dict, verbose: bool = True
) -> list[GateResult]:
    results = [check_run(run_dir, battery_record) for run_dir in run_dirs]
    if verbose:
        for result in results:
            print(result.summary_line())
        invalid = sum(1 for r in results if not r.valid)
        print(
            f"gate: {len(results) - invalid}/{len(results)} runs valid"
            + (
                " (memory numbers from invalid runs must not be reported)"
                if invalid
                else ""
            )
        )
    return results

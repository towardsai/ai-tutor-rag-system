"""Smoke test for the zero-key path: install, then gate/grade/report offline.

Everything here runs without an API key and without network access. If this
passes from a clean virtualenv, the "run once, grade forever" half of the
package works, which is the half students touch first.

Run with:  pytest -q
"""

from __future__ import annotations

import json

import pytest

from context_lab import (
    CONFIGS,
    approx_tokens,
    build_report,
    check_probe_answer,
    check_run,
    copy_prerun,
    default_battery_path,
    estimate_cost_usd,
    grade_run,
    load_battery,
    load_bundles,
    load_grades,
    run_session,
)
from context_lab.memory import split_for_summary


@pytest.fixture(scope="session")
def record() -> dict:
    return load_battery(default_battery_path())[0]


@pytest.fixture(scope="session")
def prerun(tmp_path_factory) -> list:
    """The shipped pre-run results, copied somewhere writable."""
    return copy_prerun(tmp_path_factory.mktemp("prerun"))


# --- the shipped data ------------------------------------------------------


def test_battery_is_well_formed(record):
    assert record["session_id"]
    turns = record["turns"]
    assert len(turns) >= 10
    assert [t["turn_index"] for t in turns] == list(range(1, len(turns) + 1))
    # Probes must exist, be late, and point at real turns.
    probes = record["probes"]
    assert probes
    turn_indexes = {t["turn_index"] for t in turns}
    for probe in probes:
        assert probe["turn_index"] in turn_indexes
        assert probe["probe_type"] in {
            "fact_recall",
            "fact_update",
            "preference_compliance",
        }
        assert probe["rule"]
        assert probe["checks"]["expected_any"] or probe["checks"]["forbidden_any"]
    # Facts are planted early and probed late, or the battery measures nothing.
    assert min(p["turn_index"] for p in probes) > len(turns) // 2


def test_battery_carries_payloads(record):
    """At least one turn must carry a large payload, or `capped` tests nothing."""
    docs = [t["doc"] for t in record["turns"] if t.get("doc")]
    assert len(docs) >= 2
    assert all(len(d["content"]) > CONFIGS["capped"].doc_cap_chars for d in docs)


def test_prerun_ships_three_runs(prerun):
    assert len(prerun) == 3
    for run_dir in prerun:
        assert (run_dir / "run_meta.json").exists()
        bundles = load_bundles(run_dir)
        assert bundles
        assert all("answer" in b and "context_stats" in b for b in bundles)


# --- the validity gate -----------------------------------------------------


def test_gate_accepts_the_valid_runs_and_rejects_the_mis_sized_one(prerun, record):
    verdicts = {d.name: check_run(d, record) for d in prerun}
    assert verdicts["full_history"].valid, verdicts["full_history"].reason
    assert verdicts["summarize"].valid, verdicts["summarize"].reason
    invalid = verdicts["summarize_prod_trigger"]
    assert not invalid.valid
    assert "compaction never fired" in invalid.reason


def test_gate_rejects_a_control_arm_that_compacted(prerun, record, tmp_path):
    """A full_history run that somehow compacted must fail: it is not a control."""
    src = next(d for d in prerun if d.name == "full_history")
    dest = tmp_path / "tampered"
    dest.mkdir()
    (dest / "run_meta.json").write_text((src / "run_meta.json").read_text())
    bundles = load_bundles(src)
    bundles[3]["context_stats"]["compactions_this_turn"] = 1
    (dest / "bundles.jsonl").write_text(
        "\n".join(json.dumps(b) for b in bundles) + "\n"
    )
    result = check_run(dest, record)
    assert not result.valid
    assert "control arm compacted" in result.reason


# --- offline grading -------------------------------------------------------


def test_code_checks_are_free_and_deterministic(prerun, record):
    for run_dir in prerun:
        grade_run(run_dir, record)
        rows = load_grades(run_dir, "grades_code.jsonl")
        assert len(rows) == len(record["probes"])
        for row in rows:
            assert row["grade"] in {"pass", "fail"}
            assert row["grader"] == "code"
            assert row["graded_at"]
            assert "gate_valid" in row
    # Re-grading is idempotent on the graded verdicts.
    first = [r["grade"] for r in load_grades(prerun[0], "grades_code.jsonl")]
    grade_run(prerun[0], record)
    assert [r["grade"] for r in load_grades(prerun[0], "grades_code.jsonl")] == first


@pytest.mark.parametrize(
    "answer,checks,expected",
    [
        ("requires-python = '>=3.11'", {"expected_any": [r"3\.11"]}, "pass"),
        ("requires-python = '>=3.12'", {"expected_any": [r"3\.11"]}, "fail"),
        ("client on port 9010", {"forbidden_any": [r"8010"]}, "pass"),
        ("client on port 8010", {"forbidden_any": [r"8010"]}, "fail"),
        ("", {"expected_any": [r"anything"]}, "fail"),
    ],
)
def test_check_probe_answer(answer, checks, expected):
    grade, _ = check_probe_answer(answer, checks)
    assert grade == expected


def test_full_history_remembers_more_than_summarize(prerun, record):
    """The headline result, reproduced from the shipped bundle with no API key."""
    for run_dir in prerun:
        grade_run(run_dir, record)

    def passes(name):
        run_dir = next(d for d in prerun if d.name == name)
        rows = load_grades(run_dir, "grades_code.jsonl")
        return sum(1 for r in rows if r["grade"] == "pass")

    assert passes("full_history") > passes("summarize")


# --- reporting -------------------------------------------------------------


def test_report_blanks_memory_for_invalid_runs(prerun, record):
    for run_dir in prerun:
        grade_run(run_dir, record)
    text = build_report(prerun, record)
    assert "validity gate" in text
    assert "**INVALID**" in text
    assert "n/a (gate)" in text
    # Cost is still reported for the invalid run: the dollars were real.
    assert "est cost / turn" in text
    assert text.count("|") > 50


# --- units the rest depends on ---------------------------------------------


def test_cost_applies_the_cache_discount():
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0}
    full = estimate_cost_usd("gemini-3.5-flash-lite", usage)
    cached = estimate_cost_usd(
        "gemini-3.5-flash-lite", {**usage, "cached_input_tokens": 1_000_000}
    )
    assert full == pytest.approx(0.30)
    assert cached == pytest.approx(0.03)
    assert estimate_cost_usd("some-unpriced-model", usage) is None


def test_split_for_summary_cuts_on_a_user_boundary():
    history = []
    for i in range(6):
        history.append({"role": "user", "text": f"q{i}"})
        history.append({"role": "assistant", "text": f"a{i}"})
    older, recent = split_for_summary(history, keep_recent_turns=2)
    assert older and recent
    assert recent[0]["role"] == "user"
    assert older + recent == history
    # Too short to split: nothing is summarized.
    assert split_for_summary(history[:2], keep_recent_turns=4) == ([], history[:2])


def test_approx_tokens_is_only_used_for_triggers():
    assert approx_tokens("x" * 400) == 100


# --- the runner, with a stub provider (still no API key) -------------------


class StubProvider:
    """Deterministic stand-in so the runner is testable without a network."""

    name = "stub"
    model = "gemini-3.5-flash-lite"

    def __init__(self):
        self.calls = 0

    def generate(self, system, messages, max_output_tokens=1024):
        from context_lab.providers import GenerateResult

        self.calls += 1
        return GenerateResult(
            text=f"stub answer {self.calls} " + "padding " * 60,
            usage={
                "input_tokens": 500 * self.calls,
                "output_tokens": 100,
                "cached_input_tokens": 0,
            },
            latency_ms=10,
            model=self.model,
        )


def test_runner_writes_a_bundle_per_turn_and_caps_payloads(record, tmp_path):
    provider = StubProvider()
    run_dir = run_session(
        record, CONFIGS["capped"], provider, tmp_path / "capped", quiet=True
    )
    bundles = load_bundles(run_dir)
    assert len(bundles) == len(record["turns"])
    capped = [b for b in bundles if b["context_stats"]["cap_events"]]
    assert capped, "the capped config must have truncated at least one payload"
    assert all(b["est_cost_usd"] is not None for b in bundles)
    # A capped run is a control arm: it must never compact.
    assert check_run(run_dir, record).valid


def test_runner_compacts_under_the_summarize_config(record, tmp_path):
    provider = StubProvider()
    run_dir = run_session(
        record, CONFIGS["summarize"], provider, tmp_path / "summarize", quiet=True
    )
    bundles = load_bundles(run_dir)
    fired = [b["turn_index"] for b in bundles if b["context_stats"]["compactions_this_turn"]]
    assert fired, "summarize must compact on a session this size"
    event = next(
        b["context_stats"]["compaction_events"][0] for b in bundles if b["context_stats"]["compaction_events"]
    )
    assert event["pre_compaction_tokens_approx"] >= event["configured_trigger_tokens"]

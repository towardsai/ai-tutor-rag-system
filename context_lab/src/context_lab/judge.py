"""LLM judge with typed verdicts, adapted from the production harness.

The judge grades each probe against its rule, blind to which memory
configuration produced the answer: it sees only question, answer, and
criterion, so it cannot favor a strategy. Verdicts are structured
(``grade``/``confidence``/``reason``) via the provider's schema-constrained
output, and every row records provenance (which judge model, when).

Two honest caveats the production program takes seriously and this teaching
version inherits:

- A judge's numbers only count after the judge is validated against human
  labels (the production gate is >=90% agreement on held-out human grades).
  Nothing here validates the judge for you; read its verdicts next to the code
  checks and adjudicate disagreements yourself.
- Here the judge model defaults to the same cheap model that produced the
  answers. That is fine for learning the pattern and risky for real
  conclusions; production used a stronger, separate judge.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .gate import check_run
from .runner import load_bundles, load_meta

JUDGE_PREAMBLE = (
    "You are grading a BLINDED answer from a programming assistant. You see "
    "only the question, the answer, and one criterion; you do NOT know which "
    "system produced the answer and must not try to infer it. Decide "
    "grade=pass|fail with a one-sentence reason. An EMPTY answer fails. Grade "
    "ONLY what the criterion asks, not overall polish. Always give a definite "
    "pass/fail; use confidence=low only for genuinely ambiguous calls."
)

PROBE_RUBRIC = (
    "ITEM TYPE = session memory probe. The criterion gives the expected facts "
    "and a rule. Grade STRICTLY by the rule. fact_recall: the answer must use "
    "or reflect the earlier-stated fact(s). fact_update: the answer must use "
    "the NEW fact; building on the OLD pre-update value = fail (merely "
    "mentioning the old value historically is not a fail). "
    "preference_compliance: the answer must honor the stated preference and "
    "avoid what it forbids."
)


class Verdict(BaseModel):
    grade: str = Field(description="'pass' or 'fail'")
    confidence: str = Field(description="'high' or 'low'")
    reason: str = Field(description="one short sentence")


def build_judge_prompt(probe: dict, question: str, answer: str) -> str:
    criterion = (
        f"Expected facts: {probe.get('expected_facts', [])}. Rule: {probe['rule']}"
    )
    return (
        f"RUBRIC:\n{PROBE_RUBRIC}\n\nQUESTION:\n{question}\n\n"
        f"CRITERION:\n{criterion}\n\nANSWER TO GRADE:\n{answer or '(EMPTY)'}"
    )


class GeminiJudge:
    """Schema-constrained judge on the Gemini API (the tested judge backend)."""

    def __init__(self, model: str = "gemini-3.5-flash-lite"):
        from google import genai

        self.model = model
        self._client = genai.Client()

    def verdict(self, prompt: str) -> Verdict:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_PREAMBLE,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=Verdict,
            ),
        )
        parsed = response.parsed
        if isinstance(parsed, Verdict):
            return parsed
        return Verdict.model_validate_json(response.text)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def judge_run(
    run_dir: str | Path, battery_record: dict, judge: GeminiJudge | None = None
) -> Path:
    """Judge one run's probes; write grades_judge.jsonl with provenance."""
    run_dir = Path(run_dir)
    judge = judge or GeminiJudge()
    meta = load_meta(run_dir)
    bundles = {b["turn_index"]: b for b in load_bundles(run_dir)}
    gate = check_run(run_dir, battery_record)
    rows = []
    for probe in battery_record.get("probes", []):
        bundle = bundles.get(probe["turn_index"]) or {}
        answer = bundle.get("answer") or ""
        if not answer.strip():
            verdict = Verdict(grade="fail", confidence="high", reason="empty answer")
        else:
            verdict = judge.verdict(
                build_judge_prompt(probe, bundle.get("query", ""), answer)
            )
        rows.append(
            {
                "run_id": bundle.get("run_id"),
                "session_id": battery_record["session_id"],
                "config": meta["config"]["name"],
                "model": meta["model"],
                "turn_index": probe["turn_index"],
                "item_type": f"probe:{probe['probe_type']}",
                "criterion": probe["rule"],
                "expected_facts": probe.get("expected_facts", []),
                "grade": "pass" if verdict.grade.strip().lower() == "pass" else "fail",
                "confidence": verdict.confidence.strip().lower(),
                "reason": verdict.reason,
                "grader": "judge",
                "judge_model": judge.model,
                "graded_at": _now_iso(),
                "gate_valid": gate.valid,
                "gate_reason": gate.reason,
            }
        )
    out_path = run_dir / "grades_judge.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    npass = sum(1 for r in rows if r["grade"] == "pass")
    print(f"{run_dir.name}: judge-graded {len(rows)} probes, {npass} pass")
    return out_path

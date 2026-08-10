"""Battery loading and access to the data shipped inside the package.

A battery is a JSONL file of session records. Each record::

    {
      "session_id": str,
      "description": str,
      "turns": [
        {"turn_index": int, "text": str,
         "doc": {"title": str, "content": str} | absent},
        ...
      ],
      "probes": [
        {"turn_index": int, "probe_type": "fact_recall"|"fact_update"|
             "preference_compliance",
         "expected_facts": [str, ...],
         "rule": str,                       # what a passing answer must do
         "checks": {"expected_any": [regex, ...],
                    "forbidden_any": [regex, ...]}},
        ...
      ]
    }

A ``doc`` on a turn is a large document payload that enters history with the
user message, the stand-in for a tool output (a retrieval result, a browsed
file). In the production tutor these payloads, not the chat itself, dominate
input tokens, which is why the ``capped`` configuration targets them.

The package ships one battery (``battery_docs_v1.jsonl``) plus one pre-run
results bundle, both 100% synthetic and authored for this course. See
``prerun_dir()`` / ``copy_prerun()`` for the zero-key path.
"""

from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path


def load_battery(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for record in records:
        for key in ("session_id", "turns", "probes"):
            if key not in record:
                raise ValueError(f"Battery record missing {key!r}")
    return records


def _data_root() -> Path:
    return Path(str(resources.files("context_lab") / "data"))


def default_battery_path() -> Path:
    """The synthetic planted-facts battery shipped with the package."""
    return _data_root() / "battery_docs_v1.jsonl"


def prerun_dir() -> Path:
    """The shipped pre-run results (one run directory per configuration)."""
    return _data_root() / "prerun"


def copy_prerun(dest: str | Path) -> list[Path]:
    """Copy the shipped pre-run run directories under ``dest`` and return them.

    Installed package data may live somewhere read-only, so grading and
    reporting work on a writable copy.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for run_dir in sorted(p for p in prerun_dir().iterdir() if p.is_dir()):
        target = dest / run_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(run_dir, target)
        copied.append(target)
    return copied

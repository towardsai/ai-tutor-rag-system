"""Command-line entry point: the same four-step pipeline the real harness uses.

    context-lab run    --config full_history --config summarize --out runs/
    context-lab gate   runs/*                     # did the mechanism fire?
    context-lab grade  runs/*                     # free, offline, repeatable
    context-lab report runs/* --out report.md

Only ``run`` spends money. ``gate``, ``grade`` (code checks), and ``report`` read
saved bundles and never call a model, so they work with no API key at all. Use
``context-lab prerun`` to unpack the results bundle shipped in the package and
try that path before spending anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .battery import copy_prerun, default_battery_path, load_battery
from .gate import check_runs
from .grading import grade_run
from .memory import CONFIGS
from .report import write_report
from .runner import run_session


def _battery(args) -> tuple[dict, Path]:
    path = Path(args.battery) if args.battery else default_battery_path()
    records = load_battery(path)
    return records[0], path


def cmd_run(args) -> int:
    record, battery_path = _battery(args)
    from .providers import get_provider

    provider = get_provider(args.provider, args.model)
    out_root = Path(args.out)
    for name in args.config:
        if name not in CONFIGS:
            raise SystemExit(f"Unknown config {name!r}. Known: {', '.join(CONFIGS)}")
        print(f"\n=== {name} ({provider.name}/{provider.model}) ===")
        run_session(
            record, CONFIGS[name], provider, out_root / name, battery_path=battery_path
        )
    print(f"\nWrote runs to {out_root}/. Nothing else in this tool spends money.")
    return 0


def cmd_gate(args) -> int:
    record, _ = _battery(args)
    results = check_runs(args.run_dirs, record)
    return 0 if all(r.valid for r in results) else 1


def cmd_grade(args) -> int:
    record, _ = _battery(args)
    for run_dir in args.run_dirs:
        grade_run(run_dir, record)
    if args.judge:
        from .judge import GeminiJudge, judge_run

        judge = GeminiJudge(args.judge_model)
        for run_dir in args.run_dirs:
            judge_run(run_dir, record, judge)
    return 0


def cmd_report(args) -> int:
    record, _ = _battery(args)
    path = write_report(args.run_dirs, record, args.out)
    print(path.read_text())
    print(f"(written to {path})")
    return 0


def cmd_prerun(args) -> int:
    copied = copy_prerun(args.out)
    for path in copied:
        print(path)
    print(f"\n{len(copied)} pre-run results copied. Grade and report them for free:")
    print(f"  context-lab gate   {args.out}/*")
    print(f"  context-lab grade  {args.out}/*")
    print(f"  context-lab report {args.out}/* --out {args.out}/report.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="context-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_battery(p):
        p.add_argument("--battery", help="battery JSONL (default: the shipped one)")

    p_run = sub.add_parser("run", help="run the battery under one or more configs")
    add_battery(p_run)
    p_run.add_argument(
        "--config", action="append", required=True, help="repeatable; see --list-configs"
    )
    p_run.add_argument("--provider", default="deepseek", choices=["deepseek", "gemini", "ollama"])
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--out", default="runs")
    p_run.set_defaults(func=cmd_run)

    p_gate = sub.add_parser("gate", help="check run validity (free)")
    add_battery(p_gate)
    p_gate.add_argument("run_dirs", nargs="+")
    p_gate.set_defaults(func=cmd_gate)

    p_grade = sub.add_parser("grade", help="code-check grading (free); --judge adds the LLM judge")
    add_battery(p_grade)
    p_grade.add_argument("run_dirs", nargs="+")
    p_grade.add_argument("--judge", action="store_true", help="also run the LLM judge (costs money)")
    p_grade.add_argument("--judge-model", default="gemini-3.5-flash-lite")
    p_grade.set_defaults(func=cmd_grade)

    p_report = sub.add_parser("report", help="side-by-side report (free)")
    add_battery(p_report)
    p_report.add_argument("run_dirs", nargs="+")
    p_report.add_argument("--out", default="report.md")
    p_report.set_defaults(func=cmd_report)

    p_prerun = sub.add_parser("prerun", help="unpack the shipped pre-run results (no API key)")
    p_prerun.add_argument("--out", default="prerun_runs")
    p_prerun.set_defaults(func=cmd_prerun)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

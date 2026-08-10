"""context-lab: a teaching harness for context-engineering experiments.

Run a synthetic multi-turn planted-facts battery under two or three pluggable
memory configurations, then gate, grade, and report **offline** from saved
trace bundles. The design mirrors the production eval harness behind the
Towards AI tutor, shrunk to something a student can run for under a dollar.

The property worth internalizing is **run once, grade forever**: the only step
that spends money is ``run``. Every turn persists a JSON bundle, and the gate,
the code checks, and the report read those bundles and never call a model
again. A metric bug found next month re-grades last month's run for free.

Typical use::

    from context_lab import (
        CONFIGS, default_battery_path, load_battery,
        run_session, check_run, grade_run, build_report, get_provider,
    )

    record = load_battery(default_battery_path())[0]
    provider = get_provider("gemini")
    for name in ("full_history", "summarize"):
        run_session(record, CONFIGS[name], provider, f"runs/{name}")

No API key? Start with the results bundle shipped inside the package::

    from context_lab import copy_prerun
    run_dirs = copy_prerun("prerun_runs")
"""

from .battery import copy_prerun, default_battery_path, load_battery, prerun_dir
from .gate import GateResult, check_run, check_runs
from .grading import check_probe_answer, grade_run, load_grades
from .memory import CONFIGS, MemoryConfig, approx_tokens
from .pricing import MODEL_PRICING, estimate_cost_usd
from .report import build_report, collect_run, write_report
from .runner import load_bundles, load_meta, run_session

__version__ = "0.1.0"

__all__ = [
    "CONFIGS",
    "MODEL_PRICING",
    "GateResult",
    "MemoryConfig",
    "approx_tokens",
    "build_report",
    "check_probe_answer",
    "check_run",
    "check_runs",
    "collect_run",
    "copy_prerun",
    "default_battery_path",
    "estimate_cost_usd",
    "grade_run",
    "load_battery",
    "load_bundles",
    "load_grades",
    "load_meta",
    "prerun_dir",
    "run_session",
    "write_report",
    "__version__",
]


def get_provider(name: str = "gemini", model: str | None = None):
    """Build a provider client. Imported lazily so offline use needs no SDK."""
    from .providers import get_provider as _get_provider

    return _get_provider(name, model)

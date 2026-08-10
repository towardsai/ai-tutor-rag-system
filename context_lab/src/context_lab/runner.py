"""The session runner: a plain chat loop with a pluggable memory policy.

``run_session`` executes one battery session under one memory configuration and
persists one JSON trace bundle per turn to ``<out_dir>/bundles.jsonl``. That
file is the experiment: the gate, the grader, the judge, and the report all
read saved bundles and never call the session model again ("run once, grade
forever").

Order of operations each turn, mirroring the production middleware stack:

1. If the config summarizes and the approximate history size crossed the
   trigger, compact: one LLM call rewrites the older span into a short summary
   message that replaces it (a lossy, cache-breaking prefix rewrite).
2. Build the user message. A document payload is appended to the user text; if
   the config caps payloads, the cap is applied HERE, once, at insertion time,
   so every later call re-reads identical bytes (a stable, cache-friendly cut).
3. Call the model on [system] + history + [user message]; append both sides to
   history; write the bundle.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .battery import default_battery_path
from .memory import (
    MemoryConfig,
    SUMMARY_PROMPT,
    history_tokens,
    render_conversation,
    split_for_summary,
)
from .pricing import estimate_cost_usd

SYSTEM_PROMPT = (
    "You are a helpful, concise programming assistant for a developer working "
    "on their project. Answer technical questions directly, with short code "
    "examples where useful. Respect any constraints and preferences the "
    "developer has stated at any point in the conversation."
)

DOC_TEMPLATE = "{text}\n\n--- attached document: {title} ---\n{content}"
SUMMARY_HEADER = "[Summary of the earlier conversation]\n"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _battery_label(battery_path: str | Path) -> str:
    """Record the shipped battery by name, so saved runs stay portable.

    A run directory is meant to be shareable (that is the whole point of "run
    once, grade forever"). Baking an absolute path from the machine that
    produced it into every bundle makes the artifact machine-specific for no
    benefit, so the packaged battery is recorded as its filename.
    """
    path = Path(battery_path)
    if not battery_path:
        return ""
    try:
        packaged = path.resolve().parent == default_battery_path().resolve().parent
    except (OSError, RuntimeError):
        packaged = False
    return path.name if packaged else str(battery_path)


def maybe_compact(
    provider, config: MemoryConfig, history: list[dict], turn_index: int
) -> tuple[list[dict], list[dict], dict]:
    """Apply the summarization policy. Returns (history, events, usage)."""
    if not config.summarize:
        return history, [], {}
    pre_tokens = history_tokens(history)
    if pre_tokens < config.summarize_trigger_tokens:
        return history, [], {}
    older, recent = split_for_summary(history, config.summarize_keep_recent_turns)
    if not older:
        return history, [], {}
    prompt = SUMMARY_PROMPT.format(
        max_words=config.summary_max_words,
        conversation=render_conversation(older),
    )
    result = provider.generate(
        system="You compress conversation history.",
        messages=[{"role": "user", "text": prompt}],
        max_output_tokens=512,
    )
    summary_message = {"role": "user", "text": SUMMARY_HEADER + result.text.strip()}
    event = {
        "turn_index": turn_index,
        "pre_compaction_tokens_approx": pre_tokens,
        "configured_trigger_tokens": config.summarize_trigger_tokens,
        "summarized_messages": len(older),
        "kept_messages": len(recent),
        "summary_chars": len(result.text),
    }
    return [summary_message] + recent, [event], result.usage


def build_user_message(turn: dict, config: MemoryConfig) -> tuple[dict, dict | None]:
    """Attach the turn's document payload, capping it at insertion time."""
    doc = turn.get("doc")
    if not doc:
        return {"role": "user", "text": turn["text"]}, None
    content = doc["content"]
    cap_event = None
    if config.doc_cap_chars is not None and len(content) > config.doc_cap_chars:
        content = content[: config.doc_cap_chars] + "\n[... truncated by policy]"
        cap_event = {
            "turn_index": turn["turn_index"],
            "doc_title": doc["title"],
            "chars_full": len(doc["content"]),
            "chars_kept": config.doc_cap_chars,
        }
    text = DOC_TEMPLATE.format(text=turn["text"], title=doc["title"], content=content)
    return {"role": "user", "text": text}, cap_event


def run_session(
    session: dict,
    config: MemoryConfig,
    provider,
    out_dir: str | Path,
    battery_path: str | Path = "",
    quiet: bool = False,
) -> Path:
    """Run one session under one config; write bundles.jsonl and run_meta.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": _now_iso(),
        "session_id": session["session_id"],
        "battery_path": _battery_label(battery_path),
        "config": asdict(config),
        "provider": provider.name,
        "model": provider.model,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    history: list[dict] = []
    bundles_path = out_dir / "bundles.jsonl"
    with open(bundles_path, "w", encoding="utf-8") as bundles:
        for turn in session["turns"]:
            turn_index = turn["turn_index"]
            started = _now_iso()
            wall_start = time.monotonic()
            error = None
            answer = ""
            usage: dict = {}
            latency_ms = 0
            history, compaction_events, summary_usage = maybe_compact(
                provider, config, history, turn_index
            )
            user_message, cap_event = build_user_message(turn, config)
            try:
                result = provider.generate(
                    SYSTEM_PROMPT, history + [user_message]
                )
                answer, usage, latency_ms = result.text, result.usage, result.latency_ms
            except Exception as exc:  # record, don't crash the run
                error = f"{type(exc).__name__}: {exc}"
            history.append(user_message)
            history.append({"role": "assistant", "text": answer or "(no answer)"})
            cost = estimate_cost_usd(provider.model, usage) if usage else None
            summary_cost = (
                estimate_cost_usd(provider.model, summary_usage)
                if summary_usage
                else None
            )
            bundle = {
                "run_id": f"{session['session_id']}|turn{turn_index:02d}",
                "session_id": session["session_id"],
                "battery_path": _battery_label(battery_path),
                "config": config.name,
                "provider": provider.name,
                "model": provider.model,
                "turn_index": turn_index,
                "started_at": started,
                "latency_ms": latency_ms,
                "turn_wall_ms": int((time.monotonic() - wall_start) * 1000),
                "query": turn["text"],
                "doc_title": (turn.get("doc") or {}).get("title"),
                "answer": answer,
                "usage": usage,
                "summary_usage": summary_usage,
                "est_cost_usd": cost,
                "est_summary_cost_usd": summary_cost,
                "context_stats": {
                    "history_messages": len(history),
                    "history_tokens_approx": history_tokens(history),
                    "compactions_this_turn": len(compaction_events),
                    "compaction_events": compaction_events,
                    "cap_events": [cap_event] if cap_event else [],
                },
                "error": error,
            }
            bundles.write(json.dumps(bundle, ensure_ascii=False) + "\n")
            bundles.flush()
            if not quiet:
                marker = " [compacted]" if compaction_events else ""
                marker += " [doc capped]" if cap_event else ""
                status = "ERROR" if error else f"{usage.get('input_tokens', 0):,} in"
                print(
                    f"  turn {turn_index:2d} [{config.name}] {status}"
                    f" / {latency_ms / 1000:.1f}s{marker}"
                )
    return out_dir


def _require(run_dir: str | Path, filename: str) -> Path:
    path = Path(run_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Point this at a single run directory (one "
            f"config), not at the parent folder holding several of them. A run "
            f"directory contains run_meta.json and bundles.jsonl."
        )
    return path


def load_bundles(run_dir: str | Path) -> list[dict]:
    with open(_require(run_dir, "bundles.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_meta(run_dir: str | Path) -> dict:
    return json.loads(_require(run_dir, "run_meta.json").read_text())

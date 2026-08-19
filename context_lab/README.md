# context-lab

A teaching harness for running your own context-engineering experiment.

`context-lab` runs a synthetic multi-turn conversation under two or three
pluggable memory configurations, saves one JSON trace bundle per turn, and then
**gates, grades, and reports entirely offline** from those saved bundles. It is
a shrunk, self-contained version of the eval harness behind the Towards AI
production tutor, sized so a full student experiment costs well under a dollar.

It is the companion package for Lesson 8 of Section 13 in
[From Beginner to Advanced LLM Developer](https://academy.towardsai.net/courses/take/beginner-to-advanced-llm-dev).

## The one property worth internalizing

**Run once, grade forever.** Exactly one step spends money:

| step | spends money | what it does |
|---|---|---|
| `run` | yes | executes the battery, writes `bundles.jsonl` |
| `gate` | no | did compaction actually fire before the probes? |
| `grade` | no (code checks) | regex checks over saved answers |
| `grade --judge` | yes | adds an LLM judge with typed verdicts |
| `report` | no | side-by-side cost, cache, and recall table |

Because every turn is persisted, a metric bug you find next month re-grades last
month's run for free. That is not a convenience. In the production program it is
the difference between a $323 experiment you can re-analyze forever and a $323
experiment you have to repeat.

## Install

```bash
pip install "context-lab @ git+https://github.com/towardsai/ai-tutor-rag-system.git#subdirectory=context_lab"
```

Or, from a local clone:

```bash
pip install -e /path/to/ai-tutor-rag-system/context_lab
```

## Start with no API key at all

The package ships one pre-run results bundle, produced by a real (cheap) DeepSeek
run on 2026-08-19. Unpack it and use the whole offline half of the pipeline
before you spend anything:

```bash
context-lab prerun --out prerun_runs
context-lab gate   prerun_runs/*
context-lab grade  prerun_runs/*
context-lab report prerun_runs/* --out prerun_runs/report.md
```

The shipped bundle contains three runs, and one of them **fails the validity
gate on purpose** (`summarize_prod_trigger`, a summarization policy whose
trigger is set so high it can never fire on a short session). Seeing the gate
reject a run is part of the exercise.

## Run it yourself

```bash
export DEEPSEEK_API_KEY=...
export GEMINI_API_KEY=...     # only for `grade --judge`
context-lab run --config full_history --config summarize --out runs
context-lab gate   runs/*
context-lab grade  runs/*            # add --judge for the LLM judge
context-lab report runs/* --out runs/report.md
```

### Providers

This package sits deliberately outside the course's three-provider pattern.
Context-engineering economics are cache-discount economics, and cache discounts
live on specific providers. Worse, a provider that grants the discount does not
necessarily *report* it, and a discount you cannot see is a discount you cannot
put in a table.

| provider | status | notes |
|---|---|---|
| `deepseek` | tested, the default | `deepseek-v4-flash`. Reports `prompt_cache_hit_tokens` on every call; ~31x cache discount. Thinking is disabled (see `get_provider`). |
| `gemini` | tested, and the judge backend | `gemini-3.5-flash`. Implicit caching, reported intermittently; `gemini-3.5-flash-lite` and `gemini-3.7-flash` report nothing at all (measured 2026-08-19). |
| `ollama` | code present, untested here | free and local; set `OLLAMA_BASE_URL`. No cache accounting, and $0 anyway. |

DeepSeek and Ollama share one backend, `OpenAICompatProvider`, which speaks the
OpenAI-compatible chat API over plain HTTP. Point it at any other endpoint of
that shape by copying four lines in `get_provider`.

## The memory configurations

Each mirrors a real preset in the production tutor's `app/memory_presets.py`,
simplified to teaching size.

| config | mirrors | behavior |
|---|---|---|
| `full_history` | `full_history` | keep everything. The control arm. |
| `summarize` | the `prod` summarization family | past a trigger, one LLM call rewrites older turns into a short summary that replaces them. Lossy and cache-breaking, on purpose. |
| `capped` | `exp_fh_cap10k` | large payloads are truncated **once**, when they enter history, and never rewritten. Stable bytes, so the cache survives. |
| `summarize_prod_trigger` | a deliberately mis-sized arm | summarization with a 200k-token trigger that cannot fire on a short session. It exists so the validity gate has something to reject. |

## The battery

`battery_docs_v1.jsonl` is a 17-turn synthetic session for a fictional
documentation-search project. It is **100% synthetic and authored for this
course**: no real user, student, or customer text appears anywhere in it, and
the "documentation excerpts" are original prose written for this battery.

- **Turns 1-3 plant five facts**: the project targets Python 3.11 only; no
  dependency may need a C extension; collection names carry a `ds_` prefix; all
  code must use type hints and the `logging` module rather than `print()`; and
  pandas is banned from the image.
- **Turns 4-11 are filler** technical Q&A, three of them (turns 4, 6, and 9)
  carrying large document payloads that stand in for tool outputs.
- **Turn 9 updates a fact**: the vector store port moves from 8010 to 9010.
- **Turns 12-17 are probes**: three `fact_recall`, one `fact_update`, and two
  `preference_compliance`.

Facts planted early and probed late is the whole design. It is what makes the
difference between memory policies measurable at all.

## The validity gate

A memory probe measures whether facts survive compaction. If compaction never
fired before the probe, the probe measured nothing: the number would be a fact
about an idle code path.

- A **summarizing** config must show at least one compaction, and every probe
  must come at or after the first one.
- A **non-summarizing** config must show **zero** compactions. A control arm that
  compacted invalidates the comparison.

Runs that fail the gate keep their cost and latency numbers (those are still
real) but their memory columns read `n/a (gate)` in the report, so an invalid
memory number is hard to quote by accident.

## Bundle schema

One JSON object per turn in `bundles.jsonl`:

```json
{
  "run_id": "docsearch_v1|turn07",
  "session_id": "docsearch_v1",
  "config": "summarize",
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "turn_index": 7,
  "started_at": "2026-07-30T12:00:00+00:00",
  "latency_ms": 2140,
  "query": "...",
  "answer": "...",
  "usage": {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0},
  "est_cost_usd": 0.0,
  "context_stats": {
    "history_messages": 14,
    "history_tokens_approx": 5210,
    "compactions_this_turn": 1,
    "compaction_events": [{"pre_compaction_tokens_approx": 3400, "...": "..."}],
    "cap_events": []
  },
  "error": null
}
```

The production harness writes a wider version of exactly this record (real tool
calls, resolved sources, cache-read/cache-miss token splits, a run fingerprint).
The shape is the same because the property is the same: everything a grader
could ever need is on disk before the grader exists.

## Honest limitations

- **The judge is not validated.** In production, a judge's numbers only count
  after it reproduces human labels at a preset agreement bar. Nothing here does
  that for you. Read judge verdicts next to the code checks and adjudicate
  disagreements yourself.
- **The judge is a small model.** It defaults to `gemini-3.5-flash-lite`, which
  is at least a different family from the DeepSeek model under test, but it is
  still a cheap grader. In the shipped bundle it disagrees with the code checks
  on two `summarize` probes, and on both it is the stricter of the two.
- **Latency is confounded.** Arms run sequentially against a shared API, so
  wall-clock differences include provider load. Tokens and cost are the honest
  efficiency headline.
- **One session, one trial.** Any single run is an anecdote. The production
  screens ran 1-3 trials per cell and still label their confidence carefully.

## License

MIT.

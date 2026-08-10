"""Per-model pricing (USD per 1M tokens) and cost estimation.

Prices verified against the providers' public price pages in July 2026. When a
model is missing from the table, cost columns show ``n/a`` instead of a made-up
number. Cached input is billed at the cache-read discount; on Gemini the
discount comes from implicit caching, on DeepSeek from automatic prefix
caching. Local Ollama models cost $0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input: float  # cache-miss input, $ / 1M tokens
    output: float
    cache_read: float  # cached input, $ / 1M tokens


MODEL_PRICING: dict[str, ModelPricing] = {
    # Gemini API, standard paid tier (ai.google.dev/gemini-api/docs/pricing,
    # checked 2026-07): $0.30 in / $2.50 out / $0.03 cached.
    "gemini-3.5-flash-lite": ModelPricing(input=0.30, output=2.50, cache_read=0.03),
    "gemini-3.5-flash": ModelPricing(input=1.50, output=9.00, cache_read=0.15),
    # DeepSeek first-party API (api-docs.deepseek.com pricing, checked 2026-07):
    # $0.14 cache-miss / $0.28 out / $0.0028 cache-hit (the ~50x discount).
    "deepseek-v4-flash": ModelPricing(input=0.14, output=0.28, cache_read=0.0028),
}


def pricing_for(model: str) -> ModelPricing | None:
    """Longest-prefix match, so dated snapshots resolve to their base model."""
    best: tuple[int, ModelPricing] | None = None
    for prefix, pricing in MODEL_PRICING.items():
        if model.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), pricing)
    if best:
        return best[1]
    if model.startswith("ollama/"):
        return ModelPricing(input=0.0, output=0.0, cache_read=0.0)
    return None


def estimate_cost_usd(model: str, usage: dict) -> float | None:
    """Estimated dollars for one call's usage, cache discount applied.

    ``usage`` uses the bundle schema: input_tokens (total billed input),
    cached_input_tokens (the subset billed at the cache-read rate), and
    output_tokens.
    """
    pricing = pricing_for(model)
    if pricing is None:
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    cached = min(int(usage.get("cached_input_tokens") or 0), input_tokens)
    output_tokens = int(usage.get("output_tokens") or 0)
    return (
        (input_tokens - cached) * pricing.input
        + cached * pricing.cache_read
        + output_tokens * pricing.output
    ) / 1_000_000

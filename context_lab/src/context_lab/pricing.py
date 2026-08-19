"""Per-model pricing (USD per 1M tokens) and cost estimation.

Prices verified against the providers' public price pages on 2026-08-19. When a
model is missing from the table, cost columns show ``n/a`` instead of a made-up
number. Cached input is billed at the cache-read discount; on Gemini the
discount comes from implicit caching, on DeepSeek from automatic prefix
caching. Local Ollama models cost $0.

Model prices move, and they move in the direction that changes this lab's
answer. Re-check the two price pages before quoting any cost number here.
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
    # re-checked 2026-08-19). Both flash tiers discount cached input 10x.
    "gemini-3.5-flash-lite": ModelPricing(input=0.30, output=2.50, cache_read=0.03),
    "gemini-3.5-flash": ModelPricing(input=1.50, output=9.00, cache_read=0.15),
    # DeepSeek first-party API (api-docs.deepseek.com pricing, re-checked
    # 2026-08-19). DeepSeek bills two rates by clock: peak is 01:00-04:00 and
    # 06:00-10:00 UTC ($0.44 miss / $1.32 out / $0.014 hit) and off-peak is half
    # of that. A run straddles both, so the table carries the 7-peak/17-off-peak
    # blended average: $0.2842 miss / $0.8525 out / $0.00904 hit, a ~31x cache
    # discount. Cost rows are therefore an average-price estimate, not a bill.
    "deepseek-v4-flash": ModelPricing(input=0.2842, output=0.8525, cache_read=0.00904),
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

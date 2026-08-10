"""Memory configurations: the pluggable context-management policies under test.

Each config mirrors a production preset from the AI tutor's
``app/memory_presets.py``, simplified to teaching size:

- ``full_history``       -> production ``full_history`` (keep everything).
- ``summarize``          -> the summarization family (``prod``-style): when the
                            approximate history size crosses a trigger, an LLM
                            call rewrites the older turns into one short summary
                            that REPLACES them. Lossy on purpose.
- ``capped``             -> the stable tool-output cap (``exp_fh_cap10k``):
                            large document payloads are truncated ONCE, when
                            they enter history, and never rewritten again.
- ``summarize_prod_trigger`` -> a deliberately mis-sized arm: the summarization
                            policy with a production-scale trigger (200k tokens)
                            that can never fire on a short teaching session.
                            It exists so the validity gate has something to
                            reject.

The summary prompt is deliberately generic (like the stock middleware prompt in
LangChain, which is agentic-task flavored). Production experiments showed that
generic summaries are exactly what evicts early planted facts; a fact-preserving
prompt is a separate arm (``selective_retention``) you can build as an exercise.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


def approx_tokens(text: str) -> int:
    """Cheap token approximation (chars / 4), used only for trigger logic.

    Reported metrics always come from provider-billed usage, never from this.
    """
    return len(text) // 4


@dataclass(frozen=True)
class MemoryConfig:
    name: str
    # Summarization policy (Axis A: how conversation history is kept).
    summarize: bool = False
    summarize_trigger_tokens: int = 3_000
    # user+assistant pairs kept verbatim after the summary. Sized to be roughly
    # as aggressive, relative to session length, as the production ``aggressive``
    # preset (an 8k trigger on sessions that reach 30k+). A larger kept window
    # on a session this short leaves the planted facts inside it, and the arm
    # then measures nothing.
    summarize_keep_recent_turns: int = 2
    summary_max_words: int = 100
    # Stable insertion-time cap on document payloads (Axis B: tool-output size).
    # Applied once, when the payload enters history, so every later model call
    # re-reads byte-identical text (cache-friendly). None disables it.
    doc_cap_chars: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


CONFIGS: dict[str, MemoryConfig] = {
    "full_history": MemoryConfig(name="full_history"),
    "summarize": MemoryConfig(name="summarize", summarize=True),
    "capped": MemoryConfig(name="capped", doc_cap_chars=1_500),
    "summarize_prod_trigger": MemoryConfig(
        name="summarize_prod_trigger",
        summarize=True,
        summarize_trigger_tokens=200_000,
    ),
}


SUMMARY_PROMPT = """Summarize the conversation below in under {max_words} words.
Your summary will REPLACE these messages in the assistant's memory, so anything
you leave out is forgotten. Cover the main topics discussed, decisions made,
and any next steps. Respond with the summary only.

<conversation>
{conversation}
</conversation>"""


def render_conversation(messages: list[dict]) -> str:
    """Serialize messages for the summarizer prompt."""
    lines = []
    for message in messages:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['text']}")
    return "\n\n".join(lines)


def history_tokens(messages: list[dict]) -> int:
    return sum(approx_tokens(m["text"]) for m in messages)


def split_for_summary(
    messages: list[dict], keep_recent_turns: int
) -> tuple[list[dict], list[dict]]:
    """Split history into (older span to summarize, recent span to keep).

    The cut lands on a user-turn boundary so an assistant reply is never
    orphaned from its question. Returns ([], messages) when the history is too
    short to have an older span.
    """
    user_indexes = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_indexes) <= keep_recent_turns:
        return [], messages
    cut = user_indexes[-keep_recent_turns]
    return messages[:cut], messages[cut:]

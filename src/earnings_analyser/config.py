"""DSPy + local Ollama configuration.

Import `configure_dspy()` once at process start (Streamlit entrypoint,
scripts, tests) before touching any DSPy module or signature. Every module
in this package takes its LM from `dspy.configure`/`dspy.context` only —
never constructs its own — so configuration order and overrides stay in
one place.

The Gemini path from the original build is retired here; it only comes
back as a build-time teacher for the (separate, out-of-scope for this
module) DSPy prompt-optimization workstream.
"""

import os

import dspy
from dotenv import load_dotenv

DEFAULT_MODEL = "ollama_chat/qwen3:4b-instruct"
DEFAULT_API_BASE = "http://localhost:11434"


def configure_dspy(model: str = DEFAULT_MODEL, api_base: str = DEFAULT_API_BASE, **lm_kwargs) -> dspy.LM:
    """Load .env, wire up a local Ollama LM, and set it as the DSPy default.

    Returns the configured `dspy.LM` so callers can reuse it directly if
    needed, but nothing downstream should construct a second `dspy.LM` of
    its own — that's the exact bug this rebuild fixed (see
    `implementation-plan.md` §3).
    """
    load_dotenv()

    api_base = os.getenv("OLLAMA_API_BASE", api_base)
    model = os.getenv("OLLAMA_MODEL", model)

    # Deterministic decoding by default: the collapse signature classifies
    # into fixed discrete anchors (ScoreLabel, relevance buckets) rather
    # than freeform numbers, so temperature=0 keeps that reproducible.
    lm_kwargs.setdefault("temperature", 0.0)

    lm = dspy.LM(model, api_base=api_base, api_key=lm_kwargs.pop("api_key", "ollama"), **lm_kwargs)
    dspy.configure(lm=lm)
    return lm

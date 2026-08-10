"""DSPy + Gemini Flash configuration.

Import `configure_dspy()` once at process start (Streamlit entrypoint,
scripts, tests) before touching any DSPy module or signature.
"""

import os

import dspy
from dotenv import load_dotenv

DEFAULT_MODEL = "gemini/gemini-flash-lite-latest"


def configure_dspy(model: str = DEFAULT_MODEL, **lm_kwargs) -> dspy.LM:
    """Load .env, wire up a Gemini Flash LM, and set it as the DSPy default.

    Returns the configured `dspy.LM` so callers can also use it directly
    (e.g. to swap in a stronger model for a specific dimension/signature).
    """
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Add it to a .env file in the project root."
        )
    api_key = api_key.strip()

    # litellm's Gemini provider (used by dspy.LM) reads GEMINI_API_KEY;
    # mirror GOOGLE_API_KEY into it so the .env only needs one var.
    os.environ.setdefault("GEMINI_API_KEY", api_key)

    # Deterministic decoding by default: our signatures classify into a
    # fixed set of discrete anchor levels rather than freeform numbers, and
    # temperature=0 makes that classification reproducible run-to-run.
    lm_kwargs.setdefault("temperature", 0.0)

    lm = dspy.LM(model, api_key=api_key, **lm_kwargs)
    dspy.configure(lm=lm)
    return lm

"""DSPy + backend configuration — local Ollama or a cloud provider.

Import `configure_dspy()` once at process start (Streamlit entrypoint,
scripts, tests) before touching any DSPy module or signature. Every module
in this package takes its LM from `dspy.configure`/`dspy.context` only —
never constructs its own — so configuration order and overrides stay in
one place.

Local (Ollama) and cloud (Gemini) are both first-class runtime backends,
selected by `EARNINGS_ANALYSER_BACKEND` — not just LM choice, but pipeline
tuning too: cloud can sustain far more concurrent requests and larger
batches per call than local hardware, so `BACKEND_PROFILES` pairs each
backend's LM with the pipeline knobs (`window_size`, `branching_factor`,
`max_workers`, ...) suited to it.

`requests_per_minute` exists because thread-pool `max_workers` alone isn't
enough for a real cloud quota: measured directly against a free-tier Gemini
key, `gemini-flash-lite-latest` caps at 15 req/min, and a burst of fast
calls blew through that almost immediately even with retries. 12/min (a
margin under the measured 15) is a free-tier-safe default — a paid tier
raises this a lot; override via `EARNINGS_ANALYSER_REQUESTS_PER_MINUTE`
rather than editing the profile, since actual quota depends on the
caller's specific plan, not something this module can know in advance.
"""

import os

import dspy
from dotenv import load_dotenv

DEFAULT_MODEL = "ollama_chat/qwen3:4b-instruct"
DEFAULT_API_BASE = "http://localhost:11434"

BACKEND_PROFILES: dict[str, dict] = {
    "local": {
        "model": DEFAULT_MODEL,
        "api_base": DEFAULT_API_BASE,
        "api_key": "ollama",
        "pipeline": {
            "window_size": 8,
            "window_overlap": 0,
            "branching_factor": 16,
            "target": 5,
            "max_workers": 4,
            "requests_per_minute": None,
        },
    },
    "cloud": {
        "model": "gemini/gemini-flash-lite-latest",
        "api_base": None,
        "api_key_env": "GOOGLE_API_KEY",
        "pipeline": {
            "window_size": 20,
            "window_overlap": 0,
            "branching_factor": 24,
            "target": 5,
            "max_workers": 16,
            "requests_per_minute": 12,
        },
    },
}


def configure_dspy(backend: str | None = None, **lm_kwargs) -> tuple[dspy.LM, dict, str]:
    """Load .env, wire up the selected backend's LM, and set it as the
    DSPy default. Returns `(lm, pipeline_profile, backend)` — pass
    `pipeline_profile` straight into `pipeline.run_pipeline(**pipeline_profile)`;
    `backend` is the resolved name ("local"/"cloud"), for display purposes.

    Backend resolution: the `backend` param, else `EARNINGS_ANALYSER_BACKEND`,
    else "local". `EARNINGS_ANALYSER_MAX_WORKERS`/`EARNINGS_ANALYSER_REQUESTS_PER_MINUTE`,
    if set, override the chosen profile's values — actual safe concurrency
    and quota depend on the caller's specific plan (cloud) or
    `OLLAMA_NUM_PARALLEL` (local), neither of which this module can know
    in advance.
    """
    load_dotenv()

    backend = backend or os.getenv("EARNINGS_ANALYSER_BACKEND", "local")
    if backend not in BACKEND_PROFILES:
        raise ValueError(f"Unknown backend {backend!r} — expected one of {sorted(BACKEND_PROFILES)}")
    profile = BACKEND_PROFILES[backend]

    api_base = os.getenv("OLLAMA_API_BASE", profile["api_base"]) if backend == "local" else profile["api_base"]
    model = os.getenv("OLLAMA_MODEL", profile["model"]) if backend == "local" else profile["model"]
    api_key = lm_kwargs.pop("api_key", None) or (
        os.environ[profile["api_key_env"]] if "api_key_env" in profile else profile["api_key"]
    )

    # Deterministic decoding by default: the collapse signature classifies
    # into fixed discrete anchors (ScoreLabel, relevance buckets) rather
    # than freeform numbers, so temperature=0 keeps that reproducible.
    lm_kwargs.setdefault("temperature", 0.0)

    lm = dspy.LM(model, api_base=api_base, api_key=api_key, **lm_kwargs)
    dspy.configure(lm=lm)

    pipeline_profile = dict(profile["pipeline"])
    max_workers_override = os.getenv("EARNINGS_ANALYSER_MAX_WORKERS")
    if max_workers_override:
        pipeline_profile["max_workers"] = int(max_workers_override)
    rpm_override = os.getenv("EARNINGS_ANALYSER_REQUESTS_PER_MINUTE")
    if rpm_override:
        pipeline_profile["requests_per_minute"] = int(rpm_override)

    return lm, pipeline_profile, backend

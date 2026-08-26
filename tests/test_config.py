from earnings_analyser.config import BACKEND_PROFILES, configure_dspy


def _no_dotenv(monkeypatch):
    # A real .env on disk (e.g. EARNINGS_ANALYSER_BACKEND=cloud during local
    # dev) would otherwise refill vars these tests just cleared — these tests
    # exercise env-var resolution only, not whatever .env happens to hold.
    monkeypatch.setattr("earnings_analyser.config.load_dotenv", lambda *a, **k: None)


def test_default_backend_is_local_with_unchanged_profile(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.delenv("EARNINGS_ANALYSER_BACKEND", raising=False)
    monkeypatch.delenv("EARNINGS_ANALYSER_MAX_WORKERS", raising=False)
    monkeypatch.delenv("EARNINGS_ANALYSER_REQUESTS_PER_MINUTE", raising=False)

    lm, profile, _ = configure_dspy()

    assert lm.model == BACKEND_PROFILES["local"]["model"]
    assert profile == BACKEND_PROFILES["local"]["pipeline"]


def test_cloud_backend_resolves_cloud_profile(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("EARNINGS_ANALYSER_BACKEND", "cloud")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")
    monkeypatch.delenv("EARNINGS_ANALYSER_MAX_WORKERS", raising=False)
    monkeypatch.delenv("EARNINGS_ANALYSER_REQUESTS_PER_MINUTE", raising=False)

    lm, profile, _ = configure_dspy()

    assert lm.model == BACKEND_PROFILES["cloud"]["model"]
    assert profile == BACKEND_PROFILES["cloud"]["pipeline"]
    assert profile["requests_per_minute"] == 12


def test_max_workers_env_override_wins(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("EARNINGS_ANALYSER_BACKEND", "local")
    monkeypatch.setenv("EARNINGS_ANALYSER_MAX_WORKERS", "9")
    monkeypatch.delenv("EARNINGS_ANALYSER_REQUESTS_PER_MINUTE", raising=False)

    _, profile, _backend = configure_dspy()

    assert profile["max_workers"] == 9


def test_requests_per_minute_env_override_wins(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("EARNINGS_ANALYSER_BACKEND", "cloud")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")
    monkeypatch.setenv("EARNINGS_ANALYSER_REQUESTS_PER_MINUTE", "60")

    _, profile, _backend = configure_dspy()

    assert profile["requests_per_minute"] == 60


def test_unknown_backend_raises(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("EARNINGS_ANALYSER_BACKEND", "carrier-pigeon")
    try:
        configure_dspy()
        assert False, "expected ValueError"
    except ValueError:
        pass

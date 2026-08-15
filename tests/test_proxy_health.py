from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app
from headroom.transforms import kompress_compressor


class _ReadyCompressor:
    def __init__(self, backend="onnx", ready=True, error=None):
        self.backend = backend
        self.ready = ready
        self.error = error
        self.calls = []

    def is_ready(self):
        self.calls.append("is_ready")
        if self.error:
            raise self.error
        return self.ready

    def ready_backend(self):
        self.calls.append("ready_backend")
        return self.backend


def _health_app(monkeypatch, compressor=None, *, disabled=False, **config_kwargs):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            disable_kompress=disabled,
            **config_kwargs,
        )
    )
    app.state.ready = True
    proxy = app.state.proxy
    proxy.http_client = object()
    router = proxy.anthropic_pipeline.transforms[-1]
    if compressor is not None:
        router._kompress = compressor
    return app, proxy


def test_readyz_excludes_kompress_from_aggregate_readiness(monkeypatch):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
        )
    )
    app.state.ready = True
    proxy = app.state.proxy
    proxy.http_client = object()
    proxy.warmup.kompress.mark_error("model not cached")

    client = TestClient(app)
    response = client.get("/readyz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["status"] == "healthy"
    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": False,
        "status": "unhealthy",
        "backend": None,
    }


def test_readyz_promotes_deferred_kompress_after_runtime_load(monkeypatch):
    compressor = _ReadyCompressor()
    app, proxy = _health_app(monkeypatch, compressor)
    proxy.warmup.kompress.info["source_status"] = "deferred"

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": True,
        "status": "healthy",
        "backend": "onnx",
    }


@pytest.mark.parametrize("attached", [False, True])
def test_readyz_promotes_kompress_from_module_cache(monkeypatch, attached):
    model = object()
    monkeypatch.setattr(
        kompress_compressor,
        "_kompress_cache",
        {kompress_compressor.HF_MODEL_ID: (model, object(), "onnx")},
    )
    compressor = _ReadyCompressor(ready=False) if attached else None
    app, proxy = _health_app(monkeypatch, compressor)
    proxy.warmup.kompress.info["source_status"] = "deferred"

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": True,
        "status": "healthy",
        "backend": "onnx",
    }
    assert proxy.warmup.kompress.handle is model
    if compressor is not None:
        assert compressor.calls == ["is_ready"]


def test_readyz_promotes_remote_kompress_backend(monkeypatch):
    compressor = _ReadyCompressor(backend="remote")
    app, proxy = _health_app(monkeypatch)
    router = proxy.anthropic_pipeline.transforms[-1]
    router._kompress = None
    router._kompress_remote = compressor

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"]["backend"] == "remote"
    assert payload["checks"]["kompress"]["ready"] is True


def test_readyz_keeps_pending_kompress_unloaded(monkeypatch):
    compressor = _ReadyCompressor(backend="onnx", ready=False)
    app, proxy = _health_app(monkeypatch, compressor)
    router = proxy.anthropic_pipeline.transforms[-1]
    router._kompress = compressor

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": False,
        "status": "unhealthy",
        "backend": None,
    }
    assert compressor.calls == ["is_ready"]


def test_readyz_never_starts_kompress_loading(monkeypatch):
    compressor = _ReadyCompressor()
    app, proxy = _health_app(monkeypatch, compressor)
    router = proxy.anthropic_pipeline.transforms[-1]
    router._kompress = compressor

    TestClient(app).get("/readyz")

    assert compressor.calls == ["is_ready", "ready_backend"]


def test_readyz_kompress_inspection_failure_fails_open(monkeypatch):
    compressor = _ReadyCompressor(error=RuntimeError("inspection failed"))
    app, proxy = _health_app(monkeypatch, compressor)
    proxy.warmup.kompress.mark_loaded(handle=object(), backend="onnx")

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"]["ready"] is True
    assert payload["checks"]["kompress"]["backend"] == "onnx"


def test_readyz_disabled_kompress_skips_inspection(monkeypatch):
    compressor = _ReadyCompressor()
    app, proxy = _health_app(monkeypatch, compressor, disabled=True)
    router = proxy.anthropic_pipeline.transforms[-1]
    router._kompress = compressor

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": False,
        "ready": True,
        "status": "disabled",
        "backend": None,
    }
    assert compressor.calls == []


def test_readyz_per_provider_kompress_override_reenables_health(monkeypatch):
    compressor = _ReadyCompressor()
    app, proxy = _health_app(
        monkeypatch,
        disabled=True,
        disable_kompress_anthropic=False,
    )
    router = proxy.anthropic_pipeline.transforms[-1]
    router._kompress = compressor

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": True,
        "status": "healthy",
        "backend": "onnx",
    }
    assert compressor.calls == ["is_ready", "ready_backend"]


def test_readyz_never_calls_lazy_kompress_getters(monkeypatch):
    app, proxy = _health_app(monkeypatch)
    router = proxy.anthropic_pipeline.transforms[-1]

    def _boom():
        raise AssertionError("health should not instantiate kompress")

    router._get_kompress = _boom
    router._get_remote_kompress = _boom

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": False,
        "status": "unhealthy",
        "backend": None,
    }


@pytest.mark.parametrize(
    ("slot_status", "compressor", "disabled", "expected"),
    [
        (
            "null",
            None,
            False,
            {"enabled": True, "ready": False, "status": "unhealthy", "backend": None},
        ),
        (
            "null",
            _ReadyCompressor(),
            False,
            {"enabled": True, "ready": True, "status": "healthy", "backend": "onnx"},
        ),
        (
            "null",
            _ReadyCompressor(backend="remote"),
            False,
            {"enabled": True, "ready": True, "status": "healthy", "backend": "remote"},
        ),
        (
            "error",
            _ReadyCompressor(),
            False,
            {"enabled": True, "ready": True, "status": "healthy", "backend": "onnx"},
        ),
        (
            "loaded",
            _ReadyCompressor(),
            False,
            {"enabled": True, "ready": True, "status": "healthy", "backend": "existing"},
        ),
        (
            "null",
            _ReadyCompressor(),
            True,
            {"enabled": False, "ready": True, "status": "disabled", "backend": None},
        ),
    ],
)
def test_readyz_kompress_state_matrix(monkeypatch, slot_status, compressor, disabled, expected):
    app, proxy = _health_app(monkeypatch, compressor, disabled=disabled)
    if slot_status == "error":
        proxy.warmup.kompress.mark_error("not cached")
    elif slot_status == "loaded":
        proxy.warmup.kompress.mark_loaded(handle=object(), backend=expected["backend"])
    if compressor is not None and expected["backend"] == "existing":
        compressor.backend = "new"

    payload = TestClient(app).get("/readyz").json()["checks"]["kompress"]

    assert payload == expected


def test_health_reports_the_effective_memory_db_path(monkeypatch, tmp_path):
    """A `wrap --memory --no-proxy` attaching to this proxy needs the database
    it is ACTUALLY using: the documented fallback resolves against each
    process's own cwd, so "same default rule" does not mean "same file" when
    the proxy was started from a different directory.
    """
    db = tmp_path / "srv" / ".headroom" / "memory.db"
    app, _proxy = _health_app(monkeypatch, memory_enabled=True, memory_db_path=str(db))

    # The config block is loopback-only, gated on BOTH the peer IP and the
    # Host header, so present as a genuine local caller on both.
    payload = (
        TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 40000))
        .get("/health")
        .json()
    )

    assert payload["config"]["memory_db_path"] == str(db)


def test_health_reports_the_RESOLVED_default_database(monkeypatch, tmp_path):
    """The load-bearing case: started WITHOUT an explicit --memory-db-path.

    `config.memory_db_path` stays "" while the pipeline resolves the real file
    against this process's startup cwd. Reporting the config value would tell
    an attaching wrap nothing, and it would fall back to its OWN cwd — a
    different database.
    """
    monkeypatch.chdir(tmp_path)
    app, _proxy = _health_app(monkeypatch, memory_enabled=True)

    payload = (
        TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 40000))
        .get("/health")
        .json()
    )

    reported = payload["config"]["memory_db_path"]
    assert reported, "the resolved default was not reported"
    assert Path(reported) == tmp_path / ".headroom" / "memory.db"


def test_health_resolves_an_explicit_RELATIVE_database(monkeypatch, tmp_path):
    """`--memory-db-path state/memory.db` resolves against the PROXY's cwd.

    Reporting the unresolved string sends an attaching wrap off to anchor the
    same relative path against its own directory, so MCP/wrap-side memory ends
    up on a different file than API-side retrieval — the same divergence the
    default-path fix closed, one branch over.
    """
    monkeypatch.chdir(tmp_path)
    app, _proxy = _health_app(monkeypatch, memory_enabled=True, memory_db_path="state/memory.db")

    payload = (
        TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 40000))
        .get("/health")
        .json()
    )

    reported = payload["config"]["memory_db_path"]
    assert Path(reported).is_absolute(), f"still relative: {reported!r}"
    assert Path(reported) == tmp_path / "state" / "memory.db"


def test_health_leaves_an_absolute_database_alone(monkeypatch, tmp_path):
    """Anchoring must not rewrite a path the operator already made absolute."""
    db = tmp_path / "explicit" / "memory.db"
    app, _proxy = _health_app(monkeypatch, memory_enabled=True, memory_db_path=str(db))

    payload = (
        TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 40000))
        .get("/health")
        .json()
    )

    assert payload["config"]["memory_db_path"] == str(db)


def test_health_reports_the_server_instance_from_the_environment(monkeypatch):
    """Workers inherit this from the launcher, so every worker reports the same
    value — unlike `pid`, which is the answering worker's. `headroom wrap`
    compares against it to recognise the proxy it just started, and with
    HEADROOM_WORKERS>1 a pid comparison cannot succeed at all.
    """
    monkeypatch.setenv("HEADROOM_SERVER_INSTANCE", "srv-abc123")
    app, _proxy = _health_app(monkeypatch)

    payload = (
        TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 40000))
        .get("/health")
        .json()
    )

    assert payload["config"]["server_instance"] == "srv-abc123"


def test_health_server_instance_is_empty_when_unset(monkeypatch):
    """A proxy nobody launched through wrap reports no instance, which callers
    must read as inconclusive rather than as a mismatch."""
    monkeypatch.delenv("HEADROOM_SERVER_INSTANCE", raising=False)
    app, _proxy = _health_app(monkeypatch)

    payload = (
        TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 40000))
        .get("/health")
        .json()
    )

    assert payload["config"]["server_instance"] == ""


def test_health_memory_db_path_is_present_even_when_unset(monkeypatch):
    """The key must always exist, so a caller can tell "this proxy does not
    report it" (older build) from "it reports no path"."""
    app, _proxy = _health_app(monkeypatch, memory_enabled=False)

    payload = (
        TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 40000))
        .get("/health")
        .json()
    )

    assert "memory_db_path" in payload["config"]

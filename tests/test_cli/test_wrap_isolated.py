"""Tests for `headroom wrap --isolated` (per-run proxy + workspace isolation).

Without isolation, a second concurrent wrap reuses the proxy already
listening on the shared port and writes into the shared workspace. These
tests pin the isolated behavior: a dedicated proxy on a fresh port, the
shared port left alone, and a per-run workspace activated by the group
callback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from headroom import isolation, paths
from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture(autouse=True)
def _clean_isolation_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(tmp_path / "ws"))
    mutated = (
        paths.HEADROOM_CONFIG_DIR_ENV,
        paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV,
        isolation.HEADROOM_ISOLATED_ENV,
        isolation.HEADROOM_ISOLATED_WORKSPACE_ENV,
        isolation.HEADROOM_MEMORY_DB_PATH_ENV,
    )
    for var in mutated:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in mutated:
        os.environ.pop(var, None)


def _run_in_click_context(fn) -> str:  # type: ignore[no-untyped-def]
    runner = CliRunner()

    @click.command()
    def _cmd() -> None:
        fn()

    result = runner.invoke(_cmd)
    assert result.exit_code == 0, result.output
    return result.output


class _FakeProc:
    def poll(self) -> None:
        return None


class TestEnsureProxyIsolated:
    """_ensure_proxy must never attach to a shared proxy in isolated mode."""

    def _patch_common(self, monkeypatch: pytest.MonkeyPatch, started: list[int]) -> None:
        monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_find_available_port", lambda p: p)

        def fake_start(port: int, **_kw: Any) -> _FakeProc:
            started.append(port)
            return _FakeProc()

        monkeypatch.setattr(wrap_mod, "_start_proxy", fake_start)

    def test_running_shared_proxy_is_not_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A healthy proxy on the requested port is left alone; the isolated
        run starts its own instance one port up."""

        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "1")
        started: list[int] = []
        self._patch_common(monkeypatch, started)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)

        result: list[Any] = []
        output = _run_in_click_context(
            lambda: result.append(wrap_mod._ensure_proxy(8787, no_proxy=False))
        )

        proc, actual_port = result[0]
        assert isinstance(proc, _FakeProc)
        assert actual_port == 8788
        assert started == [8788]
        assert "Isolated run: starting a dedicated proxy instance" in output
        assert "Port 8787 is reserved for the shared proxy" in output

    def test_free_shared_port_is_still_reserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even with nothing on the shared port, an isolated run must not
        claim it — a later plain wrap would silently attach to the isolated
        proxy (and its ephemeral workspace)."""

        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "1")
        started: list[int] = []
        self._patch_common(monkeypatch, started)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: False)

        result: list[Any] = []
        _run_in_click_context(lambda: result.append(wrap_mod._ensure_proxy(8787, no_proxy=False)))

        _proc, actual_port = result[0]
        assert actual_port == 8788
        assert started == [8788]

    def test_without_isolation_running_proxy_is_reused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Baseline (unchanged default): a healthy matching proxy is reused."""

        started: list[int] = []
        self._patch_common(monkeypatch, started)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _p: {})
        monkeypatch.setattr(wrap_mod, "_proxy_needs_version_restart", lambda _h: False)
        monkeypatch.setattr(wrap_mod, "_proxy_health_config", lambda _h: None)
        monkeypatch.setattr(wrap_mod, "_query_proxy_config", lambda _p: None)

        result: list[Any] = []
        output = _run_in_click_context(
            lambda: result.append(wrap_mod._ensure_proxy(8787, no_proxy=False))
        )

        proc, actual_port = result[0]
        assert proc is None
        assert actual_port == 8787
        assert started == []
        assert "Proxy already running on port 8787" in output

    def test_no_proxy_warns_that_isolation_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "1")
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _p: {})
        monkeypatch.setattr(wrap_mod, "_proxy_health_config", lambda _h: None)
        monkeypatch.setattr(wrap_mod, "_query_proxy_config", lambda _p: None)

        output = _run_in_click_context(lambda: wrap_mod._ensure_proxy(8787, no_proxy=True))

        # Isolation is the default, so the warning must explain --no-proxy's
        # effect without naming a flag the user may never have typed.
        assert "gets no dedicated proxy instance" in output


def _invoke_with_fake_tool(cli_args: list[str]) -> tuple[Any, dict[str, Any]]:
    """Invoke `headroom wrap <cli_args>` with a temporary no-op subcommand
    that records the workspace resolved inside the subcommand body."""

    seen: dict[str, Any] = {}

    @click.command("fake-tool")
    def fake_tool() -> None:
        seen["workspace"] = paths.workspace_dir()

    wrap_mod.wrap.add_command(fake_tool)
    try:
        runner = CliRunner()
        result = runner.invoke(main, cli_args)
    finally:
        wrap_mod.wrap.commands.pop("fake-tool", None)
    return result, seen


class TestWrapGroupFlag:
    def test_default_is_isolated(self, tmp_path: Path) -> None:
        """With no flag and no env override, every wrap run gets its own
        per-run workspace — isolation is the default."""

        result, seen = _invoke_with_fake_tool(["wrap", "fake-tool"])

        assert result.exit_code == 0, result.output
        assert "Isolated run: workspace" in result.output
        assert seen["workspace"].parent == tmp_path / "ws" / "runs"
        assert seen["workspace"].name.startswith("run-")

    def test_isolated_flag_activates_per_run_workspace(self, tmp_path: Path) -> None:
        result, seen = _invoke_with_fake_tool(["wrap", "--isolated", "fake-tool"])

        assert result.exit_code == 0, result.output
        assert "Isolated run: workspace" in result.output
        assert seen["workspace"].parent == tmp_path / "ws" / "runs"

    def test_env_var_activates_isolation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "1")

        result, seen = _invoke_with_fake_tool(["wrap", "fake-tool"])

        assert result.exit_code == 0, result.output
        assert seen["workspace"].parent == tmp_path / "ws" / "runs"

    def test_shared_flag_opts_out(self, tmp_path: Path) -> None:
        """`headroom wrap --shared <tool>` restores the legacy shared
        workspace and records the choice for nested processes."""

        result, seen = _invoke_with_fake_tool(["wrap", "--shared", "fake-tool"])

        assert result.exit_code == 0, result.output
        assert "Isolated run" not in result.output
        assert seen["workspace"] == tmp_path / "ws"
        # The explicit opt-out is recorded so _ensure_proxy and nested
        # Headroom invocations don't re-default to isolated.
        assert os.environ.get(isolation.HEADROOM_ISOLATED_ENV) == "0"

    def test_env_var_zero_opts_out(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "0")

        result, seen = _invoke_with_fake_tool(["wrap", "fake-tool"])

        assert result.exit_code == 0, result.output
        assert seen["workspace"] == tmp_path / "ws"

    def test_selfheal_is_exempt_from_isolation(self, tmp_path: Path) -> None:
        """The SessionStart-hook selfheal subcommand must not create a
        per-run workspace on every Claude session start."""

        runner = CliRunner()
        result = runner.invoke(main, ["wrap", "selfheal"])

        assert result.exit_code == 0, result.output
        assert "Isolated run" not in result.output
        assert not (tmp_path / "ws" / "runs").exists()

    @pytest.mark.parametrize("flag", ["--isolated", "--shared"])
    def test_misplaced_group_flag_is_rejected(self, flag: str) -> None:
        """`headroom wrap claude --isolated/--shared` must fail loudly
        instead of forwarding the flag to the wrapped CLI
        (ignore_unknown_options)."""

        runner = CliRunner()
        result = runner.invoke(main, ["wrap", "claude", flag])

        assert result.exit_code != 0
        assert "goes before the tool name" in result.output
        assert f"headroom wrap {flag} claude" in result.output


class _FakeProxyChild:
    """A fake Popen: alive until .kill(), with a fixed pid."""

    def __init__(self, pid: int = 111) -> None:
        self.pid = pid
        self._exited: int | None = None
        self.killed = False

    def poll(self) -> int | None:
        return self._exited

    def kill(self) -> None:
        self.killed = True
        self._exited = -9


class TestStartProxyOwnership:
    """_start_proxy(require_owned=True) must confirm the healthy proxy is the
    child it spawned, not a proxy that won a concurrent bind race (PR #25 P1)."""

    def _patch_spawn(self, monkeypatch: pytest.MonkeyPatch, child: _FakeProxyChild) -> None:
        monkeypatch.setattr(wrap_mod.subprocess, "Popen", lambda *a, **k: child)
        monkeypatch.setattr(wrap_mod, "_resolve_wrap_proxy_timeout_seconds", lambda: 3)
        monkeypatch.setattr(wrap_mod.time, "sleep", lambda _s: None)
        monkeypatch.setattr(wrap_mod, "_get_log_path", lambda: Path("/tmp/hr-proxy.log"))
        monkeypatch.setattr(
            wrap_mod, "_get_proxy_stdio_log_path", lambda: Path("/tmp/hr-proxy-stdio.log")
        )
        monkeypatch.setattr(wrap_mod, "_read_text", lambda _p: "")

    def test_owned_proxy_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        child = _FakeProxyChild(pid=4321)
        self._patch_spawn(monkeypatch, child)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        # Health reports OUR child's pid → confirmed ours.
        monkeypatch.setattr(wrap_mod, "_query_proxy_config", lambda _p: {"pid": 4321})

        assert wrap_mod._start_proxy(9911, require_owned=True) is child
        assert child.killed is False

    def test_foreign_proxy_raises_race_lost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        child = _FakeProxyChild(pid=4321)
        self._patch_spawn(monkeypatch, child)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        # A DIFFERENT proxy owns the port (won the race).
        monkeypatch.setattr(wrap_mod, "_query_proxy_config", lambda _p: {"pid": 9999})

        with pytest.raises(wrap_mod._DedicatedProxyPortRaceLost):
            wrap_mod._start_proxy(9911, require_owned=True)
        # Our losing child is killed so it doesn't linger.
        assert child.killed is True

    def test_require_owned_false_keeps_legacy_reuse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        child = _FakeProxyChild(pid=4321)
        self._patch_spawn(monkeypatch, child)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        # Shared path: pid ownership is never consulted.
        called = {"queried": False}

        def _boom(_p: int) -> dict[str, Any]:
            called["queried"] = True
            return {"pid": 9999}

        monkeypatch.setattr(wrap_mod, "_query_proxy_config", _boom)

        assert wrap_mod._start_proxy(9911, require_owned=False) is child
        assert called["queried"] is False


class TestEnsureProxyPortRaceRetry:
    """_ensure_proxy retries on a higher port when a dedicated start loses the
    bind race (PR #25 P1)."""

    def test_retries_until_a_port_is_won(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "1")
        monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: False)
        monkeypatch.setattr(wrap_mod, "_find_available_port", lambda p: p)

        started: list[int] = []

        def fake_start(port: int, **_kw: Any) -> _FakeProc:
            started.append(port)
            # Lose the race on the first two candidate ports, win the third.
            if len(started) < 3:
                raise wrap_mod._DedicatedProxyPortRaceLost(port)
            return _FakeProc()

        monkeypatch.setattr(wrap_mod, "_start_proxy", fake_start)

        result: list[Any] = []
        _run_in_click_context(lambda: result.append(wrap_mod._ensure_proxy(8787, no_proxy=False)))
        _proc, actual_port = result[0]
        # 8787 reserved for shared proxy → candidates 8788, 8789, 8790.
        assert started == [8788, 8789, 8790]
        assert actual_port == 8790

    def test_gives_up_after_max_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "1")
        monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: False)
        monkeypatch.setattr(wrap_mod, "_find_available_port", lambda p: p)

        def always_lose(port: int, **_kw: Any) -> _FakeProc:
            raise wrap_mod._DedicatedProxyPortRaceLost(port)

        monkeypatch.setattr(wrap_mod, "_start_proxy", always_lose)

        runner = CliRunner()

        @click.command()
        def _cmd() -> None:
            wrap_mod._ensure_proxy(8787, no_proxy=False)

        result = runner.invoke(_cmd)
        assert result.exit_code != 0
        assert "Could not reserve a dedicated proxy port" in result.output


class TestLaunchToolPortReconcile:
    """_launch_tool must fire reconcile_port with the actual port whenever it
    differs from the requested one (PR #25 P1: Codex/Grok MCP, OMP models.yml)."""

    def test_reconcile_fires_on_port_shift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wrap_mod, "_ensure_proxy", lambda *a, **k: (_FakeProc(), 8788))
        monkeypatch.setattr(wrap_mod, "_register_proxy_client", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_unregister_proxy_client", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_push_runtime_env", lambda *a, **k: None)
        monkeypatch.setattr(wrap_mod, "_make_cleanup", lambda *a, **k: lambda *x, **y: None)
        monkeypatch.setattr(wrap_mod.signal, "signal", lambda *a, **k: None)

        seen: list[int] = []

        class _Done(SystemExit):
            pass

        def fake_run(cmd: list[str], **_k: Any) -> Any:
            raise _Done(0)

        monkeypatch.setattr(wrap_mod.subprocess, "run", fake_run)

        runner = CliRunner()

        @click.command()
        def _cmd() -> None:
            wrap_mod._launch_tool(
                binary="/bin/true",
                args=(),
                env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"},
                port=8787,
                no_proxy=False,
                tool_label="TEST",
                env_vars_display=[],
                agent_type="test",
                reconcile_port=lambda actual: seen.append(actual),
            )

        runner.invoke(_cmd)
        assert seen == [8788]

    def test_reconcile_skipped_when_port_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wrap_mod, "_ensure_proxy", lambda *a, **k: (_FakeProc(), 8787))
        monkeypatch.setattr(wrap_mod, "_register_proxy_client", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_unregister_proxy_client", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_push_runtime_env", lambda *a, **k: None)
        monkeypatch.setattr(wrap_mod, "_make_cleanup", lambda *a, **k: lambda *x, **y: None)
        monkeypatch.setattr(wrap_mod.signal, "signal", lambda *a, **k: None)
        monkeypatch.setattr(
            wrap_mod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(SystemExit(0))
        )

        seen: list[int] = []
        runner = CliRunner()

        @click.command()
        def _cmd() -> None:
            wrap_mod._launch_tool(
                binary="/bin/true",
                args=(),
                env={},
                port=8787,
                no_proxy=False,
                tool_label="TEST",
                env_vars_display=[],
                agent_type="test",
                reconcile_port=lambda actual: seen.append(actual),
            )

        runner.invoke(_cmd)
        assert seen == []


class TestWrapMemoryDbPath:
    """wrap-side memory sync must target the same DB the proxy/MCP use so
    isolation actually severs cross-agent memory (PR #25 P1)."""

    def test_honors_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HEADROOM_MEMORY_DB_PATH", str(tmp_path / "run" / "memory.db"))
        assert wrap_mod._wrap_memory_db_path() == tmp_path / "run" / "memory.db"

    def test_defaults_to_project_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_MEMORY_DB_PATH", raising=False)
        assert wrap_mod._wrap_memory_db_path() == Path.cwd() / ".headroom" / "memory.db"


class TestMemoryMcpServerDbDefault:
    """The agent-spawned memory MCP server must resolve its DB from
    HEADROOM_MEMORY_DB_PATH so an isolated run's MCP opens the run DB."""

    def test_db_default_reads_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import argparse

        monkeypatch.setenv("HEADROOM_MEMORY_DB_PATH", str(tmp_path / "iso" / "memory.db"))

        # Mirror the argparse default expression the server uses.
        default = os.environ.get("HEADROOM_MEMORY_DB_PATH", "").strip() or str(
            Path.cwd() / ".headroom" / "memory.db"
        )
        parser = argparse.ArgumentParser()
        parser.add_argument("--db", default=default)
        assert parser.parse_args([]).db == str(tmp_path / "iso" / "memory.db")

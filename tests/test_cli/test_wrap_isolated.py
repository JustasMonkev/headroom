"""Tests for `headroom wrap --isolated` (per-run proxy + workspace isolation).

Without isolation, a second concurrent wrap reuses the proxy already
listening on the shared port and writes into the shared workspace. These
tests pin the isolated behavior: a dedicated proxy on a fresh port, the
shared port left alone, and a per-run workspace activated by the group
callback.
"""

from __future__ import annotations

import json
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
        paths.HEADROOM_SETTINGS_PATH_ENV,
        isolation.HEADROOM_ISOLATED_ENV,
        isolation.HEADROOM_ISOLATED_WORKSPACE_ENV,
        isolation.HEADROOM_MEMORY_DB_PATH_ENV,
        isolation.HEADROOM_ISOLATED_AGENT_HOMES_ENV,
        # Agent config homes: `_isolate_*_home()` writes these directly into
        # os.environ, so without an explicit clean they leak a per-run path
        # into every later test in the session (observed breaking
        # tests/test_install/test_providers.py, which writes Codex config).
        "CODEX_HOME",
        "GROK_HOME",
        "PI_CODING_AGENT_DIR",
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
        monkeypatch.setattr(wrap_mod, "_find_available_port", lambda p, max_attempts=100: p)

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
        monkeypatch.setattr(wrap_mod, "_find_available_port", lambda p, max_attempts=100: p)

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
        monkeypatch.setattr(wrap_mod, "_find_available_port", lambda p, max_attempts=100: p)

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


class TestDedicatedPortSearch:
    """Dedicated proxies never claim the reserved base port, and --port 65535
    (max, Click-accepted) must still work under the isolated default (P2)."""

    @pytest.mark.parametrize(
        "base,after",
        [(8787, None), (8787, 8790), (65535, None), (65535, 65534), (65535, 65533), (1, None)],
    )
    def test_search_range_never_contains_reserved_port(self, base: int, after: int | None) -> None:
        start = wrap_mod._dedicated_port_search_start(base, after=after)
        attempts = wrap_mod._dedicated_port_attempts(base, start)
        lo, hi = start, start + attempts - 1

        assert not (lo <= base <= hi), f"reserved port {base} inside {lo}..{hi}"
        assert hi <= 65535, f"range {lo}..{hi} exceeds the max port"
        assert lo >= 1

    def test_max_port_searches_below_instead_of_failing(self) -> None:
        """65535 + 1 would be an empty range; fall back to a window below."""
        start = wrap_mod._dedicated_port_search_start(65535)
        assert start < 65535
        assert wrap_mod._dedicated_port_attempts(65535, start) == 65535 - start

    def test_normal_port_searches_above(self) -> None:
        assert wrap_mod._dedicated_port_search_start(8787) == 8788

    def test_ensure_proxy_at_max_port_still_starts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: an isolated run with --port 65535 gets a dedicated proxy
        below the reserved port instead of "No available port found"."""

        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "1")
        monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: False)
        monkeypatch.setattr(wrap_mod, "_find_available_port", lambda p, max_attempts=100: p)
        started: list[int] = []

        def fake_start(port: int, **_kw: Any) -> _FakeProc:
            started.append(port)
            return _FakeProc()

        monkeypatch.setattr(wrap_mod, "_start_proxy", fake_start)

        result: list[Any] = []
        _run_in_click_context(lambda: result.append(wrap_mod._ensure_proxy(65535, no_proxy=False)))

        _proc, actual_port = result[0]
        assert actual_port != 65535
        assert actual_port < 65535
        assert started == [actual_port]


class TestClaudeWrapMarkerConcurrency:
    """The project-local .claude/settings.local.json is shared by concurrent
    isolated runs; write/restore must be ownership-aware (PR #25 review P1)."""

    def _settings(self, tmp_path: Path) -> Path:
        p = tmp_path / ".claude" / "settings.local.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _marker(self, settings: Path) -> dict[str, Any]:
        return json.loads(settings.parent.joinpath(".headroom_wrap_marker.json").read_text())

    def test_marker_records_the_port_it_was_given(self, tmp_path: Path) -> None:
        """The self-heal hook probes the marker's port; it must be the port the
        proxy actually bound, not the requested one."""
        settings = self._settings(tmp_path)

        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )

        assert self._marker(settings)["port"] == 8788
        assert self._marker(settings)["pid"] == os.getpid()

    def test_second_run_inherits_the_original_previous_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run B must record the user's ORIGINAL value as `previous`, not run
        A's proxy URL — otherwise B's exit restores a dead endpoint."""
        settings = self._settings(tmp_path)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://upstream"}}))

        # Run A (a different, still-live pid) claims the file.
        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )
        marker = self._marker(settings)
        marker["pid"] = 424242  # pretend a different process owns it
        settings.parent.joinpath(".headroom_wrap_marker.json").write_text(json.dumps(marker))
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _m: False)

        # Run B now writes; it should inherit A's recorded original.
        previous_for_b = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8789", settings_path=settings, port=8789
        )

        assert previous_for_b == "https://upstream"

    def test_restore_defers_to_a_live_peer_owner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exiting run A must not clobber live run B's URL or marker."""
        settings = self._settings(tmp_path)
        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8789", settings_path=settings, port=8789
        )
        marker = self._marker(settings)
        marker["pid"] = 424242  # run B owns the marker now
        settings.parent.joinpath(".headroom_wrap_marker.json").write_text(json.dumps(marker))
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _m: False)

        wrap_mod._restore_claude_wrap_base_url(None, settings_path=settings)

        # B's URL and marker survive A's exit.
        payload = json.loads(settings.read_text())
        assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8789"
        assert settings.parent.joinpath(".headroom_wrap_marker.json").exists()

    def test_restore_proceeds_when_we_own_the_marker(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path)
        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )

        wrap_mod._restore_claude_wrap_base_url(None, settings_path=settings)

        payload = json.loads(settings.read_text()) if settings.exists() else {}
        assert "ANTHROPIC_BASE_URL" not in payload.get("env", {})
        assert not settings.parent.joinpath(".headroom_wrap_marker.json").exists()

    def test_force_overrides_the_ownership_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """unwrap / stale-cleanup must still be able to restore."""
        settings = self._settings(tmp_path)
        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8789", settings_path=settings, port=8789
        )
        marker = self._marker(settings)
        marker["pid"] = 424242
        settings.parent.joinpath(".headroom_wrap_marker.json").write_text(json.dumps(marker))
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _m: False)

        wrap_mod._restore_claude_wrap_base_url(None, settings_path=settings, _force=True)

        payload = json.loads(settings.read_text()) if settings.exists() else {}
        assert "ANTHROPIC_BASE_URL" not in payload.get("env", {})


class TestIsolateOmpAgentDir:
    """Concurrent isolated OMP runs must not fight over ~/.omp/agent/models.yml."""

    def test_noop_without_isolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
        assert wrap_mod._isolate_omp_agent_dir() is None
        assert "PI_CODING_AGENT_DIR" not in os.environ

    def test_isolated_run_gets_its_own_agent_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
        run_dir = isolation.activate_isolated_workspace()

        target = wrap_mod._isolate_omp_agent_dir()

        assert target == run_dir / "omp-agent"
        assert target.is_dir()
        assert os.environ["PI_CODING_AGENT_DIR"] == str(target)
        # models.yml now resolves inside the run dir, so a concurrent run's
        # rewrite cannot touch this run's endpoint.
        from headroom.providers.omp import models_yml_path

        assert models_yml_path() == target / "models.yml"

    def test_seeds_from_the_users_existing_agent_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The copy preserves omp's bundled catalog and stored credentials."""
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
        fake_home = tmp_path / "home"
        source = fake_home / ".omp" / "agent"
        source.mkdir(parents=True)
        (source / "models.yml").write_text("providers: {anthropic: {}}\n")
        (source / "credentials.json").write_text("{}")
        monkeypatch.setattr(wrap_mod.Path, "home", classmethod(lambda _cls: fake_home))
        isolation.activate_isolated_workspace()

        target = wrap_mod._isolate_omp_agent_dir()

        assert target is not None
        assert (target / "models.yml").read_text() == "providers: {anthropic: {}}\n"
        assert (target / "credentials.json").exists()

    def test_explicit_user_agent_dir_wins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "mine"))
        isolation.activate_isolated_workspace()

        assert wrap_mod._isolate_omp_agent_dir() is None
        assert os.environ["PI_CODING_AGENT_DIR"] == str(tmp_path / "mine")


class TestCopilotBackendProbeGate:
    """`wrap copilot` must only inherit a running proxy's backend when it will
    actually REUSE that proxy. An isolated run (the default) starts its own
    dedicated proxy with the requested/env backend (PR #25 review P1)."""

    def _run(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str], running_backend: str = "anyllm"
    ) -> list[Any]:
        seen: list[Any] = []

        class _Stop(Exception):
            pass

        monkeypatch.setattr(wrap_mod.shutil, "which", lambda _n: "/usr/bin/copilot")
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_detect_running_proxy_backend", lambda _p: running_backend)

        def _validate(**kwargs: Any) -> None:
            seen.append(kwargs.get("backend"))
            raise _Stop()

        monkeypatch.setattr(wrap_mod, "_validate_copilot_configuration", _validate)

        runner = CliRunner()
        runner.invoke(main, argv, catch_exceptions=True)
        return seen

    def test_isolated_run_does_not_inherit_shared_proxy_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._run(monkeypatch, ["wrap", "--isolated", "copilot"])

        # An unrelated anyllm proxy on 8787 must not become our backend: this
        # run gets its own dedicated proxy on the default (Anthropic) backend.
        assert seen == [None]

    def test_shared_run_still_inherits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._run(monkeypatch, ["wrap", "--shared", "copilot"])

        # Shared mode genuinely reuses the running proxy, so inheriting is right.
        assert seen == ["anyllm"]

    def test_no_proxy_inherits_even_when_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._run(monkeypatch, ["wrap", "--isolated", "copilot", "--no-proxy"])

        # --no-proxy explicitly attaches to the running proxy.
        assert seen == ["anyllm"]


class TestPrepareOnlyIsExemptFromIsolation:
    """`--prepare-only` is a machine-readable config step whose stdout is piped
    into `openclaw config set --strict-json` by scripts/install.sh. It must not
    be isolated (no run dir to GC) and must not have its stdout polluted
    (PR #25 review round 3, P1)."""

    def test_detects_prepare_only_from_argv(self) -> None:
        ctx = click.Context(click.Command("wrap"))
        assert (
            wrap_mod._prepare_only_invocation(
                ctx, argv=["headroom", "wrap", "openclaw", "--prepare-only"]
            )
            is True
        )
        assert wrap_mod._prepare_only_invocation(ctx, argv=["headroom", "wrap", "claude"]) is False

    def test_openclaw_prepare_only_emits_pure_json_and_no_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The real default: isolation ON (the suite-wide conftest pins it off).
        monkeypatch.delenv(isolation.HEADROOM_ISOLATED_ENV, raising=False)
        # Mirror the argv scripts/install.sh actually produces.
        monkeypatch.setattr(
            wrap_mod.sys, "argv", ["headroom", "wrap", "openclaw", "--prepare-only"]
        )

        runner = CliRunner()
        result = runner.invoke(main, ["wrap", "openclaw", "--prepare-only"])

        assert result.exit_code == 0, result.output
        # stdout must parse as JSON with nothing prepended.
        payload = json.loads(result.stdout)
        assert "config" in payload
        # ...and no ephemeral run directory was created for it.
        assert not (tmp_path / "ws" / "runs").exists()
        assert isolation.HEADROOM_ISOLATED_ENV not in os.environ


class TestLaunchToolSighup:
    def test_launch_tool_registers_sighup(self) -> None:
        """Closing the terminal sends SIGHUP; without a handler the wrapper
        dies and its detached dedicated proxy survives forever on the per-run
        port (round 3, P2). claude() already does this."""
        import inspect

        src = inspect.getsource(wrap_mod._launch_tool)
        assert 'hasattr(signal, "SIGHUP")' in src
        assert "signal.signal(signal.SIGHUP, cleanup)" in src


class TestIsolateAgentConfigHomes:
    """Codex/Grok/OMP keep their endpoint + Headroom MCP entry in one shared
    config file their processes re-read, so concurrent isolated runs need
    private copies rather than a post-hoc rewrite (round 3, P1)."""

    def _fake_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        home = tmp_path / "home"
        monkeypatch.setattr(wrap_mod.Path, "home", classmethod(lambda _cls: home))
        return home

    @pytest.mark.parametrize(
        "helper,env_var,rel,dir_name",
        [
            ("_isolate_codex_home", "CODEX_HOME", (".codex",), "codex-home"),
            ("_isolate_grok_home", "GROK_HOME", (".grok",), "grok-home"),
            ("_isolate_omp_agent_dir", "PI_CODING_AGENT_DIR", (".omp", "agent"), "omp-agent"),
        ],
    )
    def test_isolated_run_gets_private_config_home(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        helper: str,
        env_var: str,
        rel: tuple[str, ...],
        dir_name: str,
    ) -> None:
        monkeypatch.delenv(env_var, raising=False)
        home = self._fake_home(monkeypatch, tmp_path)
        source = home.joinpath(*rel)
        source.mkdir(parents=True)
        (source / "config.toml").write_text("user = true\n")
        run_dir = isolation.activate_isolated_workspace()

        target = getattr(wrap_mod, helper)()

        assert target == run_dir / dir_name
        assert os.environ[env_var] == str(target)
        # Seeded from the user's own config so catalog/credentials survive.
        assert (target / "config.toml").read_text() == "user = true\n"

    @pytest.mark.parametrize(
        "helper,env_var",
        [
            ("_isolate_codex_home", "CODEX_HOME"),
            ("_isolate_grok_home", "GROK_HOME"),
            ("_isolate_omp_agent_dir", "PI_CODING_AGENT_DIR"),
        ],
    )
    def test_noop_without_isolation(
        self, monkeypatch: pytest.MonkeyPatch, helper: str, env_var: str
    ) -> None:
        monkeypatch.delenv(env_var, raising=False)
        assert getattr(wrap_mod, helper)() is None
        assert env_var not in os.environ

    @pytest.mark.parametrize(
        "helper,env_var",
        [
            ("_isolate_codex_home", "CODEX_HOME"),
            ("_isolate_grok_home", "GROK_HOME"),
        ],
    )
    def test_explicit_user_value_is_respected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, helper: str, env_var: str
    ) -> None:
        monkeypatch.setenv(env_var, str(tmp_path / "mine"))
        isolation.activate_isolated_workspace()

        assert getattr(wrap_mod, helper)() is None
        assert os.environ[env_var] == str(tmp_path / "mine")

    def test_two_isolated_runs_get_distinct_codex_configs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CODEX_HOME", raising=False)
        home = self._fake_home(monkeypatch, tmp_path)
        (home / ".codex").mkdir(parents=True)

        isolation.activate_isolated_workspace(run_id="a")
        first = wrap_mod._isolate_codex_home()

        # A second process inherits none of the first run's markers.
        for var in (
            isolation.HEADROOM_ISOLATED_WORKSPACE_ENV,
            isolation.HEADROOM_MEMORY_DB_PATH_ENV,
            paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV,
            paths.HEADROOM_SETTINGS_PATH_ENV,
            "CODEX_HOME",
        ):
            os.environ.pop(var, None)
        os.environ[paths.HEADROOM_WORKSPACE_DIR_ENV] = str(tmp_path / "ws")
        isolation.activate_isolated_workspace(run_id="b")
        second = wrap_mod._isolate_codex_home()

        assert first is not None and second is not None
        assert first != second

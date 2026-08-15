"""Tests for `headroom wrap --isolated` (per-run proxy + workspace isolation).

Without isolation, a second concurrent wrap reuses the proxy already
listening on the shared port and writes into the shared workspace. These
tests pin the isolated behavior: a dedicated proxy on a fresh port, the
shared port left alone, and a per-run workspace activated by the group
callback.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal as signal_mod
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from headroom import _filelock, isolation, paths
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
        # Agent config homes. Isolation no longer relocates these (see the
        # revert in the isolation docs), but keep them scrubbed so a stray
        # value in the developer's shell cannot steer these tests.
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
    # `_ensure_proxy` records the started proxy's PID into the run dir so GC
    # never deletes a live detached proxy's workspace, so the stub needs one.
    pid = 4242

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
    def test_misplaced_group_flag_is_rejected(
        self, flag: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`headroom wrap claude --isolated/--shared` must fail loudly
        instead of forwarding the flag to the wrapped CLI
        (ignore_unknown_options)."""

        # Detection reads the real argv so a flag forwarded after `--` can be
        # told apart from a misplaced one; mirror the true invocation.
        monkeypatch.setattr(wrap_mod.sys, "argv", ["headroom", "wrap", "claude", flag])
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
        # The marker is an owner STACK; re-own run A's entry to another pid.
        marker["owners"][-1]["pid"] = 424242
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
        marker["owners"][-1]["pid"] = 424242  # run B owns the stack entry now
        settings.parent.joinpath(".headroom_wrap_marker.json").write_text(json.dumps(marker))
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _m: False)
        # Handover now requires the peer's proxy to still answer; these
        # ports are fixtures with nothing listening.
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda _port, **_k: True)

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

    @pytest.mark.parametrize(
        "argv,expected,why",
        [
            (["headroom", "wrap", "openclaw", "--prepare-only"], True, "Headroom's own flag"),
            (["headroom", "wrap", "claude"], False, "flag absent"),
            (
                ["headroom", "wrap", "codex", "--prepare-only", "--", "--foo"],
                True,
                "own flag, before the delimiter",
            ),
            # Everything after `--` is forwarded verbatim to the wrapped CLI,
            # so a match there is the CHILD's flag. Treating it as Headroom's
            # would skip isolation for an ordinary launch and silently put it
            # back on the shared workspace + proxy (round 7, P2).
            (
                ["headroom", "wrap", "codex", "--", "--prepare-only"],
                False,
                "forwarded to the child after --",
            ),
            (
                ["headroom", "wrap", "claude", "--", "-p", "--prepare-only"],
                False,
                "deep inside child args",
            ),
        ],
    )
    def test_detects_only_headrooms_own_prepare_only(
        self, argv: list[str], expected: bool, why: str
    ) -> None:
        ctx = click.Context(click.Command("wrap"))
        assert wrap_mod._prepare_only_invocation(ctx, argv=argv) is expected, why

    def test_child_forwarded_flag_does_not_disable_isolation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End-to-end: `wrap <tool> -- --prepare-only` must still isolate."""
        monkeypatch.delenv(isolation.HEADROOM_ISOLATED_ENV, raising=False)
        monkeypatch.setattr(
            wrap_mod.sys, "argv", ["headroom", "wrap", "fake-tool", "--", "--prepare-only"]
        )

        result, seen = _invoke_with_fake_tool(["wrap", "fake-tool"])

        assert result.exit_code == 0, result.output
        assert seen["workspace"].parent == tmp_path / "ws" / "runs"

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
    def test_launch_tool_registers_a_sighup_handler_that_exits(self) -> None:
        """Closing the terminal sends SIGHUP; without a handler the wrapper
        dies and its detached dedicated proxy survives forever on the per-run
        port (round 3, P2). claude() already does this.

        Registering `cleanup` alone is NOT enough (round 19, P2): it returns,
        and during proxy startup `proxy_holder[0]` is still None, so it does
        nothing and the wrapper carries on to launch the proxy and the agent
        after the terminal has closed. The handler must raise.
        """
        import inspect

        src = inspect.getsource(wrap_mod._launch_tool)
        assert 'hasattr(signal, "SIGHUP")' in src
        assert "signal.signal(signal.SIGHUP, _hangup)" in src
        assert "raise SystemExit(0)" in src

    def test_claude_sighup_handler_also_exits(self) -> None:
        """`claude()` has its own copy of the pattern and the same startup
        window, so it needs the same raising handler."""
        import inspect

        src = inspect.getsource(wrap_mod.claude.callback)
        assert "signal.signal(signal.SIGHUP, _claude_hangup)" in src
        assert "raise SystemExit(0)" in src

    def test_the_sighup_handler_cleans_up_then_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Behavioural check on the handler `_launch_tool` installs."""
        installed: dict[str, Any] = {}
        cleaned: list[bool] = []

        def capture(sig: Any, handler: Any) -> Any:
            if sig == getattr(signal_mod, "SIGHUP", None):
                installed["handler"] = handler
            return None

        monkeypatch.setattr(wrap_mod.signal, "signal", capture)
        monkeypatch.setattr(
            wrap_mod, "_make_cleanup", lambda *_a, **_k: lambda *_x: cleaned.append(True)
        )
        monkeypatch.setattr(wrap_mod, "_reject_misplaced_isolated_flag", lambda *_a, **_k: None)
        monkeypatch.setattr(
            wrap_mod,
            "_ensure_proxy",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stop here")),
        )

        # `_launch_tool` converts a proxy failure into its own exit; all this
        # test needs is for the handler to have been installed before then.
        with contextlib.suppress(BaseException):
            wrap_mod._launch_tool("true", (), {}, 8787, False, "tool", [])

        handler = installed.get("handler")
        assert handler is not None, "no SIGHUP handler was installed"
        cleaned.clear()  # `_launch_tool`'s own error path already ran cleanup

        with pytest.raises(SystemExit):
            handler(1, None)

        assert cleaned == [True], "the handler exited without cleaning up"


class TestMcpRegistrationForcesActualPort:
    """A dedicated proxy binds a port that differs from any previously
    registered `headroom` MCP entry. Without force the registrar reports
    MISMATCH and leaves the STALE endpoint, so retrieval would hit the shared
    proxy, another run's workspace, or a dead port (round 6, P1)."""

    def test_every_registrar_call_site_forces(self) -> None:
        import inspect

        src = inspect.getsource(wrap_mod)
        calls = [
            line.strip()
            for line in src.splitlines()
            if "_setup_headroom_mcp(" in line and "def _setup_headroom_mcp" not in line
        ]
        assert calls, "expected to find _setup_headroom_mcp call sites"
        unforced = [c for c in calls if "force=True" not in c]
        assert not unforced, f"registration without force=True: {unforced}"

    def test_claude_registration_is_forced(self) -> None:
        import inspect

        src = inspect.getsource(wrap_mod.claude.callback)
        assert "_setup_headroom_mcp(ClaudeRegistrar(), actual_port" in src
        assert "force=True" in src


class TestProxyOnlyWatcherSighup:
    """cursor / grok-build / cline / zcode / continue start a DETACHED
    dedicated proxy and then just wait. Closing the terminal sends SIGHUP, so
    without a handler the watcher exits and the proxy lives on forever on the
    per-run port (round 6, P2)."""

    def test_watcher_registers_sighup_and_exits(self) -> None:
        import inspect

        src = inspect.getsource(wrap_mod._run_proxy_only_watcher)
        assert 'hasattr(signal, "SIGHUP")' in src
        assert "signal.signal(signal.SIGHUP" in src
        # Cleanup alone is not enough — it must not fall back into the loop.
        assert "SystemExit" in src


class TestWrapMarkerOwnerStack:
    """Concurrent Claude wraps in one project form an owner STACK. Whichever
    run exits, routing must be handed to a surviving live owner rather than
    reset to the pre-Headroom value (round 8, P1 — reverse exit order)."""

    def _project(self, tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user-gateway"}}))
        return settings, settings.parent / ".headroom_wrap_marker.json"

    def _url(self, settings: Path) -> str | None:
        if not settings.exists():
            return None
        return json.loads(settings.read_text())["env"].get("ANTHROPIC_BASE_URL")

    def _two_live_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> tuple[Path, Path, str | None, str | None]:
        settings, marker = self._project(tmp_path)
        # Handover now requires the peer's proxy to still answer; these
        # ports are fixtures with nothing listening.
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda _port, **_k: True)
        prev_a = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )
        payload = json.loads(marker.read_text())
        payload["owners"][-1]["pid"] = 111  # run A is a different live process
        marker.write_text(json.dumps(payload))
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)
        prev_b = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8789", settings_path=settings, port=8789
        )
        assert self._url(settings) == "http://127.0.0.1:8789"
        return settings, marker, prev_a, prev_b

    def test_newer_run_exiting_hands_back_to_live_older_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The case the single-marker design could not express."""
        settings, marker, _prev_a, prev_b = self._two_live_runs(monkeypatch, tmp_path)

        wrap_mod._restore_claude_wrap_base_url(prev_b, settings_path=settings)

        # Routing returns to run A's proxy — NOT the user's original value,
        # which would make A's daemon workers bypass Headroom entirely.
        assert self._url(settings) == "http://127.0.0.1:8788"
        assert marker.exists(), "run A must still own the marker"

    def test_older_run_exiting_leaves_newer_owner_untouched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker, prev_a, _prev_b = self._two_live_runs(monkeypatch, tmp_path)
        # Make THIS process look like run A so its exit is the one under test.
        payload = json.loads(marker.read_text())
        for owner in payload["owners"]:
            owner["pid"] = 222 if owner["pid"] == os.getpid() else os.getpid()
        marker.write_text(json.dumps(payload))

        wrap_mod._restore_claude_wrap_base_url(prev_a, settings_path=settings)

        assert self._url(settings) == "http://127.0.0.1:8789"
        assert marker.exists()

    def test_last_owner_restores_the_original(self, tmp_path: Path) -> None:
        settings, marker = self._project(tmp_path)
        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )

        wrap_mod._restore_claude_wrap_base_url(previous, settings_path=settings)

        assert self._url(settings) == "https://user-gateway"
        assert not marker.exists()

    def test_legacy_single_dict_marker_still_reads(self, tmp_path: Path) -> None:
        """A marker written by an older Headroom has no `owners` list."""
        settings, marker = self._project(tmp_path)
        marker.write_text(
            json.dumps(
                {"pid": os.getpid(), "port": 8788, "key": "ANTHROPIC_BASE_URL", "previous": None}
            )
        )

        owners = wrap_mod._wrap_marker_owners(settings)

        assert len(owners) == 1
        assert owners[0]["port"] == 8788

    def test_dead_owners_are_dropped(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings, marker = self._project(tmp_path)
        marker.write_text(
            json.dumps(
                {
                    "owners": [
                        {"pid": 2147480000, "port": 8788, "key": "ANTHROPIC_BASE_URL"},
                        {"pid": os.getpid(), "port": 8789, "key": "ANTHROPIC_BASE_URL"},
                    ]
                }
            )
        )

        owners = wrap_mod._wrap_marker_owners(settings)

        assert [o["port"] for o in owners] == [8789]


class TestMisplacedFlagRespectsDelimiter:
    """`--isolated`/`--shared` after `--` belong to the wrapped CLI and must
    pass through verbatim, not raise a usage error (round 8, P2)."""

    @pytest.mark.parametrize(
        "argv,should_raise,why",
        [
            (["headroom", "wrap", "claude", "--isolated"], True, "misplaced after tool"),
            (["headroom", "wrap", "claude", "--shared"], True, "misplaced after tool"),
            (["headroom", "wrap", "--isolated", "claude"], False, "correct group position"),
            (["headroom", "wrap", "--shared", "claude"], False, "correct group position"),
            (["headroom", "wrap", "claude", "--", "--shared"], False, "forwarded to child"),
            (
                ["headroom", "wrap", "claude", "--", "-p", "--isolated"],
                False,
                "deep in child args",
            ),
            (["headroom", "wrap", "claude"], False, "absent"),
        ],
    )
    def test_only_pre_delimiter_flags_are_rejected(
        self, argv: list[str], should_raise: bool, why: str
    ) -> None:
        if should_raise:
            with pytest.raises(click.UsageError):
                wrap_mod._reject_misplaced_isolated_flag((), "claude", argv=argv)
        else:
            wrap_mod._reject_misplaced_isolated_flag((), "claude", argv=argv)


class TestCrashedOwnerHandsBackToLivePeer:
    """Crash cleanup must remove only the DEAD owner. Wiping the whole marker
    strands a still-live peer outside Headroom (round 9, P1)."""

    def _project(self, tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user-gateway"}}))
        return settings, settings.parent / ".headroom_wrap_marker.json"

    def _url(self, settings: Path) -> str | None:
        if not settings.exists():
            return None
        return json.loads(settings.read_text())["env"].get("ANTHROPIC_BASE_URL")

    def _stack(self, marker: Path, *owners: dict[str, Any]) -> None:
        marker.write_text(json.dumps({**owners[-1], "owners": list(owners)}))

    def test_stale_cleanup_hands_back_to_live_owner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker = self._project(tmp_path)
        live = {
            "pid": os.getpid(),  # run A: alive
            "port": 8788,
            "key": "ANTHROPIC_BASE_URL",
            "url": "http://127.0.0.1:8788",
            "previous": "https://user-gateway",
        }
        dead = {
            "pid": 2147480000,  # run B: killed without cleanup
            "port": 8789,
            "key": "ANTHROPIC_BASE_URL",
            "url": "http://127.0.0.1:8789",
            "previous": "https://user-gateway",
        }
        self._stack(marker, live, dead)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8789"}}))

        restored = wrap_mod._check_and_clear_stale_wrap_marker(settings, key="ANTHROPIC_BASE_URL")

        # Routing goes to the surviving run A, NOT the user's original.
        assert restored == "http://127.0.0.1:8788"
        assert self._url(settings) == "http://127.0.0.1:8788"
        assert marker.exists(), "the live owner must keep the marker"
        owners = json.loads(marker.read_text())["owners"]
        assert [o["pid"] for o in owners] == [os.getpid()]

    def test_stale_cleanup_restores_original_when_none_survive(self, tmp_path: Path) -> None:
        settings, marker = self._project(tmp_path)
        self._stack(
            marker,
            {
                "pid": 2147480000,
                "port": 8788,
                "key": "ANTHROPIC_BASE_URL",
                "url": "http://127.0.0.1:8788",
                "previous": "https://user-gateway",
            },
        )
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788"}}))

        restored = wrap_mod._check_and_clear_stale_wrap_marker(settings, key="ANTHROPIC_BASE_URL")

        assert restored == "https://user-gateway"
        assert self._url(settings) == "https://user-gateway"
        assert not marker.exists()

    def test_dead_proxy_cleanup_requires_a_live_port(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """#2221: a peer whose PID looks alive but whose port is dead (PID
        reuse after reboot) must not be handed the entry."""
        settings, marker = self._project(tmp_path)
        self._stack(
            marker,
            {
                "pid": os.getpid(),
                "port": 8788,
                "key": "ANTHROPIC_BASE_URL",
                "url": "http://127.0.0.1:8788",
                "previous": "https://user-gateway",
            },
        )
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788"}}))
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda _p, **_k: False)

        restored = wrap_mod._check_and_clear_dead_wrap_marker(settings, key="ANTHROPIC_BASE_URL")

        assert restored == "https://user-gateway"
        assert not marker.exists()


class TestMemoryCliHonoursIsolatedDb:
    def test_override_wins_over_project_local(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A nested `headroom memory ...` inside an isolated run must read the
        run's DB, not the project-local legacy one (round 9, P2)."""
        from headroom.cli import memory as memory_cli

        project = tmp_path / "proj"
        (project / ".headroom").mkdir(parents=True)
        (project / ".headroom" / "memory.db").write_text("")
        monkeypatch.chdir(project)
        run_db = tmp_path / "ws" / "runs" / "run-x" / "memory.db"
        monkeypatch.setenv("HEADROOM_MEMORY_DB_PATH", str(run_db))

        assert memory_cli._default_db_path() == str(run_db)

    def test_project_local_still_wins_without_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from headroom.cli import memory as memory_cli

        project = tmp_path / "proj"
        (project / ".headroom").mkdir(parents=True)
        (project / ".headroom" / "memory.db").write_text("")
        monkeypatch.chdir(project)
        monkeypatch.delenv("HEADROOM_MEMORY_DB_PATH", raising=False)

        assert memory_cli._default_db_path() == str(project / ".headroom" / "memory.db")


class TestWrapMarkerLocking:
    """The owner stack is a read-modify-write on a file shared by every
    concurrent wrap in a project (round 10, P1).

    Unserialized, two sessions read the same pre-image, each append
    themselves, and the later write drops the other LIVE owner — whose exit
    then restores the pre-Headroom URL out from under a running peer. The
    whole cycle must run inside `_wrap_marker_lock`.
    """

    @staticmethod
    def _project(tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user-gateway"}}))
        return settings, wrap_mod._wrap_marker_path(settings)

    @staticmethod
    def _peer_can_lock(settings: Path) -> bool:
        """Whether an *independent* handle can take the lock right now.

        flock is held per open file description, so a fresh handle conflicts
        with a held lock even inside this same process — which makes the
        critical section observable without spawning anything.
        """
        path = wrap_mod._wrap_marker_lock_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+", encoding="utf-8") as handle:
            if not _filelock.acquire(handle, timeout=0):
                return False
            _filelock.release(handle)
            return True

    def test_lock_is_observable_when_held(self, tmp_path: Path) -> None:
        settings, _marker = self._project(tmp_path)

        assert self._peer_can_lock(settings) is True
        with wrap_mod._wrap_marker_lock(settings):
            assert self._peer_can_lock(settings) is False
        assert self._peer_can_lock(settings) is True

    def test_push_holds_the_lock_across_read_and_write(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, _marker = self._project(tmp_path)
        observed: list[bool] = []
        # The push reads the PERSISTED stack (so a crashed other-key owner is
        # not dropped), so that is where the critical section is observed.
        real = wrap_mod._raw_wrap_marker_owners

        def probing_owners(path: Path) -> Any:
            observed.append(self._peer_can_lock(path))
            return real(path)

        monkeypatch.setattr(wrap_mod, "_raw_wrap_marker_owners", probing_owners)
        wrap_mod._write_wrap_marker(settings, port=8788, key="ANTHROPIC_BASE_URL", previous=None)

        assert observed and not any(observed), "the stack read must already be inside the lock"

    def test_restore_holds_the_lock_across_read_and_write(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, _marker = self._project(tmp_path)
        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )
        observed: list[bool] = []
        real = wrap_mod._wrap_marker_owners

        def probing_owners(path: Path) -> Any:
            observed.append(self._peer_can_lock(path))
            return real(path)

        monkeypatch.setattr(wrap_mod, "_wrap_marker_owners", probing_owners)
        wrap_mod._restore_claude_wrap_base_url(previous, settings_path=settings)

        assert observed and not any(observed)

    def test_write_base_url_holds_the_lock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The peer-marker read, the settings rewrite and the stack push are
        one atomic step — a peer must not observe the file mid-update."""
        settings, _marker = self._project(tmp_path)
        observed: list[bool] = []
        real = wrap_mod._read_wrap_marker

        def probing_read(path: Path) -> Any:
            observed.append(self._peer_can_lock(path))
            return real(path)

        monkeypatch.setattr(wrap_mod, "_read_wrap_marker", probing_read)
        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )

        assert observed and not any(observed)

    def test_lock_is_reentrant_within_one_process(self, tmp_path: Path) -> None:
        """`_handover_to_live_owner` calls back into the restorer, which locks
        again. flock would deadlock on a second handle, so the guard must
        short-circuit a nested acquisition."""
        settings, _marker = self._project(tmp_path)

        with wrap_mod._wrap_marker_lock(settings):
            with wrap_mod._wrap_marker_lock(settings, timeout=0):
                assert _filelock._held[str(wrap_mod._wrap_marker_lock_path(settings))] == 1

        assert not _filelock._held

    def test_handover_through_the_restorer_does_not_deadlock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The real nested path: stale cleanup -> handover -> forced restore.
        Each frame locks; a non-reentrant lock would hang here."""
        settings, marker = self._project(tmp_path)
        marker.write_text(
            json.dumps(
                {
                    "owners": [
                        {
                            "pid": os.getpid(),
                            "port": 8788,
                            "key": "ANTHROPIC_BASE_URL",
                            "previous": "https://user-gateway",
                            "url": "http://127.0.0.1:8788",
                        },
                        {
                            "pid": 2147480000,  # crashed peer
                            "port": 8789,
                            "key": "ANTHROPIC_BASE_URL",
                            "previous": "https://user-gateway",
                            "url": "http://127.0.0.1:8789",
                        },
                    ],
                    "pid": 2147480000,
                    "port": 8789,
                    "key": "ANTHROPIC_BASE_URL",
                    "previous": "https://user-gateway",
                }
            )
        )

        handover = wrap_mod._check_and_clear_stale_wrap_marker(settings, key="ANTHROPIC_BASE_URL")

        assert handover == "http://127.0.0.1:8788"
        assert not _filelock._held

    def test_body_still_runs_when_the_lock_cannot_be_taken(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Marker bookkeeping is best-effort: a wedged holder (or an
        unwritable lock path) must never stop a wrap from launching."""
        settings, marker = self._project(tmp_path)
        monkeypatch.setattr(_filelock, "acquire", lambda *a, **k: False)

        wrap_mod._write_wrap_marker(settings, port=8788, key="ANTHROPIC_BASE_URL", previous=None)

        assert json.loads(marker.read_text())["owners"][-1]["port"] == 8788
        assert not _filelock._held

    def test_lock_open_failure_is_survivable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker = self._project(tmp_path)

        def boom(*_a: Any, **_k: Any) -> Any:
            raise OSError("no lock file for you")

        monkeypatch.setattr("builtins.open", boom)
        with wrap_mod._wrap_marker_lock(settings):
            pass

        assert not _filelock._held

    def test_concurrent_pushes_keep_every_live_owner(self, tmp_path: Path) -> None:
        """End-to-end proof across real processes: two wraps racing to push
        must both end up on the stack. Without the lock the later writer's
        pre-image is empty and it silently drops the first owner."""
        import subprocess
        import sys
        import time

        settings, marker = self._project(tmp_path)
        child = tmp_path / "push.py"
        child.write_text(
            "import sys, time\n"
            "from pathlib import Path\n"
            "from headroom.cli import wrap\n"
            "settings, port, start_at = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])\n"
            "real = wrap._wrap_marker_owners\n"
            "def slow(path):\n"
            "    owners = real(path)\n"
            "    time.sleep(0.5)  # widen the read->write window\n"
            "    return owners\n"
            "wrap._wrap_marker_owners = slow\n"
            "while time.time() < start_at:\n"
            "    time.sleep(0.005)\n"
            "wrap._write_wrap_marker("
            "settings, port=port, key='ANTHROPIC_BASE_URL', previous=None)\n"
            # Stay alive: a peer drops owners whose PID is gone, so exiting
            # here would make the drop legitimate rather than a lost write.
            "time.sleep(2.5)\n"
        )
        # Absolute start times absorb interpreter/import startup cost, so the
        # interleaving is decided by the delays above, not by process spawn
        # latency.
        base = time.time() + 6.0
        procs = [
            subprocess.Popen(
                [sys.executable, str(child), str(settings), str(port), str(base + offset)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for port, offset in ((8788, 0.0), (8789, 0.1))
        ]
        for proc in procs:
            _out, err = proc.communicate(timeout=120)
            assert proc.returncode == 0, err.decode()

        recorded = json.loads(marker.read_text())["owners"]
        assert sorted(o["port"] for o in recorded) == [8788, 8789]


class TestNestedWrapDoesNotChainProxies:
    """Isolation puts every run on its own port, so a nested `wrap claude`
    inherits the PARENT wrap's ANTHROPIC_BASE_URL on a DIFFERENT port. The
    equal-port guard alone misses that, and the inherited URL reads as a user
    gateway — chaining two Headroom pipelines (round 10, P2).
    """

    def test_inherited_parent_proxy_is_not_treated_as_a_gateway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")
        monkeypatch.setattr(
            wrap_mod, "_query_proxy_health", lambda _p: {"service": "headroom-proxy"}
        )

        assert wrap_mod._detect_inbound_anthropic_upstream(8788) is None

    def test_own_port_is_still_ignored_without_probing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8788")

        def unexpected(_port: int) -> Any:
            raise AssertionError("self-referential URL must short-circuit before probing")

        monkeypatch.setattr(wrap_mod, "_query_proxy_health", unexpected)

        assert wrap_mod._detect_inbound_anthropic_upstream(8788) is None

    def test_local_user_gateway_is_still_inherited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A LiteLLM on localhost is NOT a Headroom proxy — issue #1353 must
        keep working."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:4000")
        monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _p: {"service": "litellm"})

        assert wrap_mod._detect_inbound_anthropic_upstream(8788) == "http://localhost:4000"

    def test_unreachable_local_port_is_still_inherited(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing listening yet (gateway starts later) must not silently drop
        the user's configured upstream."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:4000")
        monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _p: None)

        assert wrap_mod._detect_inbound_anthropic_upstream(8788) == "http://127.0.0.1:4000"

    def test_remote_gateway_is_never_probed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com")

        def unexpected(_port: int) -> Any:
            raise AssertionError("remote hosts must not be probed")

        monkeypatch.setattr(wrap_mod, "_query_proxy_health", unexpected)

        assert wrap_mod._detect_inbound_anthropic_upstream(8788) == "https://gateway.example.com"

    def test_portless_local_url_is_not_probed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No explicit port means a default 80/443 listener — never a Headroom
        proxy, and probing it would be a surprise connection."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost/v1")

        def unexpected(_port: int) -> Any:
            raise AssertionError("portless URLs must not be probed")

        monkeypatch.setattr(wrap_mod, "_query_proxy_health", unexpected)

        assert wrap_mod._detect_inbound_anthropic_upstream(8788) == "http://localhost/v1"

    def test_is_local_headroom_proxy_signature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            wrap_mod, "_query_proxy_health", lambda _p: {"service": "headroom-proxy"}
        )
        assert wrap_mod._is_local_headroom_proxy(8787) is True

        monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _p: {})
        assert wrap_mod._is_local_headroom_proxy(8787) is False

        monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _p: None)
        assert wrap_mod._is_local_headroom_proxy(8787) is False


class TestDedicatedProxyPidIsRecorded:
    """`_ensure_proxy` must pin the run dir to the proxy it starts, so GC
    cannot delete a live detached proxy's workspace (round 10, P2)."""

    def test_ensure_proxy_records_the_started_proxy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_dir = isolation.activate_isolated_workspace()

        class _Proc:
            pid = 4242

        monkeypatch.setattr(wrap_mod, "_find_available_port", lambda *a, **k: 8788)
        monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: False)
        monkeypatch.setattr(wrap_mod, "_start_proxy", lambda *a, **k: _Proc())

        proc, port = wrap_mod._ensure_proxy(8787, False)

        assert port == 8788 and proc is not None
        assert isolation._run_dir_proxy_pid(run_dir) == 4242


class TestMarkerLockLivesOutsideTheProject:
    """Nothing ever deletes a lock file, so keeping it beside the marker would
    leave a permanent untracked artifact in every wrapped repo (round 11, P2).
    """

    def test_lock_is_not_written_into_the_project(self, tmp_path: Path) -> None:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {}}))

        with wrap_mod._wrap_marker_lock(settings):
            pass

        assert wrap_mod._wrap_marker_lock_path(settings).exists()
        assert list(settings.parent.iterdir()) == [settings]

    def test_lock_resolves_under_the_shared_root(self, tmp_path: Path) -> None:
        """Per-run roots differ between concurrent isolated wraps, so a per-run
        lock would hand every racer a private file and serialize nothing."""
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        shared = paths.shared_workspace_dir()

        before = wrap_mod._wrap_marker_lock_path(settings)
        assert before.parent == shared / "locks"

        isolation.activate_isolated_workspace()
        assert paths.workspace_dir() != shared

        assert wrap_mod._wrap_marker_lock_path(settings) == before

    def test_distinct_projects_get_distinct_locks(self, tmp_path: Path) -> None:
        a = tmp_path / "a" / ".claude" / "settings.local.json"
        b = tmp_path / "b" / ".claude" / "settings.local.json"

        assert wrap_mod._wrap_marker_lock_path(a) != wrap_mod._wrap_marker_lock_path(b)

    def test_same_project_gets_the_same_lock_via_different_paths(self, tmp_path: Path) -> None:
        """Two wraps naming one project differently (symlink, `..`) must still
        contend on the same lock, or the serialization is vacuous."""
        real = tmp_path / "proj" / ".claude"
        real.mkdir(parents=True)
        direct = real / "settings.local.json"
        indirect = tmp_path / "proj" / "sub" / ".." / ".claude" / "settings.local.json"
        (tmp_path / "proj" / "sub").mkdir()

        assert wrap_mod._wrap_marker_lock_path(direct) == wrap_mod._wrap_marker_lock_path(indirect)


class TestOwnerStackIsPerEndpointKey:
    """One project-local file holds owners for every Claude endpoint key. A run
    exiting must not delete another key's live owner (round 11, P2)."""

    @staticmethod
    def _project(tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user-gateway"}}))
        return settings, wrap_mod._wrap_marker_path(settings)

    def _seed_vertex_peer(self, marker: Path) -> None:
        """A concurrent live run owning the Vertex key, not ours."""
        marker.write_text(
            json.dumps(
                {
                    "owners": [
                        {
                            "pid": 111,
                            "port": 8790,
                            "key": "ANTHROPIC_VERTEX_BASE_URL",
                            "previous": None,
                            "url": "http://127.0.0.1:8790",
                        }
                    ],
                    "pid": 111,
                    "port": 8790,
                    "key": "ANTHROPIC_VERTEX_BASE_URL",
                    "previous": None,
                }
            )
        )

    def test_exiting_run_keeps_another_keys_owner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker = self._project(tmp_path)
        self._seed_vertex_peer(marker)
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)

        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )
        wrap_mod._restore_claude_wrap_base_url(previous, settings_path=settings)

        assert marker.exists(), "the Vertex run's crash record must survive our exit"
        owners = json.loads(marker.read_text())["owners"]
        assert [o["key"] for o in owners] == ["ANTHROPIC_VERTEX_BASE_URL"]
        # Our own key is restored to the user's original value regardless.
        env = json.loads(settings.read_text())["env"]
        assert env["ANTHROPIC_BASE_URL"] == "https://user-gateway"

    def test_last_owner_of_the_only_key_still_clears_the_marker(self, tmp_path: Path) -> None:
        settings, marker = self._project(tmp_path)
        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )

        wrap_mod._restore_claude_wrap_base_url(previous, settings_path=settings)

        assert not marker.exists()

    def test_only_our_pid_and_key_entry_is_popped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A same-key live peer AND another key's owner both survive."""
        settings, marker = self._project(tmp_path)
        self._seed_vertex_peer(marker)
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)
        # Handover now requires the peer's proxy to still answer; these
        # ports are fixtures with nothing listening.
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda _port, **_k: True)
        payload = json.loads(marker.read_text())
        payload["owners"].append(
            {
                "pid": 222,
                "port": 8789,
                "key": "ANTHROPIC_BASE_URL",
                "previous": "https://user-gateway",
                "url": "http://127.0.0.1:8789",
            }
        )
        marker.write_text(json.dumps(payload))

        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )
        wrap_mod._restore_claude_wrap_base_url(previous, settings_path=settings)

        owners = json.loads(marker.read_text())["owners"]
        assert sorted(o["pid"] for o in owners) == [111, 222]
        # Handover went to the live same-key peer, not the Vertex owner.
        env = json.loads(settings.read_text())["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8789"


class TestCrashHandoverPreservesOtherKeys:
    """The crash-cleanup handover rewrote the stack to just its own key,
    deleting a concurrent other-key run's record (round 12, P2)."""

    @staticmethod
    def _project(tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user-gateway"}}))
        return settings, wrap_mod._wrap_marker_path(settings)

    @staticmethod
    def _owner(pid: int, port: int, key: str) -> dict[str, Any]:
        return {
            "pid": pid,
            "port": port,
            "key": key,
            "previous": "https://user-gateway",
            "url": f"http://127.0.0.1:{port}",
        }

    def test_vertex_owner_survives_a_base_url_crash_handover(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Newest Base owner crashed; an older live Base owner takes over and
        the live Vertex owner must still be on the stack afterwards."""
        settings, marker = self._project(tmp_path)
        live_base = self._owner(os.getpid(), 8788, "ANTHROPIC_BASE_URL")
        live_vertex = self._owner(111, 8790, "ANTHROPIC_VERTEX_BASE_URL")
        dead_base = self._owner(2147480000, 8789, "ANTHROPIC_BASE_URL")
        marker.write_text(json.dumps({"owners": [live_base, live_vertex, dead_base], **dead_base}))
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda o: o.get("pid") == 2147480000)

        handover = wrap_mod._handover_to_live_owner(settings, key="ANTHROPIC_BASE_URL")

        assert handover == "http://127.0.0.1:8788"
        owners = json.loads(marker.read_text())["owners"]
        assert sorted(o["pid"] for o in owners) == [111, os.getpid()]
        # The handover target must remain newest, since the top-level mirror
        # is owners[-1] and self-heal reads it.
        assert owners[-1]["pid"] == os.getpid()

    def test_other_key_survives_when_no_same_key_owner_remains(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker = self._project(tmp_path)
        live_vertex = self._owner(111, 8790, "ANTHROPIC_VERTEX_BASE_URL")
        dead_base = self._owner(2147480000, 8789, "ANTHROPIC_BASE_URL")
        marker.write_text(json.dumps({"owners": [live_vertex, dead_base], **dead_base}))
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda o: o.get("pid") == 2147480000)

        handover = wrap_mod._handover_to_live_owner(settings, key="ANTHROPIC_BASE_URL")

        assert handover is None, "no live Base owner to hand to"
        assert marker.exists(), "the Vertex run's record must not be collateral damage"
        assert [o["key"] for o in json.loads(marker.read_text())["owners"]] == [
            "ANTHROPIC_VERTEX_BASE_URL"
        ]


class TestPreviousComesFromTheMatchingOwner:
    """`previous` was read from the top-level mirror, which is just owners[-1]
    across ALL keys — so an interleaved run inherited the wrong original
    (round 12, P2)."""

    @staticmethod
    def _project(tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user-gateway"}}))
        return settings, wrap_mod._wrap_marker_path(settings)

    def test_interleaved_keys_do_not_poison_previous(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Base A, then Vertex V, then Base B: B must inherit A's recorded
        original, not A's proxy URL (which is what the file holds)."""
        settings, marker = self._project(tmp_path)
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)

        # Run A takes the Base key.
        prev_a = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )
        assert prev_a == "https://user-gateway"
        payload = json.loads(marker.read_text())
        payload["owners"][-1]["pid"] = 111  # A is another live process
        marker.write_text(json.dumps(payload))

        # Run V takes the Vertex key, landing on top of the stack/mirror.
        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8790", settings_path=settings, port=8790, vertex_mode=True
        )
        payload = json.loads(marker.read_text())
        payload["owners"][-1]["pid"] = 222
        marker.write_text(json.dumps(payload))
        assert payload["owners"][-1]["key"] == "ANTHROPIC_VERTEX_BASE_URL"

        # Run B takes the Base key again — the mirror names V, not A.
        prev_b = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8789", settings_path=settings, port=8789
        )

        assert prev_b == "https://user-gateway", "B inherited a dead proxy URL as the original"

    def test_first_owner_still_reads_the_files_real_value(self, tmp_path: Path) -> None:
        settings, _marker = self._project(tmp_path)

        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )

        assert previous == "https://user-gateway"

    def test_other_keys_owner_alone_does_not_supply_previous(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Only a Vertex owner exists; a fresh Base run must read the file."""
        settings, marker = self._project(tmp_path)
        marker.write_text(
            json.dumps(
                {
                    "owners": [
                        {
                            "pid": 111,
                            "port": 8790,
                            "key": "ANTHROPIC_VERTEX_BASE_URL",
                            "previous": "https://somebody-elses-original",
                            "url": "http://127.0.0.1:8790",
                        }
                    ]
                }
            )
        )
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)

        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )

        assert previous == "https://user-gateway"


class TestCloudModesAlsoRejectNestedProxies:
    """Foundry and Vertex have their own endpoint variables, and a nested wrap
    inherits whichever its parent set (round 12, P2)."""

    @staticmethod
    def _is_headroom(monkeypatch: pytest.MonkeyPatch, *, yes: bool) -> None:
        monkeypatch.setattr(
            wrap_mod,
            "_query_proxy_health",
            lambda _p: {"service": "headroom-proxy" if yes else "litellm"},
        )

    def test_url_predicate_covers_every_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._is_headroom(monkeypatch, yes=True)

        assert wrap_mod._url_is_local_headroom_proxy("http://127.0.0.1:8788") is True
        # Foundry appends a path component; it must not defeat the check.
        assert wrap_mod._url_is_local_headroom_proxy("http://127.0.0.1:8788/anthropic") is True
        assert wrap_mod._url_is_local_headroom_proxy("https://foo.services.ai.azure.com") is False
        assert wrap_mod._url_is_local_headroom_proxy("http://localhost/v1") is False
        assert wrap_mod._url_is_local_headroom_proxy("") is False
        assert wrap_mod._url_is_local_headroom_proxy(None) is False

    def test_vertex_ignores_an_inherited_parent_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_VERTEX_BASE_URL", "http://127.0.0.1:8788")
        self._is_headroom(monkeypatch, yes=True)

        assert wrap_mod._vertex_target_api_url_from_claude_env("http://127.0.0.1:8789") is None

    def test_vertex_explicit_target_ignores_an_inherited_parent_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VERTEX_TARGET_API_URL", "http://127.0.0.1:8788")
        self._is_headroom(monkeypatch, yes=True)

        assert wrap_mod._vertex_target_api_url_from_claude_env("http://127.0.0.1:8789") is None

    def test_vertex_keeps_a_real_gateway(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_VERTEX_BASE_URL", "https://vertex.example.com")

        def unexpected(_port: int) -> Any:
            raise AssertionError("remote hosts must not be probed")

        monkeypatch.setattr(wrap_mod, "_query_proxy_health", unexpected)

        assert (
            wrap_mod._vertex_target_api_url_from_claude_env("http://127.0.0.1:8789")
            == "https://vertex.example.com"
        )

    def test_vertex_keeps_a_local_non_headroom_gateway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_VERTEX_BASE_URL", "http://127.0.0.1:4000")
        self._is_headroom(monkeypatch, yes=False)

        assert (
            wrap_mod._vertex_target_api_url_from_claude_env("http://127.0.0.1:8789")
            == "http://127.0.0.1:4000"
        )


class TestClearMarkerIsKeyScoped:
    """`_clear_wrap_marker` unlinked the whole file based on the top-level
    mirror. One marker serves every endpoint key, so the forced stale-cleanup
    path could destroy a live Vertex run's record as collateral damage."""

    @staticmethod
    def _project(tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user-gateway"}}))
        return settings, wrap_mod._wrap_marker_path(settings)

    def test_only_the_named_keys_owners_are_removed(self, tmp_path: Path) -> None:
        settings, marker = self._project(tmp_path)
        marker.write_text(
            json.dumps(
                {
                    "owners": [
                        {"pid": 111, "port": 8790, "key": "ANTHROPIC_VERTEX_BASE_URL"},
                        {"pid": 222, "port": 8789, "key": "ANTHROPIC_BASE_URL"},
                    ],
                    "pid": 222,
                    "key": "ANTHROPIC_BASE_URL",
                }
            )
        )

        wrap_mod._clear_wrap_marker(settings, key="ANTHROPIC_BASE_URL")

        assert marker.exists()
        assert [o["key"] for o in json.loads(marker.read_text())["owners"]] == [
            "ANTHROPIC_VERTEX_BASE_URL"
        ]

    def test_file_is_removed_once_nothing_remains(self, tmp_path: Path) -> None:
        settings, marker = self._project(tmp_path)
        marker.write_text(json.dumps({"pid": 222, "port": 8789, "key": "ANTHROPIC_BASE_URL"}))

        wrap_mod._clear_wrap_marker(settings, key="ANTHROPIC_BASE_URL")

        assert not marker.exists()

    def test_other_keys_marker_is_untouched(self, tmp_path: Path) -> None:
        settings, marker = self._project(tmp_path)
        marker.write_text(
            json.dumps({"pid": 111, "port": 8790, "key": "ANTHROPIC_VERTEX_BASE_URL"})
        )

        wrap_mod._clear_wrap_marker(settings, key="ANTHROPIC_BASE_URL")

        assert marker.exists()

    def test_forced_stale_cleanup_spares_a_live_other_key_owner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End-to-end: a crashed Base run's cleanup must not take the live
        Vertex run's record with it."""
        settings, marker = self._project(tmp_path)
        marker.write_text(
            json.dumps(
                {
                    "owners": [
                        {
                            "pid": 111,
                            "port": 8790,
                            "key": "ANTHROPIC_VERTEX_BASE_URL",
                            "previous": None,
                            "url": "http://127.0.0.1:8790",
                        },
                        {
                            "pid": 2147480000,
                            "port": 8789,
                            "key": "ANTHROPIC_BASE_URL",
                            "previous": "https://user-gateway",
                            "url": "http://127.0.0.1:8789",
                        },
                    ],
                    "pid": 2147480000,
                    "port": 8789,
                    "key": "ANTHROPIC_BASE_URL",
                    "previous": "https://user-gateway",
                }
            )
        )
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda o: o.get("pid") == 2147480000)

        restored = wrap_mod._check_and_clear_stale_wrap_marker(settings, key="ANTHROPIC_BASE_URL")

        assert restored == "https://user-gateway"
        assert marker.exists(), "the live Vertex run's record must survive"
        assert [o["key"] for o in json.loads(marker.read_text())["owners"]] == [
            "ANTHROPIC_VERTEX_BASE_URL"
        ]


class TestGuardsSelectTheirOwnKey:
    """The stale/dead-marker guards keyed off the top-level mirror, which is
    owners[-1] across ALL keys — so a newer cloud-mode owner made them go
    blind to a crashed owner of their own key (round 13, P2)."""

    @staticmethod
    def _project(tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8789"}}))
        return settings, wrap_mod._wrap_marker_path(settings)

    @staticmethod
    def _stack(marker: Path, owners: list[dict[str, Any]]) -> None:
        marker.write_text(json.dumps({"owners": owners, **owners[-1]}))

    def _crashed_base_under_live_vertex(self, marker: Path) -> None:
        """A crashed Base owner, with a LIVE Vertex owner newer than it — so
        the mirror names the Vertex key."""
        self._stack(
            marker,
            [
                {
                    "pid": 2147480000,
                    "port": 8789,
                    "key": "ANTHROPIC_BASE_URL",
                    "previous": "https://user-gateway",
                    "url": "http://127.0.0.1:8789",
                },
                {
                    "pid": 111,
                    "port": 8790,
                    "key": "ANTHROPIC_VERTEX_BASE_URL",
                    "previous": None,
                    "url": "http://127.0.0.1:8790",
                },
            ],
        )

    def test_stale_guard_sees_past_a_newer_cloud_owner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker = self._project(tmp_path)
        self._crashed_base_under_live_vertex(marker)
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda o: o.get("pid") == 2147480000)

        restored = wrap_mod._check_and_clear_stale_wrap_marker(settings, key="ANTHROPIC_BASE_URL")

        assert restored == "https://user-gateway"
        env = json.loads(settings.read_text())["env"]
        assert env["ANTHROPIC_BASE_URL"] == "https://user-gateway"
        # ...and the live Vertex owner is still recorded.
        assert [o["key"] for o in json.loads(marker.read_text())["owners"]] == [
            "ANTHROPIC_VERTEX_BASE_URL"
        ]

    def test_dead_guard_sees_past_a_newer_cloud_owner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker = self._project(tmp_path)
        self._crashed_base_under_live_vertex(marker)
        # Base's port is dead; Vertex's answers.
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda port, **_k: port == 8790)
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)

        restored = wrap_mod._check_and_clear_dead_wrap_marker(settings, key="ANTHROPIC_BASE_URL")

        assert restored == "https://user-gateway"

    def test_a_live_owner_of_our_key_is_still_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker = self._project(tmp_path)
        self._stack(
            marker,
            [
                {
                    "pid": 222,
                    "port": 8789,
                    "key": "ANTHROPIC_BASE_URL",
                    "previous": "https://user-gateway",
                    "url": "http://127.0.0.1:8789",
                },
                {
                    "pid": 111,
                    "port": 8790,
                    "key": "ANTHROPIC_VERTEX_BASE_URL",
                    "previous": None,
                    "url": "http://127.0.0.1:8790",
                },
            ],
        )
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)

        assert (
            wrap_mod._check_and_clear_stale_wrap_marker(settings, key="ANTHROPIC_BASE_URL") is None
        )
        env = json.loads(settings.read_text())["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8789"

    def test_no_owner_of_our_key_is_a_noop(self, tmp_path: Path) -> None:
        settings, marker = self._project(tmp_path)
        self._stack(
            marker,
            [{"pid": 111, "port": 8790, "key": "ANTHROPIC_VERTEX_BASE_URL", "previous": None}],
        )

        assert (
            wrap_mod._check_and_clear_stale_wrap_marker(settings, key="ANTHROPIC_BASE_URL") is None
        )
        assert marker.exists()


class TestExitPreservesCrashedOtherKeyOwners:
    """The pop was rebuilt from the staleness-FILTERED view, so a CRASHED
    cloud-mode owner was silently dropped by an unrelated key's clean exit —
    destroying the one record its self-heal needed (round 13, P2)."""

    @staticmethod
    def _project(tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user-gateway"}}))
        return settings, wrap_mod._wrap_marker_path(settings)

    def test_crashed_vertex_record_survives_a_clean_base_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker = self._project(tmp_path)
        crashed_vertex = {
            "pid": 2147480000,
            "port": 8790,
            "key": "ANTHROPIC_VERTEX_BASE_URL",
            "previous": "https://vertex-original",
            "url": "http://127.0.0.1:8790",
        }
        marker.write_text(json.dumps({"owners": [crashed_vertex], **crashed_vertex}))
        # Only the Vertex owner is dead; ours (this process) is alive.
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda o: o.get("pid") == 2147480000)

        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )
        wrap_mod._restore_claude_wrap_base_url(previous, settings_path=settings)

        assert marker.exists(), "the crashed Vertex run's self-heal record was deleted"
        owners = json.loads(marker.read_text())["owners"]
        assert [o["key"] for o in owners] == ["ANTHROPIC_VERTEX_BASE_URL"]
        assert owners[0]["previous"] == "https://vertex-original"


class TestFoundryHandoverKeepsThePathPrefix:
    """Owner records stored a URL rebuilt from the port alone, but Foundry
    writes `http://127.0.0.1:<port>/anthropic` into settings. A handover
    therefore dropped the prefix the SDK appends /v1/messages to (round 13)."""

    @staticmethod
    def _project(tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {}}))
        return settings, wrap_mod._wrap_marker_path(settings)

    def test_marker_records_the_exact_settings_value(self, tmp_path: Path) -> None:
        settings, marker = self._project(tmp_path)
        foundry_url = wrap_mod._foundry_proxy_url(wrap_mod._claude_proxy_base_url(8788))
        assert foundry_url.endswith("/anthropic")

        wrap_mod._write_claude_wrap_base_url(
            foundry_url, settings_path=settings, port=8788, foundry_mode=True
        )

        owner = json.loads(marker.read_text())["owners"][-1]
        assert owner["key"] == "ANTHROPIC_FOUNDRY_BASE_URL"
        assert owner["url"] == foundry_url

    def test_handover_restores_the_prefixed_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker = self._project(tmp_path)
        url_a = wrap_mod._foundry_proxy_url(wrap_mod._claude_proxy_base_url(8788))
        url_b = wrap_mod._foundry_proxy_url(wrap_mod._claude_proxy_base_url(8789))
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)
        # Handover now requires the peer's proxy to still answer; these
        # ports are fixtures with nothing listening.
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda _port, **_k: True)

        wrap_mod._write_claude_wrap_base_url(
            url_a, settings_path=settings, port=8788, foundry_mode=True
        )
        payload = json.loads(marker.read_text())
        payload["owners"][-1]["pid"] = 111  # run A is another live process
        marker.write_text(json.dumps(payload))
        prev_b = wrap_mod._write_claude_wrap_base_url(
            url_b, settings_path=settings, port=8789, foundry_mode=True
        )

        # B exits; routing hands back to the still-live A.
        wrap_mod._restore_claude_wrap_base_url(prev_b, settings_path=settings, foundry_mode=True)

        restored = json.loads(settings.read_text())["env"]["ANTHROPIC_FOUNDRY_BASE_URL"]
        assert restored == url_a
        assert restored.endswith("/anthropic"), "SDK would append /v1/messages to the wrong route"

    def test_bare_url_is_still_the_default_for_standard_mode(self, tmp_path: Path) -> None:
        settings, marker = self._project(tmp_path)

        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )

        assert json.loads(marker.read_text())["owners"][-1]["url"] == "http://127.0.0.1:8788"

    def test_explicit_url_is_optional(self, tmp_path: Path) -> None:
        """A direct _write_wrap_marker caller that omits `url` keeps the
        port-derived value, so older call sites are unaffected."""
        settings, marker = self._project(tmp_path)

        wrap_mod._write_wrap_marker(settings, port=8788, key="ANTHROPIC_BASE_URL", previous=None)

        assert json.loads(marker.read_text())["owners"][-1]["url"] == "http://127.0.0.1:8788"


class TestHandoverRequiresALiveProxy:
    """A live WRAPPER is not enough to hand routing back to (round 14, P2).

    The wrapper stays blocked on the agent process long after its dedicated
    proxy dies, so handing settings.local.json to its recorded URL would point
    every later conversation at a dead port.
    """

    @staticmethod
    def _project(tmp_path: Path) -> tuple[Path, Path]:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user-gateway"}}))
        return settings, wrap_mod._wrap_marker_path(settings)

    def _peer_then_us(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> tuple[Path, Path, str | None]:
        settings, marker = self._project(tmp_path)
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)
        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788
        )
        payload = json.loads(marker.read_text())
        payload["owners"][-1]["pid"] = 111  # peer A: a different LIVE wrapper
        marker.write_text(json.dumps(payload))
        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8789", settings_path=settings, port=8789
        )
        return settings, marker, previous

    @staticmethod
    def _url(settings: Path) -> str | None:
        if not settings.exists():
            return None
        return json.loads(settings.read_text())["env"].get("ANTHROPIC_BASE_URL")

    def test_dead_peer_proxy_falls_back_to_the_original(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, _marker, previous = self._peer_then_us(monkeypatch, tmp_path)
        # Peer A's wrapper is alive, but its proxy no longer answers.
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda _port, **_k: False)

        wrap_mod._restore_claude_wrap_base_url(previous, settings_path=settings)

        assert self._url(settings) == "https://user-gateway", (
            "routing was handed to a wrapper whose proxy is dead"
        )

    def test_live_peer_proxy_still_receives_the_handover(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings, marker, previous = self._peer_then_us(monkeypatch, tmp_path)
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda _port, **_k: True)

        wrap_mod._restore_claude_wrap_base_url(previous, settings_path=settings)

        assert self._url(settings) == "http://127.0.0.1:8788"
        assert marker.exists()

    def test_only_the_peer_with_a_live_port_is_chosen(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Two live peers, one dead proxy: the handover must skip it."""
        settings, marker = self._project(tmp_path)
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda port, **_k: port == 8788)
        for port, pid in ((8788, 111), (8790, 222)):
            wrap_mod._write_claude_wrap_base_url(
                f"http://127.0.0.1:{port}", settings_path=settings, port=port
            )
            payload = json.loads(marker.read_text())
            payload["owners"][-1]["pid"] = pid
            marker.write_text(json.dumps(payload))
        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8789", settings_path=settings, port=8789
        )

        wrap_mod._restore_claude_wrap_base_url(previous, settings_path=settings)

        # 8790 is newer but dead; 8788 is the newest owner that still answers.
        assert self._url(settings) == "http://127.0.0.1:8788"

    def test_a_portless_legacy_owner_is_not_excluded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No recorded port means no liveness signal — don't invent one."""
        settings, marker = self._project(tmp_path)
        marker.write_text(
            json.dumps(
                {
                    "owners": [
                        {
                            "pid": 111,
                            "key": "ANTHROPIC_BASE_URL",
                            "previous": "https://user-gateway",
                            "url": "http://127.0.0.1:8788",
                        }
                    ]
                }
            )
        )
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)

        def unexpected(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("a portless owner must not be probed")

        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", unexpected)
        previous = wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8789", settings_path=settings, port=8789
        )

        wrap_mod._restore_claude_wrap_base_url(previous, settings_path=settings)

        assert self._url(settings) == "http://127.0.0.1:8788"


class TestSelfhealHookWriteIsSerialized:
    """The hook installer rewrites the WHOLE settings payload it read, and it
    touches the same project-local file the base_url write guards. Unlocked, a
    peer's write landing between our read and our write is clobbered by our
    stale snapshot (round 15, P1)."""

    @staticmethod
    def _project(tmp_path: Path) -> Path:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user-gateway"}}))
        return settings

    @staticmethod
    def _peer_can_lock(settings: Path) -> bool:
        path = wrap_mod._wrap_marker_lock_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+", encoding="utf-8") as handle:
            if not _filelock.acquire(handle, timeout=0):
                return False
            _filelock.release(handle)
            return True

    def test_hook_install_holds_the_lock_across_read_and_write(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings = self._project(tmp_path)
        observed: list[bool] = []
        real = wrap_mod._read_text

        def probing_read(path: Path) -> Any:
            if Path(path) == settings:
                observed.append(self._peer_can_lock(settings))
            return real(path)

        monkeypatch.setattr(wrap_mod, "_read_text", probing_read)
        wrap_mod._ensure_claude_wrap_selfheal_hook(settings)

        assert observed and not any(observed), "the settings read must be inside the lock"

    def test_hook_install_preserves_a_concurrently_written_base_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Companion to the lock-coverage test above: because the installer
        rewrites the WHOLE payload, it must read the file as it stands at
        install time. Here a peer's base_url is already committed before we
        run — our write must carry it forward, not an older snapshot."""
        settings = self._project(tmp_path)
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)
        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8790", settings_path=settings, port=8790
        )

        wrap_mod._ensure_claude_wrap_selfheal_hook(settings)

        payload = json.loads(settings.read_text())
        assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8790"
        assert "hooks" in payload

    def test_the_hook_is_still_installed_and_idempotent(self, tmp_path: Path) -> None:
        settings = self._project(tmp_path)

        wrap_mod._ensure_claude_wrap_selfheal_hook(settings)
        wrap_mod._ensure_claude_wrap_selfheal_hook(settings)

        entries = json.loads(settings.read_text())["hooks"]["SessionStart"]
        assert len(entries) == 1
        assert wrap_mod._WRAP_SELFHEAL_HOOK_MARKER in entries[0]["hooks"][0]["command"]
        # ...and the env block it shares the file with is untouched.
        assert json.loads(settings.read_text())["env"]["ANTHROPIC_BASE_URL"] == (
            "https://user-gateway"
        )

    def test_nested_under_the_base_url_write_does_not_deadlock(self, tmp_path: Path) -> None:
        """Both take the same lock; the real call order is write-then-hook."""
        settings = self._project(tmp_path)

        with wrap_mod._wrap_marker_lock(settings):
            wrap_mod._ensure_claude_wrap_selfheal_hook(settings)

        assert not _filelock._held
        assert "hooks" in json.loads(settings.read_text())


class TestUnwrapRestoresEachKey:
    """`unwrap` read `_prior` from the top-level mirror, so with interleaved
    keys every non-mirrored one got None — and `_force=True` then DELETED the
    user's pre-existing gateway instead of restoring it (round 16, P2)."""

    def test_each_key_restores_its_own_recorded_previous(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project = tmp_path / "proj"
        settings = project / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(
            json.dumps(
                {
                    "env": {
                        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8788",
                        "ANTHROPIC_VERTEX_BASE_URL": "http://127.0.0.1:8790",
                    }
                }
            )
        )
        marker = wrap_mod._wrap_marker_path(settings)
        base_owner = {
            "pid": 111,
            "port": 8788,
            "key": "ANTHROPIC_BASE_URL",
            "previous": "https://user-gateway",
            "url": "http://127.0.0.1:8788",
        }
        vertex_owner = {
            "pid": 222,
            "port": 8790,
            "key": "ANTHROPIC_VERTEX_BASE_URL",
            "previous": "https://vertex-original",
            "url": "http://127.0.0.1:8790",
        }
        # Vertex is newest, so the mirror names it and Base is invisible there.
        marker.write_text(json.dumps({"owners": [base_owner, vertex_owner], **vertex_owner}))

        for foundry, vertex in ((False, False), (True, False), (False, True)):
            key = wrap_mod._claude_wrap_base_url_env_key(foundry_mode=foundry, vertex_mode=vertex)
            prior_owner = wrap_mod._newest_persisted_owner(settings, key=key)
            prior = prior_owner.get("previous") if prior_owner is not None else None
            wrap_mod._restore_claude_wrap_base_url(
                prior,
                foundry_mode=foundry,
                vertex_mode=vertex,
                settings_path=settings,
                _force=True,
            )

        env = json.loads(settings.read_text())["env"]
        assert env["ANTHROPIC_BASE_URL"] == "https://user-gateway"
        assert env["ANTHROPIC_VERTEX_BASE_URL"] == "https://vertex-original"

    def test_newest_persisted_owner_ignores_the_mirror(self, tmp_path: Path) -> None:
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        marker = wrap_mod._wrap_marker_path(settings)
        base = {"pid": 111, "key": "ANTHROPIC_BASE_URL", "previous": "https://user-gateway"}
        vertex = {"pid": 222, "key": "ANTHROPIC_VERTEX_BASE_URL", "previous": None}
        marker.write_text(json.dumps({"owners": [base, vertex], **vertex}))

        found = wrap_mod._newest_persisted_owner(settings, key="ANTHROPIC_BASE_URL")

        assert found is not None and found["previous"] == "https://user-gateway"


class TestInterruptedProxyIsReaped:
    """SIGHUP can land inside the readiness loop, after the DETACHED child is
    spawned but before `_ensure_proxy` returns it — the caller's cleanup never
    sees it and GC never learns about it (round 16, P2)."""

    def test_child_is_killed_when_the_wait_is_interrupted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        killed: list[bool] = []

        class _Proc:
            pid = 4242
            returncode = None

            def poll(self) -> None:
                return None

            def kill(self) -> None:
                killed.append(True)

        monkeypatch.setattr(wrap_mod.subprocess, "Popen", lambda *a, **k: _Proc())
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: False)

        def interrupt(_seconds: float) -> None:
            raise SystemExit(0)  # what the SIGHUP handler raises

        monkeypatch.setattr(wrap_mod.time, "sleep", interrupt)
        monkeypatch.setattr(wrap_mod, "_build_proxy_env", lambda *a, **k: {}, raising=False)

        with pytest.raises(SystemExit):
            wrap_mod._start_proxy(8788)

        assert killed == [True], "the detached child was abandoned"

    def test_a_spawn_failure_still_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`proc` is unbound if Popen itself raised; the reaper must not mask
        the original error with a NameError."""

        def boom(*_a: Any, **_k: Any) -> Any:
            raise OSError("no exec for you")

        monkeypatch.setattr(wrap_mod.subprocess, "Popen", boom)

        with pytest.raises(OSError, match="no exec for you"):
            wrap_mod._start_proxy(8788)


class TestOpenclawBootstrapIsExempt:
    """`wrap openclaw` is a durable installer that hands off to a gateway whose
    autoStart launches its own detached proxy — that proxy must not land on a
    run directory this wrapper never records (round 16, P2)."""

    def test_openclaw_is_exempt_from_isolation(self) -> None:
        assert "openclaw" in wrap_mod._WRAP_ISOLATION_EXEMPT_SUBCOMMANDS

    def test_a_wrapper_owned_session_is_not_exempt(self) -> None:
        for name in ("claude", "codex", "copilot", "grok"):
            assert name not in wrap_mod._WRAP_ISOLATION_EXEMPT_SUBCOMMANDS

    def test_openclaw_run_creates_no_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv(isolation.HEADROOM_ISOLATED_ENV, raising=False)
        runner = CliRunner()

        result = runner.invoke(main, ["wrap", "openclaw", "--prepare-only"])

        runs = tmp_path / "ws" / "runs"
        assert not runs.exists() or not list(runs.iterdir()), result.output


class TestParentUpstreamSurvivesNesting:
    """A nested wrap correctly refuses to chain onto the parent's proxy, but
    the parent's REAL upstream was never propagated — so the child silently
    fell back to api.anthropic.com (round 16, P2)."""

    def test_inherited_upstream_is_adopted_instead_of_the_parent_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")
        monkeypatch.setenv(wrap_mod._PARENT_UPSTREAM_ENV, "https://litellm.example.com")
        monkeypatch.setattr(
            wrap_mod, "_query_proxy_health", lambda _p: {"service": "headroom-proxy"}
        )

        assert wrap_mod._detect_inbound_anthropic_upstream(8788) == "https://litellm.example.com"

    def test_without_an_inherited_upstream_it_is_still_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")
        monkeypatch.delenv(wrap_mod._PARENT_UPSTREAM_ENV, raising=False)
        monkeypatch.setattr(
            wrap_mod, "_query_proxy_health", lambda _p: {"service": "headroom-proxy"}
        )

        assert wrap_mod._detect_inbound_anthropic_upstream(8788) is None

    def test_a_real_gateway_still_wins_over_the_inherited_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://direct-gateway.example.com")
        monkeypatch.setenv(wrap_mod._PARENT_UPSTREAM_ENV, "https://litellm.example.com")

        assert (
            wrap_mod._detect_inbound_anthropic_upstream(8788)
            == "https://direct-gateway.example.com"
        )

    def test_export_sets_and_clears(self) -> None:
        env: dict[str, str] = {}
        wrap_mod._export_parent_upstream(env, "https://litellm.example.com")
        assert env[wrap_mod._PARENT_UPSTREAM_ENV] == "https://litellm.example.com"

        # A grandparent's value must not be adopted by our child as ours.
        wrap_mod._export_parent_upstream(env, None)
        assert wrap_mod._PARENT_UPSTREAM_ENV not in env

    def test_foundry_falls_back_to_the_inherited_upstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(wrap_mod._PARENT_UPSTREAM_ENV, "https://foo.services.ai.azure.com")
        monkeypatch.setattr(
            wrap_mod, "_query_proxy_health", lambda _p: {"service": "headroom-proxy"}
        )

        assert wrap_mod._url_is_local_headroom_proxy("http://127.0.0.1:8788/anthropic") is True
        assert wrap_mod._inherited_parent_upstream() == "https://foo.services.ai.azure.com"


class TestExemptSubcommandsLeaveInheritedIsolation:
    """Declining to ACTIVATE isolation is not enough when nested inside an
    already-isolated agent: the parent's per-run workspace is already in the
    environment, inherited (round 17, P2)."""

    def test_nested_openclaw_returns_to_shared_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A parent wrap already activated isolation in this process tree.
        run_dir = isolation.activate_isolated_workspace()
        shared = paths.shared_workspace_dir()
        assert paths.workspace_dir() == run_dir

        wrap_mod._leave_inherited_isolation()

        assert paths.workspace_dir() == shared
        assert isolation.active_isolated_workspace() is None
        assert isolation.isolation_requested() is False

    def test_it_is_a_noop_at_top_level(self, tmp_path: Path) -> None:
        before = paths.workspace_dir()

        wrap_mod._leave_inherited_isolation()

        assert paths.workspace_dir() == before
        # `disable_isolation` was never called, so no explicit opt-out was
        # recorded for children either.
        assert isolation.HEADROOM_ISOLATED_ENV not in os.environ

    def test_the_gateway_would_inherit_the_shared_workspace(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """What actually matters: the env a spawned gateway/proxy receives."""
        isolation.activate_isolated_workspace()

        wrap_mod._leave_inherited_isolation()

        inherited = os.environ[paths.HEADROOM_WORKSPACE_DIR_ENV]
        assert "runs" not in Path(inherited).parts
        assert isolation.HEADROOM_MEMORY_DB_PATH_ENV not in os.environ


class TestPortRewriteOnlyTouchesOurOwnVariables:
    """An isolated run binds a shifted port on every launch, so the port-fixup
    path is routine. A blanket replace across the inherited environment
    silently redirects the child's unrelated traffic (round 18, P2)."""

    @staticmethod
    def _rewrite(env: dict[str, str], port: int, actual_port: int) -> dict[str, str]:
        """The real implementation — not a mirror, which could drift."""
        wrap_mod._repoint_own_endpoints(env, port, actual_port)
        return env

    def test_an_inherited_http_proxy_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8787")
        monkeypatch.setenv("DATABASE_URL", "postgres://127.0.0.1:8787/app")
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:8787"

        result = self._rewrite(env, 8787, 8788)

        assert result["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8788"
        assert result["HTTP_PROXY"] == "http://127.0.0.1:8787"
        assert result["DATABASE_URL"] == "postgres://127.0.0.1:8787/app"

    def test_inherited_headroom_service_urls_are_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `HEADROOM_*` prefix does NOT mean proxy routing: plenty of them
        address other local services, and re-pointing those at the proxy sends
        the agent and its MCP children to the wrong one."""
        monkeypatch.setenv("HEADROOM_REDIS_URL", "redis://127.0.0.1:8787/0")
        monkeypatch.setenv("HEADROOM_QDRANT_URL", "http://127.0.0.1:8787")
        monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT", "http://127.0.0.1:8787/compress")
        env = os.environ.copy()

        result = self._rewrite(env, 8787, 8788)

        assert result["HEADROOM_REDIS_URL"] == "redis://127.0.0.1:8787/0"
        assert result["HEADROOM_QDRANT_URL"] == "http://127.0.0.1:8787"
        assert result["HEADROOM_KOMPRESS_ENDPOINT"] == "http://127.0.0.1:8787/compress"

    def test_a_wrapper_set_headroom_var_is_rewritten(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What matters is that WE wrote it, not what it is called."""
        monkeypatch.delenv("HEADROOM_PROXY_URL", raising=False)
        env = os.environ.copy()
        env["HEADROOM_PROXY_URL"] = "http://127.0.0.1:8787"

        result = self._rewrite(env, 8787, 8788)

        assert result["HEADROOM_PROXY_URL"] == "http://127.0.0.1:8788"

    def test_a_wrapper_set_value_is_rewritten(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = "http://127.0.0.1:8787/v1"

        result = self._rewrite(env, 8787, 8788)

        assert result["OPENAI_BASE_URL"] == "http://127.0.0.1:8788/v1"

    def test_the_launch_path_delegates_here(self) -> None:
        """Guard against `_launch_tool` drifting back to a blanket replace."""
        import inspect

        source = inspect.getsource(wrap_mod._launch_tool)
        assert "_repoint_own_endpoints(" in source
        # ...and hands it the authorship record, not just the env.
        assert "written=_wrapper_written_keys(env_vars_display, env)" in source


class TestVertexUpstreamSurvivesNesting:
    """Vertex travels to the proxy on its own parameter, so it bypassed the
    parent-upstream export entirely (round 18, P2)."""

    def test_inherited_upstream_is_adopted_for_the_vertex_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_VERTEX_BASE_URL", "http://127.0.0.1:8788")
        monkeypatch.setenv(wrap_mod._PARENT_UPSTREAM_ENV, "https://vertex-gw.example.com")
        monkeypatch.setattr(
            wrap_mod, "_query_proxy_health", lambda _p: {"service": "headroom-proxy"}
        )

        assert (
            wrap_mod._vertex_target_api_url_from_claude_env("http://127.0.0.1:8789")
            == "https://vertex-gw.example.com"
        )

    def test_inherited_upstream_is_adopted_for_the_explicit_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VERTEX_TARGET_API_URL", "http://127.0.0.1:8788")
        monkeypatch.setenv(wrap_mod._PARENT_UPSTREAM_ENV, "https://vertex-gw.example.com")
        monkeypatch.setattr(
            wrap_mod, "_query_proxy_health", lambda _p: {"service": "headroom-proxy"}
        )

        assert (
            wrap_mod._vertex_target_api_url_from_claude_env("http://127.0.0.1:8789")
            == "https://vertex-gw.example.com"
        )

    def test_without_an_inherited_upstream_it_is_still_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_VERTEX_BASE_URL", "http://127.0.0.1:8788")
        monkeypatch.delenv(wrap_mod._PARENT_UPSTREAM_ENV, raising=False)
        monkeypatch.setattr(
            wrap_mod, "_query_proxy_health", lambda _p: {"service": "headroom-proxy"}
        )

        assert wrap_mod._vertex_target_api_url_from_claude_env("http://127.0.0.1:8789") is None

    def test_the_export_carries_the_vertex_upstream(self) -> None:
        """Vertex resolves outside `upstream_for_proxy`, so the export must
        fall back to it explicitly."""
        import inspect

        source = inspect.getsource(wrap_mod.claude.callback)
        assert "_export_parent_upstream(env, upstream_for_proxy or vertex_upstream)" in source


class TestReusedProxyMemoryDbIsQueried:
    """ "Same default rule" is not "same file": both fallbacks resolve
    `<cwd>/.headroom/memory.db` against their OWN cwd (round 18, P2)."""

    def test_health_reported_path_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            wrap_mod,
            "_query_proxy_health",
            lambda _p: {"config": {"memory_db_path": "/srv/app/.headroom/memory.db"}},
        )

        assert wrap_mod._proxy_memory_db_path(8787) == "/srv/app/.headroom/memory.db"

    def test_an_older_proxy_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _p: {"config": {}})
        assert wrap_mod._proxy_memory_db_path(8787) is None

        monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _p: None)
        assert wrap_mod._proxy_memory_db_path(8787) is None

    def test_a_blank_reported_path_reads_as_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            wrap_mod, "_query_proxy_health", lambda _p: {"config": {"memory_db_path": "  "}}
        )
        assert wrap_mod._proxy_memory_db_path(8787) is None

    def test_the_reported_path_is_adopted_by_alignment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        isolation.activate_isolated_workspace()
        reported = tmp_path / "elsewhere" / ".headroom" / "memory.db"

        assert isolation.align_memory_db_with_reused_proxy(str(reported)) is True

        assert Path(os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV]) == reported.resolve()


class TestReusedPortNeedsProxyIdentity:
    """An accepting socket is not proof OUR proxy is the one accepting.
    Isolated runs allocate transient nearby ports, so another project's proxy
    can take a dead session's port before self-heal runs (round 19, P2)."""

    @staticmethod
    def _owner(pid: int, *, proxy_pid: int | None = None) -> dict[str, Any]:
        """``pid`` is the WRAPPER; the listener's identity is ``proxy_pid``.

        They are always different processes — the proxy is detached — which is
        why the check compares against ``proxy_pid`` (round 23, P1). Defaults
        to a distinct value so no test can pass by conflating the two.
        """
        return {
            "pid": pid,
            "proxy_pid": pid + 1 if proxy_pid is None else proxy_pid,
            "port": 8788,
            "key": "ANTHROPIC_BASE_URL",
        }

    def test_another_headroom_proxy_on_the_port_reads_as_dead(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: 999999)

        assert wrap_mod._wrap_proxy_alive(8788, owner=self._owner(1234)) is False

    def test_our_own_proxy_reads_as_alive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: 4242)

        assert wrap_mod._wrap_proxy_alive(8788, owner=self._owner(1234, proxy_pid=4242)) is True

    def test_an_older_proxy_reporting_no_pid_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Never claim "not ours" without proof — a false negative clears a
        LIVE session's routing."""
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: None)

        assert wrap_mod._wrap_proxy_alive(8788, owner=self._owner(1234)) is True

    def test_a_non_headroom_listener_is_dead_only_once_the_owner_is_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A socket that accepts TCP but does not answer /health may still be
        our proxy mid-startup, so it only reads as foreign once the recorded
        process is provably gone."""
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", lambda _p: False)

        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)
        assert wrap_mod._wrap_proxy_alive(8788, owner=self._owner(1234)) is True

        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: True)
        assert wrap_mod._wrap_proxy_alive(8788, owner=self._owner(1234)) is False

    def test_without_an_owner_a_bare_connect_still_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Callers with no recorded identity keep the old behaviour."""
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)

        def unexpected(_port: int) -> Any:
            raise AssertionError("nothing to compare against; must not probe")

        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", unexpected)

        assert wrap_mod._wrap_proxy_alive(8788) is True

    def test_an_owner_without_a_pid_is_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)

        assert wrap_mod._wrap_proxy_alive(8788, owner={"port": 8788}) is True

    def test_the_dead_marker_check_passes_the_owner_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []

        def capture(port: int, **kwargs: Any) -> bool:
            seen.append(kwargs.get("owner"))
            return True

        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", capture)
        marker = self._owner(1234)

        wrap_mod._wrap_marker_proxy_is_dead(marker)

        assert seen == [marker], "identity was not handed to the liveness probe"


class TestLegacyOwnerHandoverRebuildsTheURL:
    """A marker written before owners recorded their exact `url` has only
    `port` + `key`. Treating that as "nothing to hand back to" leaves
    settings.local.json naming the EXITING run's proxy — which is about to
    die — while the marker names the live owner, so SessionStart self-heal
    probes that owner's live port and never repairs the dead URL
    (round 20, P2)."""

    @staticmethod
    def _legacy(pid: int, port: int, key: str) -> dict[str, Any]:
        return {"pid": pid, "port": port, "key": key, "previous": "https://user-gateway"}

    def test_a_legacy_owner_rebuilds_the_standard_url(self) -> None:
        owner = self._legacy(1234, 8788, "ANTHROPIC_BASE_URL")

        assert (
            wrap_mod._owner_handover_url(owner, key="ANTHROPIC_BASE_URL") == "http://127.0.0.1:8788"
        )

    def test_a_legacy_foundry_owner_keeps_the_anthropic_suffix(self) -> None:
        """The Anthropic SDK appends /v1/messages to this value; dropping
        /anthropic points Foundry mode at the wrong path."""
        owner = self._legacy(1234, 8788, "ANTHROPIC_FOUNDRY_BASE_URL")

        assert (
            wrap_mod._owner_handover_url(owner, key="ANTHROPIC_FOUNDRY_BASE_URL")
            == "http://127.0.0.1:8788/anthropic"
        )

    def test_a_recorded_url_still_wins(self) -> None:
        """Reconstruction is the fallback, never an override — a Foundry run
        that recorded its exact URL must not be second-guessed."""
        owner = {**self._legacy(1234, 8788, "ANTHROPIC_BASE_URL"), "url": "http://gateway:9/x"}

        assert wrap_mod._owner_handover_url(owner, key="ANTHROPIC_BASE_URL") == "http://gateway:9/x"

    def test_no_port_and_no_url_stays_inconclusive(self) -> None:
        assert wrap_mod._owner_handover_url({"pid": 1234}, key="ANTHROPIC_BASE_URL") is None

    def test_exit_hands_a_legacy_live_peer_its_reconstructed_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End-to-end on the reported path: we exit, a legacy-marker peer is
        still live, and settings must end up on the PEER's port, not ours."""
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8789"}}))
        marker = wrap_mod._wrap_marker_path(settings)
        peer = self._legacy(424242, 8788, "ANTHROPIC_BASE_URL")  # live, legacy: no "url"
        mine = {
            "pid": os.getpid(),
            "port": 8789,
            "key": "ANTHROPIC_BASE_URL",
            "previous": "https://user-gateway",
            "url": "http://127.0.0.1:8789",
        }
        marker.write_text(json.dumps({"owners": [peer, mine], **mine}))
        # The peer is live and its proxy answers; liveness itself is covered
        # by its own tests, so pin it here rather than spawn a real process.
        monkeypatch.setattr(wrap_mod, "_wrap_marker_owners", lambda _p: [peer, mine])
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda *_a, **_k: True)
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)

        wrap_mod._restore_claude_wrap_base_url(
            "https://user-gateway", settings_path=settings, _key_override="ANTHROPIC_BASE_URL"
        )

        assert json.loads(settings.read_text())["env"]["ANTHROPIC_BASE_URL"] == (
            "http://127.0.0.1:8788"
        ), "routing must follow the surviving peer, not stay on our dying proxy"

    def test_crash_handover_also_rebuilds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The same gap existed on the crash-cleanup path."""
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8789"}}))
        marker = wrap_mod._wrap_marker_path(settings)
        live = self._legacy(424242, 8788, "ANTHROPIC_BASE_URL")
        marker.write_text(json.dumps({"owners": [live], **live}))
        monkeypatch.setattr(wrap_mod, "_wrap_marker_owners", lambda _p: [live])
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda *_a, **_k: True)
        monkeypatch.setattr(wrap_mod, "_wrap_marker_is_stale", lambda _o: False)

        handover = wrap_mod._handover_to_live_owner(settings, key="ANTHROPIC_BASE_URL")

        assert handover == "http://127.0.0.1:8788"


class TestWrapperAuthorshipBeatsValueComparison:
    """Value inequality cannot tell "we wrote it" from "we inherited it": with
    an inherited ANTHROPIC_BASE_URL already on the requested port, the wrapper
    writes the SAME string, binds its dedicated proxy elsewhere, and the
    variable reads as untouched — leaving the agent on the shared or dead port
    (round 20, P2)."""

    def test_an_identical_value_we_wrote_is_still_repointed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")
        env = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}

        wrap_mod._repoint_own_endpoints(env, 8787, 8788, written={"ANTHROPIC_BASE_URL"})

        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8788"

    def test_an_identical_value_we_did_not_write_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of the rule — authorship, not the port, decides."""
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8787")
        env = {"HTTP_PROXY": "http://127.0.0.1:8787"}

        wrap_mod._repoint_own_endpoints(env, 8787, 8788, written={"ANTHROPIC_BASE_URL"})

        assert env["HTTP_PROXY"] == "http://127.0.0.1:8787"

    def test_value_inequality_still_covers_unannounced_writes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        env = {"OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"}

        wrap_mod._repoint_own_endpoints(env, 8787, 8788, written=())

        assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8788/v1"

    def test_authorship_comes_from_the_launch_banner(self) -> None:
        env = {"ANTHROPIC_BASE_URL": "x", "HTTP_PROXY": "y"}

        written = wrap_mod._wrapper_written_keys(
            ["ANTHROPIC_BASE_URL=http://127.0.0.1:8787", "OPENAI_BASE_URL=gone"], env
        )

        assert written == {"ANTHROPIC_BASE_URL"}, "only names actually present in env count"

    def test_the_banner_lists_every_endpoint_the_launcher_writes(self) -> None:
        """The banner is load-bearing now, so a var set but not listed would
        silently lose its authorship signal. Goose and OpenHands both write
        OPENAI_API_BASE."""
        import inspect

        for cmd in (wrap_mod.goose, wrap_mod.openhands):
            fn = getattr(cmd, "callback", cmd)
            source = inspect.getsource(fn)
            written = {
                line.split('env["', 1)[1].split('"]', 1)[0]
                for line in source.splitlines()
                if 'env["' in line and "] = " in line and "BASE" in line
            }
            listed = {
                part.split("=", 1)[0].strip().strip('f"')
                for part in source.split("env_vars_display = [", 1)[1].split("]", 1)[0].split(",")
                if "=" in part
            }
            assert written <= listed, f"{fn.__name__} sets endpoints it does not announce"


class TestMemoryIsAlignedBeforeAnyMemorySetup:
    """`--memory --no-proxy` reconciles onto the reused proxy's database, but
    the reconciliation lived inside `_ensure_proxy` — which both the Claude and
    Codex flows reach only AFTER importing native memories into the throwaway
    run database. The proxy (and the MCP server started after it) never saw
    those memories (round 20, P2)."""

    def test_the_claude_flow_aligns_before_it_syncs(self) -> None:
        import inspect

        source = inspect.getsource(wrap_mod.claude.callback)
        align = source.index("_align_memory_with_reused_proxy(")
        sync = source.index("_wrap_memory_db_path()")

        assert align < sync, "the database must be settled before the import runs"

    def test_the_codex_flow_aligns_before_it_imports(self) -> None:
        import inspect

        source = inspect.getsource(wrap_mod._prepare_codex_wrap_state)
        align = source.index("_align_memory_with_reused_proxy(")
        setup = source.index("_wrap_memory_db_path()")

        assert align < setup

    def test_alignment_is_a_no_op_outside_memory_no_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("must not probe the proxy")

        monkeypatch.setattr(wrap_mod, "_proxy_memory_db_path", unexpected)

        wrap_mod._align_memory_with_reused_proxy(8787, memory=False, no_proxy=True)
        wrap_mod._align_memory_with_reused_proxy(8787, memory=True, no_proxy=False)

    def test_alignment_adopts_the_database_the_proxy_reports(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_dir = isolation.activate_isolated_workspace()
        assert run_dir is not None
        theirs = tmp_path / "theirs" / "memory.db"
        monkeypatch.setattr(wrap_mod, "_proxy_memory_db_path", lambda _p: str(theirs))

        wrap_mod._align_memory_with_reused_proxy(8787, memory=True, no_proxy=True)

        assert os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV] == str(theirs)

    def test_alignment_is_idempotent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """It runs twice on every flow now (early, then from `_ensure_proxy`);
        the second call must not re-report or re-decide."""
        isolation.activate_isolated_workspace()
        theirs = tmp_path / "theirs" / "memory.db"
        monkeypatch.setattr(wrap_mod, "_proxy_memory_db_path", lambda _p: str(theirs))

        wrap_mod._align_memory_with_reused_proxy(8787, memory=True, no_proxy=True)
        monkeypatch.setattr(wrap_mod, "_proxy_memory_db_path", lambda _p: str(tmp_path / "other"))
        wrap_mod._align_memory_with_reused_proxy(8787, memory=True, no_proxy=True)

        assert os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV] == str(theirs)


class TestIsolatedMCPRegistrationIsPortAgnostic:
    """Every registrar writes into user-scoped config that all runs share, with
    one slot per server name. Baking a per-run port in there lets a concurrent
    launch overwrite it — run A's agent then starts its retrieval MCP against
    run B's workspace, and once B exits the entry names a dead port for
    everyone (round 20, P2)."""

    class _Registrar:
        name = "claude"
        display_name = "Claude Code"

        def __init__(self) -> None:
            self.spec: Any = None

        def detect(self) -> bool:
            return True

        def register_server(self, spec: Any, *, force: bool = False) -> Any:
            from headroom.mcp_registry import RegisterResult, RegisterStatus

            self.spec = spec
            return RegisterResult(RegisterStatus.ALREADY, "ok")

    def test_an_isolated_run_pins_no_port_in_the_shared_config(self) -> None:
        isolation.activate_isolated_workspace()
        registrar = self._Registrar()

        wrap_mod._setup_headroom_mcp(registrar, 8788, force=True)

        assert "HEADROOM_PROXY_URL" not in registrar.spec.env, (
            "a per-run port in shared config is what concurrent runs fight over"
        )

    def test_a_shared_run_still_pins_its_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wrap_mod, "_isolation_requested", lambda: False)
        registrar = self._Registrar()

        wrap_mod._setup_headroom_mcp(registrar, 8788, force=True)

        assert registrar.spec.env["HEADROOM_PROXY_URL"] == "http://127.0.0.1:8788"

    def test_two_isolated_runs_write_the_same_entry(self) -> None:
        """The point of dropping the pin: nothing left to clobber."""
        isolation.activate_isolated_workspace()
        first, second = self._Registrar(), self._Registrar()

        wrap_mod._setup_headroom_mcp(first, 8788, force=True)
        wrap_mod._setup_headroom_mcp(second, 8791, force=True)

        assert first.spec == second.spec

    def test_the_launcher_exports_the_real_port_instead(self) -> None:
        """The server is stdio, spawned as a child of the agent, and
        `headroom mcp serve --proxy-url` reads this from its environment — so
        this export is the only thing carrying the per-run port now."""
        import inspect

        for fn in (wrap_mod._launch_tool, wrap_mod.claude.callback):
            source = inspect.getsource(fn)
            assert 'env["HEADROOM_PROXY_URL"] = f"http://127.0.0.1:{actual_port}"' in source

    def test_headroom_mcp_serve_reads_that_variable(self) -> None:
        """Guard the other end of the contract: if the CLI stops honouring the
        env var, the isolated entry silently points every run at 8787."""
        from headroom.cli.mcp import mcp as mcp_group

        serve = mcp_group.commands["serve"]
        proxy_url = next(p for p in serve.params if p.name == "proxy_url")

        assert proxy_url.envvar == "HEADROOM_PROXY_URL"


class TestHighPortFallsBackBelowTheReservedBase:
    """`--port 65535` leaves no room above and already fell back below. The
    gap was the *almost* no room case: `--port 65534` leaves exactly one
    candidate, so a single busy port exhausted the upward window and failed the
    launch outright — even though isolation was never explicitly requested and
    plenty of ports below the base were free (round 21, P2)."""

    def test_the_upward_budget_never_exceeds_the_port_space(self) -> None:
        assert wrap_mod._dedicated_port_attempts(65534, 65535) == 1
        assert wrap_mod._dedicated_port_attempts(8787, 8788) == 100

    def test_an_exhausted_upward_window_continues_below(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probed: list[int] = []

        def busy_at_the_top(start: int, max_attempts: int = 100) -> int:
            probed.append(start)
            if start > 65534:
                raise RuntimeError(f"No available port found in range {start}-65535")
            return start

        monkeypatch.setattr(wrap_mod, "_find_available_port", busy_at_the_top)

        port = wrap_mod._find_dedicated_port(65534, 65535)

        assert port == 65534 - wrap_mod._DEDICATED_PORT_WINDOW
        assert probed == [65535, 65534 - wrap_mod._DEDICATED_PORT_WINDOW]

    def test_the_lower_window_never_hands_back_the_reserved_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The base port belongs to the shared proxy even when free."""
        budgets: list[tuple[int, int]] = []

        def record(start: int, max_attempts: int = 100) -> int:
            budgets.append((start, max_attempts))
            if start > 65534:
                raise RuntimeError("exhausted")
            return start

        monkeypatch.setattr(wrap_mod, "_find_available_port", record)
        wrap_mod._find_dedicated_port(65534, 65535)

        start, attempts = budgets[-1]
        assert start + attempts == 65534, "the probe window must stop short of the base port"

    def test_exhausting_the_lower_window_still_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falling back is not the same as never failing."""

        def always_busy(start: int, max_attempts: int = 100) -> int:
            raise RuntimeError("exhausted")

        monkeypatch.setattr(wrap_mod, "_find_available_port", always_busy)

        with pytest.raises(RuntimeError):
            wrap_mod._find_dedicated_port(65534, 65535)

    def test_an_ordinary_port_does_not_go_looking_below(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probed: list[int] = []

        def first_free(start: int, max_attempts: int = 100) -> int:
            probed.append(start)
            return start

        monkeypatch.setattr(wrap_mod, "_find_available_port", first_free)

        assert wrap_mod._find_dedicated_port(8787, 8788) == 8788
        assert probed == [8788]

    def test_the_launch_path_delegates_to_the_fallback(self) -> None:
        import inspect

        source = inspect.getsource(wrap_mod._ensure_proxy)
        assert "_find_dedicated_port(port, search_from)" in source


class TestProxyIdentityComparesTheProxyNotTheWrapper:
    """The marker's `pid` is the WRAPPER's; `/health` reports the PROXY's, and
    the proxy is a separate detached process — so comparing them could never
    match, and every live session's own proxy read as a foreign listener. The
    SessionStart self-heal hook then cleared a working base URL and cut
    daemon-spawned conversation workers off from the proxy (round 23, P1)."""

    @staticmethod
    def _marker(*, wrapper_pid: int, proxy_pid: Any) -> dict[str, Any]:
        return {
            "pid": wrapper_pid,
            "proxy_pid": proxy_pid,
            "port": 8788,
            "key": "ANTHROPIC_BASE_URL",
        }

    def test_our_own_live_proxy_is_recognised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The regression itself: wrapper 111, proxy 222, /health says 222."""
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: 222)

        assert (
            wrap_mod._wrap_proxy_alive(8788, owner=self._marker(wrapper_pid=111, proxy_pid=222))
            is True
        )

    def test_another_projects_proxy_on_a_reused_port_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The behaviour the comparison exists for must still work."""
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: 999)

        assert (
            wrap_mod._wrap_proxy_alive(8788, owner=self._marker(wrapper_pid=111, proxy_pid=222))
            is False
        )

    def test_the_wrapper_pid_is_never_what_gets_compared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the actual defect: a listener reporting the WRAPPER's pid is
        not our proxy, however plausible the number looks."""
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: 111)

        assert (
            wrap_mod._wrap_proxy_alive(8788, owner=self._marker(wrapper_pid=111, proxy_pid=222))
            is False
        )

    def test_a_legacy_marker_without_a_proxy_pid_is_inconclusive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Markers written before this field existed must not be condemned —
        clearing a live session's routing is the worse failure."""
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: 222)

        assert (
            wrap_mod._wrap_proxy_alive(8788, owner=self._marker(wrapper_pid=111, proxy_pid=None))
            is True
        )

    def test_the_marker_records_the_listeners_pid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End-to-end: what gets written must be the proxy's identity, not
        this process's — otherwise the comparison above can never succeed."""
        settings = tmp_path / "proj" / ".claude" / "settings.local.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"env": {}}))

        wrap_mod._write_claude_wrap_base_url(
            "http://127.0.0.1:8788", settings_path=settings, port=8788, proxy_pid=222
        )

        owner = json.loads(wrap_mod._wrap_marker_path(settings).read_text())["owners"][-1]
        assert owner["proxy_pid"] == 222
        assert owner["pid"] == os.getpid() != owner["proxy_pid"]

    def test_the_claude_flow_asks_the_listener_for_it(self) -> None:
        """It must come from the listener, not our own Popen: `--no-proxy`
        attaches to a proxy this wrapper never started."""
        import inspect

        source = inspect.getsource(wrap_mod.claude.callback)
        assert "proxy_pid=_proxy_reported_pid(actual_port)" in source


class TestOwnershipSurvivesMultipleWorkers:
    """`proc.pid` is uvicorn's PARENT; /health answers `os.getpid()` from
    whichever worker took the request. With HEADROOM_WORKERS>1 those never
    match, so the launcher declared its own healthy proxy foreign, killed it,
    and retried until every isolated wrap failed to start (round 26, P2)."""

    def test_a_worker_reporting_its_own_pid_is_still_our_launch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wrap_mod, "_proxy_reported_instance", lambda _p: "abc123")
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: 777)  # a worker

        assert wrap_mod._listener_is_our_launch(8788, "abc123", 4242) is True

    def test_a_different_server_is_still_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The race detection the check exists for must survive the fix."""
        monkeypatch.setattr(wrap_mod, "_proxy_reported_instance", lambda _p: "someone-else")

        assert wrap_mod._listener_is_our_launch(8788, "abc123", 4242) is False

    def test_an_older_proxy_falls_back_to_the_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wrap_mod, "_proxy_reported_instance", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: 4242)

        assert wrap_mod._listener_is_our_launch(8788, "abc123", 4242) is True

    def test_a_silent_health_endpoint_is_inconclusive_not_foreign(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A proxy still starting up must not read as a lost race."""
        monkeypatch.setattr(wrap_mod, "_proxy_reported_instance", lambda _p: None)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: None)

        assert wrap_mod._listener_is_our_launch(8788, "abc123", 4242) is None

    def test_the_launcher_exports_the_instance_to_the_proxy(self) -> None:
        import inspect

        source = inspect.getsource(wrap_mod._start_proxy)
        assert "proxy_env[_SERVER_INSTANCE_ENV] = instance_id" in source
        assert "_listener_is_our_launch(port, instance_id, proc.pid)" in source

    def test_the_marker_prefers_the_instance_over_the_pid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same defect on the self-heal path: a marker holding one worker's pid
        would not match the next worker to answer."""
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_instance", lambda _p: "srv-1")
        monkeypatch.setattr(wrap_mod, "_proxy_reported_pid", lambda _p: 999999)
        owner = {
            "pid": 111,
            "proxy_pid": 222,  # a different worker than the one answering now
            "proxy_instance": "srv-1",
            "port": 8788,
            "key": "ANTHROPIC_BASE_URL",
        }

        assert wrap_mod._wrap_proxy_alive(8788, owner=owner) is True

    def test_a_marker_from_another_server_still_reads_as_dead(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_is_local_headroom_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_reported_instance", lambda _p: "srv-2")
        owner = {
            "pid": 111,
            "proxy_pid": 222,
            "proxy_instance": "srv-1",
            "port": 8788,
            "key": "ANTHROPIC_BASE_URL",
        }

        assert wrap_mod._wrap_proxy_alive(8788, owner=owner) is False


class TestALocalGatewayOnTheReservedPortSurvives:
    """A dedicated run reserves the requested port and binds elsewhere, so a
    genuine gateway there (LiteLLM on 127.0.0.1:8787) is a perfectly good
    upstream. Treating an equal-port URL as self-referential discarded it and
    sent the session to api.anthropic.com instead (round 27, P1)."""

    def test_a_gateway_on_the_reserved_port_becomes_the_upstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")
        monkeypatch.setattr(wrap_mod, "_url_is_local_headroom_proxy", lambda _u: False)

        upstream = wrap_mod._detect_inbound_anthropic_upstream(8787, binds_port=False)

        assert upstream == "http://127.0.0.1:8787"

    def test_another_headroom_proxy_there_is_still_not_chained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The identity probe is what distinguishes the two, and it must still
        refuse to stack two Headroom pipelines."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")
        monkeypatch.setattr(wrap_mod, "_url_is_local_headroom_proxy", lambda _u: True)
        monkeypatch.setattr(wrap_mod, "_inherited_parent_upstream", lambda: None)

        assert wrap_mod._detect_inbound_anthropic_upstream(8787, binds_port=False) is None

    def test_a_run_that_binds_the_port_still_treats_it_as_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--shared` and `--no-proxy` really do use that listener, so an
        equal-port URL there would be a forwarding loop."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")
        monkeypatch.setattr(wrap_mod, "_inherited_parent_upstream", lambda: None)

        def unexpected(_url: str) -> bool:
            raise AssertionError("must not probe: this run owns that port")

        monkeypatch.setattr(wrap_mod, "_url_is_local_headroom_proxy", unexpected)

        assert wrap_mod._detect_inbound_anthropic_upstream(8787, binds_port=True) is None

    def test_the_claude_flow_passes_binds_port_from_the_run_shape(self) -> None:
        import inspect

        source = inspect.getsource(wrap_mod.claude.callback)
        assert "binds_port=no_proxy or not _isolation_requested()" in source


class TestADedicatedRunDoesNotClaimTheSharedPort:
    """`_launch_tool` registered as a client of the REQUESTED port before
    `_ensure_proxy` started a dedicated proxy elsewhere. The last shared
    wrapper exiting in that window saw an attached client, left its detached
    proxy running, and once the marker moved nobody was responsible for
    stopping it (round 27, P2)."""

    def test_an_isolated_run_does_not_claim_the_requested_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        claimed: list[int] = []
        monkeypatch.setattr(wrap_mod, "_register_proxy_client", lambda p: claimed.append(p))
        isolation.activate_isolated_workspace()
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "1")

        assert wrap_mod._register_launch_proxy_client(8787, no_proxy=False) is False
        assert claimed == []

    def test_no_proxy_still_claims_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--no-proxy` genuinely reuses that proxy and must keep it alive."""
        claimed: list[int] = []
        monkeypatch.setattr(wrap_mod, "_register_proxy_client", lambda p: claimed.append(p))
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "1")

        assert wrap_mod._register_launch_proxy_client(8787, no_proxy=True) is True
        assert claimed == [8787]

    def test_a_shared_run_still_claims_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        claimed: list[int] = []
        monkeypatch.setattr(wrap_mod, "_register_proxy_client", lambda p: claimed.append(p))
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "0")

        assert wrap_mod._register_launch_proxy_client(8787, no_proxy=False) is True
        assert claimed == [8787]

    def test_the_launch_paths_use_the_gated_registration(self) -> None:
        import inspect

        for fn in (wrap_mod._launch_tool, wrap_mod.claude.callback):
            source = inspect.getsource(fn)
            assert "_register_launch_proxy_client(port, no_proxy)" in source
            assert "\n        _register_proxy_client(port)\n" not in source

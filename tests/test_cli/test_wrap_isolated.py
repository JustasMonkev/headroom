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
    def test_launch_tool_registers_sighup(self) -> None:
        """Closing the terminal sends SIGHUP; without a handler the wrapper
        dies and its detached dedicated proxy survives forever on the per-run
        port (round 3, P2). claude() already does this."""
        import inspect

        src = inspect.getsource(wrap_mod._launch_tool)
        assert 'hasattr(signal, "SIGHUP")' in src
        assert "signal.signal(signal.SIGHUP, cleanup)" in src


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
        monkeypatch.setattr(wrap_mod, "_wrap_proxy_alive", lambda _p: False)

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

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
        isolation.HEADROOM_ISOLATED_ENV,
        isolation.HEADROOM_ISOLATED_WORKSPACE_ENV,
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

        assert "--isolated has no effect with --no-proxy" in output


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

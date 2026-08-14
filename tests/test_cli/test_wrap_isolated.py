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


class TestWrapGroupFlag:
    def test_isolated_flag_activates_per_run_workspace(self, tmp_path: Path) -> None:
        """`headroom wrap --isolated <tool>` activates a fresh workspace
        before the subcommand body runs."""

        seen: dict[str, Any] = {}

        @click.command("fake-tool")
        def fake_tool() -> None:
            seen["workspace"] = paths.workspace_dir()

        wrap_mod.wrap.add_command(fake_tool)
        try:
            runner = CliRunner()
            result = runner.invoke(main, ["wrap", "--isolated", "fake-tool"])
        finally:
            wrap_mod.wrap.commands.pop("fake-tool", None)

        assert result.exit_code == 0, result.output
        assert "Isolated run: workspace" in result.output
        assert seen["workspace"].parent == tmp_path / "ws" / "runs"
        assert seen["workspace"].name.startswith("run-")

    def test_env_var_activates_isolation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, "1")
        seen: dict[str, Any] = {}

        @click.command("fake-tool")
        def fake_tool() -> None:
            seen["workspace"] = paths.workspace_dir()

        wrap_mod.wrap.add_command(fake_tool)
        try:
            runner = CliRunner()
            result = runner.invoke(main, ["wrap", "fake-tool"])
        finally:
            wrap_mod.wrap.commands.pop("fake-tool", None)

        assert result.exit_code == 0, result.output
        assert seen["workspace"].parent == tmp_path / "ws" / "runs"

    def test_without_flag_workspace_is_shared(self, tmp_path: Path) -> None:
        seen: dict[str, Any] = {}

        @click.command("fake-tool")
        def fake_tool() -> None:
            seen["workspace"] = paths.workspace_dir()

        wrap_mod.wrap.add_command(fake_tool)
        try:
            runner = CliRunner()
            result = runner.invoke(main, ["wrap", "fake-tool"])
        finally:
            wrap_mod.wrap.commands.pop("fake-tool", None)

        assert result.exit_code == 0, result.output
        assert seen["workspace"] == tmp_path / "ws"

    def test_misplaced_isolated_flag_is_rejected(self) -> None:
        """`headroom wrap claude --isolated` must fail loudly instead of
        forwarding --isolated to the wrapped CLI (ignore_unknown_options)."""

        runner = CliRunner()
        result = runner.invoke(main, ["wrap", "claude", "--isolated"])

        assert result.exit_code != 0
        assert "goes before the tool name" in result.output
        assert "headroom wrap --isolated claude" in result.output

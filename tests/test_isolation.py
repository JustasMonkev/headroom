"""Tests for per-run isolation (headroom.isolation).

Two concurrent ``headroom wrap`` runs share ``~/.headroom`` and the proxy
port by default. Isolated mode gives each run its own workspace and a
dedicated proxy; these tests pin the workspace half of that contract.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from headroom import isolation, paths

_MUTATED_VARS = (
    paths.HEADROOM_CONFIG_DIR_ENV,
    isolation.HEADROOM_ISOLATED_ENV,
    isolation.HEADROOM_ISOLATED_WORKSPACE_ENV,
)


@pytest.fixture(autouse=True)
def _clean_isolation_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Route the workspace to tmp and clear all isolation env state.

    ``activate_isolated_workspace`` writes to ``os.environ`` directly, and
    ``monkeypatch.delenv(raising=False)`` records nothing for an absent
    variable — so the explicit pops after the yield keep activations made
    inside a test from leaking into the rest of the suite (monkeypatch then
    restores any pre-test values on teardown, after the pops).
    """

    monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(tmp_path / "ws"))
    for var in _MUTATED_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in _MUTATED_VARS:
        os.environ.pop(var, None)


class TestIsolationRequested:
    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, value)
        assert isolation.isolation_requested() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "banana"])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(isolation.HEADROOM_ISOLATED_ENV, value)
        assert isolation.isolation_requested() is False

    def test_unset(self) -> None:
        assert isolation.isolation_requested() is False


class TestActivateIsolatedWorkspace:
    def test_creates_unique_run_dir_under_runs(self, tmp_path: Path) -> None:
        run_dir = isolation.activate_isolated_workspace()

        assert run_dir.is_dir()
        assert run_dir.parent == tmp_path / "ws" / "runs"
        assert run_dir.name.startswith("run-")
        # The canonical workspace root now resolves to the per-run dir, so
        # every paths.py helper (memory DB, savings, logs, ...) is isolated.
        assert paths.workspace_dir() == run_dir
        assert paths.memory_db_path() == run_dir / "memory.db"

    def test_config_dir_stays_shared(self, tmp_path: Path) -> None:
        """Read-mostly config must keep resolving to the pre-activation root,
        not to an empty per-run config copy."""

        shared_config = paths.config_dir()
        assert shared_config == tmp_path / "ws" / "config"

        isolation.activate_isolated_workspace()

        assert paths.config_dir() == shared_config

    def test_explicit_config_dir_wins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(paths.HEADROOM_CONFIG_DIR_ENV, str(tmp_path / "cfg"))

        isolation.activate_isolated_workspace()

        assert paths.config_dir() == tmp_path / "cfg"

    def test_exports_isolation_markers(self) -> None:
        run_dir = isolation.activate_isolated_workspace()

        assert isolation.isolation_requested() is True
        assert isolation.active_isolated_workspace() == run_dir

    def test_activation_is_idempotent(self) -> None:
        first = isolation.activate_isolated_workspace()
        second = isolation.activate_isolated_workspace()

        assert first == second
        assert paths.workspace_dir() == first

    def test_two_activations_in_distinct_processes_get_distinct_dirs(self) -> None:
        """Simulate two concurrent runs: each activation (with the marker
        cleared, as in a separate process) must land in its own directory."""

        first = isolation.activate_isolated_workspace(run_id="a")
        # A second process would not inherit the first one's marker.
        os.environ.pop(isolation.HEADROOM_ISOLATED_WORKSPACE_ENV, None)
        os.environ[paths.HEADROOM_WORKSPACE_DIR_ENV] = str(first.parent.parent)
        second = isolation.activate_isolated_workspace(run_id="b")

        assert first != second
        assert first.parent == second.parent

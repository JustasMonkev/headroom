"""Tests for per-run isolation (headroom.isolation).

Two concurrent ``headroom wrap`` runs share ``~/.headroom`` and the proxy
port by default. Isolated mode gives each run its own workspace and a
dedicated proxy; these tests pin the workspace half of that contract.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from headroom import isolation, paths

_MUTATED_VARS = (
    paths.HEADROOM_CONFIG_DIR_ENV,
    paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV,
    isolation.HEADROOM_ISOLATED_ENV,
    isolation.HEADROOM_ISOLATED_WORKSPACE_ENV,
    isolation.HEADROOM_MEMORY_DB_PATH_ENV,
    isolation.HEADROOM_PREISOLATION_WORKSPACE_ENV,
    paths.HEADROOM_SETTINGS_PATH_ENV,
    "CODEX_HOME",
    "GROK_HOME",
    "PI_CODING_AGENT_DIR",
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


def _make_stale(path: Path) -> None:
    stale = 1_000_000_000.0  # 2001 — far past any GC cutoff
    os.utime(path, (stale, stale))


class TestPruneStaleRuns:
    def test_removes_only_stale_run_dirs(self, tmp_path: Path) -> None:
        runs = tmp_path / "ws" / "runs"
        old = runs / "run-old"
        old.mkdir(parents=True)
        fresh = runs / "run-fresh"
        fresh.mkdir()
        unrelated = runs / "keep-me"
        unrelated.mkdir()
        _make_stale(old)
        _make_stale(unrelated)

        isolation.prune_stale_runs(runs)

        assert not old.exists()
        assert fresh.exists()
        # Only run-* dirs are eligible; anything else is never touched.
        assert unrelated.exists()

    def test_recent_child_activity_protects_stale_dir_mtime(self, tmp_path: Path) -> None:
        """A long-lived run whose root dir mtime is old but whose memory.db
        was written recently must survive GC (file writes don't bump the
        parent dir's mtime)."""

        runs = tmp_path / "ws" / "runs"
        live = runs / "run-live"
        live.mkdir(parents=True)
        (live / "memory.db").write_text("x")
        _make_stale(live)  # dir mtime stale, child mtime fresh

        isolation.prune_stale_runs(runs)

        assert live.exists()

    def test_missing_runs_root_is_a_noop(self, tmp_path: Path) -> None:
        isolation.prune_stale_runs(tmp_path / "does-not-exist")

    def test_activation_garbage_collects_stale_runs(self, tmp_path: Path) -> None:
        runs = tmp_path / "ws" / "runs"
        old = runs / "run-old"
        old.mkdir(parents=True)
        _make_stale(old)

        run_dir = isolation.activate_isolated_workspace()

        assert not old.exists()
        assert run_dir.exists()


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


class TestSharedWorkspacePinning:
    """Persistent, cross-run resources must resolve against the pre-isolation
    workspace, not the ephemeral per-run directory (PR #25 review, P1/P2)."""

    def test_activation_records_shared_root(self, tmp_path: Path) -> None:
        pre = paths.workspace_dir()
        run_dir = isolation.activate_isolated_workspace()

        assert run_dir != pre
        assert paths.workspace_dir() == run_dir
        # Shared root pinned to the pre-isolation workspace.
        assert os.environ[paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV] == str(pre)
        assert paths.shared_workspace_dir() == pre

    def test_managed_binaries_stay_on_shared_root(self, tmp_path: Path) -> None:
        pre = paths.workspace_dir()
        isolation.activate_isolated_workspace()

        # rtk/lean-ctx binary dir must not move into the run dir.
        assert paths.bin_dir() == pre / "bin"
        assert paths.rtk_path().parent == pre / "bin"
        assert paths.license_cache_path() == pre / "license_cache.json"

    def test_copilot_auth_stays_on_shared_root(self, tmp_path: Path) -> None:
        from headroom.copilot_auth import headroom_copilot_auth_path

        pre = paths.workspace_dir()
        isolation.activate_isolated_workspace()

        assert headroom_copilot_auth_path() == pre / "copilot_auth.json"

    def test_mcp_ledger_stays_on_shared_root(self, tmp_path: Path) -> None:
        from headroom.mcp_registry.ledger import ledger_path

        pre = paths.workspace_dir()
        isolation.activate_isolated_workspace()

        assert ledger_path() == pre / "mcp_installs.json"

    def test_shared_root_falls_back_when_unset(self, tmp_path: Path) -> None:
        # No isolation active: shared root IS the workspace root.
        assert paths.shared_workspace_dir() == paths.workspace_dir()


class TestMemoryIsolation:
    def test_activation_pins_memory_db_into_run_dir(self, tmp_path: Path) -> None:
        run_dir = isolation.activate_isolated_workspace()

        assert os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV] == str(run_dir / "memory.db")

    def test_user_memory_db_override_is_respected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(isolation.HEADROOM_MEMORY_DB_PATH_ENV, str(tmp_path / "mine.db"))

        isolation.activate_isolated_workspace()

        # setdefault: user's explicit DB path must win.
        assert os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV] == str(tmp_path / "mine.db")

    def test_two_runs_get_distinct_memory_dbs(self, tmp_path: Path) -> None:
        first = isolation.activate_isolated_workspace(run_id="a")
        first_db = os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV]

        # Simulate a second process (no inherited markers).
        for var in (
            isolation.HEADROOM_ISOLATED_WORKSPACE_ENV,
            isolation.HEADROOM_MEMORY_DB_PATH_ENV,
            paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV,
        ):
            os.environ.pop(var, None)
        os.environ[paths.HEADROOM_WORKSPACE_DIR_ENV] = str(first.parent.parent)

        isolation.activate_isolated_workspace(run_id="b")
        second_db = os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV]

        assert first_db != second_db


class TestDisableIsolationRestore:
    def test_nested_shared_restores_workspace(self, tmp_path: Path) -> None:
        """A --shared run launched from an isolated parent must fall back to
        the shared workspace, not keep writing the parent's run dir."""

        pre = paths.workspace_dir()
        run_dir = isolation.activate_isolated_workspace()
        assert paths.workspace_dir() == run_dir  # now isolated

        isolation.disable_isolation()

        assert isolation.isolation_requested() is False
        assert paths.workspace_dir() == pre
        assert isolation.active_isolated_workspace() is None
        # The isolation-set memory path is dropped so memory resolves shared.
        assert isolation.HEADROOM_MEMORY_DB_PATH_ENV not in os.environ

    def test_disable_preserves_user_memory_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(isolation.HEADROOM_MEMORY_DB_PATH_ENV, str(tmp_path / "mine.db"))
        isolation.activate_isolated_workspace()

        isolation.disable_isolation()

        # User's explicit DB path survives the opt-out.
        assert os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV] == str(tmp_path / "mine.db")

    def test_top_level_shared_does_not_clobber_workspace(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # No parent isolation: disable_isolation must not touch the workspace.
        explicit = tmp_path / "explicit-ws"
        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(explicit))

        isolation.disable_isolation()

        assert paths.workspace_dir() == explicit


class TestPrunePidLiveness:
    @staticmethod
    def _make_stale(path: Path) -> None:
        stale = 1_000_000_000.0
        os.utime(path, (stale, stale))

    def test_owner_pid_parsed_from_name(self, tmp_path: Path) -> None:
        d = tmp_path / "run-20260814-061101-17840-4988a4"
        assert isolation._run_dir_owner_pid(d) == 17840
        assert isolation._run_dir_owner_pid(tmp_path / "keep-me") is None
        assert isolation._run_dir_owner_pid(tmp_path / "run-garbage") is None

    def test_live_owner_dir_is_never_pruned(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        # Embed THIS process's PID (guaranteed alive) in the run name.
        live = runs / f"run-20200101-000000-{os.getpid()}-abcdef"
        live.mkdir(parents=True)
        self._make_stale(live)

        isolation.prune_stale_runs(runs)

        assert live.exists()

    def test_dead_owner_stale_dir_is_pruned(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        # PID 2 is init-adjacent and effectively never a live wrap owner;
        # use a clearly-dead high PID sentinel instead.
        dead = runs / "run-20200101-000000-2147480000-abcdef"
        dead.mkdir(parents=True)
        self._make_stale(dead)

        isolation.prune_stale_runs(runs)

        assert not dead.exists()

    def test_recent_dead_owner_dir_is_kept(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        recent = runs / "run-20200101-000000-2147480000-abcdef"
        recent.mkdir(parents=True)
        # Fresh mtime → below the age cutoff → kept regardless of PID.

        isolation.prune_stale_runs(runs)

        assert recent.exists()


class TestPruneRespectsDetachedProxy:
    """A dedicated proxy is spawned detached, so it outlives its wrapper.

    GC keyed only on the wrapper PID baked into the run-dir name would delete
    a still-serving proxy's workspace — its DBs, caches and logs — once the
    age cutoff passed. ``record_run_proxy`` pins the dir to the proxy too.
    """

    @staticmethod
    def _make_stale(path: Path) -> None:
        stale = 1_000_000_000.0
        for target in (path, *path.rglob("*")):
            os.utime(target, (stale, stale))

    def test_live_proxy_pins_a_dir_whose_wrapper_is_gone(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        # Wrapper PID is a dead sentinel; the recorded proxy PID is this
        # process, i.e. provably alive.
        run = runs / "run-20200101-000000-2147480000-abcdef"
        run.mkdir(parents=True)
        isolation.record_run_proxy(os.getpid(), 8788, run_dir=run)
        self._make_stale(run)

        isolation.prune_stale_runs(runs)

        assert run.exists()

    def test_dead_proxy_and_dead_wrapper_is_pruned(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        run = runs / "run-20200101-000000-2147480000-abcdef"
        run.mkdir(parents=True)
        isolation.record_run_proxy(2147480001, 8788, run_dir=run)
        self._make_stale(run)

        isolation.prune_stale_runs(runs)

        assert not run.exists()

    def test_record_writes_pid_and_port(self, tmp_path: Path) -> None:
        run = tmp_path / "run-20200101-000000-1-abcdef"
        run.mkdir()
        isolation.record_run_proxy(4321, 8791, run_dir=run)

        record = json.loads((run / isolation._PROXY_STATE_FILE).read_text())
        # Newest entry mirrored at the top level for older readers, with the
        # full list alongside (a nested wrap adds a second proxy).
        assert record["pid"] == 4321
        assert record["port"] == 8791
        assert record["proxies"] == [{"pid": 4321, "port": 8791}]
        assert isolation._run_dir_proxy_pid(run) == 4321

    def test_record_targets_the_active_run_when_not_given(self, tmp_path: Path) -> None:
        run_dir = isolation.activate_isolated_workspace()
        isolation.record_run_proxy(4321, 8792)

        assert isolation._run_dir_proxy_pid(run_dir) == 4321

    def test_record_is_a_noop_outside_isolation(self, tmp_path: Path) -> None:
        # No active isolated workspace → nothing to pin, and no crash.
        isolation.record_run_proxy(4321, 8793)

        assert isolation.active_isolated_workspace() is None

    def test_missing_or_corrupt_record_reads_as_absent(self, tmp_path: Path) -> None:
        run = tmp_path / "run-20200101-000000-1-abcdef"
        run.mkdir()
        assert isolation._run_dir_proxy_pid(run) is None

        (run / isolation._PROXY_STATE_FILE).write_text("{not json")
        assert isolation._run_dir_proxy_pid(run) is None

        (run / isolation._PROXY_STATE_FILE).write_text('["not", "a", "dict"]')
        assert isolation._run_dir_proxy_pid(run) is None

        (run / isolation._PROXY_STATE_FILE).write_text('{"pid": "nope"}')
        assert isolation._run_dir_proxy_pid(run) is None


class TestProxyClientMarkersShared:
    """Client markers reference-count a machine-wide proxy instance, so every
    client of a given port must register in ONE directory (PR #25 review P2)."""

    def test_clients_dir_follows_shared_root(self, tmp_path: Path) -> None:
        pre = paths.workspace_dir()
        isolation.activate_isolated_workspace()

        # An isolated --no-proxy run attaches to the SHARED proxy on 8787; its
        # marker must be visible to the shared-mode wrapper that owns it.
        assert paths.proxy_clients_dir(8787) == pre / "clients" / "8787"

    def test_dedicated_port_still_gets_its_own_dir(self, tmp_path: Path) -> None:
        pre = paths.workspace_dir()
        isolation.activate_isolated_workspace()

        # Keyed by port, so a dedicated proxy never collides with the shared one.
        assert paths.proxy_clients_dir(8788) == pre / "clients" / "8788"
        assert paths.proxy_clients_dir(8788) != paths.proxy_clients_dir(8787)


class TestBlankEnvOverridesTreatedAsUnset:
    """`paths._env` treats a blank/whitespace override as unset, so isolation
    must pin those roots rather than let `setdefault` see a key and skip
    (PR #25 review round 3, P1/P2)."""

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_config_dir_is_still_pinned(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, blank: str
    ) -> None:
        monkeypatch.setenv(paths.HEADROOM_CONFIG_DIR_ENV, blank)
        pre_config = paths.config_dir()

        isolation.activate_isolated_workspace()

        # Config must NOT follow the workspace into the empty run dir.
        assert paths.config_dir() == pre_config
        assert paths.config_dir() != paths.workspace_dir() / "config"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_shared_root_is_still_pinned(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, blank: str
    ) -> None:
        monkeypatch.setenv(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, blank)
        pre = paths.workspace_dir()

        isolation.activate_isolated_workspace()

        assert paths.shared_workspace_dir() == pre
        # ...so managed binaries / auth / ledger stay off the run dir.
        assert paths.bin_dir() == pre / "bin"

    @pytest.mark.parametrize("blank", ["", "  "])
    def test_blank_memory_db_path_is_still_pinned(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, blank: str
    ) -> None:
        monkeypatch.setenv(isolation.HEADROOM_MEMORY_DB_PATH_ENV, blank)

        run_dir = isolation.activate_isolated_workspace()

        # A blank value is not a user override; memory must still isolate.
        assert os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV] == str(run_dir / "memory.db")


class TestSettingsAndSharedCaches:
    """Persistent, machine-wide state must not land in a run dir (round 3)."""

    def test_dashboard_settings_stay_shared(self, tmp_path: Path) -> None:
        pre_settings = paths.settings_path()

        isolation.activate_isolated_workspace()

        assert paths.settings_path() == pre_settings
        assert paths.settings_path().parent != paths.workspace_dir()

    def test_settings_path_pinned_in_env_for_children(self, tmp_path: Path) -> None:
        pre_settings = paths.settings_path()

        isolation.activate_isolated_workspace()

        # The proxy subprocess resolves it from the environment.
        assert os.environ[paths.HEADROOM_SETTINGS_PATH_ENV] == str(pre_settings)

    def test_update_check_cache_stays_shared(self, tmp_path: Path) -> None:
        from headroom.update_check import _cache_path

        pre = _cache_path()
        isolation.activate_isolated_workspace()

        assert _cache_path() == pre

    def test_legacy_models_json_stays_shared(self, tmp_path: Path) -> None:
        """The supported legacy ~/.headroom/models.json fallback must keep
        resolving against the shared root, or a user's custom context limits
        and pricing silently vanish under the isolated default."""
        pre = paths.workspace_dir()
        isolation.activate_isolated_workspace()

        assert paths.shared_workspace_dir() / "models.json" == pre / "models.json"


class TestDisableIsolationScope:
    def test_top_level_shared_keeps_two_distinct_configured_roots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A user may configure workspace and shared roots as different
        directories; `--shared` with no isolated parent must not collapse
        the former onto the latter (round 3, P2)."""
        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(tmp_path / "run-data"))
        monkeypatch.setenv(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, str(tmp_path / "persistent"))

        isolation.disable_isolation()

        assert paths.workspace_dir() == tmp_path / "run-data"
        assert paths.shared_workspace_dir() == tmp_path / "persistent"

    def test_nested_shared_still_restores(self, tmp_path: Path) -> None:
        """The nested opt-out path (an isolated parent) must still restore."""
        pre = paths.workspace_dir()
        isolation.activate_isolated_workspace()

        isolation.disable_isolation()

        assert paths.workspace_dir() == pre


class TestPathsPublicApi:
    def test_shared_workspace_env_const_is_exported(self) -> None:
        """`from headroom.paths import *` must expose the third canonical root
        constant alongside the other two (round 4, P2)."""
        for name in (
            "HEADROOM_CONFIG_DIR_ENV",
            "HEADROOM_WORKSPACE_DIR_ENV",
            "HEADROOM_SHARED_WORKSPACE_DIR_ENV",
        ):
            assert name in paths.__all__, f"{name} missing from paths.__all__"

        # The contract that actually matters: a star-import sees it.
        namespace: dict[str, object] = {}
        exec("from headroom.paths import *", namespace)  # noqa: S102
        assert namespace["HEADROOM_SHARED_WORKSPACE_DIR_ENV"] == "HEADROOM_SHARED_WORKSPACE_DIR"
        assert "shared_workspace_dir" in namespace


class TestPreIsolationWorkspaceRestore:
    """`--shared` must restore the workspace isolation actually took over,
    not assume it equals the shared-resource root (round 5, P2)."""

    def test_distinct_roots_restore_to_the_workspace_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(tmp_path / "run-data"))
        monkeypatch.setenv(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, str(tmp_path / "persistent"))

        isolation.activate_isolated_workspace(run_id="x")
        assert os.environ[isolation.HEADROOM_PREISOLATION_WORKSPACE_ENV] == str(
            tmp_path / "run-data"
        )

        isolation.disable_isolation()

        # Ordinary workspace state goes back to run-data, NOT the persistent bucket.
        assert paths.workspace_dir() == tmp_path / "run-data"
        assert paths.shared_workspace_dir() == tmp_path / "persistent"
        assert isolation.HEADROOM_PREISOLATION_WORKSPACE_ENV not in os.environ

    def test_identical_roots_still_restore(self, tmp_path: Path) -> None:
        pre = paths.workspace_dir()
        isolation.activate_isolated_workspace()

        isolation.disable_isolation()

        assert paths.workspace_dir() == pre


class TestSettingsSaveCreatesItsParent:
    def test_save_into_a_missing_shared_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Relocating settings to the shared root means its parent may not
        exist; `mkstemp(dir=parent)` would fail outright (round 5, P2)."""
        from headroom import settings_store

        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(tmp_path / "ws"))
        monkeypatch.setenv(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, str(tmp_path / "not-yet"))
        assert not (tmp_path / "not-yet").exists()

        settings_store.save({})

        assert paths.settings_path() == tmp_path / "not-yet" / "settings.json"
        assert paths.settings_path().exists()


class TestGcVerifiesProcessIdentity:
    """PID liveness alone is not proof an owner survives (round 11, P2).

    A PID is recycled freely long before the 7-day cutoff, so an unrelated
    long-lived process inheriting a dead wrapper's or proxy's number would pin
    its run dir forever and let ``runs/`` grow without bound.
    """

    @staticmethod
    def _make_stale(path: Path) -> None:
        stale = 1_000_000_000.0
        for target in (path, *path.rglob("*")):
            os.utime(target, (stale, stale))

    def test_recycled_wrapper_pid_does_not_pin_the_dir(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        # Name carries OUR pid (alive), but the recorded identity is a start
        # time no live process can have — i.e. the number was recycled.
        run = runs / f"run-20200101-000000-{os.getpid()}-abcdef"
        run.mkdir(parents=True)
        (run / isolation._OWNER_STATE_FILE).write_text(
            json.dumps({"pid": os.getpid(), "start_src": "proc", "start_time": 1.0})
        )
        self._make_stale(run)

        isolation.prune_stale_runs(runs)

        assert not run.exists()

    def test_matching_wrapper_identity_still_pins_the_dir(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        run = runs / f"run-20200101-000000-{os.getpid()}-abcdef"
        run.mkdir(parents=True)
        isolation.record_run_owner(run)
        self._make_stale(run)

        isolation.prune_stale_runs(runs)

        assert run.exists()

    def test_recycled_proxy_pid_does_not_pin_the_dir(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        run = runs / "run-20200101-000000-2147480000-abcdef"
        run.mkdir(parents=True)
        (run / isolation._PROXY_STATE_FILE).write_text(
            json.dumps({"pid": os.getpid(), "port": 8788, "start_src": "proc", "start_time": 1.0})
        )
        self._make_stale(run)

        isolation.prune_stale_runs(runs)

        assert not run.exists()

    def test_legacy_record_without_identity_falls_back_to_liveness(self, tmp_path: Path) -> None:
        """A run dir written before identities were recorded (or on a platform
        that cannot report start times) must keep the old behavior, not become
        newly prunable while its owner is alive."""
        runs = tmp_path / "runs"
        run = runs / f"run-20200101-000000-{os.getpid()}-abcdef"
        run.mkdir(parents=True)
        self._make_stale(run)

        isolation.prune_stale_runs(runs)

        assert run.exists()

    def test_activation_stamps_the_owner_identity(self, tmp_path: Path) -> None:
        run_dir = isolation.activate_isolated_workspace()

        record = json.loads((run_dir / isolation._OWNER_STATE_FILE).read_text())
        assert record["pid"] == os.getpid()
        # start_src/start_time are absent only where the platform cannot
        # report them; on this runner they must round-trip as a live match.
        assert isolation._owner_is_live(os.getpid(), record) is True

    def test_identity_fields_are_omitted_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(isolation, "proc_identity", lambda _pid: None)

        assert isolation._identity_fields(1234) == {}
        # ...and a record without them still reads as live.
        assert isolation._owner_is_live(os.getpid(), {"pid": os.getpid()}) is True

    def test_dead_pid_is_never_live_regardless_of_identity(self) -> None:
        assert isolation._owner_is_live(2147480000, None) is False
        assert isolation._owner_is_live(None, None) is False


class TestIsolatedWorkspaceIsAbsolute:
    """Every exported path is inherited by subprocesses that resolve it
    against THEIR cwd (round 13, P2).

    With a relatively-configured HEADROOM_WORKSPACE_DIR, a nested Headroom
    command run from another directory would open a different workspace and
    memory.db than the proxy — silently defeating process-tree isolation.
    """

    def test_relative_workspace_still_exports_absolute_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, "relative-ws")

        run_dir = isolation.activate_isolated_workspace()

        assert run_dir.is_absolute()
        for var in (
            paths.HEADROOM_WORKSPACE_DIR_ENV,
            isolation.HEADROOM_ISOLATED_WORKSPACE_ENV,
            isolation.HEADROOM_MEMORY_DB_PATH_ENV,
        ):
            assert Path(os.environ[var]).is_absolute(), f"{var} must not be cwd-relative"

    def test_a_child_in_another_cwd_resolves_the_same_workspace(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The actual failure mode: resolve the exported value from a
        different working directory and it must name the same run dir."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, "relative-ws")
        run_dir = isolation.activate_isolated_workspace()

        elsewhere = tmp_path / "some" / "other" / "dir"
        elsewhere.mkdir(parents=True)
        monkeypatch.chdir(elsewhere)

        assert paths.workspace_dir().resolve() == run_dir
        assert paths.memory_db_path().resolve() == run_dir / "memory.db"

    def test_absolute_workspace_is_unaffected(self, tmp_path: Path) -> None:
        run_dir = isolation.activate_isolated_workspace()

        assert run_dir.is_absolute()
        assert run_dir.parent == (tmp_path / "ws" / "runs").resolve()


class TestPinnedRootsAreAbsolute:
    """Not just the run dir — every pin is inherited by subprocesses that may
    run from another cwd (round 14, P2).

    A relative HEADROOM_CONFIG_DIR / HEADROOM_SHARED_WORKSPACE_DIR would make a
    nested command miss the intended model config and redirect settings,
    managed binaries, the MCP ledger and marker locks into a second tree.
    """

    def test_every_exported_pin_is_absolute(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, "relative-ws")

        isolation.activate_isolated_workspace()

        for var in (
            paths.HEADROOM_CONFIG_DIR_ENV,
            paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV,
            paths.HEADROOM_SETTINGS_PATH_ENV,
            isolation.HEADROOM_PREISOLATION_WORKSPACE_ENV,
            paths.HEADROOM_WORKSPACE_DIR_ENV,
            isolation.HEADROOM_ISOLATED_WORKSPACE_ENV,
            isolation.HEADROOM_MEMORY_DB_PATH_ENV,
        ):
            assert Path(os.environ[var]).is_absolute(), f"{var} must not be cwd-relative"

    def test_shared_resources_survive_a_child_chdir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, "relative-ws")
        isolation.activate_isolated_workspace()
        shared = paths.shared_workspace_dir()
        config = paths.config_dir()

        elsewhere = tmp_path / "deep" / "elsewhere"
        elsewhere.mkdir(parents=True)
        monkeypatch.chdir(elsewhere)

        assert paths.shared_workspace_dir() == shared
        assert paths.config_dir() == config
        assert paths.bin_dir() == shared / "bin"

    def test_shared_opt_out_restores_an_absolute_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """`--shared` restores HEADROOM_PREISOLATION_WORKSPACE, which must be
        absolute too or the opt-out lands somewhere cwd-dependent."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, "relative-ws")
        isolation.activate_isolated_workspace()

        isolation.disable_isolation()

        assert paths.workspace_dir().is_absolute()
        assert paths.workspace_dir() == (tmp_path / "relative-ws").resolve()


class TestPersistentInstallMemoryDb:
    """`headroom install apply --memory` from inside an isolated agent baked
    <run>/memory.db into the deployment manifest — a database no top-level run
    shares and that GC later deletes (round 14, P2)."""

    def test_isolated_run_resolves_to_the_shared_db(self, tmp_path: Path) -> None:
        shared = paths.workspace_dir()
        run_dir = isolation.activate_isolated_workspace()

        # The RUN's own DB is still isolated...
        assert paths.memory_db_path() == run_dir / "memory.db"
        # ...but anything persistent must reference the shared one.
        assert isolation.persistent_memory_db_path() == shared / "memory.db"

    def test_outside_isolation_it_is_the_ordinary_db(self, tmp_path: Path) -> None:
        assert isolation.persistent_memory_db_path() == paths.workspace_dir() / "memory.db"

    def test_a_user_pinned_db_is_honored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Only the path ISOLATION chose is overridden; an explicit user
        override means what it says."""
        pinned = tmp_path / "mine" / "memory.db"
        monkeypatch.setenv(isolation.HEADROOM_MEMORY_DB_PATH_ENV, str(pinned))
        isolation.activate_isolated_workspace()

        assert isolation.persistent_memory_db_path() == pinned

    def test_planner_manifest_uses_the_shared_db(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from headroom.install import planner

        shared = paths.workspace_dir()
        run_dir = isolation.activate_isolated_workspace()

        manifest = planner.build_manifest(
            profile="default",
            preset="persistent-docker",
            runtime_kind="docker",
            scope="user",
            provider_mode="manual",
            targets=["claude"],
            port=8787,
            backend="anthropic",
            anyllm_provider=None,
            region=None,
            proxy_mode="token",
            memory_enabled=True,
            telemetry_enabled=False,
            image="ghcr.io/headroomlabs-ai/headroom:latest",
        )

        assert manifest.memory_db_path == str(shared / "memory.db")
        assert str(run_dir) not in " ".join(manifest.proxy_args)
        assert str(shared / "memory.db") in manifest.proxy_args


class TestLearnedVerbosityProfileIsShared:
    """`learn --verbosity --apply` promises the saved profile applies to FUTURE
    proxies, but the shaper resolved it from workspace_dir() — which isolation
    relocates to a fresh, empty run dir (round 15, P2)."""

    def test_profile_path_follows_the_shared_root(self, tmp_path: Path) -> None:
        shared = paths.workspace_dir()
        run_dir = isolation.activate_isolated_workspace()

        assert paths.verbosity_profile_path() == shared / "verbosity.json"
        assert paths.verbosity_profile_path().parent != run_dir

    def test_shaper_finds_a_profile_saved_before_isolation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End-to-end: write the profile where `learn --apply` puts it, then
        activate isolation and resolve the level the way the proxy does."""
        from headroom.proxy.output_shaper import resolve_verbosity_level

        shared = paths.ensure_workspace_dir()
        (shared / "verbosity.json").write_text(json.dumps({"verbosity_level": 1}))
        isolation.activate_isolated_workspace()

        settings = type("S", (), {"verbosity_level": 3})()
        level, source = resolve_verbosity_level(settings)  # type: ignore[arg-type]

        assert (level, source) == (1, "learned"), "the learned profile was not found"

    def test_no_profile_still_falls_back_to_the_default(self, tmp_path: Path) -> None:
        from headroom.proxy.output_shaper import resolve_verbosity_level

        isolation.activate_isolated_workspace()

        settings = type("S", (), {"verbosity_level": 3})()
        assert resolve_verbosity_level(settings) == (3, "default")  # type: ignore[arg-type]

    def test_controller_state_stays_per_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The AIMD controller is live per-proxy tuning state, NOT a persisted
        preference — each isolated proxy must tune itself independently."""
        from headroom.proxy.output_shaper import resolve_verbosity_level

        shared = paths.ensure_workspace_dir()
        (shared / "verbosity_controller.json").write_text(json.dumps({"level": 0}))
        run_dir = isolation.activate_isolated_workspace()
        monkeypatch.setenv("HEADROOM_VERBOSITY_AUTOTUNE", "1")

        settings = type("S", (), {"verbosity_level": 3})()
        # The shared controller file is invisible to this run...
        assert resolve_verbosity_level(settings)[1] != "controller"  # type: ignore[arg-type]

        # ...but its own is used.
        (run_dir / "verbosity_controller.json").write_text(json.dumps({"level": 4}))
        assert resolve_verbosity_level(settings) == (4, "controller")  # type: ignore[arg-type]


class TestExistingOverridesAreNormalized:
    """A nonblank existing override was kept verbatim, so a RELATIVE one
    bypassed every `_abs` call and was re-resolved against each subprocess's
    cwd (round 16, P2)."""

    @pytest.mark.parametrize(
        "var",
        [
            paths.HEADROOM_CONFIG_DIR_ENV,
            paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV,
            paths.HEADROOM_SETTINGS_PATH_ENV,
            isolation.HEADROOM_MEMORY_DB_PATH_ENV,
        ],
    )
    def test_a_relative_override_is_absolutized(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, var: str
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(var, "relative-value")

        isolation.activate_isolated_workspace()

        assert Path(os.environ[var]).is_absolute()
        assert Path(os.environ[var]) == (tmp_path / "relative-value").resolve()

    def test_an_absolute_override_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pinned = tmp_path / "mine" / "config"
        pinned.mkdir(parents=True)
        monkeypatch.setenv(paths.HEADROOM_CONFIG_DIR_ENV, str(pinned))

        isolation.activate_isolated_workspace()

        assert paths.config_dir() == pinned

    def test_a_tilde_override_is_expanded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(paths.HEADROOM_CONFIG_DIR_ENV, "~/cfg")

        isolation.activate_isolated_workspace()

        assert Path(os.environ[paths.HEADROOM_CONFIG_DIR_ENV]).is_absolute()
        assert "~" not in os.environ[paths.HEADROOM_CONFIG_DIR_ENV]


class TestMemoryAlignsWithReusedProxy:
    """`--memory --no-proxy` under isolation gave the session two conflicting
    memory views. Substituting a workspace path does NOT reconcile them: the
    proxy's default is `{cwd}/.headroom/memory.db`, so picking a workspace root
    invents a third store (round 17, P2 — correcting round 16's fix)."""

    def test_the_pin_is_removed_not_replaced(self, tmp_path: Path) -> None:
        run_dir = isolation.activate_isolated_workspace()
        assert os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV] == str(run_dir / "memory.db")

        assert isolation.align_memory_db_with_reused_proxy() is True

        # Unset, so every consumer falls back to the SAME default the reused
        # proxy resolved — not to a path we chose.
        assert isolation.HEADROOM_MEMORY_DB_PATH_ENV not in os.environ

    def test_a_known_manifest_database_is_adopted(self, tmp_path: Path) -> None:
        """A persistent deployment records the DB it was started with; that is
        the reused proxy's ACTUAL store, so prefer it over any fallback."""
        isolation.activate_isolated_workspace()
        manifest_db = tmp_path / "deploy" / "memory.db"

        assert isolation.align_memory_db_with_reused_proxy(str(manifest_db)) is True

        exported = Path(os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV])
        assert exported.is_absolute()
        assert exported == manifest_db.resolve()

    def test_a_blank_manifest_value_falls_back_to_unset(self, tmp_path: Path) -> None:
        isolation.activate_isolated_workspace()

        assert isolation.align_memory_db_with_reused_proxy("   ") is True
        assert isolation.HEADROOM_MEMORY_DB_PATH_ENV not in os.environ

    def test_a_user_pinned_db_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pinned = tmp_path / "mine" / "memory.db"
        monkeypatch.setenv(isolation.HEADROOM_MEMORY_DB_PATH_ENV, str(pinned))
        isolation.activate_isolated_workspace()

        assert isolation.align_memory_db_with_reused_proxy() is False
        assert os.environ[isolation.HEADROOM_MEMORY_DB_PATH_ENV] == str(pinned)

    def test_outside_isolation_it_is_a_noop(self) -> None:
        assert isolation.align_memory_db_with_reused_proxy() is False


class TestLearnProvenanceIsShared:
    """The learn sidecar carries per-section token estimates used to decide
    what survives an over-budget rewrite. Run-scoped, a later `learn` reads
    zero and `_apply_block_cap` can evict the highest-value rules first
    (round 17, P2)."""

    def test_sidecar_follows_the_shared_root(self, tmp_path: Path) -> None:
        from headroom.learn.writer import _sidecar_path

        shared = paths.workspace_dir()
        target = tmp_path / "CLAUDE.local.md"
        before = _sidecar_path(target)
        assert before.parent == shared / "learn"

        run_dir = isolation.activate_isolated_workspace()

        after = _sidecar_path(target)
        assert after == before, "the sidecar moved into the run directory"
        assert run_dir not in after.parents

    def test_a_sidecar_written_before_isolation_is_still_read(self, tmp_path: Path) -> None:
        """The actual failure: estimates saved by an earlier run must survive
        into an isolated one, or carried-forward sections score zero."""
        from headroom.learn.writer import _load_sidecar, _sidecar_path

        target = tmp_path / "CLAUDE.local.md"
        target.write_text("x")
        path = _sidecar_path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sections": {"rules": 1234}}))

        isolation.activate_isolated_workspace()

        assert _load_sidecar(target) == {"rules": 1234}


class TestLearnedBaselineIsShared:
    """`learn --apply` seeds a synthetic-control baseline for future proxies,
    but an isolated proxy read output_savings.json from its own run dir and
    never saw it (round 16, P2)."""

    def test_baseline_path_follows_the_shared_root(self, tmp_path: Path) -> None:
        shared = paths.workspace_dir()
        isolation.activate_isolated_workspace()

        assert paths.output_savings_baseline_path() == shared / "output_savings.json"

    def test_recorder_reads_the_shared_baseline_and_writes_run_local(self, tmp_path: Path) -> None:
        from headroom.proxy.output_savings import SavingsLedger, SavingsRecorder
        from headroom.proxy.output_savings_policy import stratum_label

        shared = paths.ensure_workspace_dir()
        seeded = SavingsLedger()
        for _ in range(5):
            seeded.baseline.observe("s", 100)
        seeded.save(shared / "output_savings.json")
        run_dir = isolation.activate_isolated_workspace()

        recorder = SavingsRecorder(
            run_dir / "output_savings.json",
            baseline_path=paths.output_savings_baseline_path(),
        )
        # Visible IMMEDIATELY, before any record/flush: a proxy that never
        # reaches a flush must still estimate against the seeded baseline.
        assert recorder._ledger.baseline.total_samples == 5

        recorder.record_from_labels([stratum_label("treatment", "s")], 42)
        recorder.flush()

        # ...and still after the flush cycle re-reads it.
        assert recorder._ledger.baseline.total_samples == 5
        # ...observations land in the RUN's file...
        assert (run_dir / "output_savings.json").exists()
        # ...and the shared baseline file is never rewritten by the proxy.
        reloaded = SavingsLedger.load(shared / "output_savings.json")
        assert reloaded.baseline.total_samples == 5
        assert reloaded.treatment == {}

    def test_a_relearn_while_the_proxy_is_live_is_picked_up(self, tmp_path: Path) -> None:
        """`learn --apply` rewrites the SHARED baseline while an isolated proxy
        holds its own observations file. The periodic reload must re-read the
        shared copy, or the new baseline never takes effect until a restart."""
        from headroom.proxy.output_savings import SavingsLedger, SavingsRecorder
        from headroom.proxy.output_savings_policy import stratum_label

        shared = paths.ensure_workspace_dir()
        seeded = SavingsLedger()
        for _ in range(5):
            seeded.baseline.observe("s", 100)
        seeded.save(shared / "output_savings.json")
        run_dir = isolation.activate_isolated_workspace()
        recorder = SavingsRecorder(
            run_dir / "output_savings.json",
            baseline_path=paths.output_savings_baseline_path(),
        )
        recorder.record_from_labels([stratum_label("treatment", "s")], 42)
        recorder.flush()

        # A fresh `learn --apply` lands on the shared root mid-session.
        relearned = SavingsLedger()
        for _ in range(9):
            relearned.baseline.observe("s", 80)
        relearned.save(shared / "output_savings.json")
        recorder.record_from_labels([stratum_label("treatment", "s")], 42)
        recorder.flush()

        assert recorder._ledger.baseline.total_samples == 9, (
            "the proxy kept using the stale baseline from its own run file"
        )

    def test_same_file_when_not_isolated(self, tmp_path: Path) -> None:
        from headroom.proxy.output_savings import SavingsRecorder

        recorder = SavingsRecorder(paths.workspace_dir() / "output_savings.json")

        assert recorder._baseline_path == recorder._path


class TestEveryProxySharingARunDirIsTracked:
    """Re-entrant activation deliberately reuses the parent's run directory, so
    a nested wrap starts a SECOND dedicated proxy against it. Overwriting the
    single record would let GC delete a still-serving proxy's workspace once
    the newer one exits (round 18, P2)."""

    @staticmethod
    def _make_stale(path: Path) -> None:
        stale = 1_000_000_000.0
        for target in (path, *path.rglob("*")):
            os.utime(target, (stale, stale))

    def test_a_second_proxy_does_not_evict_the_first(self, tmp_path: Path) -> None:
        run = tmp_path / "run-20200101-000000-1-abcdef"
        run.mkdir()
        # Two distinct live processes stand in for the two dedicated proxies.
        isolation.record_run_proxy(os.getppid(), 8788, run_dir=run)
        isolation.record_run_proxy(os.getpid(), 8789, run_dir=run)

        ports = [rec["port"] for rec in isolation._recorded_run_proxies(run)]
        assert sorted(ports) == [8788, 8789]

    def test_re_recording_the_same_proxy_does_not_duplicate(self, tmp_path: Path) -> None:
        run = tmp_path / "run-20200101-000000-1-abcdef"
        run.mkdir()
        isolation.record_run_proxy(os.getpid(), 8788, run_dir=run)
        isolation.record_run_proxy(os.getpid(), 8788, run_dir=run)

        assert len(isolation._recorded_run_proxies(run)) == 1

    def test_a_live_older_proxy_still_pins_the_dir(self, tmp_path: Path) -> None:
        """The reported failure: nested proxy dead, wrapper dead, original
        still serving — the workspace must survive."""
        runs = tmp_path / "runs"
        run = runs / "run-20200101-000000-2147480000-abcdef"  # wrapper is dead
        run.mkdir(parents=True)
        isolation.record_run_proxy(os.getpid(), 8788, run_dir=run)  # original, alive
        # A nested proxy recorded afterwards, since exited.
        record = json.loads((run / isolation._PROXY_STATE_FILE).read_text())
        record["proxies"].append({"pid": 2147480001, "port": 8789})
        record.update({"pid": 2147480001, "port": 8789})
        (run / isolation._PROXY_STATE_FILE).write_text(json.dumps(record))
        self._make_stale(run)

        isolation.prune_stale_runs(runs)

        assert run.exists(), "a still-serving proxy's workspace was deleted"

    def test_all_dead_proxies_still_prune(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        run = runs / "run-20200101-000000-2147480000-abcdef"
        run.mkdir(parents=True)
        (run / isolation._PROXY_STATE_FILE).write_text(
            json.dumps(
                {
                    "pid": 2147480002,
                    "proxies": [{"pid": 2147480001}, {"pid": 2147480002}],
                }
            )
        )
        self._make_stale(run)

        isolation.prune_stale_runs(runs)

        assert not run.exists()

    def test_dead_entries_are_pruned_on_write(self, tmp_path: Path) -> None:
        """The list must not grow without bound across a long-lived run."""
        run = tmp_path / "run-20200101-000000-1-abcdef"
        run.mkdir()
        (run / isolation._PROXY_STATE_FILE).write_text(
            json.dumps({"pid": 2147480001, "proxies": [{"pid": 2147480001, "port": 1}]})
        )

        isolation.record_run_proxy(os.getpid(), 8788, run_dir=run)

        assert [r["pid"] for r in isolation._recorded_run_proxies(run)] == [os.getpid()]

    def test_a_legacy_single_dict_record_still_reads(self, tmp_path: Path) -> None:
        run = tmp_path / "run-20200101-000000-1-abcdef"
        run.mkdir()
        (run / isolation._PROXY_STATE_FILE).write_text(json.dumps({"pid": 4321, "port": 8788}))

        assert [r["pid"] for r in isolation._recorded_run_proxies(run)] == [4321]
        assert isolation._run_dir_proxy_pid(run) == 4321


class TestPersistentInstallPathIsAbsolute:
    """A top-level `install apply` never activates isolation, so nothing
    normalized a relative override — and the planner embeds it in a manifest
    that systemd/cron/Docker start from another cwd (round 18, P2)."""

    def test_a_relative_override_is_absolutized(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(isolation.HEADROOM_MEMORY_DB_PATH_ENV, "state/memory.db")

        resolved = isolation.persistent_memory_db_path()

        assert resolved.is_absolute()
        assert resolved == (tmp_path / "state" / "memory.db").resolve()

    def test_the_default_is_absolute_too(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, "relative-ws")

        assert isolation.persistent_memory_db_path().is_absolute()

    def test_the_planner_never_embeds_a_relative_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from headroom.install import planner

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(isolation.HEADROOM_MEMORY_DB_PATH_ENV, "state/memory.db")

        manifest = planner.build_manifest(
            profile="default",
            preset="persistent-docker",
            runtime_kind="docker",
            scope="user",
            provider_mode="manual",
            targets=["claude"],
            port=8787,
            backend="anthropic",
            anyllm_provider=None,
            region=None,
            proxy_mode="token",
            memory_enabled=True,
            telemetry_enabled=False,
            image="ghcr.io/headroomlabs-ai/headroom:latest",
        )

        assert Path(manifest.memory_db_path).is_absolute()
        assert manifest.memory_db_path in manifest.proxy_args

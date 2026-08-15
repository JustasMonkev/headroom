"""omp's `models.yml` is a single durable endpoint every omp process reads.

The wrap's write-and-leave contract is right for `--shared`: the shared proxy's
port outlives the session, so a persisted override stays true. It is wrong for
an isolated run, whose dedicated proxy port dies with the run — leaving every
later omp process, and any concurrent run, pointed at a dead endpoint
(round 23, P2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from headroom.providers.omp import runtime as omp


@pytest.fixture(autouse=True)
def _omp_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "omp-agent"))
    return tmp_path / "omp-agent"


def _base_url() -> str:
    import yaml

    data = yaml.safe_load(omp.models_yml_path().read_text(encoding="utf-8"))
    return data["providers"]["anthropic"]["baseUrl"]


def _pre_wrap_file(body: str = "providers:\n  custom:\n    baseUrl: https://mine\n") -> Path:
    path = omp.models_yml_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class TestAnIsolatedRunDoesNotLeaveADeadPort:
    def test_holding_then_releasing_restores_the_pre_wrap_file(self) -> None:
        original = _pre_wrap_file().read_text()

        omp.hold_models_override(8788, "proj")
        assert "127.0.0.1:8788" in _base_url()

        assert omp.release_models_override() == "restored"
        assert omp.models_yml_path().read_text() == original

    def test_a_wrap_created_file_is_removed_on_release(self) -> None:
        assert not omp.models_yml_path().exists()

        omp.hold_models_override(8788, "proj")

        assert omp.release_models_override() == "removed"
        assert not omp.models_yml_path().exists()

    def test_the_owner_sidecar_is_cleaned_up(self) -> None:
        _pre_wrap_file()
        omp.hold_models_override(8788, "proj")
        assert omp.owners_path(omp.models_yml_path()).exists()

        omp.release_models_override()

        assert not omp.owners_path(omp.models_yml_path()).exists()

    def test_releasing_without_holding_is_harmless(self) -> None:
        original = _pre_wrap_file().read_text()

        assert omp.release_models_override() == "noop"
        assert omp.models_yml_path().read_text() == original


class TestAConcurrentRunIsNotStripped:
    @staticmethod
    def _peer(port: int) -> dict:
        """A live peer that is not this process (its own PID is filtered out)."""
        from headroom._subprocess import proc_identity

        ident = proc_identity(os.getppid())
        return {
            "pid": os.getppid(),
            "port": port,
            "project": "peer",
            "start_src": ident[0] if ident else None,
            "start_time": ident[1] if ident else None,
        }

    def test_exiting_hands_the_override_to_a_live_peer(self) -> None:
        _pre_wrap_file()
        omp.hold_models_override(8788, "proj")
        owners = json.loads(omp.owners_path(omp.models_yml_path()).read_text())
        omp.owners_path(omp.models_yml_path()).write_text(json.dumps([self._peer(8790), *owners]))

        assert omp.release_models_override() == "handover"
        assert "127.0.0.1:8790" in _base_url(), "the peer's port must survive our exit"

    def test_a_dead_peer_does_not_block_the_restore(self) -> None:
        original = _pre_wrap_file().read_text()
        omp.hold_models_override(8788, "proj")
        dead = {"pid": 2147480000, "port": 8790, "project": "gone"}
        owners = json.loads(omp.owners_path(omp.models_yml_path()).read_text())
        omp.owners_path(omp.models_yml_path()).write_text(json.dumps([dead, *owners]))

        assert omp.release_models_override() == "restored"
        assert omp.models_yml_path().read_text() == original

    def test_the_surviving_peer_stays_registered(self) -> None:
        _pre_wrap_file()
        omp.hold_models_override(8788, "proj")
        owners = json.loads(omp.owners_path(omp.models_yml_path()).read_text())
        omp.owners_path(omp.models_yml_path()).write_text(json.dumps([self._peer(8790), *owners]))

        omp.release_models_override()

        remaining = json.loads(omp.owners_path(omp.models_yml_path()).read_text())
        assert [o["pid"] for o in remaining] == [os.getppid()]


class TestTheSharedContractIsUnchanged:
    def test_a_plain_inject_registers_no_owner(self) -> None:
        """`--shared` writes a durable port and must keep write-and-leave."""
        _pre_wrap_file()

        omp.inject_models_override(8787, "proj")

        assert not omp.owners_path(omp.models_yml_path()).exists()

    def test_the_backup_is_still_byte_for_byte(self) -> None:
        original = _pre_wrap_file("providers:\r\n  custom:\r\n    baseUrl: https://mine\r\n")
        raw = original.read_bytes()

        omp.hold_models_override(8788, "proj")

        assert omp.backup_path(omp.models_yml_path()).read_bytes() == raw

    def test_reinjecting_does_not_clobber_the_pristine_backup(self) -> None:
        raw = _pre_wrap_file().read_bytes()

        omp.hold_models_override(8788, "proj")
        omp.hold_models_override(8791, "proj")  # the reconcile-port rewrite

        assert omp.backup_path(omp.models_yml_path()).read_bytes() == raw
        assert "127.0.0.1:8791" in _base_url()


class TestWritesAreSerialized:
    def test_the_lock_is_held_across_inject(self) -> None:
        """Concurrent wraps otherwise read the same pre-image, and the later
        write drops the other's edits."""
        from headroom import _filelock

        lock = _filelock.lock_path_for(omp.models_yml_path())
        lock.parent.mkdir(parents=True, exist_ok=True)
        observed: list[bool] = []
        real = omp._inject_locked

        def probing(port: int, project: str | None = None):
            with open(lock, "a+", encoding="utf-8") as handle:
                free = _filelock.acquire(handle, timeout=0)
                if free:
                    _filelock.release(handle)
                observed.append(free)
            return real(port, project)

        omp._inject_locked = probing  # type: ignore[assignment]
        try:
            omp.inject_models_override(8787, "proj")
        finally:
            omp._inject_locked = real  # type: ignore[assignment]

        assert observed == [False]

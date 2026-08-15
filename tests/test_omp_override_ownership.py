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


class TestHandoverRequiresALiveProxy:
    """A peer's wrapper stays blocked on its omp child long after a detached
    proxy dies. Handing `models.yml` back on wrapper liveness alone rewrites a
    machine-global file to a dead port, breaking omp for everyone until that
    wrapper finally exits (round 25, P2)."""

    @staticmethod
    def _live_peer(port: int, proxy_pid: int | None = 4242) -> dict:
        from headroom._subprocess import proc_identity

        ident = proc_identity(os.getppid())
        return {
            "pid": os.getppid(),
            "proxy_pid": proxy_pid,
            "port": port,
            "project": "peer",
            "start_src": ident[0] if ident else None,
            "start_time": ident[1] if ident else None,
        }

    def _hold_with_peer(self, peer: dict) -> None:
        _pre_wrap_file()
        omp.hold_models_override(8788, "proj", 1111)
        owners = json.loads(omp.owners_path(omp.models_yml_path()).read_text())
        omp.owners_path(omp.models_yml_path()).write_text(json.dumps([peer, *owners]))

    def test_a_peer_whose_proxy_died_is_not_handed_the_override(self) -> None:
        original = omp.models_yml_path()
        self._hold_with_peer(self._live_peer(8790))
        pre_wrap = omp.backup_path(original).read_text()

        status = omp.release_models_override(proxy_probe=lambda _port, _pid, _inst: False)

        assert status == "restored"
        assert original.read_text() == pre_wrap

    def test_a_peer_with_a_live_proxy_still_gets_it(self) -> None:
        self._hold_with_peer(self._live_peer(8790))

        assert (
            omp.release_models_override(proxy_probe=lambda _port, _pid, _inst: True) == "handover"
        )
        assert "127.0.0.1:8790" in _base_url()

    def test_the_probe_is_asked_about_the_peers_port_and_proxy(self) -> None:
        seen: list[tuple[int, int | None, str | None]] = []
        peer = self._live_peer(8790, proxy_pid=777)
        peer["proxy_instance"] = "srv-peer"
        self._hold_with_peer(peer)

        omp.release_models_override(
            proxy_probe=lambda port, pid, inst: seen.append((port, pid, inst)) or True  # type: ignore[func-returns-value]
        )

        assert seen == [(8790, 777, "srv-peer")], "the probe must target the PEER's listener"

    def test_without_a_probe_the_wrapper_check_still_stands(self) -> None:
        """Callers with no way to probe keep the previous behaviour."""
        self._hold_with_peer(self._live_peer(8790))

        assert omp.release_models_override() == "handover"

    def test_the_holder_records_its_own_proxy_identity(self) -> None:
        _pre_wrap_file()

        omp.hold_models_override(8788, "proj", 9999, "srv-mine")

        owners = json.loads(omp.owners_path(omp.models_yml_path()).read_text())
        assert owners[-1]["proxy_pid"] == 9999
        assert owners[-1]["proxy_instance"] == "srv-mine"
        assert owners[-1]["pid"] == os.getpid() != 9999

    def test_the_instance_is_what_survives_multiple_workers(self) -> None:
        """A pid recorded from one worker cannot match the next worker to
        answer; the instance is the same for every worker of one server."""
        peer = self._live_peer(8790, proxy_pid=777)
        peer["proxy_instance"] = "srv-peer"
        self._hold_with_peer(peer)

        def only_the_instance_matches(_port: int, _pid: int | None, inst: str | None) -> bool:
            assert inst == "srv-peer", "the probe was given no instance to compare"
            return True

        assert omp.release_models_override(proxy_probe=only_the_instance_matches) == "handover"

    def test_the_launcher_passes_a_real_probe(self) -> None:
        """The wiring, not a reimplementation of it: `wrap omp` must hand the
        release a probe rather than relying on wrapper liveness."""
        import inspect

        from headroom.cli import wrap as wrap_mod

        source = inspect.getsource(wrap_mod.omp.callback)
        assert "_release_omp_models_override(_omp_proxy_still_serving)" in source
        assert "_wrap_proxy_alive(" in source

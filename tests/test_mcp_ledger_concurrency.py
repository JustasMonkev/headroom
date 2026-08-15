"""The MCP ownership ledger is shared state, so its mutations are races.

It moved to the shared root so an isolated wrap's ownership records survive for
a later `unwrap`. That makes every mutation a multi-process read-modify-write:
two wraps registering different servers can each merge onto the same pre-image,
and the lost entry can then never be proven Headroom-owned — `unwrap` leaves a
server installed in the user's config for good (round 27, P2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom import _filelock, paths
from headroom.mcp_registry import ledger
from headroom.mcp_registry.base import ServerSpec


@pytest.fixture(autouse=True)
def _shared_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, str(tmp_path / "shared"))


def _spec(name: str) -> ServerSpec:
    return ServerSpec(name=name, command="headroom", args=("mcp", "serve"), env={})


class TestLedgerMutationsAreSerialized:
    def test_the_lock_is_held_across_the_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The lost update happens at the read, so locking only the write
        would fix nothing."""
        observed: list[bool] = []
        real = ledger._read_ledger

        def probing(path: Path) -> dict:
            lock = _filelock.lock_path_for(ledger.ledger_path())
            lock.parent.mkdir(parents=True, exist_ok=True)
            with open(lock, "a+", encoding="utf-8") as handle:
                free = _filelock.acquire(handle, timeout=0)
                if free:
                    _filelock.release(handle)
                observed.append(free)
            return real(path)

        monkeypatch.setattr(ledger, "_read_ledger", probing)
        ledger.record_install("claude", _spec("headroom"))

        assert observed and not any(observed), "the merge read ran outside the lock"

    def test_clearing_is_serialized_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ledger.record_install("claude", _spec("headroom"))
        observed: list[bool] = []
        real = ledger._read_ledger

        def probing(path: Path) -> dict:
            lock = _filelock.lock_path_for(ledger.ledger_path())
            with open(lock, "a+", encoding="utf-8") as handle:
                free = _filelock.acquire(handle, timeout=0)
                if free:
                    _filelock.release(handle)
                observed.append(free)
            return real(path)

        monkeypatch.setattr(ledger, "_read_ledger", probing)
        ledger.clear_install("claude", "headroom")

        assert observed and not any(observed)

    def test_concurrent_registrations_keep_both_entries(self, tmp_path: Path) -> None:
        """End-to-end across real processes. Each child widens its own
        read-modify-write window so the merges genuinely overlap; without the
        lock the later writer's pre-image is empty and drops the other."""
        import subprocess
        import sys

        child = tmp_path / "register.py"
        child.write_text(
            "import sys, time\n"
            "from headroom.mcp_registry import ledger\n"
            "from headroom.mcp_registry.base import ServerSpec\n"
            "real = ledger._read_ledger\n"
            "def slow(path):\n"
            "    data = real(path)\n"
            "    time.sleep(1.0)\n"
            "    return data\n"
            "ledger._read_ledger = slow\n"
            "ledger.record_install(sys.argv[1], ServerSpec(name=sys.argv[2],"
            " command='headroom', args=('mcp','serve'), env={}))\n"
        )
        env = {
            **dict(__import__("os").environ),
            paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV: str(tmp_path / "shared"),
        }
        procs = [
            subprocess.Popen([sys.executable, str(child), agent, name], env=env)
            for agent, name in (("claude", "headroom"), ("codex", "serena"))
        ]
        for proc in procs:
            assert proc.wait(timeout=60) == 0

        data = json.loads(ledger.ledger_path().read_text())
        assert "headroom" in data["agents"].get("claude", {}), data
        assert "serena" in data["agents"].get("codex", {}), data


class TestTheLedgerIsPublishedAtomically:
    def test_a_reader_never_sees_a_partial_ledger(self) -> None:
        """A truncated parse reads as "Headroom owns nothing", which is how a
        server survives an unwrap that should have removed it."""
        ledger.record_install("claude", _spec("headroom"))

        assert ledger.headroom_installed_matching("claude", _spec("headroom")) is True
        assert list(ledger.ledger_path().parent.glob("*.tmp")) == []

    def test_the_temp_name_carries_the_writer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two publishers sharing one temp name can break each other's
        replace and leave no ledger at all."""
        import os

        seen: list[str] = []
        real = os.replace

        def capturing(src, dst):  # type: ignore[no-untyped-def]
            seen.append(Path(src).name)
            return real(src, dst)

        monkeypatch.setattr(os, "replace", capturing)
        ledger.record_install("claude", _spec("headroom"))

        assert seen, "the ledger was not published through a temp file"
        assert str(os.getpid()) in seen[0], f"temp name carries no writer identity: {seen[0]}"

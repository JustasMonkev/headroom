"""The update-check cache is one shared file every isolated wrap refreshes.

Resolving it against the shared root fixed the "permanently stale" half. The
other half is concurrency: once the TTL expires, every launch in a fan-out
passes `should_check()` before any of them writes, so each independently hits
PyPI — and they all wrote through the same `.json.tmp`, which can leave the
cache absent and send the NEXT fan-out to PyPI too (round 24, P2).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from headroom import _filelock, paths, update_check


@pytest.fixture(autouse=True)
def _shared_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, str(tmp_path / "shared"))
    monkeypatch.setenv("HEADROOM_UPDATE_CHECK", "on")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)


class TestTheRefreshIsSerialized:
    def test_a_peer_that_refreshed_while_we_waited_saves_us_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: the second process must reuse the first's result
        instead of making its own PyPI request."""
        update_check.write_cache("9.9.9")
        calls: list[int] = []

        def counted(**_k: Any) -> str:
            calls.append(1)
            return "1.2.3"

        monkeypatch.setattr(update_check, "fetch_latest_version", counted)

        assert update_check.run_check() == "9.9.9"
        assert calls == [], "a fresh cache written by a peer must short-circuit the fetch"

    def test_a_stale_cache_is_still_refreshed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Serializing must not turn into never checking."""
        update_check.write_cache("9.9.9", now=time.time() - update_check._CHECK_TTL_SECONDS - 10)
        monkeypatch.setattr(update_check, "fetch_latest_version", lambda **_k: "1.2.3")

        assert update_check.run_check() == "1.2.3"
        assert json.loads(update_check._cache_path().read_text())["latest_version"] == "1.2.3"

    def test_the_fetch_runs_inside_the_lock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The redundant-request window is between the staleness check and the
        write, so the critical section has to span the fetch."""
        observed: list[bool] = []

        def probing(**_k: Any) -> str:
            lock = _filelock.lock_path_for(update_check._cache_path())
            lock.parent.mkdir(parents=True, exist_ok=True)
            with open(lock, "a+", encoding="utf-8") as handle:
                free = _filelock.acquire(handle, timeout=0)
                if free:
                    _filelock.release(handle)
                observed.append(free)
            return "1.2.3"

        monkeypatch.setattr(update_check, "fetch_latest_version", probing)

        update_check.run_check()

        assert observed == [False], "the PyPI fetch ran outside the refresh lock"

    def test_an_explicit_now_is_never_second_guessed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`headroom update` asks for a check on purpose."""
        update_check.write_cache("9.9.9")
        monkeypatch.setattr(update_check, "fetch_latest_version", lambda **_k: "1.2.3")

        assert update_check.run_check(now=time.time()) == "1.2.3"


class TestConcurrentWritersDoNotCorruptTheCache:
    def test_the_temp_file_is_unique_per_writer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A shared `.json.tmp` lets one writer's replace fail and leave the
        cache missing entirely."""
        import json as _json
        import os
        import threading

        seen: list[str] = []
        real_dump = _json.dump

        def capturing(payload: Any, fh: Any, **kw: Any) -> Any:
            seen.append(Path(fh.name).name)
            return real_dump(payload, fh, **kw)

        monkeypatch.setattr(_json, "dump", capturing)
        update_check.write_cache("1.2.3")

        assert seen, "no temp file was written"
        assert str(os.getpid()) in seen[0], f"temp name carries no writer identity: {seen[0]}"
        assert str(threading.get_ident()) in seen[0]
        assert list(update_check._cache_path().parent.glob("*.tmp")) == []

    def test_a_concurrent_refresh_leaves_exactly_one_valid_cache(self, tmp_path: Path) -> None:
        """End-to-end across real processes: two launches refreshing at once
        must leave a readable cache and make ONE request between them."""
        import subprocess
        import sys

        marker = tmp_path / "requests"
        marker.mkdir()
        child = tmp_path / "refresh.py"
        child.write_text(
            "import os, sys, time, uuid\n"
            "from headroom import update_check\n"
            "def fake(**_k):\n"
            "    (sys.argv[1] + '/' + uuid.uuid4().hex).encode()\n"
            "    open(sys.argv[1] + '/' + uuid.uuid4().hex, 'w').close()\n"
            "    time.sleep(1.0)\n"
            "    return '1.2.3'\n"
            "update_check.fetch_latest_version = fake\n"
            "update_check.run_check()\n"
        )
        env = {
            **dict(__import__("os").environ),
            paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV: str(tmp_path / "shared"),
            "HEADROOM_UPDATE_CHECK": "on",
        }
        env.pop("CI", None)
        procs = [
            subprocess.Popen([sys.executable, str(child), str(marker)], env=env) for _ in range(2)
        ]
        for proc in procs:
            assert proc.wait(timeout=60) == 0

        cached = json.loads(update_check._cache_path().read_text())
        assert cached["latest_version"] == "1.2.3"
        assert len(list(marker.iterdir())) == 1, "both processes hit the network"

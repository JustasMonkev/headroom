"""`settings.json` lives on the SHARED root, so every concurrent isolated
proxy's dashboard saves into the same file.

`save()` is a load-merge-write cycle. The atomic replace it ends with prevents
a torn file but does nothing about a lost update: two dashboards saving
different fields can both read the same pre-image, merge only their own key,
and whichever writes last silently discards the other's setting (round 21, P2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from headroom import _filelock, paths, settings_store


@pytest.fixture(autouse=True)
def _shared_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, str(tmp_path / "shared"))
    monkeypatch.delenv(paths.HEADROOM_SETTINGS_PATH_ENV, raising=False)


def _peer_can_lock() -> bool:
    """Whether an independent handle can take the settings lock right now.

    flock is held per open file description, so a fresh handle conflicts with a
    held lock even inside this same process — which makes the critical section
    observable without spawning anything.
    """
    lock = _filelock.lock_path_for(paths.settings_path())
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+", encoding="utf-8") as handle:
        if not _filelock.acquire(handle, timeout=0):
            return False
        _filelock.release(handle)
        return True


def _a_field() -> tuple[str, str]:
    """Two independently settable stored fields, whatever the schema calls
    them today — the test is about losing one, not about their meaning.

    A field whose default is ``None`` is unusable here: saving None CLEARS the
    key rather than storing it.
    """
    keys = [
        key
        for key, field in settings_store._BY_KEY.items()
        if not field.secret and field.default is not None
    ]
    return keys[0], keys[1]


class TestSaveIsSerialized:
    def test_the_lock_is_held_across_the_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The lost update happens at the READ, so the critical section has to
        start before it — locking only the write would fix nothing."""
        observed: list[bool] = []
        real = settings_store.load

        def probing_load() -> Any:
            observed.append(_peer_can_lock())
            return real()

        monkeypatch.setattr(settings_store, "load", probing_load)
        first, _second = _a_field()

        settings_store.save({first: settings_store._BY_KEY[first].default})

        assert observed and not any(observed), "the merge read ran outside the lock"

    def test_the_lock_is_released_afterwards(self) -> None:
        first, _second = _a_field()

        settings_store.save({first: settings_store._BY_KEY[first].default})

        assert _peer_can_lock() is True

    def test_a_save_still_merges_rather_than_replaces(self) -> None:
        first, second = _a_field()

        settings_store.save({first: settings_store._BY_KEY[first].default})
        settings_store.save({second: settings_store._BY_KEY[second].default})

        stored = json.loads(paths.settings_path().read_text())
        assert first in stored and second in stored

    def test_concurrent_saves_keep_both_fields(self, tmp_path: Path) -> None:
        """End-to-end across real processes: two dashboards saving different
        fields at once. Without the lock the later writer's pre-image is the
        empty file and it drops the other setting."""
        import subprocess
        import sys

        first, second = _a_field()
        child = tmp_path / "save.py"
        # Widen the read-modify-write window from inside the child so the two
        # saves genuinely overlap. Without this both processes finish before
        # the other starts and the race never happens — the test would pass
        # against an unlocked merge.
        child.write_text(
            "import sys, time\n"
            "from headroom import settings_store\n"
            "key, delay = sys.argv[1], float(sys.argv[2])\n"
            "real = settings_store.load\n"
            "def slow_load():\n"
            "    data = real()\n"
            "    time.sleep(delay)\n"
            "    return data\n"
            "settings_store.load = slow_load\n"
            "settings_store.save({key: settings_store._BY_KEY[key].default})\n"
        )
        env = {
            **dict(__import__("os").environ),
            paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV: str(tmp_path / "shared"),
        }
        procs = [
            subprocess.Popen([sys.executable, str(child), key, delay], env=env)
            for key, delay in ((first, "1.0"), (second, "1.0"))
        ]
        for proc in procs:
            assert proc.wait(timeout=60) == 0

        stored = json.loads(paths.settings_path().read_text())
        assert first in stored and second in stored, f"a concurrent save was lost: {stored}"

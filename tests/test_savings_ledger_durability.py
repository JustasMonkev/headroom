"""Durability fixes for the savings ledger.

- The append/compaction writers flush before releasing the advisory lock
  (buffered bytes draining at close(), after LOCK_UN, could interleave with
  another process's write).
- One undecodable byte in the file degrades to skipping that line, not to
  `headroom savings` reporting zero forever.
- Once the 30-day working set exceeds the compaction threshold, appends no
  longer rewrite the whole file every time (O(n^2) growth); compaction waits
  for meaningful regrowth.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from headroom import savings_ledger as sl


def _write_event_line(path: Path, *, ts: datetime, saved: int = 100) -> None:
    event = {
        "v": sl.SCHEMA_VERSION,
        "ts": ts.isoformat(),
        "before": saved + 10,
        "after": 10,
        "saved": saved,
        "cost_usd": 0.001,
        "model": "unknown",
        "client": "test",
        "source": "test",
        "pid": 1,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, separators=(",", ":")) + "\n")


def test_read_events_survives_undecodable_line(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    now = datetime.now(timezone.utc)
    _write_event_line(target, ts=now, saved=100)
    # A torn write from a crashed process: raw non-UTF-8 bytes mid-file.
    with open(target, "ab") as fh:
        fh.write(b"\xff\xfe garbage \xff\n")
    _write_event_line(target, ts=now, saved=200)

    report = sl.aggregate_savings(target, now=now)
    # Both valid events counted; the poisoned line is skipped, not fatal.
    assert report.lifetime["tokens_saved"] == 300
    assert report.lifetime["calls"] == 2


def test_record_flushes_before_returning(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    assert sl.record_savings_event(tokens_before=100, tokens_after=10, path=target)
    # The line must be on disk (not sitting in a buffer) by the time
    # record_savings_event returns.
    content = target.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert json.loads(content.strip())["saved"] == 90


def test_compaction_drops_expired_events(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "events.jsonl"
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=sl.DEFAULT_RETENTION_DAYS + 5)
    for _ in range(50):
        _write_event_line(target, ts=old)
    _write_event_line(target, ts=now, saved=42)

    monkeypatch.setattr(sl, "_COMPACT_SIZE_BYTES", 128)
    sl._last_compact_sizes.clear()
    sl._maybe_compact(target)

    lines = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["saved"] == 42


def test_compaction_not_rerun_until_regrowth(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "events.jsonl"
    now = datetime.now(timezone.utc)
    # All events in retention: compaction cannot shrink the file.
    for _ in range(200):
        _write_event_line(target, ts=now)

    monkeypatch.setattr(sl, "_COMPACT_SIZE_BYTES", 128)
    monkeypatch.setattr(sl, "_COMPACT_REGROWTH_BYTES", 64 * 1024)
    sl._last_compact_sizes.clear()

    sl._maybe_compact(target)
    assert sl._last_compact_sizes  # floor recorded after the first pass

    # Track rewrites: a second compaction would re-open the file "r+".
    calls: list[str] = []
    real_open = open

    def counting_open(file, mode="r", *args, **kwargs):
        if str(file) == str(target) and "+" in mode:
            calls.append(mode)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    # Appends keep coming, file stays above the size threshold, but within
    # the regrowth window — no full-file rewrite may happen.
    for _ in range(5):
        sl._maybe_compact(target)
    assert calls == []


def _stat(size: int, dev: int = 1, ino: int = 1):
    from types import SimpleNamespace

    return SimpleNamespace(st_size=size, st_dev=dev, st_ino=ino)


def test_should_compact_decision_logic(monkeypatch) -> None:
    monkeypatch.setattr(sl, "_COMPACT_SIZE_BYTES", 1000)
    monkeypatch.setattr(sl, "_COMPACT_REGROWTH_BYTES", 100)
    sl._last_compact_sizes.clear()

    # Below threshold: never compact.
    assert not sl._should_compact(Path("p"), _stat(1000))
    # Above threshold, no floor recorded: compact.
    assert sl._should_compact(Path("p"), _stat(1001))
    # Floor recorded, within the regrowth window: skip.
    sl._last_compact_sizes["p"] = (1, 1, 1500)
    assert not sl._should_compact(Path("p"), _stat(1550))
    # Regrown past the window: compact.
    assert sl._should_compact(Path("p"), _stat(1601))
    # Shrank below the floor (external compaction): stale floor is dropped
    # and compaction proceeds.
    sl._last_compact_sizes["p"] = (1, 1, 1500)
    assert sl._should_compact(Path("p"), _stat(1200))
    assert "p" not in sl._last_compact_sizes
    # Same size but a different inode (rotation/replacement): the pathname
    # refers to a never-compacted file — the floor must not suppress it.
    sl._last_compact_sizes["p"] = (1, 1, 1500)
    assert sl._should_compact(Path("p"), _stat(1550, ino=2))
    assert "p" not in sl._last_compact_sizes


def test_compaction_rechecks_under_lock(tmp_path: Path, monkeypatch) -> None:
    """Writers queued behind a first compaction must not each rewrite the
    file again: the size/floor decision is re-evaluated after LOCK_EX. Here a
    'concurrent' compaction shrinks the file between the pre-lock check and
    the open — the rewrite must be skipped."""
    target = tmp_path / "events.jsonl"
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=sl.DEFAULT_RETENTION_DAYS + 5)
    for _ in range(50):
        _write_event_line(target, ts=old)

    # Threshold sits between one event line (~156B) and the 50-event file, so
    # the pre-lock check fires but the post-shrink recheck must not.
    monkeypatch.setattr(sl, "_COMPACT_SIZE_BYTES", 512)
    sl._last_compact_sizes.clear()

    real_open = open
    truncations: list[str] = []

    def racing_open(file, mode="r", *args, **kwargs):
        if str(file) == str(target) and "+" in mode:
            # Simulate another writer compacting while we waited for the lock.
            truncations.append(mode)
            with real_open(target, "w", encoding="utf-8") as fh:
                fh.write("")
            _write_event_line(target, ts=now, saved=7)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", racing_open)
    sl._maybe_compact(target)
    monkeypatch.setattr("builtins.open", real_open)

    assert truncations  # the race actually happened
    lines = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # The under-lock recheck saw a small file and skipped the rewrite: the
    # concurrent writer's single event is intact and no floor was recorded.
    assert len(lines) == 1
    assert json.loads(lines[0])["saved"] == 7
    assert str(target) not in sl._last_compact_sizes


def test_stale_floor_is_reset_after_external_shrink(tmp_path: Path, monkeypatch) -> None:
    """A floor recorded by this process must not suppress retention after
    another process compacts (or rotates) the file to a much smaller size —
    otherwise the ledger could regrow toward the old working-set size with
    expired events never pruned."""
    target = tmp_path / "events.jsonl"
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=sl.DEFAULT_RETENTION_DAYS + 5)
    for _ in range(50):
        _write_event_line(target, ts=old)
    _write_event_line(target, ts=now, saved=42)

    monkeypatch.setattr(sl, "_COMPACT_SIZE_BYTES", 128)
    sl._last_compact_sizes.clear()
    # Simulate a floor left over from when this process saw a much larger
    # working set (before another process compacted the same inode).
    st = target.stat()
    sl._last_compact_sizes[str(target)] = (st.st_dev, st.st_ino, 100 * 1024 * 1024)

    sl._maybe_compact(target)

    lines = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1  # expired events pruned despite the stale floor
    assert sl._last_compact_sizes[str(target)][2] < 100 * 1024 * 1024


def test_floor_invalidated_when_inode_changes(tmp_path: Path, monkeypatch) -> None:
    """Replacing the ledger with a same-or-larger never-compacted file must
    not inherit the old file's floor: the identity check detects the inode
    change and retention runs on the replacement."""
    target = tmp_path / "events.jsonl"
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=sl.DEFAULT_RETENTION_DAYS + 5)
    for _ in range(50):
        _write_event_line(target, ts=old)
    _write_event_line(target, ts=now, saved=42)

    monkeypatch.setattr(sl, "_COMPACT_SIZE_BYTES", 128)
    monkeypatch.setattr(sl, "_COMPACT_REGROWTH_BYTES", 64 * 1024)
    sl._last_compact_sizes.clear()
    # Floor recorded against a DIFFERENT inode, with a size equal to the
    # current file's (the case a size-only check cannot detect).
    st = target.stat()
    sl._last_compact_sizes[str(target)] = (st.st_dev, st.st_ino + 1, st.st_size)

    sl._maybe_compact(target)

    lines = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1  # expired events pruned despite the same-size floor
    assert json.loads(lines[0])["saved"] == 42
    # The re-recorded floor now describes the real inode.
    dev, ino, _size = sl._last_compact_sizes[str(target)]
    fresh = target.stat()
    assert (dev, ino) == (fresh.st_dev, fresh.st_ino)


def test_shared_floor_suppresses_other_process_rewrites(tmp_path: Path, monkeypatch) -> None:
    """A compaction publishes its floor to a sidecar so OTHER worker
    processes (whose in-memory caches are empty) skip the redundant
    full-file rewrite within the regrowth window."""
    target = tmp_path / "events.jsonl"
    now = datetime.now(timezone.utc)
    for _ in range(200):
        _write_event_line(target, ts=now)

    monkeypatch.setattr(sl, "_COMPACT_SIZE_BYTES", 128)
    monkeypatch.setattr(sl, "_COMPACT_REGROWTH_BYTES", 64 * 1024)
    sl._last_compact_sizes.clear()

    sl._maybe_compact(target)  # "process A" compacts and publishes the floor
    assert sl._floor_sidecar_path(target).exists()

    # "Process B": empty local cache, must pick the floor up from the sidecar.
    sl._last_compact_sizes.clear()
    real_open = open
    rewrites: list[str] = []

    def counting_open(file, mode="r", *args, **kwargs):
        if str(file) == str(target) and "+" in mode:
            rewrites.append(mode)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    for _ in range(5):
        sl._maybe_compact(target)
    assert rewrites == []


def test_corrupt_utf8_line_is_skipped_not_repaired(tmp_path: Path) -> None:
    """A bad byte INSIDE a JSON string must drop that line, not survive as a
    U+FFFD-'repaired' event counted under a corrupted model bucket."""
    target = tmp_path / "events.jsonl"
    now = datetime.now(timezone.utc)
    _write_event_line(target, ts=now, saved=100)
    # A torn line whose damage (a raw 0xFF byte, invalid UTF-8) sits inside
    # the quoted model field: with errors='replace' this would decode to a
    # U+FFFD, parse as valid JSON, and be counted.
    torn = (
        b'{"v":1,"ts":"' + now.isoformat().encode() + b'","before":110,"after":10,'
        b'"saved":9999,"cost_usd":9.0,"model":"gpt\xff4o","client":"t","source":"t","pid":1}'
    )
    with open(target, "ab") as fh:
        fh.write(torn + b"\n")
    _write_event_line(target, ts=now, saved=200)

    report = sl.aggregate_savings(target, now=now)
    assert report.lifetime["tokens_saved"] == 300  # torn line dropped entirely
    assert report.lifetime["calls"] == 2


def test_stale_local_floor_defers_to_newer_sidecar(tmp_path: Path, monkeypatch) -> None:
    """Worker A's process-local floor predates worker B's compaction. A's
    local floor alone says "regrown, compact again" — but the sidecar B
    published reflects the newer compaction, so A must skip the rewrite."""
    target = tmp_path / "events.jsonl"
    now = datetime.now(timezone.utc)
    for _ in range(200):
        _write_event_line(target, ts=now)

    monkeypatch.setattr(sl, "_COMPACT_SIZE_BYTES", 128)
    monkeypatch.setattr(sl, "_COMPACT_REGROWTH_BYTES", 64 * 1024)
    sl._last_compact_sizes.clear()

    st = target.stat()
    # A's stale local floor: same inode, but far below the current size —
    # wait, a floor BELOW current-size-minus-regrowth would trigger a
    # rewrite; that is exactly the stale state being tested.
    stale_floor = st.st_size - 70 * 1024 if st.st_size > 70 * 1024 else 1
    sl._last_compact_sizes[str(target)] = (st.st_dev, st.st_ino, max(stale_floor, 1))
    # B's newer compaction published the current size as the floor.
    sl._write_shared_floor(target, (st.st_dev, st.st_ino, st.st_size))

    real_open = open
    rewrites: list[str] = []

    def counting_open(file, mode="r", *args, **kwargs):
        if str(file) == str(target) and "+" in mode:
            rewrites.append(mode)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    sl._maybe_compact(target)
    assert rewrites == []

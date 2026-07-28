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
        if str(file) == str(target) and "r+" in mode:
            calls.append(mode)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    # Appends keep coming, file stays above the size threshold, but within
    # the regrowth window — no full-file rewrite may happen.
    for _ in range(5):
        sl._maybe_compact(target)
    assert calls == []

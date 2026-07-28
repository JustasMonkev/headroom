"""CCR marker scanning must stay linear on bracket-dense content.

The generic fallback pattern used lazy dots (``\\[.*?compressed.*?hash=...``),
which restarts a forward scan at every ``[`` — quadratic on minified-JSON tool
results (measured 12.8s on a single 181KB payload, with no marker present).
The scanner runs on every message of every proxied request.
"""

from __future__ import annotations

import time

from headroom.ccr.tool_injection import CCRToolInjector


def _scan(text: str) -> list[str]:
    injector = CCRToolInjector()
    return injector.scan_for_markers([{"role": "user", "content": text}])


def test_generic_marker_still_detected() -> None:
    h = "abc123" * 4  # 24 hex chars
    assert _scan(f"[somehow Compressed weirdly hash={h}]") == [h]
    assert _scan(f"before [3 blobs compressed into one hash={h}] after") == [h]


def test_standard_markers_still_detected() -> None:
    h = "0f" * 12
    assert _scan(f"[100 items compressed to 10. Retrieve more: hash={h}]") == [h]
    assert _scan(f"[42 lines compressed. hash={h}]") == [h]


def test_no_false_positive_on_plain_brackets() -> None:
    assert _scan("[no marker here] [also compressed nothing]") == []


def test_bracket_dense_json_scans_fast() -> None:
    # 200KB+ of minified JSON, thousands of '[' — worst case for the old
    # pattern (>10s); must now complete near-instantly.
    payload = "[" + ",".join(f'[{i},"x"]' for i in range(20000)) + "]"
    start = time.perf_counter()
    assert _scan(payload) == []
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"marker scan took {elapsed:.2f}s on bracket-dense JSON"

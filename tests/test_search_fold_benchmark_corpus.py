"""The search-fold benchmark's corpus trimming.

The benchmark's numbers are only meaningful if the corpus is what it claims to
be, so the row-trimming helper gets the same regression treatment as the fold
itself: a trim that silently drops the wrong rows changes the measured before /
after ratio without failing anything.
"""

from __future__ import annotations

from benchmarks.search_fold_benchmark import _row_file, _trim_to_file_boundary


def test_row_file_returns_the_complete_hyphenated_path():
    # Splitting on '-' here reduced this to `crates/headroom`, and the caller's
    # backtrack then discarded every row under that prefix.
    row = "crates/headroom-core/src/transforms/search_compressor.rs:12:body"
    assert _row_file(row) == "crates/headroom-core/src/transforms/search_compressor.rs"


def test_row_file_is_none_for_a_context_row():
    assert _row_file("src/app.py-40-before") is None


def test_trim_only_drops_the_partial_file_at_the_cut():
    rows = (
        [f"crates/headroom-core/src/a.rs:{n}:body" for n in range(1, 6)]
        + [f"crates/headroom-parity/src/b.rs:{n}:body" for n in range(1, 6)]
        + [f"crates/headroom-proxy/src/c.rs:{n}:body" for n in range(1, 6)]
    )
    cut = _trim_to_file_boundary(rows, 12)  # lands inside the third file
    assert cut == 10  # exactly the first two files survive
    assert {_row_file(r) for r in rows[:cut]} == {
        "crates/headroom-core/src/a.rs",
        "crates/headroom-parity/src/b.rs",
    }


def test_trim_pulls_context_rows_along_with_their_file():
    rows = [
        "src/a.py-1-before",
        "src/a.py:2:MATCH",
        "src/a.py-3-after",
        "src/b.py-9-before",
        "src/b.py:10:MATCH",
        "src/b.py-11-after",
    ]
    cut = _trim_to_file_boundary(rows, 5)  # inside b.py's block
    assert rows[:cut] == rows[:3]  # b.py's leading context row goes too


def test_trim_never_returns_an_empty_corpus():
    rows = [f"only/one.py:{n}:body" for n in range(1, 6)]
    assert _trim_to_file_boundary(rows, 3) > 0

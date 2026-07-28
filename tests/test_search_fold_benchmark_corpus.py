"""The search-fold benchmark's corpus trimming.

The benchmark's numbers are only meaningful if the corpus is what it claims to
be, so the row-trimming helper gets the same regression treatment as the fold
itself: a trim that silently drops the wrong rows changes the measured before /
after ratio without failing anything.
"""

from __future__ import annotations

from benchmarks.search_fold_benchmark import _belongs, _row_file, _trim_to_file_boundary


def test_row_file_returns_the_complete_hyphenated_path():
    # Splitting on '-' here reduced this to `crates/headroom`, and the caller's
    # backtrack then discarded every row under that prefix.
    row = "crates/headroom-core/src/transforms/search_compressor.rs:12:body"
    assert _row_file(row) == "crates/headroom-core/src/transforms/search_compressor.rs"


def test_row_file_is_none_for_a_context_row():
    assert _row_file("src/app.py-40-before") is None


def test_row_file_is_none_for_a_context_row_whose_body_holds_a_colon():
    # Splitting on the first ':' alone yields `src/partial.py-11-type`, a file
    # that does not exist, and the caller then backtracks over that one row and
    # stops — leaving the rest of the file in the corpus.
    assert _row_file("src/partial.py-11-type: value") is None
    assert _row_file("src/app.py-40-    d = {'k': 12}") is None


def test_trim_does_not_cut_inside_a_file_ending_in_a_colon_bodied_context_row():
    rows = [f"src/keep.py:{n}:body" for n in range(1, 4)] + [
        "src/partial.py:9:MATCH",
        "src/partial.py-10-plain",
        "src/partial.py-11-type: value",
        "src/partial.py-12-more",
    ]
    cut = _trim_to_file_boundary(rows, 6)  # lands on the colon-bodied row
    assert not any(r.startswith("src/partial.py") for r in rows[:cut])
    assert rows[:cut] == rows[:3]


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


def test_trim_drops_a_new_file_left_holding_only_leading_context():
    # `-B`/`-C` context precedes its match row, so a cut inside a new file's
    # leading context leaves the backward search pointing at the *previous*
    # file and nothing to remove. The orphan rows — and the `--` above them —
    # would leave a file represented by context alone, which ripgrep never
    # emits.
    rows = [
        "a/grok.py:70:MATCH",
        "a/grok.py-71-after",
        "a/grok.py-72-after",
        "--",
        "a/opencode.py-127-leading",
        "a/opencode.py-128-leading",
        "a/opencode.py:129:MATCH",
    ]
    cut = _trim_to_file_boundary(rows, 6)  # inside opencode.py's leading context
    assert rows[:cut] == rows[:3]
    assert "--" not in rows[:cut]
    assert not any("opencode" in r for r in rows[:cut])


def test_trim_keeps_trailing_context_of_the_last_surviving_file():
    # The second pass must not eat the kept file's own trailing context.
    rows = ["a/grok.py:70:MATCH", "a/grok.py-71-after", "--", "a/next.py-9-leading"]
    cut = _trim_to_file_boundary(rows, 4)
    assert rows[:cut] == rows[:2]


def test_trim_keeps_a_cut_that_already_sits_on_a_boundary():
    # Trimming unconditionally deleted a complete file: the backward walk ran
    # over the last kept file whether or not the cut was inside it.
    rows = [f"a/f1.py:{n}:b" for n in range(1, 4)] + [f"a/f2.py:{n}:b" for n in range(1, 4)]
    assert _trim_to_file_boundary(rows, 3) == 3


def test_trim_keeps_a_partial_file_when_one_file_fills_the_budget():
    # A query concentrated in one large file has no earlier boundary to find.
    # The walk reached the start and left a single row — a corpus that measures
    # nothing. A partial file measures the fold imperfectly, which is better.
    rows = [f"only/big.py:{n}:body" for n in range(1, 20)]
    assert _trim_to_file_boundary(rows, 10) == 10


def test_trim_does_not_treat_a_sibling_file_as_the_target_file():
    # `a/file.py-backup:1:x` starts with `a/file.py-`, so a prefix test applied
    # to match rows made a boundary between two sibling files look like the
    # inside of the first — and the trim deleted that file whole.
    rows = (
        [f"x/other.py:{n}:b" for n in range(1, 4)]
        + [f"a/file.py:{n}:b" for n in range(1, 4)]
        + [f"a/file.py-backup:{n}:b" for n in range(1, 4)]
    )
    assert _trim_to_file_boundary(rows, 6) == 6  # already a boundary
    assert _trim_to_file_boundary(rows, 8) == 6  # inside the sibling


def test_trim_crosses_group_separators_within_one_file():
    # `-C` splits one file into non-contiguous groups divided by `--`. Stopping
    # at the separator left the file's earlier groups in while its later ones
    # were dropped — still a cut inside a file.
    rows = (
        ["z/pre.py:1:M"]
        + [f"a/f.py-{n}-ctx" for n in (8, 9)]
        + ["a/f.py:10:MATCH"]
        + [f"a/f.py-{n}-ctx" for n in (11, 12)]
        + ["--"]
        + [f"a/f.py-{n}-ctx" for n in (98, 99)]
        + ["a/f.py:100:MATCH"]
    )
    cut = _trim_to_file_boundary(rows, 9)  # inside the second group
    assert not any(r.startswith("a/f.py") for r in rows[:cut])
    assert rows[:cut] == ["z/pre.py:1:M"]


def test_trim_does_not_attribute_a_siblings_leading_context_to_this_file():
    # `a/file.py-backup-20-leading` starts with `a/file.py-`, but a context row
    # of `a/file.py` would carry the line-number marker there —
    # `a/file.py-<digits>-`. Without that the sibling's orphan leading context
    # stays in the corpus with its match row beyond the cut.
    rows = (
        [f"z/pre.py:{n}:M" for n in (1, 2)]
        + [f"a/file.py:{n}:M" for n in (10, 11)]
        + ["--"]
        + ["a/file.py-backup-20-leading", "a/file.py-backup-21-leading"]
        + ["a/file.py-backup:22:MATCH"]
    )
    cut = _trim_to_file_boundary(rows, 7)
    assert not any("backup" in r for r in rows[:cut])
    assert rows[:cut] == rows[:4]


def test_belongs_still_accepts_this_files_own_context():
    assert _belongs("a/file.py-40-before", "a/file.py")
    assert not _belongs("a/file.py-backup-20-leading", "a/file.py")
    assert not _belongs("a/file.py-backup:1:x", "a/file.py")


def test_trim_backtracks_when_the_cutoff_lands_on_the_separator():
    # The cut can point at the `--` itself rather than into the group past it.
    # Then the row it checks is the separator, which belongs to no file, so the
    # backward pass never ran and the file kept its earlier groups.
    rows = (
        ["z/pre.py:1:M"]
        + [f"a/f.py-{n}-ctx" for n in (8, 9)]
        + ["a/f.py:10:MATCH"]
        + ["--"]
        + [f"a/f.py-{n}-ctx" for n in (98, 99)]
        + ["a/f.py:100:MATCH"]
    )
    for requested in (4, 5):
        cut = _trim_to_file_boundary(rows, requested)
        assert not any(r.startswith("a/f.py") for r in rows[:cut])


def test_trim_never_returns_an_empty_corpus():
    rows = [f"only/one.py:{n}:body" for n in range(1, 6)]
    assert _trim_to_file_boundary(rows, 3) > 0

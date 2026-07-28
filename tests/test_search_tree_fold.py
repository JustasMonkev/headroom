"""The composed search fold: ``search_tree_heading`` / ``search_tree_unheading``.

``search_heading`` factors a repeated file and ``search_dir_heading`` factors a
repeated directory, and ``compact_lossless`` picks one of them — so the ordinary
``grep -rn`` shape (several matches in each of several files under a shared
directory) only ever collected one of the two savings. This fold composes them
and adds consecutive-line elision and duplicate-body merging on top.

Every test here asserts the byte-exact round-trip, because that is the whole
contract: ``compact_lossless`` throws the fold away if the inverse doesn't
reproduce the input, so a fold that silently corrupts shows up as *no savings*
rather than as wrong content reaching the model.
"""

from __future__ import annotations

import random

import pytest

from headroom.transforms.lossless_compaction import (
    compact_lossless,
    search_dir_heading,
    search_heading,
    search_tree_heading,
    search_tree_unheading,
    search_unfold,
)


def roundtrips(text: str) -> bool:
    return search_tree_unheading(search_tree_heading(text)) == text


# --- the shape the older folds each only half-handled ---


def test_folds_directory_and_file_together():
    grep = "pkg/mod/a.py:1:alpha\npkg/mod/a.py:9:beta\npkg/mod/b.py:4:gamma\npkg/mod/b.py:7:delta\n"
    folded = search_tree_heading(grep)
    assert folded == "pkg/mod/\na.py\n1:alpha\n9:beta\nb.py\n4:gamma\n7:delta\n"
    assert search_tree_unheading(folded) == grep
    # Strictly better than either single-axis fold on this shape.
    assert len(folded) < len(search_heading(grep))
    assert len(folded) < len(search_dir_heading(grep))


def test_consecutive_line_numbers_elide_to_bare_separator():
    grep = "".join(f"src/x.py:{n}:line {n}\n" for n in range(10, 15))
    folded = search_tree_heading(grep)
    assert folded == "src/\nx.py\n10:line 10\n:line 11\n:line 12\n:line 13\n:line 14\n"
    assert search_tree_unheading(folded) == grep


def test_repeated_body_in_one_file_becomes_a_line_list():
    grep = "a/f.py:3:import os\na/f.py:8:other\na/f.py:20:import os\n"
    folded = search_tree_heading(grep)
    assert "3,20:import os" in folded
    # The merge moves line 20 up next to line 3; the inverse re-sorts the block.
    assert search_tree_unheading(folded) == grep


def test_directory_header_not_repeated_across_files_in_same_dir():
    grep = "".join(f"deep/nested/dir/f{i}.py:{i}:hit {i}\n" for i in range(5))
    assert search_tree_heading(grep).count("deep/nested/dir/") == 1
    assert roundtrips(grep)


# --- ripgrep context rows: the case that used to inflate ---


def test_ripgrep_context_rows_fold_with_dash_separator():
    rg = (
        "src/app.py-40-before\n"
        "src/app.py:41:MATCH\n"
        "src/app.py-42-after\n"
        "src/other.py-7-ctx\n"
        "src/other.py:8:MATCH\n"
    )
    folded = search_tree_heading(rg)
    assert search_tree_unheading(folded) == rg
    assert len(folded) < len(rg)
    # The separator character is preserved per-row, not normalised to ':'.
    assert "-before" in folded and ":MATCH" in folded


def test_ripgrep_group_separator_survives():
    # `--` is also a syntactically valid increment row (sep '-', body '-'), so
    # both directions have to special-case it or the groups merge into one file.
    rg = "a/x.py-1-ctx\na/x.py:2:hit\n--\nb/y.py-9-ctx\nb/y.py:10:hit\n"
    folded = search_tree_heading(rg)
    assert "--" in folded
    assert search_tree_unheading(folded) == rg


def test_body_that_is_exactly_a_group_separator_is_not_elided():
    # Row whose content is '-' at a consecutive line: eliding it would emit the
    # literal '--', which the unfold reads as a group separator.
    rg = "a/x.py-1-ctx\na/x.py-2--\n"
    folded = search_tree_heading(rg)
    assert search_tree_unheading(folded) == rg


# --- paths the parser has to not mangle ---


def test_windows_drive_paths_roundtrip():
    grep = "C:\\proj\\src\\a.py:12:hit one\nC:\\proj\\src\\a.py:13:hit two\n"
    assert roundtrips(grep)


@pytest.mark.parametrize(
    "grep",
    [
        "logs/2026-05-03/app.log:12:ERROR boom\nlogs/2026-05-03/app.log:14:ERROR again\n",
        "advisories/CVE-2021-44228.md:9:log4j\n",
        "migrations/20240101-002-add_users.sql:3:CREATE TABLE\n",
        "a/.pre-commit-config.yaml:42:repos:\n",
    ],
)
def test_dashes_and_dates_in_paths_roundtrip(grep):
    assert roundtrips(grep)


def test_bare_basename_rows_roundtrip():
    grep = "README.md:1:# Title\nREADME.md:5:body\n"
    assert roundtrips(grep)


def test_directory_only_path_is_left_unfolded():
    # `src/:1:x` has no basename to head with — must not emit an empty header.
    grep = "src/:1:x\nsrc/:2:y\n"
    assert search_tree_unheading(search_tree_heading(grep)) == grep


# --- number and content shapes that would break naive handling ---


def test_zero_padded_line_numbers_are_never_merged_or_elided():
    # Both the merge form and the elision form regenerate the number through
    # int(), which would drop the padding.
    grep = "a/f.py:007:dup\na/f.py:008:dup\n"
    folded = search_tree_heading(grep)
    assert "007" in folded and "008" in folded
    assert search_tree_unheading(folded) == grep


def test_out_of_order_block_still_folds_but_does_not_merge():
    # Merging relies on re-sorting the block to undo it, so a block that wasn't
    # ascending to begin with must skip merging — and still keep the header and
    # elision savings rather than costing the whole payload its fold.
    grep = "a/f.py:106:dup\na/f.py:114:other\na/f.py:11:dup\n"
    folded = search_tree_heading(grep)
    assert "106,11" not in folded
    assert search_tree_unheading(folded) == grep
    assert len(folded) < len(grep)


def test_content_containing_colons_and_numbers_roundtrips():
    grep = "a/f.py:1:url = http://x/y:8080\na/f.py:2:d = {'k': 12}\na/f.py:3:3:not a row\n"
    assert roundtrips(grep)


def test_empty_body_rows_roundtrip():
    grep = "a/f.py:1:\na/f.py:2:\na/f.py:3:x\n"
    assert roundtrips(grep)


def test_non_ascii_bodies_roundtrip():
    grep = "a/f.py:1:# 圧縮テスト\na/f.py:2:# ünïcödé\n"
    assert roundtrips(grep)


# --- mixed / non-grep content must pass through untouched ---


def test_passthrough_lines_end_the_grouping():
    mixed = "a/x.py:1:hit\n== banner ==\na/x.py:2:hit again\n"
    assert roundtrips(mixed)
    assert "== banner ==" in search_tree_heading(mixed)


def test_plain_prose_is_returned_unchanged():
    prose = "just some text\nwith no grep rows at all\n"
    assert search_tree_heading(prose) == prose
    assert search_tree_unheading(prose) == prose


@pytest.mark.parametrize("text", ["", "\n", "a/f.py:1:only", "a/f.py:1:only\n"])
def test_degenerate_inputs_roundtrip(text):
    assert roundtrips(text)


def test_trailing_newline_presence_is_preserved():
    grep = "a/f.py:1:x\na/f.py:2:y"
    assert not search_tree_heading(grep).endswith("\n")
    assert search_tree_heading(grep + "\n").endswith("\n")
    assert roundtrips(grep) and roundtrips(grep + "\n")


def test_randomised_result_sets_roundtrip():
    rng = random.Random(20260728)
    dirs = ["src/", "src/deep/", "a-b/", "logs/2026-05-03/", ""]
    bodies = ["import os", "    return x", "", "k: 'v:1'", "-", "def f(a, b):"]
    for _ in range(200):
        rows = []
        for _ in range(rng.randint(1, 25)):
            path = f"{rng.choice(dirs)}f{rng.randint(0, 3)}.py"
            sep = rng.choice([":", "-"])
            rows.append(f"{path}{sep}{rng.randint(1, 200)}{sep}{rng.choice(bodies)}")
        rows.sort(key=lambda r: r.split(":")[0].split("-")[0])
        text = "\n".join(rows) + ("\n" if rng.random() < 0.5 else "")
        assert roundtrips(text), text


# --- integration with compact_lossless / search_unfold ---


def test_compact_lossless_prefers_the_tree_fold_and_stays_recoverable():
    # Several matches in each of several files under one directory: the shape
    # where the old folds each captured only half the repetition.
    grep = "".join(
        f"pkg/mod/f{f}.py:{ln}:    value = compute({ln})\n" for f in range(3) for ln in range(10)
    )
    out = compact_lossless(grep, "search")
    assert len(out) < len(search_heading(grep))
    assert len(out) < len(search_dir_heading(grep))
    assert search_unfold(out) == grep


def test_compact_lossless_never_grows_the_payload():
    # A result set with nothing to factor: one row, unique path, unique body.
    grep = "a/only.py:1:x\n"
    assert len(compact_lossless(grep, "search")) <= len(grep)


def test_search_unfold_recovers_every_fold_shape():
    grep = "pkg/a.py:1:one\npkg/a.py:2:two\npkg/b.py:3:three\n"
    for folded in (search_tree_heading(grep), search_heading(grep), search_dir_heading(grep)):
        assert search_unfold(folded) == grep


def test_search_unfold_passes_through_content_that_is_not_a_fold():
    for text in ("", "plain text\n", "a/f.py:1:already expanded\n"):
        assert search_unfold(text) == text

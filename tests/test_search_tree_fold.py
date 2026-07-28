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
    _tree_colon_row,
    _tree_split_row,
    compact_lossless,
    search_dir_heading,
    search_fold_recovers,
    search_heading,
    search_tree_heading,
    search_tree_unheading,
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


def test_a_file_named_like_the_group_separator_is_left_unfolded():
    # A file really can be named `--`. Heading with it would be
    # indistinguishable from ripgrep's separator between non-contiguous context
    # groups, so the rows beneath would read as orphans belonging to nothing.
    grep = "a/--:1:x\na/--:2:y\n"
    assert search_tree_heading(grep) == grep
    assert search_tree_unheading(search_tree_heading(grep)) == grep
    # The rest of a payload still folds around it.
    mixed = grep + "b/f.py:3:z\nb/f.py:4:w\n"
    folded = search_tree_heading(mixed)
    assert "\n--\n" not in folded
    assert len(folded) < len(mixed)
    assert search_tree_unheading(folded) == mixed


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


# --- digits: the scanner, the regexes and int() must agree ---


@pytest.mark.parametrize(
    "grep",
    [
        "a:\u00b2:1:x\n",  # isdigit() but not \d and not int()-able
        "a/f.py:\u00b2:hit\n",
        "a:" + "9" * 4301 + ":b\n",  # past CPython's 4300-digit int() limit
        "a/f.py:" + "9" * 4301 + ":hit\n",
        "a/f.py:\u0661\u0662:arabic-indic digits\n",
    ],
)
def test_exotic_digit_runs_never_raise_and_roundtrip(grep):
    # Anything the scanner won't take is simply not a data row, so it passes
    # through — which still round-trips.
    search_tree_heading(grep)
    search_tree_unheading(grep)
    assert roundtrips(grep)
    assert compact_lossless(grep, "search") == grep or search_fold_recovers(
        compact_lossless(grep, "search"), grep
    )


def test_line_numbers_up_to_the_digit_bound_still_fold():
    grep = f"a/f.py:{10**17}:hit one\na/f.py:{10**17 + 1}:hit two\n"
    assert roundtrips(grep)
    assert len(search_tree_heading(grep)) < len(grep)


# --- the marker scan must not be quadratic in line length ---


def test_pathological_marker_line_is_bounded():
    # A single line of nothing but accepted-then-rejected dash markers. Before
    # the scan bounds this was quadratic: ~250ms at 31 KB, 4x per doubling.
    import time

    line = "a" + "-1" * 40000 + "/x"  # ~156 KB, one line
    start = time.perf_counter()
    out = compact_lossless(line, "search")
    assert time.perf_counter() - start < 2.0
    assert out == line


def test_long_bodies_still_fold_despite_the_scan_bounds():
    # The bound is on where the marker may START, not on line length, so a row
    # with a huge body still folds.
    grep = "".join(f"a/f.py:{n}:{'x' * 20000}\n" for n in (1, 2))
    assert roundtrips(grep)
    assert len(search_tree_heading(grep)) < len(grep)


# --- context rows are anchored to files the colon pass established ---
#
# A dash is legal inside a path, so no textual rule can say which `-<digits>-`
# is the line-number marker. grep and ripgrep always print the matching line
# itself with `:`, so a file with context rows always has a match row too, and
# the fold anchors on that instead of guessing.


def test_context_row_binds_to_the_dashed_filename_not_its_first_marker():
    # `report.log-2026-backup` is the file. Guessing stopped at the extension
    # and read this as file `report.log`, line `2026` — byte-reversible, so the
    # round-trip guard passed it through to the model as a bogus file heading.
    rg = "report.log-2026-backup:1:MATCH\nreport.log-2026-backup-2-after\n"
    folded = search_tree_heading(rg)
    assert folded.startswith("report.log-2026-backup\n")
    assert "\n2026-backup" not in folded
    assert search_tree_unheading(folded) == rg


def test_unanchored_dash_row_is_left_alone():
    assert _tree_split_row("report.log-2026-backup-1-before") is None


def test_ripgrep_json_records_are_never_folded():
    # `rg --json` is a search command, so it reaches the search fold. A `-12-`
    # inside a matched string used to split the record, factoring the repeated
    # JSON prefix into headings and emitting invalid JSON fragments as rows.
    records = (
        '{"type":"match","data":{"path":{"text":"src/a.rs"},'
        '"lines":{"text":"foo-12-bar"},"line_number":7}}\n'
        '{"type":"match","data":{"path":{"text":"src/a.rs"},'
        '"lines":{"text":"foo-12-baz"},"line_number":9}}\n'
    )
    assert search_tree_heading(records) == records
    assert compact_lossless(records, "search") == records


@pytest.mark.parametrize("embedded", ["foo-12-bar", "foo:12:bar"])
def test_ripgrep_json_is_rejected_for_both_marker_forms(embedded):
    # Anchoring context rows closes the dash form only — the colon scan reaches
    # these records on its own, skipping the record's own structural colons and
    # accepting the one embedded in a matched string.
    records = "".join(
        '{"type":"match","data":{"path":{"text":"src/a.rs"},'
        f'"lines":{{"text":"{embedded}"}},"line_number":{n}}}}}\n'
        for n in (7, 9, 11)
    )
    assert search_tree_heading(records) == records
    assert compact_lossless(records, "search") == records


def test_braces_and_quotes_in_a_body_do_not_block_the_fold():
    # The structural guard only looks at the path, so ordinary code bodies —
    # dict literals, string literals — still fold.
    grep = 'a/f.py:1:def x(): return {"k": 2}\na/f.py:2:def y(): return {"k": 4}\n'
    folded = search_tree_heading(grep)
    assert len(folded) < len(grep)
    assert search_tree_unheading(folded) == grep


# A row that reads as two different files is left unparsed. Three tie-breaks
# were tried before this — longest path, then the surrounding block, then
# ascending line order — and each fixed one shape by breaking another, because
# the competing shapes are textually identical with opposite right answers.


@pytest.mark.parametrize(
    "rg",
    [
        # The same three rows in both orders: any tie-break gets one of them wrong.
        "report.log:2025:MATCH\nreport.log-2026-backup-1-before\nreport.log-2026-backup:2:OTHER\n",
        "report.log:1:MATCH one\n"
        "report.log-2026-backup:2:MATCH two\n"
        "report.log-2026-backup-1-before\n",
        # Nothing around the row at all.
        "report.log:1:a\nreport.log-2026-backup:2:b\n== banner ==\n"
        "report.log-2026-backup-1-before\n",
    ],
)
def test_rows_that_read_as_two_files_are_left_unparsed(rg):
    folded = search_tree_heading(rg)
    assert "report.log-2026-backup-1-before" in folded  # emitted verbatim
    assert search_tree_unheading(folded) == rg


def test_a_quoted_file_line_reference_in_a_context_body_is_left_unparsed():
    # `src/app.py-40-other.py:12:foo` is a match row in `src/app.py-40-other.py`
    # or context in `src/app.py`. Taking the colon reading folded the block
    # under a fabricated heading; taking the dash reading breaks the mirror
    # case below. Neither, then — and the rest of the block still folds.
    rg = "src/app.py-40-other.py:12:foo\nsrc/app.py:41:MATCH\nsrc/app.py-42-after\n"
    folded = search_tree_heading(rg)
    assert "src/app.py-40-other.py:12:foo" in folded
    assert "app.py-40-other.py\n" not in folded  # no fabricated heading
    assert search_tree_unheading(folded) == rg
    assert len(folded) < len(rg)


def test_a_match_row_extending_an_anchored_path_is_left_unparsed():
    # The mirror image: `src/a-1-b.py:12:x` is a real match row, but with
    # `src/a` anchored it also reads as context in `src/a` at line 1.
    rg = "src/a:5:established\nsrc/a-1-b.py:12:real match row\n"
    folded = search_tree_heading(rg)
    assert "src/a-1-b.py:12:real match row" in folded
    assert search_tree_unheading(folded) == rg


@pytest.mark.parametrize("body", ["d = {'k': 12}", "raise ValueError('x: 1')", "url = http://a/b"])
def test_context_bodies_holding_colons_still_fold(body):
    # The common case, and it must keep folding: nothing in these bodies reads
    # as a `path:<digits>:` marker, so there is no rival interpretation.
    rg = f"s/a.py-9-{body}\ns/a.py:10:MATCH\ns/a.py-11-{body}\n"
    folded = search_tree_heading(rg)
    assert len(folded) < len(rg)
    assert search_tree_unheading(folded) == rg


def test_a_bare_timestamp_body_declines_and_that_is_affordable():
    # `s/a.py-9-t = 12:34:56` reads as context in `s/a.py` or as a match in a
    # file called `s/a.py-9-t = 12`. Contrived as the second sounds, it is the
    # same shape as `report.log-2026-backup name:2:MATCH`, where it is the right
    # answer — so this declines rather than pick.
    #
    # Affordable because it is rare in practice, not because it is cheap per
    # row: on `rg -C 2` over headroom/ and crates/ the folded corpus is 63.6% of
    # raw whether these rows fold or decline, for both the `context` and `error`
    # queries. The surrounding rows still fold either way.
    rg = "s/a.py-9-t = 12:34:56\ns/a.py:10:MATCH\ns/a.py-11-u = 01:02:03\n"
    folded = search_tree_heading(rg)
    assert "s/a.py-9-t = 12:34:56" in folded  # verbatim, not re-filed
    assert search_tree_unheading(folded) == rg
    # Whatever candidate compact_lossless ends up picking here (the dir fold
    # wins on this shape) stays byte-recoverable.
    out = compact_lossless(rg, "search")
    assert out == rg or search_fold_recovers(out, rg)

    # In a block with something left to factor, the surrounding rows still fold
    # and only the timestamp row stays verbatim.
    bigger = rg + "".join(f"s/a.py:{n}:MATCH {n}\n" for n in range(12, 20))
    folded = search_tree_heading(bigger)
    assert "s/a.py-9-t = 12:34:56" in folded
    assert len(folded) < len(bigger)
    assert search_tree_unheading(folded) == bigger


def test_the_marker_colon_must_be_the_lines_first_colon():
    # A path cannot contain a colon — the claim the whole fold rests on. Walking
    # on to a later colon when the first isn't followed by digits drops that
    # colon into the path: `rg --heading -n -o` emits `1:see/foo:12:value`,
    # which parsed as a file named `1:see/foo` under a `1:see/` directory.
    assert _tree_colon_row("1:see/foo:12:value") is None
    headed = "src/main.rs\n1:see/foo:12:value\n2:see/foo:12:other\n"
    assert search_tree_heading(headed) == headed


def test_a_spaced_path_holding_a_dash_marker_is_not_filed_under_its_prefix():
    # The space guard suppresses the colon reading here, so without asking what
    # it would have said the dash tier files this under `report.log` at line
    # 2026 with no disagreement to notice.
    rg = "report.log:1:hit\nreport.log-2026-backup name:2:MATCH\n"
    folded = search_tree_heading(rg)
    assert "report.log-2026-backup name:2:MATCH" in folded  # verbatim
    assert "\n2026-backup name" not in folded  # not re-filed under report.log
    assert search_tree_unheading(folded) == rg


def test_a_truncated_marker_scan_declines_rather_than_guessing():
    # The scan's bounds exist for speed. A walk stopped by them cannot say
    # whether a second anchored reading lay beyond, and returning the first
    # candidate as if it were the only one filed a long file's context under a
    # short prefix that happened to be anchored.
    long_path = "a-1-" + "-".join("x" for _ in range(70)) + ".py"
    rg = f"a:5:anchor\n{long_path}:9:MATCH\n{long_path}-10-ctx\n"
    folded = search_tree_heading(rg)
    assert f"{long_path}-10-ctx" in folded  # verbatim, not filed under `a`
    assert search_tree_unheading(folded) == rg


def test_a_filename_holding_a_quote_is_not_reinterpreted_as_context():
    # The structural guard exists to keep JSON records out. A Unix filename may
    # legitimately hold a quote or brace, and rejecting its colon reading left
    # the dash tier free to claim the row as context of an anchored prefix.
    rg = 'a:5:anchor\na-1-"file0":2:MATCH\na-1-"file1":3:MATCH\n'
    folded = search_tree_heading(rg)
    assert 'a-1-"file0":2:MATCH' in folded  # verbatim
    assert 'a\n5:anchor\n1-"file0"' not in folded  # not filed under `a` at line 1
    assert search_tree_unheading(folded) == rg


def test_unambiguous_context_rows_still_fold():
    # Only one anchored path fits and the colon tier does not claim it.
    rg = "src/app.py-40-before\nsrc/app.py:41:MATCH\nsrc/app.py-42-after\n"
    folded = search_tree_heading(rg)
    assert len(folded) < len(rg)
    assert search_tree_unheading(folded) == rg


def test_a_reference_inside_a_context_body_does_not_hijack_the_parse():
    # The body of a context row quoting `other.py:12:` must not be read as a
    # path — that is what the space guard on the colon tier is for.
    assert _tree_colon_row("src/app.py-40-    see other.py:12: for details") is None


def test_paths_containing_spaces_still_fold():
    # A line with no dash marker anywhere cannot be a context row, so a space
    # before the marker is just part of the path.
    rg = (
        "very/long/directory name/f.py-1-before\n"
        "very/long/directory name/f.py:2:MATCH\n"
        "very/long/directory name/f.py-3-after\n"
        "very/long/directory name/g.py-8-before\n"
        "very/long/directory name/g.py:9:MATCH\n"
    )
    folded = compact_lossless(rg, "search")
    assert search_fold_recovers(folded, rg)
    assert len(folded) / len(rg) < 0.55


# --- savings floors: these pin the win, so a later parser change cannot
# quietly trade compression away while every round-trip test still passes ---


def _corpus(kind: str) -> str:
    if kind == "grep-rn":  # several matches in each of several files, one dir
        return "".join(
            f"headroom/transforms/module_{f}.py:{ln}:    result = compute(value_{ln})\n"
            for f in range(8)
            for ln in range(1, 13)
        )
    if kind == "context":  # rg -C 1: match rows interleaved with context rows
        return "".join(
            f"headroom/proxy/handler_{f}.py-{ln - 1}-    before line\n"
            f"headroom/proxy/handler_{f}.py:{ln}:    logger.warning(msg)\n"
            f"headroom/proxy/handler_{f}.py-{ln + 1}-    after line\n"
            for f in range(6)
            for ln in range(2, 20, 3)
        )
    if kind == "one-per-file":  # grep -rn across many files, one hit each
        return "".join(f"headroom/cli/command_{f:03d}.py:{f + 1}:import os\n" for f in range(120))
    raise AssertionError(kind)


@pytest.mark.parametrize(
    ("kind", "ceiling"),
    [
        # Ceilings sit just above what the fold achieves today, so an erosion
        # trips them while ordinary noise does not. Measured: 50.5 / 13.2 / 68.7.
        ("grep-rn", 0.53),
        ("context", 0.16),
        ("one-per-file", 0.71),
    ],
)
def test_savings_floor_per_shape(kind, ceiling):
    corpus = _corpus(kind)
    folded = compact_lossless(corpus, "search")
    assert search_fold_recovers(folded, corpus)
    ratio = len(folded) / len(corpus)
    assert ratio <= ceiling, f"{kind}: fold kept {ratio:.1%}, ceiling {ceiling:.0%}"


@pytest.mark.parametrize("kind", ["grep-rn", "context"])
def test_tree_fold_beats_both_older_folds(kind):
    corpus = _corpus(kind)
    assert len(search_tree_heading(corpus)) < len(search_heading(corpus))
    assert len(search_tree_heading(corpus)) < len(search_dir_heading(corpus))


def test_tree_fold_ties_the_dir_fold_when_every_file_has_one_match():
    # With a single row per file the tree fold spends a newline on the file
    # header and saves the basename on the row — an exact wash. This is why all
    # three folds stay as candidates rather than the tree fold replacing them.
    corpus = _corpus("one-per-file")
    assert len(search_tree_heading(corpus)) == len(search_dir_heading(corpus))
    assert len(search_tree_heading(corpus)) < len(search_heading(corpus))


def test_fold_stays_linear_in_payload_size():
    # Guards the whole pipeline, not just the marker scan: the colon pass, the
    # block walk and the three-candidate selection all have to stay linear.
    import time

    small, big = _corpus("grep-rn"), _corpus("grep-rn") * 8
    timings = []
    for corpus in (small, big):
        start = time.perf_counter()
        for _ in range(3):
            compact_lossless(corpus, "search")
        timings.append((time.perf_counter() - start) / 3)
    # 8x the input must not cost more than 24x the time (3x headroom for noise).
    assert timings[1] < timings[0] * 24, f"{timings[0]:.4f}s -> {timings[1]:.4f}s for 8x input"


# --- integration with compact_lossless ---


def test_compact_lossless_prefers_the_tree_fold_and_stays_recoverable():
    # Several matches in each of several files under one directory: the shape
    # where the old folds each captured only half the repetition.
    grep = "".join(
        f"pkg/mod/f{f}.py:{ln}:    value = compute({ln})\n" for f in range(3) for ln in range(10)
    )
    out = compact_lossless(grep, "search")
    assert len(out) < len(search_heading(grep))
    assert len(out) < len(search_dir_heading(grep))
    assert search_fold_recovers(out, grep)


def test_compact_lossless_never_grows_the_payload():
    # A result set with nothing to factor: one row, unique path, unique body.
    grep = "a/only.py:1:x\n"
    assert len(compact_lossless(grep, "search")) <= len(grep)


def test_search_fold_recovers_accepts_every_fold_shape():
    grep = "pkg/a.py:1:one\npkg/a.py:2:two\npkg/b.py:3:three\n"
    for folded in (search_tree_heading(grep), search_heading(grep), search_dir_heading(grep)):
        assert search_fold_recovers(folded, grep)


def test_search_fold_recovers_rejects_a_fold_of_different_content():
    grep = "pkg/a.py:1:one\npkg/a.py:2:two\n"
    other = "pkg/a.py:1:one\npkg/a.py:2:CHANGED\n"
    assert not search_fold_recovers(search_tree_heading(grep), other)
    assert not search_fold_recovers("totally unrelated text", grep)


@pytest.mark.parametrize(
    "original",
    [
        # A grep result preceded by a literal directory banner. The file fold
        # wins here; the tree inverse also claims the blob and expands it into
        # something that re-folds to the same bytes, so an inverse chosen
        # without the original would silently fabricate `docs/` onto both paths.
        "docs/\nREADME.md:12:hello\nREADME.md:40:world\n",
        # A rootless path beside one with a directory: the tree fold declines
        # this block, so the file fold wins.
        "setup.py:10:import os\nsrc/util.py:3:def f():\nsrc/util.py:9:    return\n",
        # A trailing passthrough line that is itself data-row-shaped.
        "b.py:1:x\nb.py:5:y\n-\n",
    ],
)
def test_ambiguous_blobs_are_still_recoverable_given_the_original(original):
    out = compact_lossless(original, "search")
    assert out == original or search_fold_recovers(out, original)


# --------------------------------------------------------------------------
# C12: the composed dir+file fold
#
# `search_heading` factors a repeated FILE and `search_dir_heading` factors a
# repeated DIRECTORY; picking whichever is smaller ALONE leaves the other axis
# unfolded. `search_tree_heading` normally composes both, but it declines on
# shapes it cannot parse unambiguously (a space in the path plus a `-<digits>-`
# marker inside it). On those, the explicit composition is the only fold left.
# --------------------------------------------------------------------------
def test_composed_dir_then_file_fold_wins_where_the_tree_fold_declines():
    from headroom.transforms.lossless_compaction import (
        compact_lossless,
        search_dir_heading,
        search_heading,
        search_tree_heading,
    )

    grep = "\n".join(
        f"my project/logs 2026-05-03/{mod}.py:{10 + i}:    value = compute(item, ctx)"
        for mod in ("alpha", "beta", "gamma")
        for i in range(4)
    )

    # The tree fold declines this shape outright.
    assert search_tree_heading(grep) == grep

    out = compact_lossless(grep, "search")
    assert len(out) < len(search_heading(grep))
    assert len(out) < len(search_dir_heading(grep))


def test_composed_fold_round_trips_exactly():
    from headroom.transforms.lossless_compaction import (
        search_dir_heading,
        search_dir_unheading,
        search_heading,
        search_unheading,
    )

    grep = "\n".join(
        f"my project/logs 2026-05-03/{mod}.py:{10 + i}:    value = compute(item, ctx)"
        for mod in ("alpha", "beta", "gamma")
        for i in range(4)
    )
    folded = search_heading(search_dir_heading(grep))
    assert search_dir_unheading(search_unheading(folded)) == grep


def test_composed_fold_never_wins_by_breaking_the_round_trip():
    """Every candidate is verified; a composition that can't invert must lose."""
    from headroom.transforms.lossless_compaction import compact_lossless

    # Content that is itself heading-shaped — the folds cannot survive it.
    grep = "src/\na.py\n12:body\nsrc/a.py:12:body\n"
    out = compact_lossless(grep, "search")
    assert len(out) <= len(grep)

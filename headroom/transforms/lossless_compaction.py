"""Format-native, reversible lossless compaction for no-CCR proxy mode.

Every helper here is pure stdlib and keeps its output *looking like its own
type* — grep stays grep, logs stay logs, diffs stay diffs. No retrieval
marker (``<<ccr:…>>`` / ``Retrieve …``) is ever emitted, so the proxy needs
no MCP retrieve round-trip to stay recoverable.

The reversible transforms ship with exact inverses and are self-checked at
runtime by :func:`compact_lossless`: if a round-trip does not reproduce the
original (modulo intentionally-dropped non-semantic bits such as ANSI color)
or the result is not actually smaller, the original content is returned
unchanged. Nothing here raises.
"""

from __future__ import annotations

import re

__all__ = [
    "strip_ansi",
    "collapse_runs",
    "expand_runs",
    "is_run_collapsed",
    "fold_repeated_blocks",
    "unfold_repeated_blocks",
    "search_heading",
    "search_unheading",
    "search_dir_heading",
    "search_dir_unheading",
    "search_tree_heading",
    "search_tree_unheading",
    "search_fold_recovers",
    "diff_strip_index",
    "compact_lossless",
]

# ANSI CSI SGR (color/style) escape sequences: ESC [ ... m. Color is
# non-semantic, so stripping it is a safe (one-way) lossless-of-meaning op.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# syslog-style run-collapse marker. The count is captured for exact inversion.
_RUN_MARKER_RE = re.compile(r"^\.\.\. \(repeated (\d+) times\)$")

# multi-line block back-reference marker. Length and distance (both in lines,
# in ORIGINAL coordinates) are captured for exact inversion: everything before
# a marker expands to the exact original prefix, so `distance` lines back in
# the expanded output is the block's first occurrence.
_BLOCK_MARKER_RE = re.compile(r"^\.\.\. \(repeats (\d+) lines from (\d+) lines back\)$")

# fold_repeated_blocks search bounds: minimum/maximum block length worth a
# marker, candidate anchors per line, and an input size cap so the scan stays
# negligible on huge payloads.
_FOLD_MIN_BLOCK = 3
_FOLD_MAX_BLOCK = 64
_FOLD_MAX_CANDIDATES = 8
_FOLD_MAX_LINES = 20_000

# grep/ripgrep default row shape: ``path:line:content``. ``line`` is digits;
# ``path`` must not itself look like ``line:content`` (i.e. not start with a
# bare number) so we don't mis-split a heading-form ``line:content`` row.
_GREP_ROW_RE = re.compile(r"^(?P<path>[^\n:]+):(?P<line>\d+):(?P<content>.*)$")
# heading-form data row (``line:content``) produced by search_heading.
_HEADING_ROW_RE = re.compile(r"^(?P<line>\d+):(?P<content>.*)$")

# unified-diff ``index <sha>..<sha> <mode>`` line. The diff still applies
# without it (git only uses it for rename/blob bookkeeping).
_DIFF_INDEX_RE = re.compile(r"^index [0-9a-fA-F]+\.\.[0-9a-fA-F]+( [0-7]+)?$")


def strip_ansi(text: str) -> str:
    """Remove ANSI CSI/SGR (color) escape sequences. Color is non-semantic."""
    return _ANSI_RE.sub("", text)


def _split_keep_trailing(text: str) -> tuple[list[str], bool]:
    """Split into lines, remembering whether a trailing newline was present.

    Returns (lines, had_trailing_newline). This lets the run helpers rejoin
    byte-exactly instead of always appending or always dropping a newline.
    """
    if text == "":
        return [], False
    had_trailing = text.endswith("\n")
    body = text[:-1] if had_trailing else text
    return body.split("\n"), had_trailing


def _join(lines: list[str], had_trailing: bool) -> str:
    out = "\n".join(lines)
    if had_trailing:
        out += "\n"
    return out


def collapse_runs(text: str) -> str:
    """Collapse runs of >=2 identical consecutive lines (syslog convention).

    A run of N (N>=2) identical lines becomes the line once followed by
    ``... (repeated N times)``. Exact inverse: :func:`expand_runs`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        j = i
        while j + 1 < n and lines[j + 1] == lines[i]:
            j += 1
        run_len = j - i + 1
        if run_len >= 2:
            out.append(lines[i])
            out.append(f"... (repeated {run_len} times)")
        else:
            out.append(lines[i])
        i = j + 1
    return _join(out, had_trailing)


def expand_runs(text: str) -> str:
    """Exact inverse of :func:`collapse_runs`."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if i + 1 < n:
            m = _RUN_MARKER_RE.match(lines[i + 1])
            if m:
                count = int(m.group(1))
                out.extend([line] * count)
                i += 2
                continue
        out.append(line)
        i += 1
    return _join(out, had_trailing)


def is_run_collapsed(text: str) -> bool:
    """True if any run-collapse marker line is present."""
    for line in text.split("\n"):
        if _RUN_MARKER_RE.match(line):
            return True
    return False


def fold_repeated_blocks(text: str) -> str:
    """Collapse multi-line blocks that repeat earlier content into back-refs.

    The block-level generalization of :func:`collapse_runs`: a run of K
    consecutive lines (K >= 3) that exactly reproduces K lines seen D lines
    earlier becomes ``... (repeats K lines from D lines back)``. The repeats
    need not be adjacent, which is what config payloads actually look like —
    k8s container stanzas repeat with only the ``name:`` line differing, so
    their identical tails fold even though no two whole stanzas are
    consecutive. Coordinates are in original lines: the fold is only taken
    when the block does not overlap its anchor (K <= D), so on expansion the
    referenced region is always already reconstructed.
    Exact inverse: :func:`unfold_repeated_blocks`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    n = len(lines)
    if n < _FOLD_MIN_BLOCK * 2 or n > _FOLD_MAX_LINES:
        return text
    positions: dict[str, list[int]] = {}
    out: list[str] = []
    i = 0
    while i < n:
        best_len = 0
        best_dist = 0
        for q in reversed(positions.get(lines[i], ())):
            max_len = min(_FOLD_MAX_BLOCK, n - i, i - q)
            length = 0
            while length < max_len and lines[q + length] == lines[i + length]:
                length += 1
            if length > best_len:
                best_len = length
                best_dist = i - q
        if best_len >= _FOLD_MIN_BLOCK:
            marker = f"... (repeats {best_len} lines from {best_dist} lines back)"
            block_chars = sum(len(lines[i + k]) + 1 for k in range(best_len))
            if block_chars > len(marker) + 1:
                out.append(marker)
                for k in range(best_len):
                    _remember(positions, lines[i + k], i + k)
                i += best_len
                continue
        _remember(positions, lines[i], i)
        out.append(lines[i])
        i += 1
    return _join(out, had_trailing)


def _remember(positions: dict[str, list[int]], line: str, index: int) -> None:
    """Track recent original positions of `line`, bounded per distinct line."""
    bucket = positions.setdefault(line, [])
    bucket.append(index)
    if len(bucket) > _FOLD_MAX_CANDIDATES:
        del bucket[0]


def unfold_repeated_blocks(text: str) -> str:
    """Exact inverse of :func:`fold_repeated_blocks`."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    for line in lines:
        m = _BLOCK_MARKER_RE.match(line)
        if m:
            length, dist = int(m.group(1)), int(m.group(2))
            start = len(out) - dist
            if start >= 0 and length <= dist:
                out.extend(out[start : start + length])
                continue
        out.append(line)
    return _join(out, had_trailing)


def search_heading(text: str) -> str:
    """Convert grep ``path:line:content`` rows into ripgrep --heading form.

    Consecutive rows sharing a path collapse to the path once on its own line
    (a *header* line), then ``line:content`` rows beneath it. Lines that don't
    match the ``path:line:content`` shape are passed through untouched. No
    blank separators are inserted (they would be ambiguous with passthrough
    content), keeping the transform exactly reversible via
    :func:`search_unheading`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_path: str | None = None
    for line in lines:
        m = _GREP_ROW_RE.match(line)
        if m:
            path = m.group("path")
            if path != current_path:
                out.append(path)
                current_path = path
            out.append(f"{m.group('line')}:{m.group('content')}")
        else:
            # Any non-grep-row line ends the current file grouping.
            out.append(line)
            current_path = None
    return _join(out, had_trailing)


def search_unheading(text: str) -> str:
    """Exact inverse of :func:`search_heading`.

    A *header* line is any line that is not itself a ``line:content`` data row
    and is immediately followed by at least one ``line:content`` data row; it
    is consumed (not re-emitted) and its text becomes the ``path`` prefix for
    the data rows that follow, until a non-data line appears.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_path: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        data = _HEADING_ROW_RE.match(line)
        if current_path is not None and data:
            out.append(f"{current_path}:{data.group('line')}:{data.group('content')}")
            i += 1
            continue
        # Not a data row under an active header. Decide if THIS line is a new
        # header: it must not be a data row itself and must be followed by a
        # data row. If so, consume it as the path prefix (do not emit).
        if not data and i + 1 < n and _HEADING_ROW_RE.match(lines[i + 1]):
            current_path = line
            i += 1
            continue
        # Plain passthrough line (or a stray data row with no header): emit it
        # verbatim and clear any active grouping.
        current_path = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


# A dir-heading data row: ``<base>:<line>:<content>`` where base has no '/'.
_DIR_DATA_RE = re.compile(r"^(?P<base>[^/\n:]+):(?P<line>\d+):(?P<content>.*)$")


def search_dir_heading(text: str) -> str:
    """Fold grep ``path:line:content`` rows by DIRECTORY.

    Consecutive rows whose path shares a parent directory collapse to that
    directory once (a header ending in ``/``), then ``base:line:content`` rows
    beneath it. Complements :func:`search_heading` (which factors a repeated
    *file*): this factors a repeated *directory* across distinct files — the
    common ``grep -rn`` case where each file has a single match, so file-heading
    saves nothing but the shared directory repeats on every row. Rows whose path
    has no ``/`` pass through untouched. Exactly reversed by
    :func:`search_dir_unheading`; ``compact_lossless`` verifies the round-trip.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_dir: str | None = None
    for line in lines:
        m = _GREP_ROW_RE.match(line)
        if m and "/" in m.group("path"):
            path = m.group("path")
            cut = path.rindex("/") + 1
            dir_part, base = path[:cut], path[cut:]
            if dir_part != current_dir:
                out.append(dir_part)
                current_dir = dir_part
            out.append(f"{base}:{m.group('line')}:{m.group('content')}")
        else:
            out.append(line)
            current_dir = None
    return _join(out, had_trailing)


def search_dir_unheading(text: str) -> str:
    """Exact inverse of :func:`search_dir_heading`.

    A *header* is a line ending in ``/`` immediately followed by a
    ``base:line:content`` data row; it is consumed and re-prefixed onto each
    following data row until a non-data line appears.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_dir: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        data = _DIR_DATA_RE.match(line)
        if current_dir is not None and data:
            out.append(f"{current_dir}{line}")
            i += 1
            continue
        if line.endswith("/") and i + 1 < n and _DIR_DATA_RE.match(lines[i + 1]):
            current_dir = line
            i += 1
            continue
        current_dir = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


# ── search_tree_heading: dir + file + line folding in one pass ─────────────
#
# `search_heading` factors a repeated FILE, `search_dir_heading` factors a
# repeated DIRECTORY, and `compact_lossless` picks whichever is smaller — so a
# result set that repeats BOTH (the normal `grep -rn` shape: several matches in
# each of several files under a shared directory) only ever gets one of the two
# savings. This fold composes them and adds two more axes that neither covers:
# consecutive line numbers, and a body repeated on several lines of one file.

# A tree-fold data row. Either an explicit line-number list
# (``12:body`` / ``12,40,73:body``) or an *increment* row (``:body``) meaning
# "previous line number + 1". The separator is ``:`` for grep match rows and
# ``-`` for ripgrep context rows (``-C``/``-A``/``-B``).
#
# Digits are ASCII and length-bounded on purpose. ``str.isdigit()`` is true for
# 108 non-decimal characters (``²``, ``፩``, …) that ``\d`` does not match and
# ``int()`` rejects, and CPython refuses ``int()`` on runs longer than 4300
# digits — either mismatch turns a parse into a ValueError halfway through the
# fold. ``[0-9]{1,18}`` makes the scanner, the regexes and ``int()`` agree on
# exactly one definition of "digit run", and 18 digits is already far past any
# line number a file can have.
_TREE_NUMS_RE = re.compile(r"^(?P<nums>[0-9]{1,18}(?:,[0-9]{1,18})*)(?P<sep>[:-])(?P<content>.*)$")
_TREE_INC_RE = re.compile(r"^(?P<sep>[:-])(?P<content>.*)$")
_TREE_MAX_DIGITS = 18
# ripgrep's between-groups separator. It is also a syntactically valid
# increment row (``-`` separator, body ``-``), so both directions special-case
# it: the fold never emits it as an increment, the unfold never reads it as one.
_RG_GROUP_SEP = "--"
# Bounds on the line-number-marker scan. A marker past `_TREE_MAX_PATH` chars is
# beyond any real path (PATH_MAX is 4096), and a line needing more than
# `_TREE_MAX_MARKERS` rejected candidates is not grep output. Without them the
# scan is quadratic in line length — `"a" + "-1"*k + "/x"` is a single line of
# nothing but accepted-then-rejected dash markers, and at 31 KB it already costs
# ~250ms with a clean 4x per doubling. Failing to parse is free: the line
# becomes passthrough, so the bounds cost savings at worst, never correctness.
_TREE_MAX_PATH = 4096
_TREE_MAX_MARKERS = 64


def _tree_digit_run_end(line: str, pos: int) -> int:
    """Index just past the digit run starting at ``pos + 1``."""
    end = pos + 1
    while end < len(line) and end - pos <= _TREE_MAX_DIGITS and line[end] in "0123456789":
        end += 1
    return end


def _tree_has_dash_marker(line: str) -> bool:
    """True when the line holds any ``-<digits>-`` triplet at all."""
    limit = min(len(line), _TREE_MAX_PATH)
    pos = line.find("-")
    seen = 0
    while pos != -1 and pos < limit and seen < _TREE_MAX_MARKERS:
        seen += 1
        end = _tree_digit_run_end(line, pos)
        if end > pos + 1 and end < len(line) and line[end] == "-":
            return True
        pos = line.find("-", pos + 1)
    return False


def _tree_structure_stop(line: str) -> int:
    """Index of the first character a foldable path would never contain.

    ``rg --json`` emits one compact record per line, and a matched string inside
    it can hold a ``:12:`` or ``-12-`` reference. The scan would skip the
    record's own structural colons, accept the embedded marker, and treat
    ``{"type":"match","data":{"lines":{"text":"foo`` as a path — reversible, so
    the round-trip guard passes it and the model gets shredded JSON. Anchoring
    context rows fixed the dash form only; the colon scan reaches these records
    on its own, so the path itself has to be rejected.

    Returns ``len(line)`` when the line holds none of them.
    """
    stop = len(line)
    for ch in '"{}':
        found = line.find(ch)
        if found != -1 and found < stop:
            stop = found
    return stop


def _tree_colon_row(line: str, *, lenient: bool = False) -> tuple[str, str, str, str] | None:
    """Parse a ``path:line:content`` match row.

    The marker has to be the line's **first** colon, because a path cannot
    contain one — the claim the rest of this fold rests on. Walking on to a
    later colon when the first one isn't followed by digits silently drops that
    colon into the path, which is how ``rg --heading -n -o`` output like
    ``1:see/foo:12:value`` came to parse as a file named ``1:see/foo`` and fold
    under a fabricated ``1:see/`` directory. The Windows drive colon is the one
    exception, and it is skipped explicitly.

    ``lenient`` lifts both guards below. Callers use it to ask "would this have
    been a match row but for the guards?" — see :func:`_tree_split_row`.
    """
    # Skip a Windows drive colon (``C:\Users\...``) so it is never mistaken for
    # the line-number marker.
    start = 2 if len(line) >= 3 and line[0].isalpha() and line[1] == ":" and line[2] in "\\/" else 0
    pos = line.find(":", start)
    if pos <= 0 or pos >= min(len(line), _TREE_MAX_PATH):
        return None
    end = _tree_digit_run_end(line, pos)
    if end == pos + 1 or end >= len(line) or line[end] != ":":
        return None
    if not lenient:
        if pos > _tree_structure_stop(line):
            return None
        # A space before the marker usually means this is the *body* of a ``-``
        # context row quoting a ``foo.py:12:`` reference, not a path. But a line
        # with no dash marker anywhere cannot be a context row, so there the
        # space is simply part of a path that has one —
        # ``very/long/directory name/file.py:2:MATCH`` is a real result row.
        first_space = line.find(" ")
        if 0 <= first_space < pos and _tree_has_dash_marker(line):
            return None
    return line[:pos], ":", line[pos + 1 : end], line[end + 1 :]


def _tree_dash_row(
    line: str,
    known_paths: frozenset[str],
    known_lengths: frozenset[int],
) -> tuple[str, str, str, str] | None:
    """Parse a ``path-line-content`` context row, anchored on ``known_paths``.

    Unlike ``:``, a dash appears inside real paths all the time
    (``logs/2026-05-03/app.log``, ``CVE-2021-44228.md``), so no purely textual
    rule can say which ``-<digits>-`` is the line-number marker. Guessing is
    what made ``report.log-2026-backup-1-before`` read as file ``report.log``
    line ``2026``, and what let a ``rg --json`` record split at a ``-12-`` inside
    a matched string — both byte-reversible, so the round-trip guard accepted
    them and the model got a bogus file heading.

    So this doesn't guess. ripgrep and grep always print the matching line
    itself with ``:`` separators, so every file that has ``-C``/``-A``/``-B``
    context rows also has at least one match row. A dash row is therefore only
    recognised when its path is one the colon pass already established. Nothing
    anchors a JSON record or a stray prose line, so they stay passthrough.

    Anchoring narrows the field but does not always empty it: a result set
    holding both ``report.log:1:…`` and ``report.log-2026-backup:2:…`` makes both
    prefixes members, and ``report.log-2026-backup-1-before`` reads equally well
    as either file. Nothing in the text settles that, so the row is left
    unparsed. Declining costs one row of folding; picking wrong puts a line
    under a file it never came from, and the byte round-trip cannot see it.

    ``known_lengths`` lets the scan reject a candidate marker on an integer
    compare instead of slicing the prefix, keeping the walk linear.

    A scan stopped by its own bounds declines too. The bounds exist for speed,
    but a truncated walk cannot say whether a *second* anchored reading lay
    beyond them — and on a path with 65-plus dash positions the first candidate
    was returned as if it were the only one, filing the long file's context
    under the short prefix. Not-yet-looked-at is not the same as not-there.
    """
    if not known_paths:
        return None
    limit = min(len(line), _TREE_MAX_PATH)
    candidates: list[tuple[str, str, str, str]] = []
    pos = line.find("-")
    seen = 0
    while pos != -1:
        if pos >= limit or seen >= _TREE_MAX_MARKERS:
            return None
        seen += 1
        if pos > 0 and pos in known_lengths:
            end = _tree_digit_run_end(line, pos)
            if end > pos + 1 and end < len(line) and line[end] == "-":
                path = line[:pos]
                if path in known_paths:
                    candidates.append((path, "-", line[pos + 1 : end], line[end + 1 :]))
                    if len(candidates) > 1:
                        return None
        pos = line.find("-", pos + 1)
    return candidates[0] if candidates else None


def _tree_split_row(
    line: str,
    known_paths: frozenset[str] = frozenset(),
    known_lengths: frozenset[int] = frozenset(),
) -> tuple[str, str, str, str] | None:
    """Split a grep/ripgrep row into ``(path, sep, line_digits, content)``.

    Returns ``None`` for any line that isn't a data row, **or that is two rows
    at once**.

    A ``:`` marker is decisive on its own — a path cannot contain one — so a row
    only the colon tier claims is a match row, and a row only the anchored dash
    tier claims is a context row. The hard case is the row both tiers claim,
    and it has no textual answer:

    * ``src/app.py-40-other.py:12:foo`` — a context line quoting a
      ``file:line:`` reference, which is what most log and error output looks
      like — reads as a match in ``src/app.py-40-other.py`` or as context in
      ``src/app.py``.
    * ``src/a-1-b.py:12:x`` with ``src/a`` anchored reads as a match in
      ``src/a-1-b.py`` or as context in ``src/a``, line 1.

    They are the same shape with opposite right answers, so no ordering of the
    tiers is correct, and successive tie-breaks — longest path, then the
    surrounding block, then ascending line order — each fixed one and broke
    another. This one declines instead: a row that reads as two different files
    is left unparsed. Passthrough costs one row of folding; a fabricated heading
    puts a line under a file it never came from, and the byte round-trip cannot
    tell the difference.
    """
    colon = _tree_colon_row(line)
    dash = _tree_dash_row(line, known_paths, known_lengths)
    if colon is not None:
        return colon if dash is None or dash[0] == colon[0] else None
    if dash is None:
        return None
    # Either guard can suppress a colon reading that was the right one, leaving
    # the dash tier to claim the row with no disagreement to notice:
    #   * the space guard, on a path holding both a space and a ``-<digits>-``
    #     (``report.log-2026-backup name:2:MATCH``);
    #   * the structural guard, on a Unix filename holding a quote or brace
    #     (``a-1-"file0":2:MATCH``) — a real result path, not a JSON record.
    # Ask what the colon tier would have said without them; a different file
    # means the row is ambiguous after all.
    unguarded = _tree_colon_row(line, lenient=True)
    return None if unguarded is not None and unguarded[0] != dash[0] else dash


def _tree_anchor_paths(lines: list[str]) -> tuple[frozenset[str], frozenset[int]]:
    """The files a payload establishes, used to anchor its context rows.

    A quoted reference in a context body can put a path in here that was never
    in the result set (``src/app.py-40-other.py``). That is harmless on its own:
    the only row such an anchor can claim is one the colon tier also claims, and
    :func:`_tree_split_row` declines those.
    """
    paths = frozenset(row[0] for line in lines if (row := _tree_colon_row(line)))
    return paths, frozenset(len(p) for p in paths)


def _tree_is_data_row(line: str) -> bool:
    return line != _RG_GROUP_SEP and bool(_TREE_NUMS_RE.match(line) or _TREE_INC_RE.match(line))


def search_tree_heading(text: str) -> str:
    """Fold grep/ripgrep rows along all four repeating axes at once.

    Output grammar (each line is one of)::

        <dir>/                      directory header, emitted on directory change
        <base>                      file header, emitted on file change
        <n>:<content>               data row
        <n1>,<n2>,<n3>:<content>    one body that matched on several lines
        :<content>                  data row at previous line + 1

    ``-`` replaces ``:`` on ripgrep context rows, exactly as in the input. Rows
    that don't parse pass through untouched and end the current grouping.

    The line-list form reorders a file's rows (all lines sharing a body move to
    the body's first position), so :func:`search_tree_unheading` re-sorts each
    file block it merged — which reproduces the input exactly because grep emits
    a file's rows in ascending line order. ``compact_lossless`` verifies that
    round-trip and drops the fold if it doesn't hold.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text

    known_paths, known_lengths = _tree_anchor_paths(lines)

    out: list[str] = []
    # Mirrors the inverse's state: "" means no directory header is in effect.
    current_dir = ""
    i = 0
    n = len(lines)
    while i < n:
        row = _tree_split_row(lines[i], known_paths, known_lengths)
        if row is None:
            out.append(lines[i])
            current_dir = ""
            i += 1
            continue

        # Collect the contiguous block of rows belonging to this one file.
        path = row[0]
        block: list[tuple[str, str, str, str]] = []
        while i < n:
            nxt = _tree_split_row(lines[i], known_paths, known_lengths)
            if nxt is None or nxt[0] != path:
                break
            block.append(nxt)
            i += 1

        cut = path.rfind("/") + 1
        dir_part, base = path[:cut], path[cut:]
        # Four shapes can't be headed, and all four are decided BEFORE any
        # header is written — a dir header emitted above an unfoldable block
        # would be consumed by the inverse and never re-emitted:
        #   * no basename (``src/:1:x``) — the header line would be empty;
        #   * a basename that itself parses as a data row
        #     (``20240101-002-add_users.sql``) — the inverse would read the
        #     header as content;
        #   * a basename of ``--`` (a file really can be named that) — the
        #     header would be indistinguishable from ripgrep's group separator,
        #     so the rows beneath it read as orphans belonging to nothing;
        #   * a path with no directory while a directory header is still in
        #     effect — there is no way to spell "back to no directory", so the
        #     inverse would keep prefixing the stale one.
        if (
            not base
            or base == _RG_GROUP_SEP
            or _tree_is_data_row(base)
            or (not dir_part and current_dir)
        ):
            out.extend(f"{p}{s}{d}{s}{c}" for p, s, d, c in block)
            current_dir = ""
            continue
        if dir_part != current_dir:
            out.append(dir_part)
            current_dir = dir_part
        out.append(base)
        out.extend(_tree_render_block(block))

    return _join(out, had_trailing)


def _tree_render_block(block: list[tuple[str, str, str, str]]) -> list[str]:
    """Render one file's rows: merge duplicate bodies, elide consecutive lines."""
    # Merging moves every line sharing a body up to that body's first position,
    # and the inverse undoes it by re-sorting the block — which only reproduces
    # the input if the block was in ascending line order to begin with. grep
    # always emits it that way; when something upstream hasn't, skip merging for
    # this block rather than hand ``compact_lossless`` a fold that fails
    # verification and costs the *whole* payload its savings.
    numbers = [int(digits) for _p, _s, digits, _c in block]
    mergeable = all(a <= b for a, b in zip(numbers, numbers[1:]))

    # Group by (separator, body) in first-appearance order. A zero-padded line
    # number gets a private, unmergeable key: the merge form re-sorts through
    # ``int`` on the way back, and the elision form regenerates the number from
    # a count — both would drop the padding.
    groups: dict[object, list[str]] = {}
    fields: dict[object, tuple[str, str]] = {}
    order: list[object] = []
    for index, (_path, sep, digits, content) in enumerate(block):
        padded = digits != str(int(digits))
        key: object = ("pad", index) if padded or not mergeable else ("body", sep, content)
        if key not in groups:
            groups[key] = []
            fields[key] = (sep, content)
            order.append(key)
        groups[key].append(digits)

    rendered: list[str] = []
    previous: int | None = None
    for key in order:
        nums = groups[key]
        sep, content = fields[key]
        padded = nums[0] != str(int(nums[0]))
        if not padded and len(nums) == 1 and previous is not None and int(nums[0]) == previous + 1:
            row = f"{sep}{content}"
            if row == _RG_GROUP_SEP:
                row = f"{nums[0]}{sep}{content}"
            rendered.append(row)
        else:
            rendered.append(f"{','.join(nums)}{sep}{content}")
        previous = int(nums[-1])
    return rendered


def search_tree_unheading(text: str) -> str:
    """Exact inverse of :func:`search_tree_heading`.

    A line ending in ``/`` followed by a non-data line is a directory header; a
    non-data line with no ``/`` followed by a data row is a file header. Both
    are consumed rather than emitted, and their text re-prefixes the data rows
    beneath. A file block that contained a merged line list is re-sorted by line
    number on the way out; blocks with nothing merged keep their emitted order.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text

    out: list[str] = []
    current_dir = ""
    current_file: str | None = None
    previous: int | None = None
    pending: list[tuple[int, str, str, str]] = []
    merged = False

    def flush() -> None:
        nonlocal pending, merged
        if not pending:
            return
        rows = sorted(pending, key=lambda r: r[0]) if merged else pending
        out.extend(f"{current_dir}{current_file}{s}{d}{s}{c}" for _, s, d, c in rows)
        pending = []
        merged = False

    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if current_file is not None and line != _RG_GROUP_SEP:
            m = _TREE_NUMS_RE.match(line)
            if m:
                digits = m.group("nums").split(",")
                if len(digits) > 1:
                    merged = True
                for d in digits:
                    pending.append((int(d), m.group("sep"), d, m.group("content")))
                previous = int(digits[-1])
                i += 1
                continue
            m = _TREE_INC_RE.match(line)
            if m and previous is not None:
                previous += 1
                pending.append((previous, m.group("sep"), str(previous), m.group("content")))
                i += 1
                continue

        flush()
        if line.endswith("/") and i + 1 < n and not _tree_is_data_row(lines[i + 1]):
            current_dir = line
            current_file = None
            previous = None
            i += 1
            continue
        if (
            line
            and line != _RG_GROUP_SEP  # the fold never heads with it; see above
            and "/" not in line
            and not _tree_is_data_row(line)
            and i + 1 < n
            and _tree_is_data_row(lines[i + 1])
        ):
            current_file = line
            previous = None
            i += 1
            continue
        current_dir = ""
        current_file = None
        previous = None
        out.append(line)
        i += 1

    flush()
    return _join(out, had_trailing)


def search_fold_recovers(folded: str, original: str) -> bool:
    """True when some search fold's inverse turns ``folded`` back into ``original``.

    This is the exact guarantee ``compact_lossless`` enforces when it picks a
    ``kind="search"`` candidate, restated so callers (chiefly tests) can assert
    it without knowing which of the three folds won.

    There is deliberately no ``search_unfold(folded) -> original`` counterpart.
    Recovering a fold *without* the original is unsound: the inverses are not
    mutually exclusive, so more than one can claim the same blob, and an inverse
    that did not produce it can still expand it into text that re-folds to
    exactly the input. That check therefore accepts a wrong answer rather than
    rejecting it — e.g. ``'docs/\\nREADME.md\\n12:h\\n40:w\\n'`` is what the file
    fold emits for a grep result preceded by a literal ``docs/`` banner line, but
    the tree inverse claims it first and fabricates ``docs/`` onto both paths
    while destroying the banner. Nothing in the compression path ever needs to
    unfold — the folded text is what the model reads — so the ambiguity is only
    worth resolving where the original is on hand, which is what this does.
    """
    for _fold, unfold in (
        (search_tree_heading, search_tree_unheading),
        (search_heading, search_unheading),
        (search_dir_heading, search_dir_unheading),
    ):
        try:
            if unfold(folded) == original:
                return True
        except Exception:
            continue
    return False


def diff_strip_index(text: str) -> str:
    """Drop ``index <sha>..<sha>`` lines from a unified diff (still applies)."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out = [line for line in lines if not _DIFF_INDEX_RE.match(line)]
    return _join(out, had_trailing)


# A whole-line file path: optional ``./``/``../`` root, >=1 directory segment,
# then a basename. No whitespace or ':' (so grep ``path:line:content`` rows —
# handled by search_heading — are excluded). Directory-only lines (trailing '/')
# don't match (empty basename), which keeps the fold unambiguous.
_PATH_ROW_RE = re.compile(r"^(?P<dir>(?:\.{0,2}/)?(?:[^/\s:]+/)+)(?P<base>[^/\s:]+)$")


def path_heading(text: str) -> str:
    """Fold a *pure* file-path listing (``find`` / ``ls -1`` / ``rg -l`` output)
    into ripgrep-heading form: each parent directory printed once on its own
    line (ending in ``/``), then the bare basenames beneath it.

    Reversibility is not assumed here — ``compact_lossless`` verifies the exact
    round-trip via :func:`path_unheading` and discards the fold on any mismatch
    (e.g. a stray no-slash line mistaken for a basename), so mixed content is
    always safe. Requires >=2 path rows or there is nothing to group.
    Complements ``search_heading``, which only handles the ``path:line:content``
    grep shape, not plain path lists.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if sum(1 for ln in lines if _PATH_ROW_RE.match(ln)) < 2:
        return text
    out: list[str] = []
    current: str | None = None
    for line in lines:
        m = _PATH_ROW_RE.match(line)
        if m:
            d = m.group("dir")
            if d != current:
                out.append(d)
                current = d
            out.append(m.group("base"))
        else:  # blank line inside/around the listing
            out.append(line)
            current = None
    return _join(out, had_trailing)


def path_unheading(text: str) -> str:
    """Exact inverse of :func:`path_heading`.

    A *header* is a line ending in ``/`` immediately followed by a basename row
    (a non-empty line with no ``/``); it is consumed and re-prefixed onto each
    following basename row until a blank line or another header.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        is_base = line != "" and "/" not in line
        if current is not None and is_base:
            out.append(current + line)
            i += 1
            continue
        if line.endswith("/") and i + 1 < n and lines[i + 1] != "" and "/" not in lines[i + 1]:
            current = line
            i += 1
            continue
        current = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


def _smaller(candidate: str, original: str) -> bool:
    return len(candidate) < len(original)


def compact_lossless(content: str, kind: str) -> str:
    """Dispatch format-native lossless compaction by ``kind``.

    ``kind`` in {'log', 'search', 'diff', 'text', 'config'}. For reversible kinds the
    round-trip is verified internally (modulo the intentionally-dropped
    non-semantic bits, e.g. ANSI color for logs); if verification fails or the
    result is not smaller, the original content is returned unchanged. Never
    raises; unknown kinds pass through.
    """
    if not content:
        return content
    try:
        if kind == "log":
            # ANSI is non-semantic and dropped one-way; run-collapse must be
            # exactly reversible against the de-ANSI'd baseline.
            baseline = strip_ansi(content)
            candidate = collapse_runs(baseline)
            if expand_runs(candidate) != baseline:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "search":
            # Three independent folds; keep the smallest that round-trips
            # exactly. search_heading factors a repeated FILE (many matches in
            # one file); search_dir_heading factors a repeated DIRECTORY (one
            # match each across many files in a dir — the grep -rn case the file
            # fold misses); search_tree_heading factors both at once and also
            # folds consecutive line numbers and repeated bodies, so it wins on
            # most real result sets. The first two stay as candidates because
            # they are strictly more permissive about what counts as a data row,
            # and so still round-trip on inputs the tree fold declines.
            best = content
            for fold, inverse in (
                (search_tree_heading, search_tree_unheading),
                (search_heading, search_unheading),
                (search_dir_heading, search_dir_unheading),
            ):
                # Size first: verifying costs about as much as folding, and a
                # candidate that isn't smaller than the incumbent can't win no
                # matter how it verifies. The tree fold leads because it is the
                # one that usually wins, which makes the other two cheap.
                candidate = fold(content)
                if _smaller(candidate, best) and inverse(candidate) == content:
                    best = candidate
            return best

        if kind == "paths":
            # Pure path listings (find/ls -1/rg -l): fold repeated parent dirs.
            candidate = path_heading(content)
            if path_unheading(candidate) != content:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "diff":
            # Purely subtractive of non-semantic bookkeeping lines; the
            # remaining hunks still apply. No exact inverse needed.
            candidate = diff_strip_index(content)
            return candidate if _smaller(candidate, content) else content

        if kind == "text":
            # Collapse blank-line runs; reversible against itself.
            candidate = collapse_runs(content)
            if expand_runs(candidate) != content:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "config":
            # Structured config (YAML/TOML/INI): single-line runs first, then
            # repeated multi-line stanzas. Inverse applies in reverse order.
            candidate = fold_repeated_blocks(collapse_runs(content))
            if expand_runs(unfold_repeated_blocks(candidate)) != content:
                return content
            return candidate if _smaller(candidate, content) else content
    except Exception:
        return content
    return content

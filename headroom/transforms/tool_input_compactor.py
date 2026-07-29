"""Compact completed tool-call inputs (arguments) via CCR.

Tool OUTPUTS are compressed by ContentRouter; tool-call INPUTS are not.
Historical Write payloads, apply_patch bodies, shell heredocs, and SQL
strings therefore stay verbatim in context for the rest of the session
even though the model already acted on their results. This pre-processing
pass replaces large, completed tool-call arguments with a compact marker
+ CCR hash, preserving the call id and tool name so provider validation
and conversation structure are untouched. Handles both wire shapes:

- OpenAI: ``message["tool_calls"][i]["function"]["arguments"]`` (JSON string)
- Anthropic: ``tool_use`` content blocks' ``input`` (object)

The original serialized arguments are stored in the CCR compression store
under the marker's hash, so ``headroom_retrieve`` / ``/v1/retrieve/{hash}``
recovers the exact bytes on demand.

Safety rules (each prevents a concrete failure mode):
- **Reproducible/read-only inputs only.** A call is compacted only when
  neither its tool name nor its serialized arguments look *mutating*
  (``Write``/``apply_patch``-family tools, SQL DML/DDL, shell heredocs,
  shell write-redirection / in-place edits, and any interpreter handed an
  arbitrary embedded program — ``python -c``, ``node -e``, ``ruby -e`` —
  whose effects cannot be read off the command line). For a mutating call the tool
  result is usually a bare acknowledgement ("File written"), so the
  arguments are the ONLY exact record of what changed — and the CCR entry
  expires (``CCRConfig.ttl_seconds``, default 1,800s), after which that
  record would be gone for good. Read-only inputs (a Grep pattern, a
  Read path, a SELECT) are always re-derivable from the tool result or by
  re-running the call, so a lapsed CCR entry costs nothing irreversible.
- **Successful CCR persistence is a precondition.** If the store is
  missing or ``store()`` raises, the call is left completely untouched.
  There is no catch-up mechanism, so emitting a marker whose entry was
  never written would silently make the historical input unrecoverable.
- Only calls whose matching tool result appears in a LATER message are
  compacted — a pending call's arguments are live working context.
- The trailing ``protect_recent_turns`` assistant messages are never
  touched: the model frequently reuses recent arguments (e.g. iterating
  on a patch), and re-deriving them from a retrieval round-trip would
  cost more than the compaction saves.
- Messages inside the provider's frozen cache prefix are never mutated —
  rewriting cached bytes busts the prefix cache, which costs more than
  the token savings.
- Already-compacted inputs (the ``_ccr`` key) are skipped, so repeated
  passes over the same conversation are idempotent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import ToolInputCompactionConfig

logger = logging.getLogger(__name__)

# Replacement key carrying the marker inside the compacted arguments.
CCR_INPUT_KEY = "_ccr"

# ---------------------------------------------------------------------------
# THE RULE: only reproducible / read-only tool inputs are compacted.
#
# A mutating call's result is typically a bare acknowledgement ("File
# written", "1 row updated"), so its ARGUMENTS are the sole exact record of
# what changed. CCR entries expire (``CCRConfig.ttl_seconds``, default
# 1,800s), so replacing those arguments with a marker would make the record
# unrecoverable once the entry lapses. Read-only inputs are always
# re-derivable (re-run the search, re-read the file), so a lapsed entry is
# merely inconvenient. Hence: name-based denylist + a content check for the
# mutation shapes that ride inside otherwise-innocuous tools (bash, sql).
# ---------------------------------------------------------------------------

#: Tool names (normalized: casefolded, ``_``/``-`` stripped) whose inputs
#: describe a mutation. Never compacted, at any size or age.
MUTATING_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "write",
        "writefile",
        "createfile",
        "edit",
        "editfile",
        "multiedit",
        "applypatch",
        "patch",
        "notebookedit",
        "strreplaceeditor",
        "strreplacebasededittool",
        "texteditor",
        "filewrite",
        "fswrite",
        "update",
        "delete",
        "deletefile",
        "removefile",
        "movefile",
        "renamefile",
        "insert",
        "createpullrequest",
        "createorupdatefile",
        "pushfiles",
    }
)

# ---------------------------------------------------------------------------
# Cost discipline. ``is_mutating_tool_input`` runs on EVERY completed tool call
# of EVERY request, over the full serialized arguments (tens of KB is normal,
# and the arguments are attacker-influenced). Two rules keep it cheap:
#
# 1. Every content check is guarded by a plain-substring precondition. Python's
#    ``str.__contains__`` is a C-speed two-way search; a Python ``re`` pass over
#    the same bytes is ~50x slower. If a required literal is absent, the shape
#    provably cannot match and the regex never runs.
# 2. No pattern may contain an unbounded gap between two literals. The original
#    in-place-edit alternative ``\bsed\b[^|;&]*\s-i\b`` was quadratic in the
#    tail — every ``sed`` occurrence scanned to end-of-string and backtracked —
#    measured at multiple seconds on a 64 KB ``sed``-dense blob. It is now a
#    linear segment scan (``_has_inplace_edit``).
#
# The detection SET is unchanged: each helper below matches exactly what its
# predecessor alternative matched.
# ---------------------------------------------------------------------------

#: SQL statements that change data or schema. Matched anywhere in the
#: serialized arguments (a query string may be nested arbitrarily).
_SQL_MUTATION_RE = re.compile(
    r"\b(?:"
    r"insert\s+into|update\s+[\"'`\w]|delete\s+from|merge\s+into|replace\s+into|"
    r"drop\s+(?:table|database|schema|index|view|column)|"
    r"alter\s+(?:table|database|schema|view)|truncate(?:\s+table)?\s+[\"'`\w]|"
    r"create\s+(?:table|database|schema|index|view|or\s+replace)|"
    r"grant\s+\w|revoke\s+\w"
    r")",
    re.IGNORECASE,
)

#: Leading verbs of every ``_SQL_MUTATION_RE`` alternative. The regex is
#: case-insensitive, so the prescreen tests a lowercased copy.
_SQL_MUTATION_VERBS: tuple[str, ...] = (
    "insert",
    "update",
    "delete",
    "merge",
    "replace",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
)

#: Shell heredoc (``cat <<'EOF' > file``) — the body IS the written content.
#: The arguments are JSON-serialized, so a real newline appears as the two
#: characters ``\`` ``n``; accept both forms.
_HEREDOC_RE = re.compile(r"<<-?\s*[\"']?[A-Za-z_][A-Za-z0-9_]*[\"']?\s*(?:\\n|\n|\\\\n)")

#: Write-redirection body, matched ANCHORED at a candidate ``>``. The caller
#: (:func:`_has_write_redirect`) supplies the ``(?<![-=<>])`` guard by checking
#: the preceding character, so arrows/comparisons (``->``, ``=>``, ``>=``,
#: ``>>=``) still never look like a write and an ordinary search pattern such
#: as ``def f() -> int`` stays compactable.
#:
#: ``["'\\]*`` after the operator is load-bearing: the text being scanned is the
#: SERIALIZED arguments, so a quoted target (``> "$file"``, ``>> 'out.txt'``)
#: reaches this pattern as ``> \"$file\"`` — a backslash and a quote sit between
#: the operator and the first word character. Without accepting them,
#: ``printf … > "$file"`` classified as read-only and its arguments — the only
#: record of what was written — became eligible for an expiring CCR marker.
#: A word character is still required after the quotes, so a dangling ``> "`` at
#: the end of a blob does not match.
#: ``\|?`` covers bash's clobber form ``>| file``.
_REDIRECT_AT_RE = re.compile(r">>?(?![=>])\|?\s*[\"'\\]*[\w./~$-]*[\w/]")

#: ``sed``/``perl`` in-place editing, split into two anchors so the gap between
#: them is bounded by one shell segment instead of the whole input.
_INPLACE_TOOL_RE = re.compile(r"\b(?:sed|perl)\b")
_INPLACE_FLAG_RE = re.compile(r"\s-i\b")
#: Shell command separators. ``[^|;&]*`` in the original pattern meant "same
#: segment"; splitting on the same three characters preserves that exactly.
_SHELL_SEGMENT_RE = re.compile(r"[|;&]")

#: Word-anchored destructive commands. Each entry is
#: ``(required-substrings, pattern)``: the pattern can only match if at least
#: one of its literals is present, so the cheap membership test gates it.
#:
#: The file-op alternation covers everything that CREATES, LINKS, MOVES,
#: TRUNCATES or CHANGES METADATA on a path — not just the obviously destructive
#: verbs. ``touch f``, ``install -m 755 src dst``, ``mkfifo p``, ``mknod``,
#: ``shred``, ``unlink``, ``chgrp``, ``chattr`` and ``rsync`` all produce a bare
#: (often empty) acknowledgement, so their arguments are the sole record of the
#: mutation — exactly the case THE RULE exists to protect.
_TEE_RE = re.compile(r"\btee\b")
_FILEOP_RE = re.compile(
    r"\b(?:rm|mv|cp|mkdir|rmdir|chmod|chown|chgrp|chattr|ln|unlink|truncate|dd|"
    r"touch|install|mkfifo|mknod|shred|rsync|setfacl)\s"
)
#: ``patch`` is a common English word, so it is anchored on the option/redirect
#: that a real invocation needs (``patch -p1 …``, ``patch < fix.diff``) rather
#: than the bare token.
#: ``patch`` is a common English word, so the bare token is not enough. Three
#: real shapes: the classic ``patch -p1 …`` / ``patch < fix.diff``; a CLI
#: subcommand (``kubectl patch deployment app``, ``oc patch``); and a payload
#: flag (``--patch '<json>'``, ``--patch-file f``) where the operand follows the
#: flag rather than the word. All three produce a bare "patched" acknowledgement,
#: so the arguments are the only record of what the patch contained.
_PATCH_CMD_RE = re.compile(
    r"\bpatch\s+[-<]"
    r"|\b(?:kubectl|oc|helm|az|gcloud|aws)\s+(?:[-\w]+\s+)*patch\b"
    r"|--patch(?:-file)?[=\s]"
)
#: State-changing git subcommands. `add` matters most in practice: `git add`
#: with hundreds of paths is exactly the long-argument, empty-result shape this
#: guard exists for, and once the CCR entry lapses the transcript no longer
#: records what was staged. Read-only porcelain (status, log, diff, show, blame,
#: grep, ls-files, rev-parse, describe) is deliberately absent so it stays
#: compactable.
_GIT_OPTION_VALUE = (
    r"""(?:\\"[^"\r\n]*\\"|"[^"\r\n]*"|'[^'\r\n]*'|"""
    r"""(?:[^\s"'\\]|\\(?!")[^\r\n])(?:[^\s"'\\]|\\[^\r\n])*)"""
)
_GIT_MUTATION_RE = re.compile(
    r"\bgit\s+"
    rf"(?:(?:(?:-C|-c)\s+{_GIT_OPTION_VALUE}|"
    rf"(?:--git-dir|--work-tree|--namespace|--config-env|--shallow-file)"
    rf"(?:=|\s+){_GIT_OPTION_VALUE}|"
    rf"--exec-path={_GIT_OPTION_VALUE}|"
    r"-p|-P|--paginate|--no-pager|--no-replace-objects|--no-lazy-fetch|"
    r"--no-optional-locks|--no-advice|--bare|--literal-pathspecs|"
    r"--glob-pathspecs|--noglob-pathspecs|--icase-pathspecs)\s+)*"
    r"(?:"
    r"add|am|apply|branch|checkout|cherry-pick|clean|commit|config|fetch|gc|init|"
    r"merge|mv|notes|prune|pull|push|rebase|remote|reset|restore|revert|rm|"
    r"sparse-checkout|stash|submodule|switch|tag|update-index|update-ref|worktree"
    r")\b"
)
#: Package managers accept options and toolchain selectors between the
#: executable and the verb — ``apt-get -y remove …``, ``npm --global uninstall``,
#: ``cargo +nightly uninstall``, ``pip -q install`` — so requiring the verb
#: immediately after the binary missed exactly the long-argument invocations
#: this guard is for. The gap allows short option/selector tokens only
#: (``-y``, ``--global``, ``+nightly``), each bounded, so nothing scans to the
#: end of a large blob.
_PKG_MUTATION_RE = re.compile(
    r"\b(?:npm|pnpm|yarn|pip|pip3|uv|cargo|apt|apt-get|brew|gem|go|dotnet|poetry)\s+"
    r"(?:[-+][-\w=]{0,30}\s+){0,4}"
    r"(?:install|add|remove|uninstall|publish|upgrade|update)\b"
)

#: Remote mutations whose terse result does not preserve the applied payload.
#: The bounded character gap admits quoted/global options
#: (``kubectl --context "$CTX" apply``) without scanning an unbounded command tail.
_DEPLOYMENT_MUTATION_RE = re.compile(
    r"\b(?:kubectl|oc)\b[^\r\n]{0,512}?\b"
    r"(?:apply|create|delete|edit|replace|patch|scale|annotate|label|taint|"
    r"cordon|uncordon|drain|set|autoscale|expose|run|rollout\s+(?:restart|undo))\b"
    r"|\bhelm\b[^\r\n]{0,512}?\b(?:install|upgrade|uninstall|rollback)\b"
)
_CURL_METHOD_RE = re.compile(r"(?:^|\s)(?:-X|--request)(?:[=\s]*)([A-Za-z]+)\b")
_CURL_GET_RE = re.compile(r"(?:^|\s)(?:-G|--get)\b")
#: ``-K``/``--config`` makes curl read arguments from a file — or, with ``-K -``,
#: from stdin. Per ``curl --manual`` those arguments are treated exactly like
#: command-line ones, except the leading dashes may be omitted and the separator
#: may be ``=`` or whitespace: ``request = POST`` is ``--request POST``. A
#: request whose method and body live only in that config therefore reads as a
#: bare ``curl -K -`` on the command line and slipped past every flag probe.
_CURL_CONFIG_RE = re.compile(r"(?:^|\s)(?:-K|--config)(?:[=\s]|$)")
#: Config-file spelling of the mutating directives. Dashes optional, `=` or
#: space separator, value on the same line.
_CURL_CONFIG_MUTATION_RE = re.compile(
    r"(?:^|[\s\\rn\"])-{0,2}(?:"
    r"request[=\s]+(?:POST|PUT|PATCH|DELETE)|"
    r"data(?:-ascii|-binary|-raw|-urlencode)?[=\s]|"
    r"json[=\s]|form(?:-string)?[=\s]|upload-file[=\s]|T[=\s]"
    r")",
    re.IGNORECASE,
)
_CURL_PAYLOAD_RE = re.compile(
    r"(?:^|\s)(?:"
    r"-d|--data(?:-ascii|-binary|-raw|-urlencode)?|--json|"
    r"-F|--form(?:-string)?|-T|--upload-file"
    r")(?:[=\s]|(?=[^-]))"
)

_WORD_MUTATION_PROBES: tuple[tuple[tuple[str, ...], re.Pattern[str]], ...] = (
    (("tee",), _TEE_RE),
    (
        (
            # NB: every literal here is a NECESSARY condition for one
            # alternative of `_FILEOP_RE`; each costs ~15us on a 64 KB blob
            # while running the regex costs ~1.2ms, so the gate stays worth it.
            # `rmdir` is deliberately absent — `rm` already covers it.
            "rm",
            "mv",
            "cp",
            "mkdir",
            "chmod",
            "chown",
            "chgrp",
            "chattr",
            "ln",
            "unlink",
            "truncate",
            "dd",
            "touch",
            "install",
            "mkfifo",
            "mknod",
            "shred",
            "rsync",
            "setfacl",
        ),
        _FILEOP_RE,
    ),
    (("patch",), _PATCH_CMD_RE),
    (("git",), _GIT_MUTATION_RE),
    (
        # Must list every binary in _PKG_MUTATION_RE: the gate is what decides
        # whether the regex runs at all, so a binary missing here is invisible
        # no matter what the pattern says.
        (
            "npm",
            "pnpm",
            "yarn",
            "pip",
            "uv",
            "cargo",
            "apt",
            "brew",
            "gem",
            "go",
            "dotnet",
            "poetry",
        ),
        _PKG_MUTATION_RE,
    ),
)


def _has_remote_mutation(text: str) -> bool:
    """Deployment commands and HTTP requests that write remote state."""
    if ("kubectl" in text or "oc" in text or "helm" in text) and _DEPLOYMENT_MUTATION_RE.search(
        text
    ):
        return True
    if "curl" not in text:
        return False
    for segment in _SHELL_SEGMENT_RE.split(text):
        curl = re.search(r"\bcurl\b", segment)
        if curl is None:
            continue
        tail = segment[curl.end() :]
        if _CURL_CONFIG_RE.search(tail):
            # The real arguments are in the config body, which for `-K -` is the
            # sibling `stdin` field of the same serialized call. Scan the whole
            # text rather than just this segment: a mutating directive found
            # anywhere in it is reason enough to keep the input, and a missed
            # fold costs only tokens where a wrong fold costs the record of what
            # was sent.
            if _CURL_CONFIG_MUTATION_RE.search(text):
                return True
        methods = _CURL_METHOD_RE.findall(tail)
        if methods:
            if methods[-1].upper() in {"POST", "PUT", "PATCH", "DELETE"}:
                return True
            continue
        if not _CURL_GET_RE.search(tail) and _CURL_PAYLOAD_RE.search(tail):
            return True
    return False


def _has_write_redirect(text: str) -> bool:
    """``> path`` / ``>> path`` write redirection.

    Walks the (rare) ``>`` positions with ``str.find`` instead of letting the
    regex engine try a lookbehind at every character. Equivalent to the former
    ``(?<![-=<>])>>?(?![=>])\\s*[\\w./~$-]*[\\w/]`` alternative: the manual
    predecessor check is the lookbehind.
    """
    pos = text.find(">")
    while pos != -1:
        if pos == 0 or text[pos - 1] not in "-=<>":
            if _REDIRECT_AT_RE.match(text, pos):
                return True
        pos = text.find(">", pos + 1)
    return False


def _has_inplace_edit(text: str) -> bool:
    """``sed … -i`` / ``perl … -i`` within a single shell command segment.

    Linear: one split on ``[|;&]``, then at most two bounded scans per segment.
    The former ``\\bsed\\b[^|;&]*\\s-i\\b`` form re-scanned the whole tail from
    every ``sed`` occurrence (quadratic, seconds on a 64 KB blob).
    """
    if "-i" not in text:
        return False
    if "sed" not in text and "perl" not in text:
        return False
    for segment in _SHELL_SEGMENT_RE.split(text):
        tool = _INPLACE_TOOL_RE.search(segment)
        # The FIRST sed/perl in the segment is the most permissive anchor: any
        # `-i` reachable from a later occurrence is reachable from this one.
        if tool is not None and _INPLACE_FLAG_RE.search(segment, tool.end()):
            return True
    return False


def _has_shell_mutation(text: str) -> bool:
    """Local or remote shell mutations whose arguments must remain durable."""
    if ">" in text and _has_write_redirect(text):
        return True
    if _has_inplace_edit(text):
        return True
    if _has_remote_mutation(text):
        return True
    for literals, pattern in _WORD_MUTATION_PROBES:
        for literal in literals:
            if literal in text:
                if pattern.search(text) or (
                    pattern is _GIT_MUTATION_RE and pattern.search(text.replace("\\\\", "\\"))
                ):
                    return True
                break
    return False


# ---------------------------------------------------------------------------
# Embedded code execution (Codex P2).
#
# Every probe above matches a mutation written in SHELL. A completed
# `Bash`/`exec_command` call can just as easily write through an embedded
# runtime, where the shell text contains no write token at all:
#
#     python -c "Path('out').write_text(body)"
#     node -e "fs.writeFileSync('out', body)"
#     ruby -e "File.write('out', body)"
#
# All three classified as read-only, so arguments over `min_chars` — the sole
# exact record of the write — were eligible for a CCR marker that expires after
# `CCRConfig.ttl_seconds`.
#
# WHY THE WHOLE SHAPE, NOT THE WRITE APIS. The obvious fix is to also recognize
# `write_text` / `writeFileSync` / `File.write` / `open(...,'w')` / `IO.write` /
# `fs.promises.writeFile` / `shutil.*` / `os.replace` / … That enumeration is
# OPEN-ENDED and unclosable: each interpreter admitted by a `-c`/`-e` flag
# brings its own standard library, its own aliases (`import os as o`), its own
# escape hatches (`os.system`, `subprocess`, `ctypes`, `sqlite3`, an HTTP POST),
# and a program can be assembled at runtime from strings this scanner will never
# resolve. Every miss is an unrecoverable record — the asymmetry THE RULE exists
# to respect. So the CONTAINER is what gets recognized: an interpreter handed an
# arbitrary program is opaque, and opaque is treated as mutating.
#
# THE RULE STATED CRISPLY: if the serialized arguments invoke a general-purpose
# interpreter with its program supplied inline (`-c`, `-e`, `--eval`, `eval`,
# `-r`, `-ne`, …) or on stdin (`python - <<PY`), the call is mutating.
#
# COST, MEASURED (real agent transcripts: 3,911 tool calls, 2.73 MB of
# serialized inputs). 676 calls exceed the 800-char `min_chars` threshold, and
# only 11 of those were compactable at all — the name denylist and the shell
# probes already claim the other 665. This rule removes 4 of the 11: 4,801 of
# 12,889 compactable chars, i.e. 0.18% of all tool-input bytes. The whole
# compaction opportunity on this corpus is 0.47% of tool-input bytes; the rule
# takes it to 0.30%. A read-only inline program that ALSO exceeds 800 characters
# and ALSO avoids every other probe is a narrow target, and every one of the 4
# lost calls here is an inline Python script — precisely the shape whose effects
# cannot be read off the command line. Trading 0.18% of bytes for a closed rule
# instead of an unclosable enumeration is the right side of THE RULE's
# asymmetry.
# ---------------------------------------------------------------------------

#: Interpreters that will execute a program handed to them as an argument.
#: Deliberately case-sensitive: an all-caps `PYTHON -c` does not occur in
#: practice, and case-folding would mean allocating a lowercased copy of a blob
#: that is routinely tens of KB (see "Cost discipline" above).
_CODE_EXEC_RE = re.compile(
    r"\b(?:python[0-9.]*|pypy[0-9.]*|node|nodejs|deno|bun|ruby|perl|php|lua|julia|"
    r"[Rr]script|osascript|tclsh|sh|bash|zsh|ksh|dash|fish|pwsh|powershell)\b"
    r"(?:"
    # Subcommand form: `deno eval 'src'`, `bun eval 'src'`.
    r"[ \t]+eval\b"
    # Inline-source flag: `python -c`, `node --eval`, `perl -ne`, `php -r`,
    # `bash -lc`, `pwsh -Command`. At most ONE intervening short option keeps
    # the gap bounded (`python -E -c …` still matches) while `python -m pytest
    # -c setup.cfg` — a config flag five tokens away — does not. A bounded gap
    # is required: an unbounded one would scan to end-of-string from every
    # interpreter occurrence, the quadratic shape `_has_inplace_edit` exists to
    # avoid.
    r"|(?:[ \t]+-[A-Za-z0-9]{1,3})?[ \t]+(?:-{1,2}(?:eval|exec|[Cc]ommand)|-[A-Za-z]{0,3}[cerE])\b"
    # Program on stdin: `python - <<PY`, `bash -s`. The heredoc probe misses
    # `python - <<'PY' 2>&1 | tail` (the delimiter is followed by a redirect,
    # not a newline), and a piped program has no heredoc at all.
    r"|[ \t]+-(?=[\s\"'\\]|$)"
    r")"
)

#: Necessary substrings for :data:`_CODE_EXEC_RE` — one per interpreter
#: alternative, minus the ones another entry already implies (`python3` contains
#: `python`, and `bash`/`zsh`/`ksh`/`dash`/`fish`/`pwsh`/`powershell`/`tclsh`
#: all contain `sh`). `sh` is a weak gate — plenty of English words contain it —
#: but a weak gate is still a correct one, and the regex it admits is linear.
_CODE_EXEC_LITERALS: tuple[str, ...] = (
    "python",
    "pypy",
    "node",
    "deno",
    "bun",
    "ruby",
    "perl",
    "php",
    "lua",
    "julia",
    "script",
    "sh",
)

#: Second, *more selective* necessary condition. Every alternative of
#: :data:`_CODE_EXEC_RE` separates the interpreter from what follows with
#: ``[ \t]+`` and then requires either ``-`` or ``eval`` — hence one of these
#: four two-character-ish literals must be present. Deliberately the reason the
#: gap is spelled ``[ \t]+`` rather than ``\s+``: a strictly-necessary literal
#: is only derivable when the separator class is known, and ``sh`` alone is far
#: too weak a gate (it occurs in "should", "shell", "finished", …). Checked
#: FIRST because ordinary prose — which contains ``sh`` constantly — almost
#: never contains " -".
_CODE_EXEC_SEPARATORS: tuple[str, ...] = (" -", "\t-", " eval", "\teval")


def _has_code_execution(text: str) -> bool:
    """An interpreter invoked with an arbitrary embedded program.

    Conservative by construction: what the embedded program does is not
    inspected, because it cannot be inspected reliably. See the block comment
    above for why the container rather than the write API is the thing matched.
    """
    if not any(sep in text for sep in _CODE_EXEC_SEPARATORS):
        return False
    for literal in _CODE_EXEC_LITERALS:
        if literal in text:
            return bool(_CODE_EXEC_RE.search(text))
    return False


def _has_sql_mutation(text: str) -> bool:
    """SQL DML/DDL anywhere in the serialized arguments."""
    lowered = text.lower()
    if not any(verb in lowered for verb in _SQL_MUTATION_VERBS):
        return False
    return bool(_SQL_MUTATION_RE.search(text))


#: Verb prefixes that mark an arbitrary (e.g. MCP) tool as mutating. A fixed
#: denylist cannot enumerate every third-party server's write operations, so
#: the leading verb is used as the safety net. Conservative by design: a false
#: positive only forgoes savings, a false negative destroys a record.
_MUTATING_NAME_PREFIXES: tuple[str, ...] = (
    "add",
    "append",
    "apply",
    "archive",
    "assign",
    "close",
    "comment",
    "create",
    "delete",
    "destroy",
    "edit",
    "insert",
    "merge",
    "modify",
    "move",
    "patch",
    "post",
    "publish",
    "push",
    "put",
    "remove",
    "rename",
    "replace",
    "save",
    "send",
    "set",
    "submit",
    "transfer",
    "update",
    "upload",
    "upsert",
    "write",
)

# Segments of a raw tool name, before `_normalize_tool_name` folds the
# separators away. Handles both spellings a tool name arrives in:
# `memory_save` -> ("memory", "save") and `TodoWrite` -> ("todo", "write").
_NAME_SEGMENTS = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")


def _name_segments(name: str) -> list[str]:
    """Lowercased word segments of a raw tool name."""
    return [part.casefold() for part in _NAME_SEGMENTS.findall(name)]


def _normalize_tool_name(name: Any) -> str:
    """Fold a tool name for denylist membership (case/separator-insensitive)."""
    if not isinstance(name, str):
        return ""
    return name.casefold().replace("_", "").replace("-", "")


def _mcp_leaf_candidates(tool_name: str) -> tuple[str, ...]:
    """Normalized leaf-name candidates for an MCP-wrapped tool name.

    Two wire spellings exist, and both are handled elsewhere in the codebase
    (``config._tool_name_aliases``, ``proxy.tool_name_policy``):

    * ``mcp__server__tool`` — unambiguous: the leaf is the text after the last
      ``__``, and the server label must never be judged (``mcp__github__get_file``
      has to stay compactable).
    * ``mcp_server_tool`` — emitted by Anthropic-speaking clients, and NOT
      unambiguous: both the server label and the leaf may contain ``_``
      (``mcp_github_create_or_update_file``, ``mcp_headroom_memory_memory_save``),
      so there is no single correct split point. Splitting only at the second
      underscore, the way ``_tool_name_aliases`` does for alias generation, is
      wrong *here* because a miss is not symmetric: THE RULE trades missed
      savings for never destroying a record. So EVERY underscore-delimited
      suffix is offered as a candidate and one mutating candidate is enough —
      ``mcp_github_create_or_update_file`` is caught via ``create_or_update_file``
      regardless of where the label ends.

    Returns an empty tuple for names that are not MCP-wrapped.
    """
    if "__" in tool_name:
        return (_normalize_tool_name(tool_name.rsplit("__", 1)[-1]),)
    if not tool_name.casefold().startswith("mcp_"):
        return ()
    rest = tool_name[4:]
    candidates: list[str] = []
    while rest:
        normalized = _normalize_tool_name(rest)
        if normalized:
            candidates.append(normalized)
        index = rest.find("_")
        if index == -1:
            break
        rest = rest[index + 1 :]
    return tuple(dict.fromkeys(candidates))


def is_mutating_tool_input(tool_name: str, serialized_args: str) -> bool:
    """True when this call's arguments are the sole record of a mutation.

    See THE RULE above. Deliberately conservative — a false positive costs
    only missed savings, a false negative costs an unrecoverable record.
    """
    normalized = _normalize_tool_name(tool_name)
    # `mcp__server__write_file` / `mcp_server_write_file`: judge the trailing
    # component(s), never the server label. See `_mcp_leaf_candidates`.
    leaves = _mcp_leaf_candidates(tool_name) or (normalized,)
    if normalized in MUTATING_TOOL_NAMES:
        return True
    for leaf in leaves:
        if leaf in MUTATING_TOOL_NAMES or leaf.startswith(_MUTATING_NAME_PREFIXES):
            return True
    # Not every name leads with its verb: `memory_save`, `page_update` and
    # `document_delete` all write, and a prefix test alone reads them as
    # read-only. Split the RAW name — `_normalize_tool_name` strips the
    # separators, so segments only survive before normalization. Whole-segment
    # equality (not prefix or suffix) is what keeps `get_asset` out while
    # catching the noun-first spelling.
    if any(part in _MUTATING_NAME_PREFIXES for part in _name_segments(tool_name)):
        return True
    # Content checks are ordered cheapest-guard-first; each is substring-gated
    # so the regex engine only ever sees inputs that could actually match.
    if _has_sql_mutation(serialized_args):
        return True
    if "<<" in serialized_args and _HEREDOC_RE.search(serialized_args):
        return True
    if _has_shell_mutation(serialized_args):
        return True
    # Last because its literal gate (`sh`) is the weakest of the set: anything
    # the shell probes already caught never reaches it.
    return _has_code_execution(serialized_args)


#: A retrievable CCR hash: 12-24 lowercase hex characters. This is the exact
#: shape every pattern in ``CCRToolInjector._marker_patterns`` captures — see
#: ``ccr/tool_injection.py``. Anything else in ``markers_inserted`` is
#: provenance metadata, not a redeemable handle. Case-insensitive because the
#: scanner's generic bracket pattern is ``re.IGNORECASE`` and can therefore
#: capture an upper-case hex hash; dropping those would weaken the fix this
#: filter protects.
_CCR_HASH_RE = re.compile(r"\A[a-fA-F0-9]{12,24}\Z")


def ccr_hashes_from_markers(markers: Any) -> list[str]:
    """The redeemable CCR hashes among a ``TransformResult.markers_inserted``.

    ``markers_inserted`` is a mixed bag: this pass and read-lifecycle put
    redeemable CCR hashes in it, but SmartCrusher appends
    ``<headroom:tool_digest sha256="…">`` provenance strings and CacheAligner
    appends ``stable_prefix_hash:…``. Only the hash-shaped entries may reach
    the injection decision.

    Why the filter is load-bearing (#1850): the merged list feeds
    ``has_new_ccr_markers``, which asks "is any of these absent from the
    previously-forwarded messages?" by re-scanning those messages with
    ``CCRToolInjector.scan_for_markers``. That scanner can only ever return
    hash-shaped strings, so a non-hash entry is *unconditionally* "new" — and
    on a byte-identical replayed prefix it would re-inject ``headroom_retrieve``
    every frozen turn, busting the tools cache segment the frozen prefix exists
    to protect. Filtering here, at the point markers are contributed, keeps a
    future emitter from reintroducing that.
    """
    hashes: list[str] = []
    seen: set[str] = set()
    for marker in markers or ():
        if not isinstance(marker, str) or marker in seen:
            continue
        if _CCR_HASH_RE.match(marker):
            seen.add(marker)
            hashes.append(marker)
    return hashes


#: A CCR hash as it appears inside the ``_ccr`` VALUE of a compacted tool
#: input. Both spellings that can legitimately land there are covered: the
#: load-bearing ``Retrieve original: hash=`` phrase written by
#: :meth:`ToolInputCompactor._store_original`, and the generic ``<<ccr:HASH>>``
#: blob marker. Mirrors the corresponding entries of
#: ``CCRToolInjector._marker_patterns`` (which only ever scans message TEXT).
_TOOL_ARG_CCR_HASH_RE = re.compile(r"(?:Retrieve original: hash=|<<ccr:)([a-fA-F0-9]{12,24})")

#: Cheap substring preconditions for :func:`_collect_tool_arg_hashes`. Even
#: inside ``_ccr`` the value is a short marker, but keeping the literal gate
#: means the regex only ever runs on text that could actually match.
_TOOL_ARG_CCR_LITERALS: tuple[str, ...] = ("hash=", "<<ccr:")


def _collect_tool_arg_hashes(text: str, out: list[str]) -> None:
    if not any(literal in text for literal in _TOOL_ARG_CCR_LITERALS):
        return
    out.extend(_TOOL_ARG_CCR_HASH_RE.findall(text))


def _collect_ccr_input_hashes(args: Any, out: list[str]) -> None:
    """Collect marker hashes from ONE tool call's ``_ccr`` property only.

    Scanning the whole arguments blob for marker text is wrong: an ordinary
    completed call can *legitimately contain* that text as data — this repo's
    own agents run ``Grep(pattern="<<ccr:")`` and search for
    ``Retrieve original: hash=`` — and a hash lifted out of a search pattern
    has no CCR entry behind it. Registering it would put ``headroom_retrieve``
    into the sticky session tracker for the rest of the session (a permanently
    useless tool, and one more toward the tool-search threshold). Only the
    ``_ccr`` key is ever written by :class:`ToolInputCompactor`, so only that
    key is read back.

    ``args`` is the OpenAI JSON *string* or the Anthropic ``input`` object. The
    JSON parse is gated on a plain substring test: transcripts with no ``_ccr``
    anywhere — the overwhelming majority — never pay for a parse. Malformed or
    non-object JSON is skipped, never raised.
    """
    if isinstance(args, str):
        if CCR_INPUT_KEY not in args:
            return
        try:
            args = json.loads(args)
        except (TypeError, ValueError):
            return  # Malformed arguments are not a marker.
    if not isinstance(args, dict):
        return
    value = args.get(CCR_INPUT_KEY)
    if isinstance(value, str):
        _collect_tool_arg_hashes(value, out)


def ccr_hashes_in_tool_arguments(messages: Any) -> list[str]:
    """Redeemable CCR hashes already present in tool-call ARGUMENTS.

    ``CCRToolInjector.scan_for_markers`` reads message text and tool-RESULT
    content; :class:`ToolInputCompactor` writes its marker into
    ``tool_calls[].function.arguments`` (OpenAI) / ``tool_use.input``
    (Anthropic). ``merge_pipeline_ccr_hashes`` closes that gap only for markers
    minted during the CURRENT pipeline run.

    One step further out lies the same bug: when a conversation that already
    contains a compacted tool input reaches a NEW worker or a restarted process,
    the compactor skips the existing ``_ccr`` value as idempotent (so the
    pipeline mints nothing), the in-memory session CCR tracker is empty (so the
    sticky injection has no history to replay), and the scanner cannot see the
    marker — the retrieval tool is never injected even though the marker and its
    persistent CCR entry are both still in the transcript, leaving the model a
    handle it cannot redeem. Scanning the transcript's tool arguments makes the
    decision depend on what is actually being forwarded rather than on
    process-local state.

    ONLY the ``_ccr`` property is inspected — never the rest of the arguments.
    Marker text appearing anywhere else is the tool call's own data (a ``Grep``
    pattern of ``<<ccr:`` or ``Retrieve original: hash=`` is exactly how these
    markers get audited), and a hash taken from there names no CCR entry; see
    :func:`_collect_ccr_input_hashes`.

    Results are shape-filtered through :func:`ccr_hashes_from_markers`, so only
    real ``[a-fA-F0-9]{12,24}`` hashes can drive injection.
    """
    found: list[str] = []
    for msg in messages or ():
        if not isinstance(msg, dict):
            continue
        # OpenAI shape: tool_calls[].function.arguments (JSON string).
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function")
                if not isinstance(func, dict):
                    continue
                _collect_ccr_input_hashes(func.get("arguments"), found)
        # Anthropic shape: tool_use content blocks with object input.
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                _collect_ccr_input_hashes(block.get("input"), found)
    return ccr_hashes_from_markers(found)


def merge_pipeline_ccr_hashes(
    detected_hashes: Any,
    pipeline_ccr_hashes: Any,
) -> list[str]:
    """Union of scanner-detected CCR hashes and hashes the pipeline minted.

    ``CCRToolInjector.scan_for_markers`` reads message TEXT and tool-RESULT
    content. This pass writes its marker into ``tool_use.input`` /
    ``tool_calls[].function.arguments``, which the scanner never visits — so on
    the first compaction of a session the scanner reports zero hashes, the
    provider handlers skip the sticky ``headroom_retrieve`` injection, and the
    stored original becomes unreachable. Handlers therefore merge the hashes
    the pipeline reported minting (``TransformResult.markers_inserted``) into
    the injection decision.

    Both sides are shape-filtered to real CCR hashes — see
    :func:`ccr_hashes_from_markers` for why anything else would re-inject the
    retrieval tool on every frozen turn. ``detected_hashes`` is already
    hash-shaped by construction; filtering it too costs nothing and makes the
    invariant hold at the choke point rather than only at the callers.

    Order-stable and de-duplicated: the merged list feeds
    ``has_new_ccr_markers``, whose result drives cache-affecting tool
    injection, so a stable order keeps that decision reproducible.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for source in (detected_hashes or (), pipeline_ccr_hashes or ()):
        for h in ccr_hashes_from_markers(source):
            if h not in seen:
                seen.add(h)
                merged.append(h)
    return merged


@dataclass
class ToolInputCompactionResult:
    """Output of the tool-input compaction pass."""

    messages: list[dict[str, Any]]
    compacted_count: int = 0
    chars_before: int = 0
    chars_after: int = 0
    transforms_applied: list[str] = field(default_factory=list)
    ccr_hashes: list[str] = field(default_factory=list)


class ToolInputCompactor:
    """Replace large completed tool-call arguments with CCR markers.

    Mirrors :class:`ReadLifecycleManager`'s shape: a pre-processing pass over
    ``messages`` run at the top of ``ContentRouter.apply``, storing originals
    in the shared compression store.
    """

    def __init__(
        self,
        config: ToolInputCompactionConfig,
        compression_store: Any | None = None,
    ):
        self.config = config
        self.store = compression_store

    def apply(
        self,
        messages: list[dict[str, Any]],
        frozen_message_count: int = 0,
    ) -> ToolInputCompactionResult:
        """Compact completed tool-call arguments in place-safe copies.

        Args:
            messages: Conversation messages (OpenAI or Anthropic shape).
            frozen_message_count: Leading messages inside the provider's
                prefix cache; never mutated.
        """
        result = ToolInputCompactionResult(messages=messages)
        if not self.config.enabled or not messages:
            return result

        completed_at = self._result_message_indices(messages)
        if not completed_at:
            return result

        protected = self._protected_assistant_indices(messages)

        new_messages: list[dict[str, Any]] | None = None
        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            if idx < frozen_message_count or idx in protected:
                continue
            compacted = self._compact_assistant_message(idx, msg, completed_at, result)
            if compacted is not None:
                if new_messages is None:
                    new_messages = list(messages)
                new_messages[idx] = compacted

        if new_messages is not None:
            result.messages = new_messages
            logger.info(
                "ToolInputCompactor: compacted %d tool inputs, %d -> %d chars",
                result.compacted_count,
                result.chars_before,
                result.chars_after,
            )
        return result

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _result_message_indices(self, messages: list[dict[str, Any]]) -> dict[str, int]:
        """Map tool_call_id -> index of the message carrying its result."""
        indices: dict[str, int] = {}
        for idx, msg in enumerate(messages):
            # OpenAI: role "tool" messages reference tool_call_id.
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if isinstance(tc_id, str) and tc_id and tc_id not in indices:
                    indices[tc_id] = idx
                continue
            # Anthropic: user messages carry tool_result content blocks.
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tc_id = block.get("tool_use_id")
                if isinstance(tc_id, str) and tc_id and tc_id not in indices:
                    indices[tc_id] = idx
        return indices

    def _protected_assistant_indices(self, messages: list[dict[str, Any]]) -> set[int]:
        """Indices of the trailing N assistant messages (never compacted)."""
        keep = self.config.protect_recent_turns
        if keep <= 0:
            return set()
        assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        return set(assistant_indices[-keep:])

    # ------------------------------------------------------------------
    # Replacement
    # ------------------------------------------------------------------

    def _compact_assistant_message(
        self,
        msg_index: int,
        msg: dict[str, Any],
        completed_at: dict[str, int],
        result: ToolInputCompactionResult,
    ) -> dict[str, Any] | None:
        """Return a compacted copy of ``msg``, or None if nothing changed."""
        changed = False
        new_msg = dict(msg)

        # OpenAI shape: tool_calls array with JSON-string arguments.
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            new_calls: list[Any] = []
            for tc in tool_calls:
                replacement = self._compact_openai_call(msg_index, tc, completed_at, result)
                if replacement is not None:
                    new_calls.append(replacement)
                    changed = True
                else:
                    new_calls.append(tc)
            if changed:
                new_msg["tool_calls"] = new_calls

        # Anthropic shape: tool_use content blocks with object input.
        content = msg.get("content")
        if isinstance(content, list):
            new_blocks: list[Any] = []
            block_changed = False
            for block in content:
                replacement = self._compact_anthropic_block(msg_index, block, completed_at, result)
                if replacement is not None:
                    new_blocks.append(replacement)
                    block_changed = True
                else:
                    new_blocks.append(block)
            if block_changed:
                new_msg["content"] = new_blocks
                changed = True

        return new_msg if changed else None

    def _compact_openai_call(
        self,
        msg_index: int,
        tc: Any,
        completed_at: dict[str, int],
        result: ToolInputCompactionResult,
    ) -> dict[str, Any] | None:
        if not isinstance(tc, dict):
            return None
        tc_id = tc.get("id")
        func = tc.get("function")
        if not isinstance(tc_id, str) or not isinstance(func, dict):
            return None
        args = func.get("arguments")
        if not isinstance(args, str) or len(args) < self.config.min_chars:
            return None
        if completed_at.get(tc_id, -1) <= msg_index:
            return None  # Pending or same-message result: arguments are live.
        if CCR_INPUT_KEY in args[:16]:
            return None  # Already compacted (idempotence).
        tool_name = str(func.get("name", ""))
        if is_mutating_tool_input(tool_name, args):
            return None  # THE RULE: mutating arguments are the only record.

        stored = self._store_original(args, tool_name=tool_name, tool_call_id=tc_id)
        if stored is None:
            return None  # Persistence failed — leave the arguments intact.
        marker, ccr_hash = stored
        replacement_args = json.dumps({CCR_INPUT_KEY: marker}, separators=(",", ":"))
        self._record(result, tool_name, len(args), len(replacement_args), ccr_hash)
        return {**tc, "function": {**func, "arguments": replacement_args}}

    def _compact_anthropic_block(
        self,
        msg_index: int,
        block: Any,
        completed_at: dict[str, int],
        result: ToolInputCompactionResult,
    ) -> dict[str, Any] | None:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            return None
        tc_id = block.get("id")
        inp = block.get("input")
        if not isinstance(tc_id, str) or not isinstance(inp, dict):
            return None
        if CCR_INPUT_KEY in inp:
            return None  # Already compacted (idempotence).
        try:
            serialized = json.dumps(inp, separators=(",", ":"), ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return None
        if len(serialized) < self.config.min_chars:
            return None
        if completed_at.get(tc_id, -1) <= msg_index:
            return None  # Pending or same-message result: arguments are live.
        tool_name = str(block.get("name", ""))
        if is_mutating_tool_input(tool_name, serialized):
            return None  # THE RULE: mutating arguments are the only record.

        stored = self._store_original(serialized, tool_name=tool_name, tool_call_id=tc_id)
        if stored is None:
            return None  # Persistence failed — leave the arguments intact.
        marker, ccr_hash = stored
        self._record(result, tool_name, len(serialized), len(marker), ccr_hash)
        return {**block, "input": {CCR_INPUT_KEY: marker}}

    def _store_original(
        self, serialized: str, *, tool_name: str, tool_call_id: str
    ) -> tuple[str, str] | None:
        """Persist the original arguments; return (marker, ccr_hash) or None.

        Successful persistence is a PRECONDITION for compaction, not a
        best-effort side effect. There is no catch-up mechanism: if the store
        is absent or ``store()`` raises, a marker would point at an entry that
        never existed and the historical arguments would be gone for good. So
        a failure returns ``None`` and the caller leaves the call untouched —
        the request still succeeds, it just saves no tokens this turn.
        """
        if self.store is None:
            return None
        ccr_hash = hashlib.sha256(serialized.encode()).hexdigest()[:24]
        try:
            ccr_hash = self.store.store(
                original=serialized,
                compressed="",
                tool_name=tool_name or "tool",
                tool_call_id=tool_call_id,
                compression_strategy="tool_input_compaction",
                explicit_hash=ccr_hash,
            )
        except Exception as e:  # noqa: BLE001 - storage failure must not break the request
            logger.warning("tool_input_compaction: CCR store failed for %s: %s", tool_call_id, e)
            return None
        if not isinstance(ccr_hash, str) or not ccr_hash:
            logger.warning("tool_input_compaction: CCR store returned no hash for %s", tool_call_id)
            return None
        # NOTE: the literal phrase "Retrieve original: hash=" is load-bearing —
        # the hash collectors in ccr/tool_injection.py match it, which keeps the
        # headroom_retrieve tool injected while compacted inputs are in context.
        marker = f"[tool input elided. Retrieve original: hash={ccr_hash}]"
        return marker, ccr_hash

    @staticmethod
    def _record(
        result: ToolInputCompactionResult,
        tool_name: str,
        chars_before: int,
        chars_after: int,
        ccr_hash: str,
    ) -> None:
        result.compacted_count += 1
        result.chars_before += chars_before
        result.chars_after += chars_after
        result.transforms_applied.append(f"tool_input_compaction:{tool_name or 'tool'}")
        result.ccr_hashes.append(ccr_hash)

"""Tests for completed tool-call input compaction (F3).

Large historical tool-call arguments are replaced with CCR markers once
their matching results have arrived; pending calls, recent turns, and the
frozen cache prefix are never touched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from headroom.config import ToolInputCompactionConfig
from headroom.transforms.tool_input_compactor import (
    CCR_INPUT_KEY,
    ToolInputCompactor,
    is_mutating_tool_input,
)

# Read-only (reproducible) arguments — the only kind that may be compacted.
LARGE_ARGS = json.dumps({"pattern": "def handler", "path": "/repo", "glob": "y" * 2000})
SMALL_ARGS = json.dumps({"pattern": "def handler"})
# A mutating call: its arguments are the sole exact record of the change.
MUTATING_ARGS = json.dumps({"file_path": "/tmp/x.py", "content": "x" * 2000})


class _FakeStore:
    def __init__(self) -> None:
        self.stored: list[dict[str, Any]] = []

    def store(self, **kwargs: Any) -> str:
        self.stored.append(kwargs)
        return kwargs["explicit_hash"]


def _cfg(**overrides: Any) -> ToolInputCompactionConfig:
    defaults: dict[str, Any] = {"enabled": True, "min_chars": 100, "protect_recent_turns": 0}
    defaults.update(overrides)
    return ToolInputCompactionConfig(**defaults)


def _openai_conversation() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "write the file"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Grep", "arguments": LARGE_ARGS},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]


def _anthropic_conversation() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "write the file"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Grep",
                    "input": {"pattern": "def handler", "path": "/repo", "glob": "y" * 2000},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
        {"role": "assistant", "content": "done"},
    ]


def test_openai_completed_call_is_compacted():
    store = _FakeStore()
    messages = _openai_conversation()
    result = ToolInputCompactor(_cfg(), compression_store=store).apply(messages)

    compacted = result.messages[1]["tool_calls"][0]
    assert compacted["id"] == "call_1"
    assert compacted["function"]["name"] == "Grep"
    args = json.loads(compacted["function"]["arguments"])
    assert set(args) == {CCR_INPUT_KEY}
    assert "Retrieve original: hash=" in args[CCR_INPUT_KEY]
    assert result.compacted_count == 1
    assert result.transforms_applied == ["tool_input_compaction:Grep"]
    assert len(result.ccr_hashes) == 1
    # Original bytes are retrievable from the store.
    assert store.stored[0]["original"] == LARGE_ARGS
    assert store.stored[0]["tool_call_id"] == "call_1"
    assert store.stored[0]["compression_strategy"] == "tool_input_compaction"
    # Input list is not mutated in place.
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == LARGE_ARGS


def test_anthropic_completed_call_is_compacted():
    store = _FakeStore()
    messages = _anthropic_conversation()
    result = ToolInputCompactor(_cfg(), compression_store=store).apply(messages)

    block = result.messages[1]["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "toolu_1"
    assert block["name"] == "Grep"
    assert set(block["input"]) == {CCR_INPUT_KEY}
    assert "Retrieve original: hash=" in block["input"][CCR_INPUT_KEY]
    assert result.compacted_count == 1
    # Original serialized input is retrievable.
    assert json.loads(store.stored[0]["original"])["pattern"] == "def handler"


def test_pending_call_is_never_compacted():
    # No tool result for the call: arguments are live working context.
    messages = _openai_conversation()
    del messages[2]  # remove the tool result
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(messages)
    assert result.compacted_count == 0
    assert result.messages is messages


def test_small_arguments_are_left_alone():
    messages = _openai_conversation()
    messages[1]["tool_calls"][0]["function"]["arguments"] = SMALL_ARGS
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(messages)
    assert result.compacted_count == 0


def test_protect_recent_turns_skips_trailing_assistant_messages():
    messages = _openai_conversation()
    # Both assistant messages fall inside the protection window of 2.
    result = ToolInputCompactor(_cfg(protect_recent_turns=2), compression_store=_FakeStore()).apply(
        messages
    )
    assert result.compacted_count == 0


def test_frozen_prefix_is_never_mutated():
    messages = _openai_conversation()
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        messages, frozen_message_count=2
    )
    assert result.compacted_count == 0


def test_disabled_is_noop():
    messages = _openai_conversation()
    result = ToolInputCompactor(_cfg(enabled=False), compression_store=_FakeStore()).apply(messages)
    assert result.messages is messages
    assert result.compacted_count == 0


def test_idempotent_on_already_compacted_input():
    store = _FakeStore()
    compactor = ToolInputCompactor(_cfg(), compression_store=store)
    first = compactor.apply(_openai_conversation())
    second = compactor.apply(first.messages)
    assert second.compacted_count == 0
    assert len(store.stored) == 1

    anthropic_first = compactor.apply(_anthropic_conversation())
    anthropic_second = compactor.apply(anthropic_first.messages)
    assert anthropic_second.compacted_count == 0


def test_store_failure_leaves_arguments_intact():
    """Codex P1: a failed store must NOT be replaced by an unredeemable marker."""

    class _BrokenStore:
        def store(self, **kwargs: Any) -> str:
            raise RuntimeError("boom")

    messages = _openai_conversation()
    result = ToolInputCompactor(_cfg(), compression_store=_BrokenStore()).apply(messages)
    assert result.compacted_count == 0
    assert result.messages is messages
    assert result.messages[1]["tool_calls"][0]["function"]["arguments"] == LARGE_ARGS


def test_missing_store_leaves_arguments_intact():
    """No store at all == no persistence == no compaction."""
    messages = _openai_conversation()
    result = ToolInputCompactor(_cfg(), compression_store=None).apply(messages)
    assert result.compacted_count == 0
    assert result.messages is messages


def test_store_returning_empty_hash_leaves_arguments_intact():
    class _EmptyHashStore:
        def store(self, **kwargs: Any) -> str:
            return ""

    result = ToolInputCompactor(_cfg(), compression_store=_EmptyHashStore()).apply(
        _openai_conversation()
    )
    assert result.compacted_count == 0


def test_result_before_call_does_not_count_as_completed():
    # A stray result at an EARLIER index must not mark the call completed.
    messages = [
        {"role": "tool", "tool_call_id": "call_1", "content": "stale"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Grep", "arguments": LARGE_ARGS},
                }
            ],
        },
        {"role": "assistant", "content": "done"},
    ]
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(messages)
    assert result.compacted_count == 0


# ---------------------------------------------------------------------------
# THE RULE: only reproducible / read-only inputs are compacted (Codex P1).
# A mutating call's result is a bare acknowledgement, so its arguments are the
# sole exact record of the change — and CCR entries expire (default 1,800s).
# ---------------------------------------------------------------------------


def _openai_call(name: str, args: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]


def test_write_tool_input_is_never_compacted():
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Write", MUTATING_ARGS)
    )
    assert result.compacted_count == 0


def test_apply_patch_input_is_never_compacted():
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("apply_patch", json.dumps({"patch": "@@\n+" + "a" * 2000}))
    )
    assert result.compacted_count == 0


def test_mcp_prefixed_write_tool_is_never_compacted():
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("mcp__fs__write_file", MUTATING_ARGS)
    )
    assert result.compacted_count == 0


def test_sql_mutation_input_is_never_compacted():
    sql = "INSERT INTO users (id, name) VALUES " + ",".join(f"({i},'n{i}')" for i in range(200))
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("run_query", json.dumps({"sql": sql}))
    )
    assert result.compacted_count == 0


def test_select_query_input_is_still_compacted():
    sql = (
        "SELECT id, name FROM users WHERE name IN (" + ",".join(f"'n{i}'" for i in range(300)) + ")"
    )
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("run_query", json.dumps({"sql": sql}))
    )
    assert result.compacted_count == 1


def test_shell_heredoc_input_is_never_compacted():
    cmd = "cat <<'EOF' > /tmp/config.yaml\n" + ("key: value\n" * 200) + "EOF"
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Bash", json.dumps({"command": cmd}))
    )
    assert result.compacted_count == 0


def test_shell_redirection_input_is_never_compacted():
    cmd = "printf '%s' '" + "x" * 2000 + "' > /tmp/out.txt"
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Bash", json.dumps({"command": cmd}))
    )
    assert result.compacted_count == 0


def test_quoted_redirection_target_is_never_compacted():
    """Codex (marked outdated, verified still broken): the redirection target's
    character class rejected the quote/escape that immediately follows `>`.

    The arguments are a serialized JSON string, so `> "$file"` arrives as
    `> \\"$file\\"`. Both the raw and the JSON-escaped spelling must classify
    as mutating — otherwise a `printf … > "$file"` write is replaced by an
    expiring CCR marker whose only companion is an empty shell result.
    """
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    for command in (
        'printf "%s" "$body" > "$file"',
        "echo hi > 'output.txt'",
        'cmd >> "output file"',
    ):
        assert is_mutating_tool_input("Bash", command) is True, f"raw: {command}"
        assert is_mutating_tool_input("Bash", json.dumps({"command": command})) is True, command

    cmd = 'printf "%s" "' + "x" * 2000 + '" > "$out_file"'
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Bash", json.dumps({"command": cmd}))
    )
    assert result.compacted_count == 0


def test_touch_style_file_mutation_is_never_compacted():
    """Codex P2: creating/linking/truncating a path is a mutation even when the
    command produces no output at all."""
    cmd = "touch " + " ".join(f"pkg{i}/__init__.py" for i in range(200))
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Bash", json.dumps({"command": cmd}))
    )
    assert result.compacted_count == 0


def test_read_only_shell_input_is_still_compacted():
    cmd = "rg -n 'def handler' " + " ".join(f"pkg{i}" for i in range(300))
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Bash", json.dumps({"command": cmd}))
    )
    assert result.compacted_count == 1


def test_anthropic_mutating_block_is_never_compacted():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Write",
                    "input": {"file_path": "/tmp/x.py", "content": "y" * 2000},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
        {"role": "assistant", "content": "done"},
    ]
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(messages)
    assert result.compacted_count == 0


def test_arrow_in_a_search_pattern_is_not_mistaken_for_redirection():
    """`->` / `=>` must not read as a write redirection (missed savings)."""
    pattern = "def handler(self) -> dict[str, Any]:  # " + "z" * 1500
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Grep", json.dumps({"pattern": pattern}))
    )
    assert result.compacted_count == 1


def test_is_mutating_tool_input_unit_cases():
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    assert is_mutating_tool_input("Write", "{}") is True
    assert is_mutating_tool_input("apply_patch", "{}") is True
    assert is_mutating_tool_input("Grep", '{"pattern":"a -> b"}') is False
    assert is_mutating_tool_input("Bash", '{"command":"ls -la | wc -l"}') is False
    assert is_mutating_tool_input("Bash", '{"command":"echo hi > out.txt"}') is True
    assert is_mutating_tool_input("Bash", '{"command":"sed -i s/a/b/ f.py"}') is True
    assert is_mutating_tool_input("Bash", '{"command":"git commit -m x"}') is True
    assert is_mutating_tool_input("Bash", '{"command":"git log --oneline -3"}') is False
    assert is_mutating_tool_input("q", '{"sql":"SELECT * FROM t"}') is False
    assert is_mutating_tool_input("q", '{"sql":"DELETE FROM t WHERE id=1"}') is True


def test_unknown_mcp_write_verbs_are_treated_as_mutating():
    """A fixed denylist can't enumerate every MCP server's write ops, so the
    leading verb is the safety net."""
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    for name in (
        "mcp__linear__create_issue",
        "mcp__notion__update_page",
        "mcp__github__push_files",
        "mcp__slack__post_message",
        "upload_artifact",
    ):
        assert is_mutating_tool_input(name, "{}") is True, name


def test_single_underscore_mcp_write_verbs_are_treated_as_mutating():
    """Codex P2: `mcp_server_tool` (single underscore) must be judged too.

    Anthropic-speaking clients emit the MCP wrapper as `mcp_github_create_file`
    rather than `mcp__github__create_file` (see `config._tool_name_aliases` and
    `proxy.tool_name_policy.normalize_headroom_tool_name`). Without leaf
    extraction the whole name normalizes to `mcpgithubcreatefile`, which starts
    with neither a denylisted name nor a mutating verb — so a write whose
    arguments contain only paths and file contents (no shell/SQL syntax to fall
    back on) was compactable, and the expiring CCR marker would leave a bare
    "success" result as the only record.
    """
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    for name in (
        "mcp_github_create_or_update_file",  # leaf itself contains underscores
        "mcp_fs_write_file",
        "mcp_notion_update_page",
        "mcp_linear_create_issue",
        "mcp_github_push_files",
        "mcp_slack_post_message",
        "mcp_github_delete_file",
        "MCP_GitHub_Create_Or_Update_File",  # casing is not significant
    ):
        assert is_mutating_tool_input(name, '{"path":"a.py","content":"x"}') is True, name


def test_read_only_mcp_tools_are_still_compactable():
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    for name in (
        "mcp__github__get_file_contents",
        "mcp__github__search_code",
        "mcp__linear__list_issues",
        # The single-underscore spelling must not become a blanket "mutating":
        # the leaf-candidate expansion is conservative, not indiscriminate.
        "mcp_github_get_file_contents",
        "mcp_github_search_code",
        "mcp_linear_list_issues",
        "Grep",
        "Read",
        "WebFetch",
    ):
        assert is_mutating_tool_input(name, "{}") is False, name


# ---------------------------------------------------------------------------
# THE RULE, pinned exhaustively. `is_mutating_tool_input` was rewritten from
# three broad regex passes to substring-gated linear scans for cost reasons
# (see the "Cost discipline" note in the module); the detection SET must not
# have moved. A false negative here silently destroys the only record of a
# mutation, so every shape gets an explicit case.
# ---------------------------------------------------------------------------

MUTATING_COMMANDS = [
    # write redirection
    "echo hi > out.txt",
    "cat a.py >> b.py",
    "printf x >/tmp/f",
    "make 2> build.log",
    "ls | tee listing.txt",
    # Quoted redirection targets (Codex, "outdated" finding — it was NOT fixed).
    # Note these are json.dumps()-ed by the test, so the quotes reach the
    # classifier escaped, exactly as they do in real serialized arguments.
    'printf "%s" "$body" > "$file"',
    "echo hi > 'output.txt'",
    'cmd >> "output file"',
    'cat a.py >"b.py"',
    "echo x >| clobbered.txt",
    # touch-style mutations: create / link / truncate / metadata (Codex P2)
    "touch /tmp/newfile",
    "install -m 755 build/app /usr/local/bin/app",
    "mkfifo /tmp/pipe",
    "mknod /tmp/dev c 1 3",
    "unlink stale.lock",
    "shred -u secret.pem",
    "chgrp staff report.txt",
    "chattr +i locked.conf",
    "rsync -a src/ dst/",
    "setfacl -m u:me:rw f",
    "patch -p1 < fix.diff",
    "patch < fix.diff",
    # in-place edits
    "sed -i s/a/b/ f.py",
    "sed --regexp-extended -i 's/a/b/' f.py",
    "grep foo f.py | sed -i s/x/y/ g.py",
    "perl -i -pe s/a/b/ f.py",
    # destructive file ops
    "rm -rf build/",
    "mv a.py b.py",
    "cp a.py b.py",
    "mkdir -p out",
    "chmod 755 run.sh",
    "chown me:me f",
    "ln -s a b",
    "truncate -s 0 log",
    "dd if=/dev/zero of=f",
    # vcs / package mutations
    "git commit -m x",
    "git apply patch.diff",
    "git checkout -- .",
    "npm install left-pad",
    "pip install requests",
    "cargo add serde",
    "apt-get remove foo",
    # heredocs (JSON-escaped and raw newline forms)
    "cat <<EOF > f\\nbody\\nEOF",
    "cat <<'EOF'\nbody\nEOF",
    "python - <<-PY\nprint(1)\nPY",
    # ---------------------------------------------------------------
    # Embedded code execution (Codex P2). Every probe above matches a
    # mutation written in SHELL; a write performed inside an interpreter's
    # inline program contains no shell write token at all, so the arguments
    # were classified read-only and the only exact record of the write became
    # eligible for an expiring CCR marker. The interpreter+inline-program
    # SHAPE is what is matched — not the write API — because the set of ways
    # an arbitrary program can write is not enumerable.
    # ---------------------------------------------------------------
    # The three examples from the finding, verbatim.
    "python -c \"Path('out').write_text(payload)\"",
    "node -e \"fs.writeFileSync('out', payload)\"",
    "ruby -e \"File.write('out', payload)\"",
    # Same shape, other write APIs the enumeration approach would have to
    # chase: none of these is spelled out anywhere in the implementation.
    "python3 -c \"open('out','w').write(payload)\"",
    'python -c "import shutil; shutil.copy(a, b)"',
    'python -c "import os; os.replace(a, b)"',
    "node -e \"require('fs').promises.writeFile(p, d)\"",
    'ruby -e "IO.write(p, d)"',
    # Escape hatches that no write-API list can cover.
    'python -c "import os; os.system(cmd)"',
    'python -c "exec(base64.b64decode(blob))"',
    'python -c "conn.execute(stmt)"',
    # Other interpreters / flag spellings.
    ".venv/bin/python -c 'print(1)'",
    "python -E -c 'print(1)'",
    "perl -ne 'print' f.txt",
    "php -r 'file_put_contents($f, $d);'",
    "lua -e 'x()'",
    "julia -e 'x()'",
    "Rscript -e 'writeLines(x, f)'",
    "osascript -e 'do shell script \"x\"'",
    "tclsh -c 'x'",
    "node --eval 'x()'",
    "deno eval 'Deno.writeTextFile(p, d)'",
    "bun eval 'x()'",
    "sh -c 'x'",
    "bash -c 'x'",
    "bash -lc 'x'",
    "zsh -c 'x'",
    "pwsh -Command 'x'",
    # Program supplied on stdin. `python - <<'PY' 2>&1 | tail` is NOT caught by
    # the heredoc probe (the delimiter is followed by a redirect, not by a
    # newline), and a piped program has no heredoc at all.
    "timeout 300 .venv/bin/python - <<'PY' 2>&1 | tail -20",
    "curl -s http://x/script.py | python -",
]

NON_MUTATING_COMMANDS = [
    "ls -la | wc -l",
    "git log --oneline -3",
    "rg 'x => y' .",
    "cat f.py",
    "grep -rn 'sed' docs/",  # mentions sed, no -i
    "git status --porcelain",
    "npm ls --depth 0",
    # The touch-family additions are word+separator anchored, so merely naming
    # one of the commands in a search pattern stays compactable.
    "grep -rn 'touch' docs/",
    "rg 'patch' src/",
    "ls -l installers/",
    # The code-execution rule is anchored on interpreter + inline-program flag,
    # so merely NAMING an interpreter — or running one with an ordinary,
    # non-program argument — stays compactable. This is what keeps the closed
    # rule from swallowing the whole Bash tool.
    "python --version",
    "node --version",
    "which python3",
    "ls scripts/*.sh",
    "head -50 run.sh",
    "wc -l tests/*.py",
]


@pytest.mark.parametrize(
    "command",
    [
        "terraform apply -var x=" + "y" * 2048,
        "docker run -e X=" + "y" * 2048,
        "ansible-playbook site.yml",
        "find . -delete",
        "awk 'BEGIN { system(\"touch marker\") }'",
        "rg x | terraform apply",
        "ls && docker run image",
        "ls\nterraform apply",
        "curl -o /tmp/out https://example.test",
        "sort -o /tmp/out input",
        "uniq input /tmp/out",
        "git diff --output=/tmp/out HEAD",
        "git log --ext-diff -1",
        "git -c diff.external=./mutator diff",
        "helm template app --post-renderer ./mutator",
        "rg --pre ./mutator needle",
        "rg --hostname-bin=./mutator needle",
        "git grep -O./mutator needle",
        "cargo tree",
        "cat =(terraform apply)",
    ],
)
def test_unknown_shell_commands_fail_closed(command: str) -> None:
    assert is_mutating_tool_input("Bash", json.dumps({"command": command})) is True


@pytest.mark.parametrize(
    "tool_name",
    ["shell", "exec_command", "run_shell_command", "terminal", "functions.exec_command"],
)
def test_shell_executor_aliases_fail_closed(tool_name: str) -> None:
    assert is_mutating_tool_input(tool_name, '{"cmd":"terraform apply"}') is True
    assert is_mutating_tool_input(tool_name, '{"cmd":"rg needle src | head"}') is False
    assert (
        is_mutating_tool_input(
            tool_name, '{"command":"rg needle src","cmd":"terraform apply"}'
        )
        is True
    )


def test_non_shell_lookalike_stays_compactable() -> None:
    assert is_mutating_tool_input("Grep", '{"pattern":"terraform apply"}') is False


@pytest.mark.parametrize("command", MUTATING_COMMANDS)
def test_mutating_shell_inputs_are_never_compacted(command: str) -> None:
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    args = json.dumps({"command": command})
    assert is_mutating_tool_input("Bash", args) is True, command


@pytest.mark.parametrize("command", NON_MUTATING_COMMANDS)
def test_read_only_shell_inputs_stay_compactable(command: str) -> None:
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    args = json.dumps({"command": command})
    assert is_mutating_tool_input("Bash", args) is False, command


MUTATING_SQL = [
    "INSERT INTO t VALUES (1)",
    "insert into t values (1)",
    "UPDATE t SET a=1",
    "DELETE FROM t WHERE id=1",
    "MERGE INTO t USING s ON (1=1)",
    "REPLACE INTO t VALUES (1)",
    "DROP TABLE t",
    "ALTER TABLE t ADD COLUMN c INT",
    "TRUNCATE TABLE t",
    "CREATE INDEX i ON t (c)",
    "CREATE OR REPLACE VIEW v AS SELECT 1",
    "GRANT SELECT ON t TO u",
    "REVOKE SELECT ON t FROM u",
]

NON_MUTATING_SQL = [
    "SELECT * FROM t",
    "SELECT count(*) FROM updates",
    "WITH x AS (SELECT 1) SELECT * FROM x",
    "SHOW TABLES",
]

#: Known over-conservative classifications, unchanged by the rewrite. A bare
#: `>` or `>>` followed by a word is read as write-redirection even inside an
#: expression, so these read-only inputs forgo compaction. Pinned so the
#: rewrite's equivalence with the original regex is explicit: the cost is
#: missed savings, never a lost record, which is the direction THE RULE picks.
OVER_CONSERVATIVE_INPUTS = [
    ("Bash", "python -c 'print(1 >> 2)'"),
    ("query", "EXPLAIN SELECT a FROM t WHERE a > 5"),
    # The code-execution rule is text-based, like every other probe here, so an
    # input that merely QUOTES an interpreter invocation is read as one. Same
    # direction the existing probes already take (`grep -rn 'rm -rf' docs/` has
    # always been classified mutating by `_FILEOP_RE`): missed savings, never a
    # lost record.
    ("Bash", 'grep -rn "python -c" docs/analysis/'),
    ("Agent", "Explain to the user why `node -e` payloads are risky."),
]


@pytest.mark.parametrize(("tool", "text"), OVER_CONSERVATIVE_INPUTS)
def test_over_conservative_cases_are_unchanged(tool: str, text: str) -> None:
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    assert is_mutating_tool_input(tool, json.dumps({"command": text})) is True


@pytest.mark.parametrize("sql", MUTATING_SQL)
def test_sql_mutations_are_never_compacted(sql: str) -> None:
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    assert is_mutating_tool_input("query", json.dumps({"sql": sql})) is True, sql


@pytest.mark.parametrize("sql", NON_MUTATING_SQL)
def test_read_only_sql_stays_compactable(sql: str) -> None:
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    assert is_mutating_tool_input("query", json.dumps({"sql": sql})) is False, sql


EMBEDDED_RUNTIME_WRITES = [
    # Codex P2, verbatim: a completed Bash/exec_command call that writes through
    # an embedded runtime. No shell write token appears anywhere in the command.
    ("Bash", "python -c \"from pathlib import Path; Path('out').write_text(body)\""),
    ("Bash", "node -e \"fs.writeFileSync('out', body)\""),
    ("exec_command", "ruby -e \"File.write('out', body)\""),
]


@pytest.mark.parametrize(("tool", "command"), EMBEDDED_RUNTIME_WRITES)
def test_embedded_runtime_write_is_never_compacted(tool: str, command: str) -> None:
    """The arguments are the ONLY record of the write, so they must survive.

    Without the code-execution rule all three classified as read-only: the
    command contains no `>`, no heredoc, no `sed -i`, no destructive verb — the
    write lives entirely inside the interpreter's inline program. Above
    `min_chars` the sole exact record of the change would be replaced by a CCR
    marker that expires after `CCRConfig.ttl_seconds` (default 1,800s).
    """
    store = _FakeStore()
    # Padded past min_chars the way a real inline script is: the payload being
    # written is what makes these calls large in the first place.
    args = json.dumps({"command": command, "payload": "x" * 2000})
    assert is_mutating_tool_input(tool, args) is True, command

    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": tool, "arguments": args}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    result = ToolInputCompactor(_cfg(), compression_store=store).apply(messages)
    assert result.compacted_count == 0
    assert result.messages[1]["tool_calls"][0]["function"]["arguments"] == args
    assert store.stored == []


def test_embedded_runtime_write_survives_padding() -> None:
    """The safety property must not be defeated by burying the call in padding."""
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    padding = ("grep -rn 'a -> b' src/ " * 3000)[:65536]
    assert is_mutating_tool_input("Bash", padding) is False
    for command, _ in ((c, t) for t, c in EMBEDDED_RUNTIME_WRITES):
        assert is_mutating_tool_input("Bash", padding + " ; " + command) is True, command
    assert is_mutating_tool_input("Bash", padding + ' ; bash -c "$SCRIPT"') is True
    assert is_mutating_tool_input("Bash", padding + " ; deno eval 'x()'") is True


def test_mutation_detection_is_bounded_on_a_hostile_blob() -> None:
    """The scan must stay linear on adversarial input.

    `\\bsed\\b[^|;&]*\\s-i\\b` re-scanned the whole tail from every `sed`
    occurrence: a 64 KB `sed`-dense argument blob took multiple SECONDS, which
    is a DoS-shaped risk since tool arguments are attacker-influenced and this
    runs on every completed call of every request.
    """
    import time

    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    blobs = [
        ("sed expression foo bar baz qux " * 2200)[:65536],  # no shell separators
        ("sed 's/foo/bar/' file.txt && " * 2400)[:65536],
        ("grep -rn 'a -> b' src/ " * 3000)[:65536],  # `->`-dense, redirect branch
        ("perl print scalar keys " * 3000)[:65536],
        # Code-execution branch: interpreter-dense text where the inline-source
        # flag is always just out of reach. The gap between the interpreter and
        # the flag is bounded to one short option precisely so these cannot
        # degrade into a scan-to-end-of-string from every occurrence.
        ("python -m pytest -q --tb=line tests/ " * 1800)[:65536],
        ("the shell should show a finished dashboard " * 1600)[:65536],
        ("node modules ruby gems perl modules php files " * 1400)[:65536],
        ("python3 nonsense ruby gemfile perl module " * 1600)[:65536],
    ]
    for blob in blobs:
        start = time.perf_counter()
        is_mutating_tool_input("Bash", blob)
        elapsed = time.perf_counter() - start
        # Comfortably above the ~1 ms the linear scan needs, far below the
        # multi-second quadratic blowup it replaced.
        assert elapsed < 0.25, f"{elapsed:.3f}s on a 64 KB blob: {blob[:40]!r}"


def test_hostile_blob_still_detects_a_trailing_mutation() -> None:
    """Bounded cost must not come from truncating the inspected text."""
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    padding = ("grep -rn 'a -> b' src/ " * 3000)[:65536]
    assert is_mutating_tool_input("Bash", padding) is False
    assert is_mutating_tool_input("Bash", padding + " ; echo x > out.txt") is True
    assert is_mutating_tool_input("Bash", padding + " ; sed -i s/a/b/ f.py") is True
    assert is_mutating_tool_input("Bash", padding + " ; DROP TABLE t") is True
    # Shapes added for the Codex P2 findings must be caught after padding too.
    assert is_mutating_tool_input("Bash", padding + ' ; echo x > "$out"') is True
    assert is_mutating_tool_input("Bash", padding + " ; touch marker") is True
    assert is_mutating_tool_input("Bash", padding + " ; mkfifo p") is True


@pytest.mark.parametrize(
    "tool_name",
    [
        # Noun-first spelling: the verb is the trailing segment, so a prefix
        # test alone reads these as read-only.
        "memory_save",
        "memory_update",
        "memory_delete",
        "page_update",
        "document_delete",
        "mcp__headroom-memory__memory_save",
        "mcp_headroom_memory_memory_save",
        # camelCase has no separators at all to split on.
        "TodoWrite",
        "NotebookEdit",
        "MultiEdit",
    ],
)
def test_noun_first_and_camel_case_names_are_mutating(tool_name: str) -> None:
    assert is_mutating_tool_input(tool_name, '{"content": "x"}') is True


@pytest.mark.parametrize(
    "tool_name",
    [
        "Read",
        "Grep",
        "Glob",
        "WebFetch",
        "WebSearch",
        "BashOutput",
        "read_file",
        "search_files",
        "list_directory",
        "list_issues",
        "query_database",
        "memory_search",
        # Whole-segment equality, not prefix/suffix: "asset" must not match
        # the "set" verb, and "settings" must not match either.
        "get_asset",
        "get_settings",
    ],
)
def test_read_only_names_stay_compactable(tool_name: str) -> None:
    assert is_mutating_tool_input(tool_name, '{"path": "x"}') is False


@pytest.mark.parametrize(
    "command",
    [
        # `git add` with hundreds of paths is the long-argument, empty-result
        # shape this guard exists for: once the CCR entry lapses, nothing
        # records what was staged.
        "git add -A",
        "git add src/a.py src/b.py src/c.py",
        "git stash push -u",
        "git switch -c feature/x",
        "git cherry-pick abc123",
        "git restore --staged file.py",
        "git rm -r build/",
        "git mv old.py new.py",
        "git tag -a v1.0 -m release",
        "git worktree add ../wt",
        "git submodule update --init",
        "git config user.name someone",
        "git update-ref refs/heads/x abc",
        "git sparse-checkout set src",
        "git -C /repo add src/a.py",
        "git -c user.name=x commit -m x",
        "git --git-dir=/repo/.git reset --hard HEAD",
        'git -C "/tmp/my repo" add x',
        'git -c "user.name=Jane Doe" commit -m x',
        "git --git-dir repo reset --hard HEAD",
        "git --work-tree tree --git-dir repo add x",
        "git --namespace ns push",
        r"git -C /tmp/my\ repo add x",
        r"git -c user.name=Jane\ Doe commit -m x",
        r"git --git-dir=/tmp/my\ repo reset --hard HEAD",
        "git --literal-pathspecs add :(literal)file",
        "git --glob-pathspecs add *.py",
        "git --noglob-pathspecs add *.py",
        "git --icase-pathspecs add README.md",
        "git --shallow-file shallow reset --hard HEAD",
        "git --shallow-file=shallow reset --hard HEAD",
    ],
)
def test_state_changing_git_subcommands_are_mutating(command: str) -> None:
    assert is_mutating_tool_input("Bash", command) is True


@pytest.mark.parametrize(
    "command",
    [
        # Read-only porcelain must stay compactable — it is reproducible by
        # re-running the command.
        "git status",
        "git log --oneline -5",
        "git diff HEAD",
        "git show abc123",
        "git blame file.py",
        "git grep pattern",
        "git ls-files",
        "git rev-parse HEAD",
        "git describe --tags",
        "git -C /repo status",
        "git --git-dir=/repo/.git diff HEAD",
        "git --help add",
        "git --version reset",
        "git --html-path commit",
    ],
)
def test_read_only_git_subcommands_stay_compactable(command: str) -> None:
    assert is_mutating_tool_input("Bash", command) is False


def test_git_mutation_detection_has_no_global_option_count_ceiling() -> None:
    options = " ".join(f"-c key{i}=value" for i in range(9))
    assert is_mutating_tool_input("Bash", f"git {options} commit -m x") is True
    quoted = json.dumps({"command": 'git -C "/tmp/my repo" add x'})
    assert is_mutating_tool_input("Bash", quoted) is True
    for command in (
        r"git -C /tmp/my\ repo add x",
        r"git -c user.name=Jane\ Doe commit -m x",
        r"git --git-dir=/tmp/my\ repo reset --hard HEAD",
        r"git -C /tmp/foo\\ add x",
        r"git -c key=value\\ commit -m x",
        r"git --git-dir=/tmp/foo\\ reset --hard HEAD",
    ):
        assert is_mutating_tool_input("Bash", command) is True
        assert is_mutating_tool_input("Bash", json.dumps({"command": command})) is True

    long_value = "git -c test.value=" + "x" * 513 + " commit -m x"
    assert is_mutating_tool_input("Bash", long_value) is True
    assert is_mutating_tool_input("Bash", json.dumps({"command": long_value})) is True


def test_git_config_options_fail_closed_without_backtracking() -> None:
    options = " ".join('-c "key=value"' for _ in range(30))
    assert is_mutating_tool_input("Bash", f"git {options} status") is True


@pytest.mark.parametrize(
    "command",
    [
        # `patch` as a CLI subcommand: the word is followed by a resource type,
        # not by `-`/`<`.
        'kubectl patch deployment app --patch \'{"spec": {"replicas": 3}}\'',
        "oc patch svc web -p '{}'",
        "kubectl apply --patch-file overlay.yaml",
        # ...and the classic forms still match.
        "patch -p1 < fix.diff",
        "patch < fix.diff",
        # Options and toolchain selectors between the package manager and verb.
        "apt-get -y remove pkg-a pkg-b pkg-c",
        "npm --global uninstall left-pad",
        "cargo +nightly uninstall ripgrep",
        "pip -q install requests",
        "poetry add flask",
        "gem install rails",
        "dotnet add package Newtonsoft.Json",
    ],
)
def test_patch_and_package_mutations_are_detected(command: str) -> None:
    assert is_mutating_tool_input("Bash", command) is True


@pytest.mark.parametrize(
    "command",
    [
        # "patch" as an ordinary word, and read-only package/cluster queries.
        "grep -rn patch /repo",
        "cat patchnotes.md",
        "echo the patch was reviewed",
        "kubectl get pods",
        "kubectl describe deployment app",
        "npm ls",
        "pip list",
        "poetry show",
        "gem list",
    ],
)
def test_patch_lookalikes_and_queries_stay_compactable(command: str) -> None:
    assert is_mutating_tool_input("Bash", command) is False


@pytest.mark.parametrize(
    "command",
    [
        "kubectl apply -f -",
        "kubectl --context prod -n api create configmap settings --from-file app.env",
        'kubectl --context "$CTX" apply -f -',
        "kubectl replace -f deployment.yaml",
        "kubectl scale deployment api --replicas=3",
        "kubectl rollout restart deployment/api",
        "oc delete route app",
        "helm upgrade --install app ./chart",
        'helm --kube-context "$CTX" upgrade app ./chart',
        "helm uninstall app",
        'curl -X POST https://api.example.test/items -d \'{"name":"x"}\'',
        "curl --request=PUT https://api.example.test/items/1 --data-binary @item.json",
        'curl -XPATCH https://api.example.test/items/1 --json \'{"name":"y"}\'',
        "curl -X DELETE https://api.example.test/items/1",
        "curl https://api.example.test/items --data-urlencode name=x",
        "curl -F artifact=@build.tgz https://api.example.test/upload",
        "curl -T build.tgz https://api.example.test/upload",
        "curl -K -",
        "curl -K-",
        "curl -Krequest.conf",
        "curl -K request.conf",
        "curl -sK-",
        "curl -fsK -",
        "curl -#K-",
        "curl --config request.conf",
    ],
)
def test_remote_mutating_shell_commands_are_detected(command: str) -> None:
    assert is_mutating_tool_input("Bash", json.dumps({"command": command})) is True


@pytest.mark.parametrize(
    "command",
    [
        "kubectl get pods",
        "kubectl describe deployment api",
        "kubectl logs deployment/api",
        "helm list",
        "helm status app",
    ],
)
def test_remote_read_commands_stay_compactable(command: str) -> None:
    assert is_mutating_tool_input("Bash", json.dumps({"command": command})) is False


@pytest.mark.parametrize(
    "args",
    [
        json.dumps(
            {
                "command": "kubectl apply -f -",
                "stdin": "apiVersion: v1\nkind: ConfigMap\ndata:\n  body: " + "x" * 2000,
            }
        ),
        json.dumps(
            {
                "command": "curl -X POST https://api.example.test/items -d '" + "x" * 2000 + "'",
            }
        ),
        json.dumps(
            {
                "command": "curl -K -",
                "stdin": (
                    "url = https://api.example.test/items\nrequest = POST\ndata = " + "x" * 2000
                ),
            }
        ),
        json.dumps(
            {
                "command": "curl --config -",
                "stdin": "url = https://api.example.test/items\nrequest = GET",
            },
            separators=(",", ":"),
        ),
        json.dumps(
            {
                "command": "curl -fsK -",
                "stdin": "url = https://api.example.test/items\nrequest = POST",
            }
        ),
        json.dumps(
            {
                "command": "curl -#K-",
                "stdin": "url = https://api.example.test/items\nrequest = POST",
            }
        ),
    ],
)
def test_remote_mutation_payload_is_never_compacted(args: str) -> None:
    store = _FakeStore()
    result = ToolInputCompactor(_cfg(), compression_store=store).apply(_openai_call("Bash", args))
    assert result.compacted_count == 0
    assert result.messages[1]["tool_calls"][0]["function"]["arguments"] == args
    assert store.stored == []

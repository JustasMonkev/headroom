"""Guard: the Codex WS frame-compression success path stays reachable.

The ``/v1/responses`` WS frame handler compresses each frame and, on failure,
forwards the raw bytes with a ``compression_exception`` reason. That failure
``return`` was once dedented to the try/except's own level, so it ran on the
SUCCESS path too — every frame returned uncompressed and the whole
compression-success block below it was unreachable dead code.

This parses the module (no import — it pulls in the ML stack) and asserts the
``compression_exception`` ``return`` lives inside an ``except`` handler, so the
success path after it can run.
"""

from __future__ import annotations

import ast
from pathlib import Path

_OPENAI = (
    Path(__file__).resolve().parents[1] / "headroom" / "proxy" / "handlers" / "openai.py"
)


def _returns_compression_exception(node: ast.Return) -> bool:
    return any(
        isinstance(c, ast.Constant) and c.value == "compression_exception"
        for c in ast.walk(node)
    )


def test_frame_compression_exception_return_is_inside_except():
    tree = ast.parse(_OPENAI.read_text(encoding="utf-8"))

    # Every `return ... "compression_exception" ...` must be a direct child of an
    # `except` handler body — never a sibling of the success path in a `try`
    # body or the function body (which would make the success path unreachable).
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Return) and _returns_compression_exception(stmt):
                found = True

    stray = []
    for node in ast.walk(tree):
        # Statement lists that are NOT except-handler bodies.
        if isinstance(node, ast.ExceptHandler):
            continue
        for attr in ("body", "orelse", "finalbody"):
            stmts = getattr(node, attr, None)
            if not isinstance(stmts, list):
                continue
            for i, stmt in enumerate(stmts):
                if (
                    isinstance(stmt, ast.Return)
                    and _returns_compression_exception(stmt)
                    and i < len(stmts) - 1  # something follows it -> it's unreachable-making
                ):
                    stray.append(stmt.lineno)

    assert found, "expected the compression_exception return inside an except handler"
    assert not stray, f"compression_exception return orphans following code at lines {stray}"

"""Compression summary generator — describes what was compressed.

When content is compressed, the LLM needs to know what it's missing so it can
decide whether to call headroom_retrieve and what to search for.

Used by:
- CodeCompressor: lists removed function/class names (from AST, language-agnostic)

A `summarize_dropped_items` sibling used to live here. It had zero production
consumers — its would-be consumer was the dead `CCRConfig.marker_template`
(both removed per docs/token-efficiency-review.md "Dead code to remove").
"""

from __future__ import annotations

import re


def summarize_compressed_code(
    function_bodies: list[tuple[str, str, int]],
    compressed_bodies_count: int,
) -> str:
    """Generate a summary of compressed code sections from AST data.

    Language-agnostic: works with any language tree-sitter supports because
    it reads function signatures directly from the CodeCompressor's AST output.

    Args:
        function_bodies: List of (signature, body, line) from CodeStructure.
        compressed_bodies_count: Number of bodies that were compressed.

    Returns:
        Summary string like "5 bodies: authenticate, validate_token" or empty
        string.

    The phrasing is deliberately bare. This lands inside a marker that already
    opens with "N tokens compressed", so a second "compressed" was a duplicate
    verb; the `()` suffix cost ~1 token per name to say "function", which the
    word "bodies" already said; and `(+3 more)` cost ~5 tokens to announce that
    a list of examples is a list of examples.
    """
    if not function_bodies or compressed_bodies_count == 0:
        return ""

    # Extract short names from signatures
    names = []
    for sig, _body, _line in function_bodies:
        name = _extract_name_from_signature(sig)
        if name:
            names.append(name)

    if not names:
        return f"{compressed_bodies_count} bodies"

    return f"{compressed_bodies_count} bodies: {', '.join(names[:_MAX_NAMES_LISTED])}"


# ---- Internal helpers ----

# Names listed in a compressed-code summary. Examples, not an index.
_MAX_NAMES_LISTED = 6


def _extract_name_from_signature(sig: str) -> str:
    """Extract the function/method name from a signature string.

    Works for any language because it looks for common patterns:
    - Python: "def authenticate(", "async def fetch("
    - JavaScript: "function authenticate(", "async function fetch("
    - Go: "func (s *Server) HandleRequest("
    - Rust: "fn authenticate("
    - Java/C++: "public void authenticate("

    Returns a BARE name. The `()` suffix this used to append cost ~1 token per
    name to say "this is a function", which the caller's wording already says.
    """
    # Try common function definition patterns
    match = re.search(r"(?:def|func|fn|function)\s+(?:\([^)]*\)\s*)?(\w+)", sig)
    if match:
        return match.group(1)

    # Try method patterns: "public static void methodName("
    match = re.search(r"(?:public|private|protected|static|async|export)\s+.*?(\w+)\s*\(", sig)
    if match:
        return match.group(1)

    # Try class patterns
    match = re.search(r"class\s+(\w+)", sig)
    if match:
        return match.group(1)

    # Fallback: last word before (
    match = re.search(r"(\w+)\s*\(", sig)
    if match:
        return match.group(1)

    return ""

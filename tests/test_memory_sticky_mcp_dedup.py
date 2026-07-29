"""A2: sticky memory tool injection must dedupe Headroom-owned mcp__ aliases."""
from headroom.memory.tools import MEMORY_TOOLS_OPTIMIZED
from headroom.proxy.helpers import apply_session_sticky_memory_tools


def _names(tools):
    out = []
    for t in tools:
        out.append(t.get("name") or t.get("function", {}).get("name"))
    return out


def _inject(existing, session_id):
    tools, injected = apply_session_sticky_memory_tools(
        existing_tools=existing,
        memory_tools_to_inject=list(MEMORY_TOOLS_OPTIMIZED),
        provider="anthropic",
        session_id=session_id,
        inject_this_turn=True,
        request_id="r1",
    )
    return _names(tools), injected


def test_headroom_owned_mcp_alias_dedupes():
    existing = [{"name": "mcp__headroom-memory__memory_save", "input_schema": {}}]
    names, _ = _inject(existing, "sticky-owned")
    assert names.count("memory_save") == 0, names


def test_foreign_mcp_server_does_not_dedupe():
    existing = [{"name": "mcp__other-memory__memory_save", "input_schema": {}}]
    names, _ = _inject(existing, "sticky-foreign")
    assert "memory_save" in names, names

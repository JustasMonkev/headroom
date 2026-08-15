"""Headroom-owned MCP install ledger.

The ledger tracks MCP servers that Headroom registered on the user's behalf
when the target agent config cannot carry Headroom-specific ownership markers.
It lets unwrap remove only entries still matching the spec Headroom installed,
preserving user-managed MCP servers with the same name.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from headroom import _filelock, paths

from .base import ServerSpec

_LEDGER_FILE = "mcp_installs.json"


def ledger_path() -> Path:
    """Return the Headroom MCP install ledger path.

    The ledger records MCP servers Headroom installed into the user's global
    agent config, which is persistent and shared. It therefore resolves
    against the shared workspace, not a per-run isolated workspace: otherwise
    an isolated wrap would write ownership records into a throwaway directory,
    and a later ``headroom unwrap`` (reading the shared ledger) could not
    prove ownership and would leave Headroom-installed entries behind.
    """
    return paths.shared_workspace_dir() / _LEDGER_FILE


def spec_fingerprint(spec: ServerSpec) -> str:
    """Stable fingerprint for a registered MCP server spec."""
    payload = {
        "name": spec.name,
        "command": spec.command,
        "args": list(spec.args),
        "env": dict(sorted(spec.env.items())),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def record_install(agent: str, spec: ServerSpec, *, path: Path | None = None) -> None:
    """Record that Headroom installed ``spec`` for ``agent``.

    Serialized: the ledger is on the SHARED root, so concurrent isolated wraps
    registering different servers each read the same pre-image and the later
    write drops the other's entry — after which `unwrap` cannot prove Headroom
    owns the lost one and leaves it installed for good.
    """
    ledger_file = path or ledger_path()
    with _filelock.exclusive(_filelock.lock_path_for(ledger_file)):
        data = _read_ledger(ledger_file)
        agents = data.setdefault("agents", {})
        agent_entry = agents.setdefault(agent, {})
        agent_entry[spec.name] = {
            "fingerprint": spec_fingerprint(spec),
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_ledger(ledger_file, data)


def clear_install(agent: str, server_name: str, *, path: Path | None = None) -> None:
    """Remove one ledger entry if present.

    Serialized for the same reason as :func:`record_install` — an unlocked
    removal can also carry a peer's just-recorded entry away with it.
    """
    ledger_file = path or ledger_path()
    with _filelock.exclusive(_filelock.lock_path_for(ledger_file)):
        data = _read_ledger(ledger_file)
        agents = data.get("agents")
        if not isinstance(agents, dict):
            return
        agent_entry = agents.get(agent)
        if not isinstance(agent_entry, dict) or server_name not in agent_entry:
            return
        del agent_entry[server_name]
        if not agent_entry:
            del agents[agent]
        if not agents:
            data.pop("agents", None)
        _write_ledger(ledger_file, data)


def headroom_installed_matching(
    agent: str,
    current_spec: ServerSpec | None,
    *,
    path: Path | None = None,
) -> bool:
    """Return True when the ledger says Headroom installed ``current_spec``."""
    if current_spec is None:
        return False
    ledger_file = path or ledger_path()
    data = _read_ledger(ledger_file)
    try:
        entry = data["agents"][agent][current_spec.name]
    except (KeyError, TypeError):
        return False
    if not isinstance(entry, dict):
        return False
    return entry.get("fingerprint") == spec_fingerprint(current_spec)


def _read_ledger(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_ledger(path: Path, data: dict[str, Any]) -> None:
    """Publish the ledger atomically.

    A reader that catches a partial write parses nothing and concludes Headroom
    owns none of the entries, which is how a user-visible MCP server survives
    an `unwrap` that should have removed it. The temp name carries this
    writer's pid so two publishers cannot break each other's replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

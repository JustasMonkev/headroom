"""Ownership handoff for a durable, single-slot agent config file.

Several agents resolve their endpoint from ONE user-scoped file that every
process of theirs re-reads — omp's ``models.yml``, Grok Build's
``config.toml``. Writing a wrap's port there is a durable contract and the
right one for ``--shared``: that port outlives the session, so the persisted
override stays true.

It is wrong for an isolated run, whose dedicated proxy port dies with the run.
Two failures follow, and this module exists for both:

* the port outlives nothing — every later process of that agent, and any
  concurrent run, is left pointing at a dead endpoint;
* a concurrent run's exit would restore the pre-wrap file out from under a
  peer that is still serving.

So a run HOLDS the override and RELEASES it on exit: the release hands the
file to a still-live peer when there is one, and otherwise restores the
pre-wrap state. Shared sessions register as owners too — they are a valid
handover target — but keep their durable contract and never release.

Every mutation is serialized on the config file, because the owner list and
the file itself must not be read and rewritten by two runs at once.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from headroom import _filelock
from headroom._subprocess import identity_mismatch, pid_alive, proc_identity

OWNERS_SUFFIX = ".headroom-owners.json"

# ``(port, proxy_pid, proxy_instance) -> bool``: is that listener still serving?
ProxyProbe = Callable[[int, "int | None", "str | None"], bool]

# ``(port, project) -> Any``: write the override for this port.
ApplyOverride = Callable[[int, "str | None"], Any]

# ``() -> str``: undo the override, returning a status string.
RestoreOverride = Callable[[], Any]


def owners_path(config_file: Path) -> Path:
    """Sidecar listing the wrap sessions holding ``config_file``."""
    return config_file.with_name(config_file.name + OWNERS_SUFFIX)


def read_owners(config_file: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(owners_path(config_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return [o for o in raw if isinstance(o, dict) and isinstance(o.get("pid"), int)]


def write_owners(config_file: Path, owners: list[dict[str, Any]]) -> None:
    path = owners_path(config_file)
    if not owners:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(owners), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def self_owner(
    port: int,
    project: str | None = None,
    proxy_pid: int | None = None,
    proxy_instance: str | None = None,
) -> dict[str, Any]:
    ident = proc_identity(os.getpid())
    return {
        "pid": os.getpid(),
        "port": port,
        # Identity of the process LISTENING on `port` — a different process
        # from `pid`, and the one that has to be alive for a handover to this
        # owner to mean anything. The instance survives uvicorn workers; the
        # pid is kept for records written before it existed.
        "proxy_pid": proxy_pid,
        "proxy_instance": proxy_instance,
        "project": project,
        "start_src": ident[0] if ident else None,
        "start_time": ident[1] if ident else None,
    }


def owner_is_live(owner: dict[str, Any], proxy_probe: ProxyProbe | None = None) -> bool:
    """Whether ``owner`` can still serve the override we would hand it.

    The wrapper being alive is NOT enough: it stays blocked on its agent long
    after a detached proxy dies, and handing the file to that session points
    every process of the agent at a dead port.
    """
    pid = owner.get("pid")
    if not isinstance(pid, int) or not pid_alive(pid):
        return False
    if identity_mismatch(owner.get("start_src"), owner.get("start_time"), pid):
        return False
    if proxy_probe is None:
        return True
    port = owner.get("port")
    if not isinstance(port, int):
        return False
    proxy_pid = owner.get("proxy_pid")
    instance = owner.get("proxy_instance")
    return proxy_probe(
        port,
        proxy_pid if isinstance(proxy_pid, int) else None,
        instance if isinstance(instance, str) and instance else None,
    )


def hold(
    config_file: Path,
    *,
    apply: ApplyOverride,
    port: int,
    project: str | None = None,
    proxy_pid: int | None = None,
    proxy_instance: str | None = None,
) -> Any:
    """Write the override and register this session as an owner."""
    with _filelock.exclusive(_filelock.lock_path_for(config_file)):
        result = apply(port, project)
        others = [o for o in read_owners(config_file) if o.get("pid") != os.getpid()]
        write_owners(config_file, [*others, self_owner(port, project, proxy_pid, proxy_instance)])
    return result


def release(
    config_file: Path,
    *,
    apply: ApplyOverride,
    restore: RestoreOverride,
    proxy_probe: ProxyProbe | None = None,
) -> str:
    """Drop this session's hold, handing the file to a live peer if any.

    Returns ``"handover"`` when a surviving owner's port was written back,
    otherwise whatever ``restore`` reports. Restoring unconditionally would
    strip a concurrent run's routing; doing nothing would strand every later
    process of that agent on a dead port.
    """
    with _filelock.exclusive(_filelock.lock_path_for(config_file)):
        survivors = [
            o
            for o in read_owners(config_file)
            if o.get("pid") != os.getpid() and owner_is_live(o, proxy_probe)
        ]
        write_owners(config_file, survivors)
        if survivors:
            newest = survivors[-1]
            apply(int(newest["port"]), newest.get("project"))
            return "handover"
        return str(restore())


__all__ = [
    "OWNERS_SUFFIX",
    "ProxyProbe",
    "hold",
    "owner_is_live",
    "owners_path",
    "read_owners",
    "release",
    "self_owner",
    "write_owners",
]

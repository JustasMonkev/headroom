"""Per-run isolation for concurrent Headroom sessions.

By default every ``headroom wrap``/``headroom proxy`` run shares one
workspace (``~/.headroom`` — savings ledger, memory DB, TOIN telemetry,
logs, caches) and reuses any proxy already listening on the requested
port. Running two coding agents at once therefore routes both through the
same proxy instance and interleaves their state.

Isolated mode (``headroom wrap --isolated <tool>`` or
``HEADROOM_ISOLATED=1``) gives the current run:

* its own workspace under ``<workspace>/runs/run-<timestamp>-<pid>-<rand>``
  (exported as ``HEADROOM_WORKSPACE_DIR`` so the proxy subprocess, the
  wrapped agent, and any MCP children all inherit it), and
* a dedicated proxy instance — an already-running proxy is never reused,
  and the base port is left to the shared proxy (see
  ``_ensure_proxy`` in :mod:`headroom.cli.wrap`).

The read-mostly config root intentionally stays shared: before the
workspace override is applied, the resolved config dir is pinned via
``HEADROOM_CONFIG_DIR`` so model catalogs and plugin settings keep coming
from the user's existing configuration instead of an empty per-run copy.

Note that isolation severs cross-agent memory by design — each isolated
run gets its own ``memory.db``.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from headroom import paths

HEADROOM_ISOLATED_ENV = "HEADROOM_ISOLATED"

# Set by activate_isolated_workspace() so re-entrant activation (e.g. a wrap
# subcommand that shells back into the CLI) reuses the same run directory
# instead of nesting a second one.
HEADROOM_ISOLATED_WORKSPACE_ENV = "HEADROOM_ISOLATED_WORKSPACE"

_RUNS_DIR = "runs"


def isolation_requested() -> bool:
    """True when this process should run with per-run isolation."""

    value = os.environ.get(HEADROOM_ISOLATED_ENV, "").strip().lower()
    return value in ("1", "true", "yes", "on")


def active_isolated_workspace() -> Path | None:
    """Return the already-activated per-run workspace, if any."""

    value = os.environ.get(HEADROOM_ISOLATED_WORKSPACE_ENV, "").strip()
    return Path(value) if value else None


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def activate_isolated_workspace(run_id: str | None = None) -> Path:
    """Create a unique workspace for this run and point Headroom at it.

    Idempotent per process tree: if a previous activation already exported
    ``HEADROOM_ISOLATED_WORKSPACE``, that directory is returned unchanged.

    Environment mutations (all inherited by subprocesses):

    * ``HEADROOM_CONFIG_DIR`` — pinned to the pre-activation config dir so
      read-mostly configuration stays shared (only set if not already set).
    * ``HEADROOM_WORKSPACE_DIR`` — the fresh per-run directory.
    * ``HEADROOM_ISOLATED`` — set to ``1`` so downstream code (and nested
      CLI invocations) can detect isolated mode.
    * ``HEADROOM_ISOLATED_WORKSPACE`` — re-activation marker.
    """

    existing = active_isolated_workspace()
    if existing is not None:
        return existing

    # Pin the config root BEFORE relocating the workspace: config_dir()
    # derives from HEADROOM_WORKSPACE_DIR when HEADROOM_CONFIG_DIR is unset,
    # and the per-run workspace must not shadow the user's real config.
    os.environ.setdefault(paths.HEADROOM_CONFIG_DIR_ENV, str(paths.config_dir()))

    run_dir = paths.workspace_dir() / _RUNS_DIR / f"run-{run_id or _new_run_id()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ[paths.HEADROOM_WORKSPACE_DIR_ENV] = str(run_dir)
    os.environ[HEADROOM_ISOLATED_ENV] = "1"
    os.environ[HEADROOM_ISOLATED_WORKSPACE_ENV] = str(run_dir)
    return run_dir


__all__ = [
    "HEADROOM_ISOLATED_ENV",
    "HEADROOM_ISOLATED_WORKSPACE_ENV",
    "isolation_requested",
    "active_isolated_workspace",
    "activate_isolated_workspace",
]

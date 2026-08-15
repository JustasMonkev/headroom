"""Per-run isolation for concurrent Headroom sessions.

Historically every ``headroom wrap``/``headroom proxy`` run shared one
workspace (``~/.headroom`` — savings ledger, memory DB, TOIN telemetry,
logs, caches) and reused any proxy already listening on the requested
port, so running two coding agents at once routed both through the same
proxy instance and interleaved their state.

Isolated mode is now the DEFAULT for ``headroom wrap``: opt back into the
shared proxy/workspace with ``headroom wrap --shared <tool>`` or
``HEADROOM_ISOLATED=0``. Isolation gives the current run:

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
import shutil
import time
import uuid
from pathlib import Path

from headroom import paths
from headroom._subprocess import pid_alive

HEADROOM_ISOLATED_ENV = "HEADROOM_ISOLATED"

# The proxy and the memory MCP resolve their SQLite DB from this env var
# (default: <cwd>/.headroom/memory.db — project-local, NOT workspace-relative).
# Relocating HEADROOM_WORKSPACE_DIR alone therefore does NOT isolate memory:
# two isolated runs in one project would share ./.headroom/memory.db. Isolation
# pins this into the run directory so each run gets its own database.
HEADROOM_MEMORY_DB_PATH_ENV = "HEADROOM_MEMORY_DB_PATH"

# Set by activate_isolated_workspace() so re-entrant activation (e.g. a wrap
# subcommand that shells back into the CLI) reuses the same run directory
# instead of nesting a second one.
HEADROOM_ISOLATED_WORKSPACE_ENV = "HEADROOM_ISOLATED_WORKSPACE"

# Comma-separated list of agent config-home env vars that isolation pointed at
# a per-run copy (CODEX_HOME / GROK_HOME / PI_CODING_AGENT_DIR — see
# `_isolate_agent_config_home` in headroom/cli/wrap.py). Recorded in the
# environment so it is inherited by children: a nested `wrap --shared` must
# hand the child back the user's SHARED agent config instead of letting it read
# — and rewrite — the still-live outer session's private endpoint config.
HEADROOM_ISOLATED_AGENT_HOMES_ENV = "HEADROOM_ISOLATED_AGENT_HOMES"

_RUNS_DIR = "runs"
_MEMORY_DB_FILE = "memory.db"

# Per-run workspaces are ephemeral; anything older than this is garbage
# collected on the next activation. Generous enough that a week-long
# session's proxy is never deleted out from under it.
_RUN_DIR_MAX_AGE_SECONDS = 7 * 24 * 3600


def _trimmed_env(name: str) -> str:
    """Environment value with surrounding whitespace stripped, or ``""``.

    Mirrors ``paths._env``: the filesystem contract treats a blank or
    whitespace-only override as *unset*, so isolation must use the same
    definition rather than mere key presence.
    """

    return os.environ.get(name, "").strip()


def _pin_env(name: str, value: str) -> None:
    """Set ``name`` unless it already holds a MEANINGFUL value.

    ``os.environ.setdefault`` is wrong here: it treats a present-but-blank
    variable as already configured and refuses to write. Every consumer reads
    blank as unset, so the pin would be skipped while the resource still
    resolved against the relocated (per-run) workspace — silently making
    config, Copilot auth, managed binaries, the MCP ledger, or the memory DB
    ephemeral.
    """

    if not _trimmed_env(name):
        os.environ[name] = value


def isolation_requested() -> bool:
    """True when this process should run with per-run isolation.

    Reads the explicit env state only (set by the ``wrap`` group callback or
    exported by the user); the "isolated unless ``--shared``" default lives
    in the CLI layer, which records its decision here for nested processes.
    """

    value = os.environ.get(HEADROOM_ISOLATED_ENV, "").strip().lower()
    return value in ("1", "true", "yes", "on")


def disable_isolation() -> None:
    """Record an explicit shared-mode choice (``--shared``) in the environment.

    Isolation is the CLI default, so opting out must be stated explicitly:
    nested Headroom invocations and ``_ensure_proxy`` consult
    :func:`isolation_requested` via the environment, and an unset variable
    would otherwise be re-defaulted to isolated by the next CLI layer.

    When ``--shared`` is invoked from an *already isolated* parent (a wrapped
    agent shelling back into the CLI), the process inherits the parent's
    per-run ``HEADROOM_WORKSPACE_DIR`` and ``HEADROOM_MEMORY_DB_PATH``. Left
    in place, the "shared" run would use the shared proxy while still reading
    and writing the parent's isolated state. Restore the workspace (and drop
    the isolated memory path) to the recorded shared root so ``--shared``
    genuinely means the legacy shared workspace.
    """

    os.environ[HEADROOM_ISOLATED_ENV] = "0"
    isolated_ws = os.environ.pop(HEADROOM_ISOLATED_WORKSPACE_ENV, None)

    # ONLY undo an actual isolation activation. A top-level user may legitimately
    # configure HEADROOM_WORKSPACE_DIR and HEADROOM_SHARED_WORKSPACE_DIR as two
    # distinct roots; collapsing the former onto the latter just because
    # `--shared` was passed would silently redirect all their workspace state.
    if not isolated_ws:
        return

    shared = _trimmed_env(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV)
    if shared:
        os.environ[paths.HEADROOM_WORKSPACE_DIR_ENV] = shared

    # Hand back the user's shared Codex/Grok/OMP config, so this invocation
    # does not read (and rewrite) the still-live outer session's private
    # per-run endpoint config.
    _release_isolated_agent_homes(isolated_ws)

    # Only clear the memory path if *isolation* set it (== <run_dir>/memory.db);
    # never discard a user-supplied HEADROOM_MEMORY_DB_PATH.
    isolated_db = str(Path(isolated_ws) / _MEMORY_DB_FILE)
    if os.environ.get(HEADROOM_MEMORY_DB_PATH_ENV) == isolated_db:
        os.environ.pop(HEADROOM_MEMORY_DB_PATH_ENV, None)


def record_isolated_agent_home(env_var: str) -> None:
    """Note that isolation pointed ``env_var`` at a per-run agent config home.

    Called by the wrap layer right after it assigns the variable, so
    :func:`disable_isolation` knows which agent homes to hand back when a
    nested invocation opts into shared mode.
    """

    recorded = _recorded_agent_homes()
    if env_var not in recorded:
        recorded.append(env_var)
    os.environ[HEADROOM_ISOLATED_AGENT_HOMES_ENV] = ",".join(recorded)


def _recorded_agent_homes() -> list[str]:
    return [v for v in _trimmed_env(HEADROOM_ISOLATED_AGENT_HOMES_ENV).split(",") if v]


def _release_isolated_agent_homes(isolated_ws: str) -> None:
    """Unset agent config homes that isolation pointed into ``isolated_ws``.

    Isolation only ever assigns one of these when it was previously UNSET (an
    explicit user value is always respected), so handing back the user's shared
    agent config means unsetting the variable. Guarded on the value still
    pointing inside the run directory being left, so a value the child set for
    itself is never clobbered.
    """

    try:
        run_root = Path(isolated_ws).resolve(strict=False)
    except OSError:
        return
    for env_var in _recorded_agent_homes():
        value = _trimmed_env(env_var)
        if not value:
            continue
        try:
            if Path(value).resolve(strict=False).is_relative_to(run_root):
                os.environ.pop(env_var, None)
        except (OSError, ValueError):
            continue
    os.environ.pop(HEADROOM_ISOLATED_AGENT_HOMES_ENV, None)


def active_isolated_workspace() -> Path | None:
    """Return the already-activated per-run workspace, if any."""

    value = os.environ.get(HEADROOM_ISOLATED_WORKSPACE_ENV, "").strip()
    return Path(value) if value else None


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def _run_dir_owner_pid(run_dir: Path) -> int | None:
    """Parse the launching PID out of a ``run-<ts>-<pid>-<rand>`` name.

    The timestamp itself contains a dash (``%Y%m%d-%H%M%S``), so the PID is
    the second-to-last dash-delimited field. Returns None when the name does
    not match the expected shape.
    """

    parts = run_dir.name.split("-")
    if len(parts) < 4 or parts[0] != "run":
        return None
    try:
        return int(parts[-2])
    except ValueError:
        return None


def _run_dir_last_activity(run_dir: Path) -> float:
    """Best-effort recency for a run dir: newest mtime seen while walking the
    tree (file writes don't bump ancestor directory mtimes, so a long-lived
    run whose only recent writes are nested log lines still reads as recent).
    """

    newest = run_dir.stat().st_mtime
    for dirpath, _dirnames, filenames in os.walk(run_dir):
        for name in (dirpath, *(os.path.join(dirpath, f) for f in filenames)):
            try:
                newest = max(newest, os.stat(name).st_mtime)
            except OSError:
                continue
    return newest


def prune_stale_runs(runs_root: Path, *, max_age_seconds: float = _RUN_DIR_MAX_AGE_SECONDS) -> None:
    """Best-effort GC of old per-run workspaces.

    Isolation is the default, so every wrap creates a run dir; without
    pruning, ``runs/`` grows without bound. A ``run-*`` directory is removed
    only when BOTH its last activity predates the cutoff AND the process that
    created it (the PID embedded in its name) is no longer alive — a paused
    but still-running long session must never have its proxy databases, logs,
    and MCP state deleted out from under it. Every failure is swallowed; GC
    must never break a launch.
    """

    cutoff = time.time() - max_age_seconds
    try:
        entries = list(runs_root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if not (entry.is_dir() and entry.name.startswith("run-")):
                continue
            owner_pid = _run_dir_owner_pid(entry)
            if owner_pid is not None and pid_alive(owner_pid):
                # A live owner still holds this run — never GC it, regardless
                # of how quiet its files have been.
                continue
            if _run_dir_last_activity(entry) < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


def activate_isolated_workspace(run_id: str | None = None) -> Path:
    """Create a unique workspace for this run and point Headroom at it.

    Idempotent per process tree: if a previous activation already exported
    ``HEADROOM_ISOLATED_WORKSPACE``, that directory is returned unchanged.

    Environment mutations (all inherited by subprocesses):

    * ``HEADROOM_CONFIG_DIR`` — pinned to the pre-activation config dir so
      read-mostly configuration stays shared (only set if not already set).
    * ``HEADROOM_SHARED_WORKSPACE_DIR`` — pinned to the pre-activation
      workspace so persistent cross-run resources (managed binaries, Copilot
      auth, MCP ledger, license cache) keep resolving there, and so
      ``--shared`` can restore it (only set if not already set).
    * ``HEADROOM_WORKSPACE_DIR`` — the fresh per-run directory.
    * ``HEADROOM_MEMORY_DB_PATH`` — the per-run memory DB, so ``--memory``
      genuinely isolates (only set if the user did not pin one).
    * ``HEADROOM_ISOLATED`` — set to ``1`` so downstream code (and nested
      CLI invocations) can detect isolated mode.
    * ``HEADROOM_ISOLATED_WORKSPACE`` — re-activation marker.
    """

    existing = active_isolated_workspace()
    if existing is not None:
        return existing

    # Pin config + shared roots BEFORE relocating the workspace: both derive
    # from HEADROOM_WORKSPACE_DIR when unset, and the per-run workspace must
    # not shadow the user's real config or persistent shared state.
    # `_pin_env`, not `setdefault` — a present-but-blank override reads as
    # unset everywhere else and must not block the pin.
    _pin_env(paths.HEADROOM_CONFIG_DIR_ENV, str(paths.config_dir()))
    _pin_env(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, str(paths.workspace_dir()))
    # The dashboard-managed settings file is a persistent user preference, not
    # run state: pin it to the shared root so an edit made from an isolated
    # run's dashboard is not written into a directory GC later deletes.
    _pin_env(paths.HEADROOM_SETTINGS_PATH_ENV, str(paths.settings_path()))

    runs_root = paths.workspace_dir() / _RUNS_DIR
    prune_stale_runs(runs_root)
    run_dir = runs_root / f"run-{run_id or _new_run_id()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ[paths.HEADROOM_WORKSPACE_DIR_ENV] = str(run_dir)
    # Route memory into the run dir unless the user pinned an explicit DB path.
    _pin_env(HEADROOM_MEMORY_DB_PATH_ENV, str(run_dir / _MEMORY_DB_FILE))
    os.environ[HEADROOM_ISOLATED_ENV] = "1"
    os.environ[HEADROOM_ISOLATED_WORKSPACE_ENV] = str(run_dir)
    return run_dir


__all__ = [
    "HEADROOM_ISOLATED_ENV",
    "HEADROOM_ISOLATED_WORKSPACE_ENV",
    "HEADROOM_ISOLATED_AGENT_HOMES_ENV",
    "HEADROOM_MEMORY_DB_PATH_ENV",
    "isolation_requested",
    "record_isolated_agent_home",
    "disable_isolation",
    "active_isolated_workspace",
    "activate_isolated_workspace",
    "prune_stale_runs",
]

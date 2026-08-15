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

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from headroom import paths
from headroom._subprocess import identity_mismatch, pid_alive, proc_identity

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

# The workspace root as it stood BEFORE activation. Usually identical to the
# shared root, but a user may configure HEADROOM_WORKSPACE_DIR and
# HEADROOM_SHARED_WORKSPACE_DIR as two distinct directories — in which case
# restoring `--shared` from the shared root would silently redirect ordinary
# workspace state into the persistent-resource bucket. Recorded separately so
# the opt-out returns to the exact directory isolation took over from.
HEADROOM_PREISOLATION_WORKSPACE_ENV = "HEADROOM_PREISOLATION_WORKSPACE"

_RUNS_DIR = "runs"
_MEMORY_DB_FILE = "memory.db"

# Sidecar recording the dedicated proxy this run started. `_start_proxy`
# launches the proxy DETACHED (``start_new_session`` / CREATE_BREAKAWAY), so it
# can outlive the wrapper whose PID is baked into the run directory name — and
# a proxy that has simply been idle is otherwise indistinguishable from an
# abandoned run. GC reads this file so a live proxy pins its own workspace.
_PROXY_STATE_FILE = ".proxy.json"

# Start identity of the wrapper that created the run dir. The directory name
# carries only its PID, and a PID is recycled freely long before the 7-day GC
# cutoff — without an identity to compare against, an unrelated long-lived
# process inheriting that number would pin the directory forever and let
# ``runs/`` grow without bound.
_OWNER_STATE_FILE = ".owner.json"

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


def _abs(path: Path) -> str:
    """``path`` as an absolute string, resolving symlinks when possible.

    Everything isolation exports is inherited by subprocesses that resolve
    relative values against THEIR working directory, and the wrapped agent
    routinely runs from elsewhere. A relatively configured
    ``HEADROOM_WORKSPACE_DIR`` must therefore be absolutized before export, or
    a nested Headroom command silently opens a different tree. Falls back to
    ``absolute()`` when the path cannot be resolved (missing parents, an
    unreadable link) — still absolute, just not symlink-collapsed.
    """

    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


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


def _pin_path_env(name: str, value: str) -> None:
    """:func:`_pin_env` for PATH variables, normalizing what it keeps.

    Preserving a user's existing override verbatim is right, but keeping it
    *relative* is not: the value is exported into a process tree whose members
    run from other directories, so a nested Headroom or MCP process would
    resolve `HEADROOM_CONFIG_DIR=config` (or shared root, settings, memory DB)
    beneath ITS cwd and use different state than the proxy. An existing
    override is therefore absolutized in place rather than left alone.
    """

    existing = _trimmed_env(name)
    if not existing:
        os.environ[name] = value
        return
    os.environ[name] = _abs(Path(existing).expanduser())


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

    # Restore the workspace isolation actually took over from. Fall back to the
    # shared root only for a run activated before this was recorded; the two
    # are identical unless the user configured them as distinct directories.
    restore_to = _trimmed_env(HEADROOM_PREISOLATION_WORKSPACE_ENV) or _trimmed_env(
        paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV
    )
    if restore_to:
        os.environ[paths.HEADROOM_WORKSPACE_DIR_ENV] = restore_to
    os.environ.pop(HEADROOM_PREISOLATION_WORKSPACE_ENV, None)

    # Only clear the memory path if *isolation* set it (== <run_dir>/memory.db);
    # never discard a user-supplied HEADROOM_MEMORY_DB_PATH.
    isolated_db = str(Path(isolated_ws) / _MEMORY_DB_FILE)
    if os.environ.get(HEADROOM_MEMORY_DB_PATH_ENV) == isolated_db:
        os.environ.pop(HEADROOM_MEMORY_DB_PATH_ENV, None)


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


def record_run_proxy(pid: int, port: int, *, run_dir: Path | None = None) -> None:
    """Record the dedicated proxy this isolated run started.

    Without this, GC's only liveness signal is the wrapper PID in the run
    directory's name. ``_start_proxy`` spawns the proxy detached
    (``start_new_session`` on POSIX, ``CREATE_BREAKAWAY_FROM_JOB`` on Windows),
    so it keeps serving after the wrapper exits — a wrapper killed without
    running its cleanup, or the proxy-only watcher flows the editor
    integrations (``wrap cursor``/``cline``/``continue``) use. A quiet-but-live
    proxy's workspace — its databases, caches and logs — could otherwise be
    deleted out from under it once the age cutoff passed.

    No-op outside isolated mode, and best-effort: never break a launch.
    """

    target = run_dir if run_dir is not None else active_isolated_workspace()
    if target is None:
        return
    try:
        target.mkdir(parents=True, exist_ok=True)
        (target / _PROXY_STATE_FILE).write_text(
            json.dumps({"pid": int(pid), "port": int(port), **_identity_fields(int(pid))}),
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError):
        pass


def _identity_fields(pid: int) -> dict[str, Any]:
    """``start_src``/``start_time`` fields for ``pid``, or ``{}`` when the
    platform cannot report a start time (mirrors the wrap marker's shape)."""

    ident = proc_identity(pid)
    if ident is None:
        return {}
    return {"start_src": ident[0], "start_time": ident[1]}


def _read_run_state(run_dir: Path, filename: str) -> dict[str, Any] | None:
    try:
        record = json.loads((run_dir / filename).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _owner_is_live(pid: int | None, record: dict[str, Any] | None) -> bool:
    """True when ``pid`` is alive AND is not a provably different process.

    ``identity_mismatch`` is conservative: a legacy record with no identity,
    or a platform that cannot report start times, yields False and we fall
    back to plain liveness — same behavior as before identities were recorded.
    """

    if pid is None or not pid_alive(pid):
        return False
    if record is None:
        return True
    return not identity_mismatch(record.get("start_src"), record.get("start_time"), pid)


def persistent_memory_db_path() -> Path:
    """The memory DB a PERSISTENT artifact should reference.

    ``headroom install apply --memory`` run from inside an isolated agent
    would otherwise bake ``<run>/memory.db`` into the deployment manifest and
    the supervised proxy's arguments: a database no top-level run shares and
    that run-dir GC deletes once the deployment has been quiet, even though
    the manifest itself lives on the shared root.

    Only the path ISOLATION chose is overridden. A user who pinned
    ``HEADROOM_MEMORY_DB_PATH`` themselves meant it, so it is returned as-is —
    the same test ``disable_isolation`` uses to decide what it may unset.
    """

    configured = _trimmed_env(HEADROOM_MEMORY_DB_PATH_ENV)
    isolated_ws = active_isolated_workspace()
    if isolated_ws is not None and configured == str(isolated_ws / _MEMORY_DB_FILE):
        return paths.shared_workspace_dir() / _MEMORY_DB_FILE
    if configured:
        return Path(configured).expanduser()
    return paths.shared_workspace_dir() / _MEMORY_DB_FILE


def restore_shared_memory_db() -> Path | None:
    """Undo the per-run memory pin, returning the shared DB it restored to.

    For a run that attaches to a proxy it did not start (``--no-proxy``), the
    isolated database is actively harmful: wrap-side sync and the agent's
    memory MCP would read and mutate the run DB while the reused proxy serves
    API-side retrieval from its own, presenting two conflicting memory views
    inside one session. Dropping the pin puts both back on the same store.

    Only the path ISOLATION chose is dropped — a user-supplied
    ``HEADROOM_MEMORY_DB_PATH`` is left alone, same test as
    :func:`disable_isolation`. Returns None when there was nothing to undo.
    """

    isolated_ws = active_isolated_workspace()
    if isolated_ws is None:
        return None
    if _trimmed_env(HEADROOM_MEMORY_DB_PATH_ENV) != str(isolated_ws / _MEMORY_DB_FILE):
        return None
    os.environ.pop(HEADROOM_MEMORY_DB_PATH_ENV, None)
    shared = paths.shared_workspace_dir() / _MEMORY_DB_FILE
    os.environ[HEADROOM_MEMORY_DB_PATH_ENV] = _abs(shared)
    return shared


def record_run_owner(run_dir: Path) -> None:
    """Stamp the creating wrapper's start identity into a fresh run dir.

    The PID alone lives in the directory name; this pins WHICH process that
    number referred to, so a recycled PID cannot keep the directory alive
    forever. Best-effort — GC degrades to plain liveness without it.
    """

    try:
        (run_dir / _OWNER_STATE_FILE).write_text(
            json.dumps({"pid": os.getpid(), **_identity_fields(os.getpid())}), encoding="utf-8"
        )
    except OSError:
        pass


def _run_dir_proxy_pid(run_dir: Path) -> int | None:
    """PID of the dedicated proxy recorded for ``run_dir``, if any."""

    record = _read_run_state(run_dir, _PROXY_STATE_FILE)
    if record is None:
        return None
    pid = record.get("pid")
    return pid if isinstance(pid, int) else None


def _run_dir_has_live_owner(run_dir: Path) -> bool:
    """True while any process still holds ``run_dir``.

    Two owners keep a run alive: the wrapper that created it (PID embedded in
    the directory name, start identity in ``.owner.json``) and the dedicated
    proxy it started (``.proxy.json``). Either one being alive is enough — but
    "alive" means the recorded process, not merely the recorded PID number, so
    an unrelated process that inherits a recycled PID cannot pin the directory
    forever.
    """

    if _owner_is_live(_run_dir_owner_pid(run_dir), _read_run_state(run_dir, _OWNER_STATE_FILE)):
        return True
    return _owner_is_live(_run_dir_proxy_pid(run_dir), _read_run_state(run_dir, _PROXY_STATE_FILE))


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
    only when BOTH its last activity predates the cutoff AND no owner is still
    alive — neither the wrapper that created it (the PID embedded in its name)
    nor the detached proxy it started (recorded in ``.proxy.json``). A paused
    but still-running long session, or a wrapper-less proxy still serving
    requests, must never have its databases, logs, and MCP state deleted out
    from under it. Every failure is swallowed; GC
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
            if _run_dir_has_live_owner(entry):
                # A live owner — the wrapper OR its detached proxy — still
                # holds this run; never GC it, regardless of how quiet its
                # files have been.
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
    # Every pin is ABSOLUTE. These are inherited by subprocesses, and the
    # wrapped agent routinely changes directory before spawning a nested
    # Headroom command — a relatively configured HEADROOM_WORKSPACE_DIR would
    # otherwise leave the child resolving the config root, the shared root and
    # the settings path beneath ITS cwd, quietly missing the intended model
    # config and redirecting managed binaries, the MCP ledger and marker locks
    # into a second tree.
    _pin_path_env(paths.HEADROOM_CONFIG_DIR_ENV, _abs(paths.config_dir()))
    _pin_path_env(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, _abs(paths.workspace_dir()))
    # The dashboard-managed settings file is a persistent user preference, not
    # run state: pin it to the shared root so an edit made from an isolated
    # run's dashboard is not written into a directory GC later deletes.
    _pin_path_env(paths.HEADROOM_SETTINGS_PATH_ENV, _abs(paths.settings_path()))
    # Record the exact workspace we are taking over, so `--shared` restores it
    # rather than assuming it equals the shared-resource root.
    os.environ[HEADROOM_PREISOLATION_WORKSPACE_ENV] = _abs(paths.workspace_dir())

    runs_root = paths.workspace_dir() / _RUNS_DIR
    prune_stale_runs(runs_root)
    run_dir = runs_root / f"run-{run_id or _new_run_id()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Resolve after mkdir so symlinks in the path collapse consistently for
    # every reader (see `_abs` for why absolute matters at all).
    run_dir = Path(_abs(run_dir))
    record_run_owner(run_dir)

    os.environ[paths.HEADROOM_WORKSPACE_DIR_ENV] = str(run_dir)
    # Route memory into the run dir unless the user pinned an explicit DB path.
    _pin_path_env(HEADROOM_MEMORY_DB_PATH_ENV, str(run_dir / _MEMORY_DB_FILE))
    os.environ[HEADROOM_ISOLATED_ENV] = "1"
    os.environ[HEADROOM_ISOLATED_WORKSPACE_ENV] = str(run_dir)
    return run_dir


__all__ = [
    "HEADROOM_ISOLATED_ENV",
    "HEADROOM_ISOLATED_WORKSPACE_ENV",
    "HEADROOM_PREISOLATION_WORKSPACE_ENV",
    "HEADROOM_MEMORY_DB_PATH_ENV",
    "isolation_requested",
    "disable_isolation",
    "active_isolated_workspace",
    "activate_isolated_workspace",
    "prune_stale_runs",
    "persistent_memory_db_path",
    "restore_shared_memory_db",
    "record_run_owner",
    "record_run_proxy",
]

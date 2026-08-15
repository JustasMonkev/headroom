"""Best-effort "is a newer Headroom released?" check.

This module is intentionally dependency-light (stdlib + ``packaging`` only —
``httpx`` lives in the ``[proxy]`` extra and must not be required by the base
CLI). It mirrors the telemetry beacon contract: opt-out, cached, fire-and-
forget, and it must never raise into a caller or block startup.

Two halves, deliberately split so a background thread never races stdout:

* :func:`maybe_check_async` performs the (rate-limited) network probe on a
  daemon thread and writes the result to a cache file. It prints nothing.
* :func:`format_update_notice` reads *only* the cache and returns a one-line
  notice string (or ``None``). Callers own the rendering.

Opt out with ``HEADROOM_UPDATE_CHECK=off``. Also skipped in ``--stateless``
mode, in CI, inside Docker, and from a git checkout (developers manage their
own tree).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PACKAGE_NAME = "headroom-ai"
_PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
_CACHE_FILE = "update_check.json"

# Probe PyPI at most once per day.
_CHECK_TTL_SECONDS = 86_400

# How long to wait for a peer's in-flight refresh before giving up and doing
# our own. Longer than `fetch_latest_version`'s own 4s timeout, so the common
# case is that the waiter outlasts the winner's request and reuses its result.
_REFRESH_LOCK_TIMEOUT_S = 6.0

# Back-off after a FAILED probe. A failure publishes a timestamp too, so the
# peers queued behind the lock reuse the attempt instead of each re-running
# `should_check()`, still seeing a stale cache, and hitting PyPI again — a
# fan-out would otherwise make one request per launch during exactly the
# outage when repeating them is least useful. Much shorter than the success
# TTL, since a transient failure should not cost a whole day of staleness.
_FAILED_CHECK_TTL_SECONDS = 900

_OFF_VALUES = frozenset(("off", "false", "0", "no", "disable", "disabled"))
_TRUE_VALUES = frozenset(("on", "true", "1", "yes", "enable", "enabled"))


def _env_off(name: str, default: str = "on") -> bool:
    """Return True when env var ``name`` is set to a falsey/off value."""
    return os.environ.get(name, default).strip().lower() in _OFF_VALUES


def _env_on(name: str) -> bool:
    """Return True when env var ``name`` is set to a truthy/on value."""
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def is_update_check_enabled() -> bool:
    """Whether the update check / banner should run at all.

    Disabled by ``HEADROOM_UPDATE_CHECK=off``, offline mode
    (``HEADROOM_OFFLINE``), stateless mode
    (``HEADROOM_STATELESS=true``/``1``/``yes``/``on``, matching the proxy's own
    parsing), or any CI environment (``CI`` set).
    """
    from headroom.offline import is_offline

    if is_offline():
        return False
    if _env_off("HEADROOM_UPDATE_CHECK"):
        return False
    if _env_on("HEADROOM_STATELESS"):
        return False
    if os.environ.get("CI", "").strip():
        return False
    return True


def _is_source_checkout() -> bool:
    """True when running from a git checkout (developers manage their tree)."""
    try:
        from headroom._version import _source_root

        return _source_root() is not None
    except Exception:
        return False


def _in_docker() -> bool:
    """Best-effort container detection — image rebuilds, not self-update."""
    try:
        return Path("/.dockerenv").exists() or bool(
            os.environ.get("HEADROOM_IN_DOCKER", "").strip()
        )
    except Exception:
        return False


def installed_version() -> str | None:
    """Return the *installed-distribution* version, or None.

    Deliberately uses ``importlib.metadata`` rather than
    ``headroom._version.get_version()`` — the latter computes a synthetic
    version from git history in a checkout, which would produce a meaningless
    comparison against PyPI.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(PACKAGE_NAME)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def _cache_path() -> Path:
    # Machine-wide rate-limit cache, resolved against the SHARED workspace:
    # under per-run isolation the worker finishes after the workspace has been
    # relocated, so a run-dir cache would leave the shared one permanently
    # stale and every wrap would re-hit PyPI.
    from headroom.paths import shared_workspace_dir

    return shared_workspace_dir() / _CACHE_FILE


def read_cache() -> dict[str, Any] | None:
    """Return the cached check result, or None if missing/unreadable."""
    try:
        path = _cache_path()
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_cache(latest_version: str | None, *, now: float | None = None) -> None:
    """Persist the latest-known version + check timestamp. Never raises.

    ``latest_version=None`` records a FAILED attempt: the timestamp is written
    so peers back off, and any previously known version is carried forward so a
    failure never erases a good answer.
    """
    try:
        # Must match _cache_path()'s root, which is the shared workspace.
        from headroom.paths import ensure_shared_workspace_dir

        ensure_shared_workspace_dir()
        stamp = now if now is not None else time.time()
        if latest_version is None:
            previous = read_cache() or {}
            known = previous.get("latest_version")
            payload = {
                "last_check": stamp,
                "latest_version": known if isinstance(known, str) else None,
                "last_failure": stamp,
            }
        else:
            payload = {"last_check": stamp, "latest_version": latest_version}
        path = _cache_path()
        # Unique per writer. A single shared `.json.tmp` lets concurrent
        # refreshes scribble over each other's partial writes and can make one
        # `replace` fail outright, leaving the cache absent — which sends the
        # NEXT launch fan-out straight back to PyPI.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(path)
    except Exception:
        logger.debug("update_check: failed to write cache", exc_info=True)


def _select_latest(data: dict[str, Any], *, allow_pre: bool) -> str | None:
    """Pick the newest non-yanked release from a PyPI JSON payload."""
    from packaging.version import InvalidVersion, Version

    releases = data.get("releases")
    candidates: list[Version] = []
    if isinstance(releases, dict):
        for ver_str, files in releases.items():
            # Skip releases whose every artifact is yanked.
            if (
                isinstance(files, list)
                and files
                and all(isinstance(f, dict) and f.get("yanked") for f in files)
            ):
                continue
            try:
                ver = Version(ver_str)
            except InvalidVersion:
                continue
            if ver.is_prerelease and not allow_pre:
                continue
            candidates.append(ver)
    if candidates:
        return str(max(candidates))

    # Fallback to info.version when releases is absent/empty.
    info = data.get("info")
    if isinstance(info, dict):
        ver_str = info.get("version")
        if isinstance(ver_str, str):
            try:
                ver = Version(ver_str)
            except InvalidVersion:
                return None
            if ver.is_prerelease and not allow_pre:
                return None
            return str(ver)
    return None


def fetch_latest_version(*, allow_pre: bool = False, timeout: float = 4.0) -> str | None:
    """Query the PyPI JSON API for the latest release. Returns None on any error.

    Uses ``urllib`` (stdlib) so the base CLI install needs no HTTP dependency.
    """
    try:
        req = urllib.request.Request(
            _PYPI_JSON_URL,
            headers={"Accept": "application/json", "User-Agent": "headroom-update-check"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https URL
            data = json.loads(resp.read().decode("utf-8"))
        return _select_latest(data, allow_pre=allow_pre)
    except Exception:
        logger.debug("update_check: PyPI fetch failed", exc_info=True)
        return None


def should_check(now: float | None = None) -> bool:
    """True when the cache is stale (older than the TTL) or absent."""
    cache = read_cache()
    if not cache:
        return True
    last = cache.get("last_check")
    if not isinstance(last, (int, float)):
        return True
    now = now if now is not None else time.time()
    # A failed attempt backs off for a short window rather than a whole day:
    # long enough that a fan-out of launches does not each retry through an
    # outage, short enough that a transient failure is not cached as if it
    # were an answer.
    ttl = _FAILED_CHECK_TTL_SECONDS if cache.get("last_failure") == last else _CHECK_TTL_SECONDS
    return (now - last) >= ttl


def run_check(*, allow_pre: bool = False, now: float | None = None) -> str | None:
    """Probe PyPI and update the cache. Returns the latest version or None.

    Synchronous — used directly by ``headroom update`` and indirectly by
    :func:`maybe_check_async`. Honors the enable gate.
    """
    if not is_update_check_enabled():
        return None
    # Serialize the whole refresh — staleness check, fetch, publish — across
    # processes. Isolation makes every wrap its own process, so once the TTL
    # expires a fan-out of launches all pass `should_check()` before any of
    # them writes and each independently hits PyPI. Holding the lock across
    # the fetch means the peers re-read the cache the winner just wrote and
    # skip their own request.
    #
    # Best-effort by design (see `_filelock`): if the lock cannot be taken the
    # refresh still runs, unserialized. A cached version number is never worth
    # failing a launch over.
    from headroom import _filelock

    cache = _cache_path()
    with _filelock.exclusive(_filelock.lock_path_for(cache), timeout=_REFRESH_LOCK_TIMEOUT_S):
        if now is None and not should_check():
            # A peer refreshed while we waited. An explicit `now` means a
            # caller (or a test) asked for this check specifically, so it is
            # not second-guessed.
            cached = read_cache() or {}
            latest_cached = cached.get("latest_version")
            return latest_cached if isinstance(latest_cached, str) else None
        latest = fetch_latest_version(allow_pre=allow_pre)
        # Record the FAILURE too, before releasing the lock. Otherwise every
        # peer queued behind us re-runs `should_check()`, still sees a stale
        # cache, and makes its own request.
        write_cache(latest, now=now)
        return latest


def maybe_check_async() -> threading.Thread | None:
    """Fire a rate-limited background check on a daemon thread.

    Returns the spawned thread (for tests) or None when the check is gated off,
    suppressed (checkout/Docker), or still within the TTL window. Never blocks
    and never raises.
    """
    try:
        if not is_update_check_enabled() or _is_source_checkout() or _in_docker():
            return None
        if not should_check():
            return None

        def _worker() -> None:
            try:
                run_check()
            except Exception:
                logger.debug("update_check: background check crashed", exc_info=True)

        thread = threading.Thread(target=_worker, name="headroom-update-check", daemon=True)
        thread.start()
        return thread
    except Exception:
        logger.debug("update_check: maybe_check_async crashed", exc_info=True)
        return None


def format_update_notice() -> str | None:
    """Return a one-line "update available" notice, or None.

    Reads only the cache (no network). Returns None when the check is disabled,
    in a checkout/Docker, when the installed version is unknown, or when already
    up to date.
    """
    try:
        if not is_update_check_enabled() or _is_source_checkout() or _in_docker():
            return None
        cache = read_cache()
        if not cache:
            return None
        latest = cache.get("latest_version")
        current = installed_version()
        if not isinstance(latest, str) or not current:
            return None

        from packaging.version import InvalidVersion, Version

        try:
            if Version(latest) <= Version(current):
                return None
        except InvalidVersion:
            return None

        # ASCII-only: some Windows consoles can't encode unicode and would raise
        # UnicodeEncodeError at the echo site, breaking a "best-effort" banner.
        return f"Update available: Headroom {latest} (you have {current}) - run: headroom update"
    except Exception:
        logger.debug("update_check: format_update_notice crashed", exc_info=True)
        return None


__all__ = [
    "PACKAGE_NAME",
    "fetch_latest_version",
    "format_update_notice",
    "installed_version",
    "is_update_check_enabled",
    "maybe_check_async",
    "read_cache",
    "run_check",
    "should_check",
    "write_cache",
]

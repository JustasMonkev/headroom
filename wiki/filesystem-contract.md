# Filesystem Contract

Headroom writes configuration, runtime state, logs, and caches to a small
set of well-known paths under the user's home directory. This page is the
source of truth for where those paths live, how to override them, and how
they behave inside Docker containers.

## Two-root model

| Variable | Default | Purpose | Typical access |
|---|---|---|---|
| `HEADROOM_CONFIG_DIR` | `~/.headroom/config` | User/admin-authored configuration (model catalogs, plugin settings, etc.) | Read-mostly |
| `HEADROOM_WORKSPACE_DIR` | `~/.headroom` | Runtime state written by the proxy and CLI (savings, logs, memory DB, telemetry, caches) | Read-write |
| `HEADROOM_SHARED_WORKSPACE_DIR` | `= HEADROOM_WORKSPACE_DIR` | Persistent, cross-run state that must never land in a throwaway per-run workspace (managed binaries, Copilot auth, MCP install ledger, license cache, learned verbosity profile, learned savings baseline, learn provenance sidecars) | Read-write |

`HEADROOM_SHARED_WORKSPACE_DIR` is normally **unset and identical to
`HEADROOM_WORKSPACE_DIR`**. Per-run isolation (`headroom wrap`, isolated by
default) relocates `HEADROOM_WORKSPACE_DIR` to
`~/.headroom/runs/run-<...>` and pins `HEADROOM_SHARED_WORKSPACE_DIR` to the
original `~/.headroom`, so a handful of persistent resources keep resolving
there while run-specific state moves into the run directory. See
[cli.md](cli.md) for the isolation model.

A run directory is garbage collected only when it has been quiet for more
than 7 days **and** none of its owners are alive. There are two owners: the
wrapper process (its PID is the second-to-last field of the
`run-<ts>-<pid>-<rand>` name) and the dedicated proxy that run started, whose
PID and port are recorded in `<run-dir>/.proxy.json`. The proxy is spawned
detached, so it routinely outlives its wrapper — without that record a live
but idle proxy could have its own workspace deleted underneath it.

Both records also carry a process *start identity*, not just a PID, because a
PID is recycled long before the 7-day cutoff and an unrelated long-lived
process inheriting one would otherwise pin the directory forever. A directory
written before identities were recorded (or on a platform that cannot report
start times) falls back to plain PID liveness.

`<shared-workspace>/locks/` holds the advisory lock files that serialize
concurrent updates to a project's `.claude/.headroom_wrap_marker.json` owner
stack, keyed by a digest of the resolved settings path. They live here rather
than in the project because nothing ever deletes a lock file, and they resolve
against the **shared** root so concurrent isolated runs contend on one file.

The per-run workspace is always exported as an absolute path, even when
`HEADROOM_WORKSPACE_DIR` is configured relatively — every exported value is
inherited by subprocesses that would otherwise resolve it against their own
working directory, so a nested Headroom command launched from elsewhere would
open a different workspace and `memory.db` than its proxy. The pinned config
and shared roots are absolutized for the same reason.

Persistent artifacts never reference a run directory. `headroom install apply
--memory` resolves its deployment manifest's memory database against the
shared root even when planned from inside an isolated run, so the supervised
proxy does not end up on a database that run-dir GC will delete.

All three variables are recognized by the Python proxy / CLI and the npm SDK.
They are **additive** — every pre-existing per-resource env var
(`HEADROOM_SAVINGS_PATH`, `HEADROOM_TOIN_PATH`,
`HEADROOM_SUBSCRIPTION_STATE_PATH`, `HEADROOM_MODEL_LIMITS`, ...)
continues to work with identical semantics.

## Precedence

For every per-resource helper, resolution follows this order:

```
explicit argument
    │ falls through when None/""
    ▼
per-resource env var (e.g. HEADROOM_SAVINGS_PATH)
    │ falls through when unset/blank
    ▼
derived from canonical root
    │ e.g. ${HEADROOM_WORKSPACE_DIR}/proxy_savings.json
    ▼
default (e.g. ~/.headroom/proxy_savings.json)
```

Examples:

- `HEADROOM_WORKSPACE_DIR=/mnt/state` → savings land at
  `/mnt/state/proxy_savings.json` unless `HEADROOM_SAVINGS_PATH` overrides.
- `HEADROOM_SAVINGS_PATH=/custom/savings.json` always wins, even when
  `HEADROOM_WORKSPACE_DIR` is set.
- Unset both and the default is `~/.headroom/proxy_savings.json`.

## Bucket assignments

### Workspace bucket (`HEADROOM_WORKSPACE_DIR`)

| Resource | Default path | Legacy env var |
|---|---|---|
| Proxy savings ledger | `${WORKSPACE_DIR}/proxy_savings.json` | `HEADROOM_SAVINGS_PATH` |
| TOIN telemetry JSON | `${WORKSPACE_DIR}/toin.json` | `HEADROOM_TOIN_PATH` |
| Subscription contribution state | `${WORKSPACE_DIR}/subscription_state.json` | `HEADROOM_SUBSCRIPTION_STATE_PATH` |
| Memory SQLite | `${WORKSPACE_DIR}/memory.db` | CLI `--memory-db-path`, env `HEADROOM_MEMORY_DB_PATH` |
| Native memory directory | `${WORKSPACE_DIR}/memories/` | `MemoryConfig.native_memory_dir` |
| Session stats JSONL | `${WORKSPACE_DIR}/session_stats.jsonl` | — |
| Memory sync state | `${WORKSPACE_DIR}/sync_state.json` | — |
| Memory bridge state | `${WORKSPACE_DIR}/bridge_state.json` | — |
| Proxy log directory | `${WORKSPACE_DIR}/logs/` | — |
| HTTP 400 debug dumps | `${WORKSPACE_DIR}/logs/debug_400/` | — |
| Deployment profiles | `${WORKSPACE_DIR}/deploy/` | — |
| Beacon lock file | `${WORKSPACE_DIR}/.beacon_lock_<port>` | — |

### Shared-workspace bucket (`HEADROOM_SHARED_WORKSPACE_DIR`)

Persistent, cross-run resources. Identical to the workspace bucket unless a
per-run isolated `headroom wrap` is active, in which case these stay on the
real `~/.headroom` while the workspace bucket above moves into the run dir.

| Resource | Default path | Legacy env var |
|---|---|---|
| Vendored `rtk` / `lean-ctx` binaries | `${SHARED_WORKSPACE_DIR}/bin/` | — |
| License cache | `${SHARED_WORKSPACE_DIR}/license_cache.json` | — |
| Copilot OAuth token | `${SHARED_WORKSPACE_DIR}/copilot_auth.json` | `HEADROOM_COPILOT_AUTH_FILE` |
| MCP install ledger | `${SHARED_WORKSPACE_DIR}/mcp_installs.json` | — |
| Proxy client markers | `${SHARED_WORKSPACE_DIR}/clients/<port>/` | — |
| Dashboard settings | `${SHARED_WORKSPACE_DIR}/settings.json` | `HEADROOM_SETTINGS_PATH` |
| Update-check cache | `${SHARED_WORKSPACE_DIR}/update_check.json` | — |
| Legacy models catalog (fallback) | `${SHARED_WORKSPACE_DIR}/models.json` | — |
| Account usage snapshot | `${SHARED_WORKSPACE_DIR}/subscription_snapshot.json` | — |
| Account poll lock | `${SHARED_WORKSPACE_DIR}/subscription_poll.lock` | — |
| Dashboard settings lock | `${SHARED_WORKSPACE_DIR}/settings.json.lock` | — |

Anything on the shared bucket is a multi-process path by definition, so a
read-modify-write on one needs an interprocess lock, not just an atomic
replace: `settings.json` saves take `settings.json.lock` across the whole
load-merge-write cycle, since an atomic replace prevents a torn file but not
a lost update. The account snapshot is only adopted when its recorded
`token_prefix` matches the polling token, so two proxies on different Claude
accounts never read each other's quota out of it. A proxy that loses the
election waits for the winner's snapshot rather than issuing its own request,
falling back to polling only if nothing is published — otherwise a cold start,
where every proxy finds the snapshot empty at the same moment, would make the
election decide nothing.

A persistent Docker deployment does not inherit these host paths: the
container mounts the host `~/.headroom` at `<container_home>/.headroom`, so
the workspace, config, shared root and settings path are all pinned to the
container's view of that mount instead of being passed through by name.

The subscription split follows the same rule as the savings baseline: the
usage windows describe an **account**, so one proxy polls
`/api/oauth/usage` under the shared lock and publishes the snapshot for the
others to adopt — otherwise a fan-out of N isolated agents on one OAuth
account makes N account-usage requests per interval. Each run's own
contribution counters stay in its private `subscription_state.json`, where
concurrent runs cannot overwrite each other's totals.

Proxy client markers reference-count a proxy instance identified by
`127.0.0.1:<port>`, which is machine-wide — every client of a given proxy
must register in the same directory, including an isolated `--no-proxy` run
attaching to the shared proxy. A dedicated proxy still gets its own
directory because the path is keyed by its distinct port.

### Config bucket (`HEADROOM_CONFIG_DIR`)

| Resource | Default path | Legacy env var |
|---|---|---|
| Models catalog | `${CONFIG_DIR}/models.json` | `HEADROOM_MODEL_LIMITS` (content override) |
| Plugin settings | `${CONFIG_DIR}/plugins/<name>/...` | — |

### Backward compatibility — models.json

`models.json` historically lived at `~/.headroom/models.json` (i.e. in the
workspace root, not in `config/`). For a seamless migration the Python
providers check **both** locations in this order:

1. `${HEADROOM_CONFIG_DIR}/models.json` (new canonical location)
2. `${HEADROOM_SHARED_WORKSPACE_DIR}/models.json` (legacy fallback — the
   *shared* root, so a per-run isolated wrap still sees a catalog kept at the
   legacy `~/.headroom/models.json` location)

Existing installs continue to work unchanged. New installs are encouraged
to put `models.json` in the config bucket.

## Plugin authors

Two helpers give plugins isolated, per-plugin directories under both
roots:

### Python

```python
from headroom import paths

cfg_dir = paths.plugin_config_dir("my-plugin")
# → ~/.headroom/config/plugins/my-plugin

state_dir = paths.plugin_workspace_dir("my-plugin")
# → ~/.headroom/plugins/my-plugin

cfg_dir.mkdir(parents=True, exist_ok=True)
(cfg_dir / "settings.json").write_text("{}")
```

### npm SDK

```typescript
import { pluginConfigDir, pluginWorkspaceDir } from "@headroom/sdk";

const cfgDir = pluginConfigDir("my-plugin");
const stateDir = pluginWorkspaceDir("my-plugin");
```

Plugin-author helpers reject names containing `/` or `\` to keep the
namespace flat.

## Docker naming overlap: `HEADROOM_WORKSPACE` vs `HEADROOM_WORKSPACE_DIR`

These are **two different variables** with different semantics, both
retained for backward compatibility:

| Variable | Scope | Meaning |
|---|---|---|
| `HEADROOM_WORKSPACE` | Host-side (Docker) | Directory to bind-mount into the container as `/workspace` (equivalent to CWD in native runs). Used by `docker-compose.native.yml`. |
| `HEADROOM_WORKSPACE_DIR` | Inside-the-container | Canonical Headroom state root. Resolves to `/tmp/headroom-home/.headroom` inside the official container image, which in turn bind-mounts to `${HOME}/.headroom` on the host. |

The official Docker bootstrap (compose file, `scripts/install.sh`, and the
Python `install` command) sets `HEADROOM_WORKSPACE_DIR` and
`HEADROOM_CONFIG_DIR` inside the container so the proxy resolves state to
the bind-mounted path without any user action.

## Project-scoped `.headroom/` directories

A few code paths deliberately use **project-local** `.headroom/` paths
resolved relative to the current working directory rather than the
canonical workspace root:

- `headroom/proxy/server.py` — project-scoped memory DB default
- `headroom/memory/mcp_server.py` — project-scoped memory DB default
- `headroom/cli/wrap.py` — project-scoped memory and hook artifacts

These **do not obey** `HEADROOM_WORKSPACE_DIR`. This is intentional: it
preserves the "project memory lives in the project directory" invariant
documented in [memory.md](memory.md). Users who want a single centrally
located memory store can pass `--memory-db-path <path>` explicitly or set
the path via the plugin API.

## Legacy per-resource env vars

Every legacy env var continues to work with its original semantics (raw
string in, raw string out — no tilde expansion, no path-separator
normalization), ensuring byte-for-byte backward compatibility.

Full list:

- `HEADROOM_SAVINGS_PATH`
- `HEADROOM_TOIN_PATH`
- `HEADROOM_SUBSCRIPTION_STATE_PATH`
- `HEADROOM_MODEL_LIMITS` (content override — JSON string or file path)

## See also

- [configuration.md](configuration.md) — general configuration reference
- [docker-install.md](docker-install.md) — Docker install details
- [persistent-installs.md](persistent-installs.md) — persistent
  deployment profiles
- [memory.md](memory.md) — memory-system paths and project scoping

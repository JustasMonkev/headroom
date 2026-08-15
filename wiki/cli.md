# CLI Reference

This page is the authoritative reference for the **Python Headroom CLI** exposed by the `headroom` console script.

## Global behavior

### Entry points

- Console script: `headroom`
- Python module entrypoint: `python -m headroom.cli`

### Global options

| Option | Scope | Meaning |
|---|---|---|
| `--help`, `-?` | root, groups, commands | Show help and exit |
| `--version`, `-v` | root only | Show the Headroom version and exit |

> `-v` is a **root-level version alias**. Inside subcommands such as `headroom wrap claude -v`, `-v` keeps its subcommand meaning (`--verbose`), not version.

## Command index

| Command | Purpose | Docker-native parity |
|---|---|---|
| `headroom install ...` | Install and manage persistent deployments | **python-native; Docker-native wrapper supports `persistent-docker` lifecycle subset** |
| `headroom proxy` | Run the Headroom proxy server | **native in container** |
| `headroom learn` | Learn from past tool-call failures | **native in container** |
| `headroom perf` | Summarize recent proxy performance | **native in container** |
| `headroom inspect` | Show original vs compressed content for recent requests | **native in container** |
| `headroom evals ...` | Run memory evaluation workflows | **native in container** |
| `headroom memory ...` | Inspect and manage stored memories | **native in container** |
| `headroom mcp ...` | Install, inspect, remove, or serve MCP integration | **native in container** |
| `headroom wrap claude` | Start proxy and launch Claude Code | **host-bridged** |
| `headroom wrap copilot` | Start proxy and launch GitHub Copilot CLI | **python-native only** |
| `headroom wrap codex` | Start proxy and launch Codex CLI | **host-bridged** |
| `headroom wrap aider` | Start proxy and launch Aider | **host-bridged** |
| `headroom wrap cursor` | Start proxy and print Cursor config guidance | **host-bridged** |
| `headroom wrap openclaw` | Install and configure the OpenClaw plugin | **host-bridged** |
| `headroom unwrap openclaw` | Disable the Headroom OpenClaw plugin | **host-bridged** |

## Captured `--help` output

The sections below capture the current top-level help output from the live CLI.

### `headroom --help`

```text
Usage: headroom [OPTIONS] COMMAND [ARGS]...

  Headroom - The Context Optimization Layer for LLM Applications.

  Manage memories, run the optimization proxy, and analyze metrics.

  Examples:
      headroom proxy              Start the optimization proxy
      headroom memory list        List stored memories
      headroom memory stats       Show memory statistics

Options:
  -v, --version  Show the version and exit.
  -?, --help     Show this message and exit.

Commands:
  evals   Memory evaluation commands.
  install Install and manage persistent Headroom deployments.
  learn   Learn from past tool call failures to prevent future ones.
  mcp     MCP server for Claude Code integration.
  memory  Manage memories stored in Headroom.
  perf    Analyze proxy performance from logs.
  proxy   Start the optimization proxy server.
  unwrap  Undo durable Headroom wrapping for supported tools.
  wrap    Wrap CLI tools to run through Headroom.
```

### Top-level command help snapshots

<details>
<summary><code>headroom proxy --help</code></summary>

```text
Usage: headroom proxy [OPTIONS]

  Start the optimization proxy server.

  Examples:
      headroom proxy                    Start proxy on port 8787
      headroom proxy --port 8080        Start proxy on port 8080
      headroom proxy --no-optimize      Passthrough mode (no optimization)

  Usage with Claude Code:
      ANTHROPIC_BASE_URL=http://localhost:8787 claude

  Usage with OpenAI-compatible clients:
      OPENAI_BASE_URL=http://localhost:8787/v1 your-app
```

</details>

<details>
<summary><code>headroom learn --help</code></summary>

```text
Usage: headroom learn [OPTIONS]

  Learn from past tool call failures to prevent future ones.
```

</details>

<details>
<summary><code>headroom perf --help</code></summary>

```text
Usage: headroom perf [OPTIONS]

  Analyze proxy performance from logs.
```

</details>

<details>
<summary><code>headroom evals --help</code></summary>

```text
Usage: headroom evals [OPTIONS] COMMAND [ARGS]...

  Memory evaluation commands.

Commands:
  memory     Run LoCoMo memory evaluation benchmark.
  memory-v2  Run LoCoMo V2 evaluation with LLM-controlled memory tools.
```

</details>

<details>
<summary><code>headroom memory --help</code></summary>

```text
Usage: headroom memory [OPTIONS] COMMAND [ARGS]...

  Manage memories stored in Headroom.

Commands:
  delete  Delete one or more memories by ID.
  edit    Edit a memory's content or importance.
  export  Export all memories to JSON.
  import  Import memories from a JSON file.
  list    List stored memories with optional filters.
  prune   Prune memories matching specified criteria.
  purge   Delete ALL memories from the database.
  show    Show full details of a single memory.
  stats   Show memory store statistics.
```

</details>

<details>
<summary><code>headroom mcp --help</code></summary>

```text
Usage: headroom mcp [OPTIONS] COMMAND [ARGS]...

  MCP server for Claude Code integration.

Commands:
  install    Install Headroom MCP server into Claude Code config.
  serve      Start the MCP server (called by Claude Code).
  status     Check Headroom MCP configuration status.
  uninstall  Remove Headroom MCP server from Claude Code config.
```

</details>

<details>
<summary><code>headroom install --help</code></summary>

```text
Usage: headroom install [OPTIONS] COMMAND [ARGS]...

  Install and manage persistent Headroom deployments.

Options:
  -?, --help  Show this message and exit.

Commands:
  apply    Install a persistent Headroom deployment.
  remove   Remove a persistent deployment and undo managed config.
  restart  Restart a persistent deployment.
  start    Start a persistent deployment.
  status   Show persistent deployment status.
  stop     Stop a persistent deployment.
```

</details>

<details>
<summary><code>headroom wrap --help</code></summary>

```text
Usage: headroom wrap [OPTIONS] COMMAND [ARGS]...

  Wrap CLI tools to run through Headroom.

Commands:
  aider     Launch aider through Headroom proxy.
  claude    Launch Claude Code through Headroom proxy.
  copilot   Launch GitHub Copilot CLI through Headroom proxy.
  codex     Launch OpenAI Codex CLI through Headroom proxy.
  cursor    Start Headroom proxy for use with Cursor.
  openclaw  Install and configure Headroom OpenClaw plugin in one command.
```

</details>

<details>
<summary><code>headroom unwrap --help</code></summary>

```text
Usage: headroom unwrap [OPTIONS] COMMAND [ARGS]...

  Undo durable Headroom wrapping for supported tools.

Commands:
  openclaw  Disable the Headroom OpenClaw plugin and restore the legacy engine slot.
```

</details>

## `headroom proxy`

Start the optimization proxy server.

```bash
headroom proxy
headroom proxy --port 8787
headroom proxy --mode cache
```

| Option | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Host interface to bind |
| `--port`, `-p` | `8787` | Port to bind |
| `--mode` | runtime default | Optimization mode: `token`, `cache`, `token_mode`, `cache_mode`, `token_savings`, `cost_savings`, `token_headroom` |
| `--no-optimize` | off | Disable optimization and operate in passthrough mode |
| `--no-cache` | off | Disable semantic caching |
| `--no-rate-limit` | off | Disable rate limiting |
| `--retry-max-attempts` | runtime default `3` | Maximum upstream retry attempts |
| `--request-timeout-seconds` | runtime default `300` | Request timeout in seconds |
| `--connect-timeout-seconds` | runtime default `10` | Upstream connection timeout |
| `--anthropic-pre-upstream-concurrency` | auto `max(2, min(8, cpu_count))` | Cap simultaneous pre-upstream work on `/v1/messages` (body read, deep copy, first compression stage, memory-context lookup, upstream connect). `0` or negative disables (unbounded); any positive integer is honoured verbatim. Prevents cold-start replay storms from starving `/livez`, `/readyz`, and new Codex WS opens. |
| `--anthropic-pre-upstream-acquire-timeout-seconds` | `15.0` | Fail fast when the Anthropic pre-upstream queue is saturated. Requests that wait longer return `503` with `Retry-After` instead of parking indefinitely. |
| `--anthropic-pre-upstream-memory-context-timeout-seconds` | `2.0` | Fail-open timeout for Anthropic memory-context lookup while the request still holds a pre-upstream slot. |
| `--log-file` | unset | JSONL log output path |
| `--budget` | unset | Daily USD budget limit |
| `--no-code-aware` | off | Disable AST-aware code compression |
| `--code-aware` | off | Enable code-aware compression in the proxy (env: HEADROOM_CODE_AWARE_ENABLED) |
| `--no-read-lifecycle` | off | Disable stale/superseded read compression |
| `--no-ccr` | off | Disable CCR entirely — no retrieval markers in content and no injected `headroom_retrieve` tool (lossy, no recovery path) |
| `--no-ccr-proactive-expansion` | off | Disable proactive CCR context expansion |
| `--memory` | off | Enable persistent user memory |
| `--memory-db-path` | `""` | Override memory DB path (help text: `{cwd}/.headroom/memory.db`) |
| `--no-memory-tools` | off | Disable automatic memory tool injection |
| `--no-memory-context` | off | Disable automatic memory context injection |
| `--memory-top-k` | `10` | Number of memories to inject |
| `--learn` | off | Enable live traffic learning |
| `--no-learn` | off | Explicitly disable traffic learning |
| `--backend` | `anthropic` | Backend: `anthropic`, `bedrock`, `openrouter`, `anyllm`, or `litellm-*` |
| `--anyllm-provider` | `openai` | Provider name for `anyllm` |
| `--anthropic-api-url` | unset | Custom Anthropic passthrough API URL |
| `--openai-api-url` | unset | Custom OpenAI passthrough API URL |
| `--anthropic-extra-headers` | unset | JSON object of extra headers merged into (and overriding) forwarded Anthropic requests |
| `--openai-extra-headers` | unset | JSON object of extra headers merged into (and overriding) forwarded OpenAI requests |
| `--gemini-api-url` | unset | Custom Gemini passthrough API URL |
| `--region` | `us-west-2` | Cloud region for Bedrock / Vertex / related backends |
| `--bedrock-region` | unset | Deprecated Bedrock region override |
| `--bedrock-profile` | unset | AWS profile name for Bedrock |
| `--telemetry` | off | Opt in to anonymous usage telemetry (off by default) |
| `--no-telemetry` | off | Force anonymous usage telemetry off (already the default) |

Notes:

- `--learn` implies memory unless `--no-learn` is also set.
- Proxy startup can also read environment variables such as `HEADROOM_HOST`, `HEADROOM_PORT`, `HEADROOM_BUDGET`, `HEADROOM_MODE`, `HEADROOM_ANYLLM_PROVIDER`, `HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY`, `HEADROOM_ANTHROPIC_PRE_UPSTREAM_ACQUIRE_TIMEOUT_SECONDS`, `HEADROOM_REQUEST_TIMEOUT`, `HEADROOM_ANTHROPIC_PRE_UPSTREAM_MEMORY_CONTEXT_TIMEOUT_SECONDS`, `ANTHROPIC_TARGET_API_URL`, `OPENAI_TARGET_API_URL`, `GEMINI_TARGET_API_URL`, `ANTHROPIC_TARGET_API_HEADERS`, and `OPENAI_TARGET_API_HEADERS`. CLI flags take precedence over environment variables.
- The default Anthropic pre-upstream cap is intentionally conservative for CPU/ONNX-heavy work. Larger containers may want to raise it after checking the resolved runtime values on `/readyz` or `/debug/warmup`.

See also: [Proxy Server](proxy.md), [Configuration](configuration.md)

## `headroom learn`

Learn from past tool-call failures and produce agent guidance.

```bash
headroom learn
headroom learn --apply
headroom learn --agent codex --all
```

| Option | Default | Meaning |
|---|---|---|
| `--project` | current project resolution | Target project path |
| `--all` | off | Analyze all discovered projects |
| `--apply` | off | Write recommendations instead of dry-run output |
| `--agent` | `auto` | Agent source: `auto`, built-ins (`claude`, `codex`, `gemini`), or plugin-provided names |
| `--model` | auto-detect | LLM model used for analysis |

Notes:

- `--agent auto` scans all detected agent data sources.
- If `--project` is omitted, Headroom resolves from the current directory upward.
- External agent integrations register through the `headroom.learn_plugin` entry point.

See also: [Failure Learning](learn.md)

## `headroom perf`

Summarize recent proxy performance from the local proxy log.

```bash
headroom perf
headroom perf --hours 24
headroom perf --raw
```

| Option | Default | Meaning |
|---|---|---|
| `--hours` | `168.0` | Time window in hours |
| `--raw` | off | Print raw PERF records instead of the summarized report |

The command reads `${HEADROOM_WORKSPACE_DIR}/logs/proxy.log` (defaults
to `~/.headroom/logs/proxy.log` — see the
[Filesystem Contract](filesystem-contract.md)).

## `headroom inspect`

Show the original vs compressed content for recent requests so you can *see*
what the compressor changed (not just the token counts). Useful for building
trust in compression and debugging quality regressions.

```bash
headroom inspect                 # inspect the most recent request
headroom inspect --last 5        # inspect the 5 most recent requests
headroom inspect --full          # include unchanged messages
headroom inspect --format json   # raw feed for piping into another tool
```

| Option | Default | Meaning |
|---|---|---|
| `--port` / `-p` | `8787` | Proxy port to query (env: `HEADROOM_PORT`) |
| `--last` | `1` | Number of most-recent requests to show |
| `--format` | `text` | `text` renders a highlighted diff; `json` emits the raw feed |
| `--full` | off | Include messages the compressor left unchanged |

`inspect` queries the running proxy's loopback `/transformations/feed` endpoint,
so the proxy must be started with `--log-messages` (or `--log-file`) for the
pre/post-compression snapshots to be captured.

## `headroom evals`

Memory evaluation command group.

### `headroom evals memory`

Run the LoCoMo memory evaluation benchmark.

```bash
headroom evals memory -n 3
headroom evals memory --answer-model gpt-4o --llm-judge
```

| Option | Default | Meaning |
|---|---|---|
| `--n-conversations`, `-n` | all available | Number of conversations to evaluate |
| `--categories` | benchmark default | Comma-separated categories |
| `--include-adversarial` | off | Include category 5 / unanswerable questions |
| `--top-k` | `10` | Memories retrieved per question |
| `--f1-threshold` | `0.5` | Threshold for correctness |
| `--answer-model` | unset | Model for answer generation |
| `--llm-judge` | off | Use LLM-as-judge scoring |
| `--judge-provider` | `litellm` | Judge provider: `openai`, `anthropic`, `litellm`, `simple` |
| `--judge-model` | `gpt-4o` | Judge model |
| `--output`, `-o` | unset | Save JSON results to a path |
| `--no-extract` | off | Disable LLM memory extraction |
| `--extraction-model` | `gpt-4o-mini` | Memory extraction model |
| `--pass-all` | off | Require all checks to pass |
| `--parallel` | `10` | Parallel worker count |
| `--debug` | off | Enable debug output |

### `headroom evals memory-v2`

Run the V2 memory evaluation flow with LLM-controlled tools.

```bash
headroom evals memory-v2
headroom evals memory-v2 --save-model gpt-4o-mini --llm-judge
```

| Option | Default | Meaning |
|---|---|---|
| `--n-conversations`, `-n` | all available | Number of conversations to evaluate |
| `--categories` | benchmark default | Comma-separated categories |
| `--include-adversarial` | off | Include adversarial questions |
| `--f1-threshold` | `0.5` | Threshold for correctness |
| `--save-model` | `gpt-4o-mini` | Model used when persisting memories |
| `--answer-model` | `gpt-4o` | Answer model |
| `--max-results` | `10` | Maximum tool results |
| `--no-graph` | off | Disable graph usage |
| `--llm-judge` | off | Use LLM-as-judge scoring |
| `--judge-model` | `gpt-4o` | Judge model |
| `--output`, `-o` | unset | Save JSON results |
| `--parallel` | `5` | Parallel worker count |
| `--debug` | off | Enable debug output |

Hidden compatibility shims exist for older command paths:

- `headroom memory-eval`
- `headroom memory-eval-v2`

These are intentionally omitted from normal usage docs.

## `headroom memory`

Memory management command group. This group is only registered when the optional memory dependencies import successfully.

### `headroom memory list`

```bash
headroom memory list
headroom memory list --scope USER --since 7d
headroom memory list -q "budget"
```

| Option | Default | Meaning |
|---|---|---|
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--limit`, `-n` | `50` | Maximum memories to show |
| `--session`, `-s` | unset | Filter by session ID |
| `--scope` | unset | `USER`, `SESSION`, `AGENT`, or `TURN` |
| `--since` | unset | Age filter using duration syntax such as `7d`, `2w`, `1m` |
| `--search`, `-q` | unset | Content search query |

### `headroom memory show <memory_id>`

```bash
headroom memory show 1234abcd
headroom memory show 1234abcd --json
```

| Argument / option | Default | Meaning |
|---|---|---|
| `memory_id` | required | Full or partial memory ID |
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--json` | off | Emit raw JSON |

### `headroom memory stats`

```bash
headroom memory stats
```

| Option | Default | Meaning |
|---|---|---|
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |

### `headroom memory edit <memory_id>`

```bash
headroom memory edit 1234abcd --content "Updated note"
headroom memory edit 1234abcd --importance 0.9
```

| Argument / option | Default | Meaning |
|---|---|---|
| `memory_id` | required | Full or partial memory ID |
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--content`, `-c` | unset | New memory content |
| `--importance`, `-i` | unset | New importance score (`0.0` to `1.0`) |

At least one of `--content` or `--importance` is required.

### `headroom memory delete <memory_ids...>`

```bash
headroom memory delete 1234abcd 5678efgh
headroom memory delete 1234abcd --force
```

| Argument / option | Default | Meaning |
|---|---|---|
| `memory_ids...` | required | One or more memory IDs |
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--force`, `-f` | off | Skip confirmation |

### `headroom memory prune`

```bash
headroom memory prune --older-than 30d --dry-run
headroom memory prune --scope SESSION --force
```

| Option | Default | Meaning |
|---|---|---|
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--older-than` | unset | Age threshold |
| `--scope` | unset | Scope filter: `USER`, `SESSION`, `AGENT`, `TURN` |
| `--low-importance` | unset | Importance cutoff |
| `--session`, `-s` | unset | Session ID filter |
| `--dry-run` | off | Show what would be removed |
| `--force`, `-f` | off | Skip confirmation |

At least one filter is required. Filters combine with **AND** semantics.

### `headroom memory purge`

```bash
headroom memory purge --confirm
```

| Option | Default | Meaning |
|---|---|---|
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--confirm` | off | Required confirmation flag |

### `headroom memory export`

```bash
headroom memory export
headroom memory export --output export.json
```

| Option | Default | Meaning |
|---|---|---|
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--output`, `-o` | stdout | Output path |

### `headroom memory import <file>`

```bash
headroom memory import export.json
headroom memory import export.json --force
```

| Argument / option | Default | Meaning |
|---|---|---|
| `file` | required | JSON file containing exported memories |
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--force`, `-f` | off | Skip confirmation |

The import expects a JSON array. Malformed entries are skipped.

## `headroom mcp`

Manage the Headroom MCP server integration.

### `headroom mcp install`

```bash
headroom mcp install
headroom mcp install --proxy-url http://127.0.0.1:9000
```

| Option | Default | Meaning |
|---|---|---|
| `--proxy-url` | `http://127.0.0.1:8787` | Proxy URL written into MCP config |
| `--force` | off | Overwrite an existing Headroom MCP config |

### `headroom mcp uninstall`

```bash
headroom mcp uninstall
```

This removes the Headroom MCP server entry from the Claude configuration.

### `headroom mcp status`

```bash
headroom mcp status
```

This inspects MCP SDK availability, Claude config state, and proxy reachability.

### `headroom mcp serve`

```bash
headroom mcp serve
headroom mcp serve --proxy-url http://127.0.0.1:9000 --debug
```

| Option | Default | Meaning |
|---|---|---|
| `--proxy-url` | `http://127.0.0.1:8787` | Proxy URL (also reads `HEADROOM_PROXY_URL`) |
| `--direct` | off | Disable stdio transport wrapping |
| `--debug` | off | Enable debug logging |

`serve` is part of the public CLI, but it is usually consumed by MCP host tooling rather than by humans directly.

See also: [MCP Tools](mcp.md)

## `headroom install`

Install and manage persistent local Headroom deployments.

### `headroom install apply --help`

```text
Usage: headroom install apply [OPTIONS]

  Install a persistent Headroom deployment.

Options:
  --preset [persistent-service|persistent-task|persistent-docker]
                                  Persistent runtime preset to install.
                                  [default: persistent-service]
  --runtime [python|docker]       Runtime used to execute Headroom for
                                  service/task modes.  [default: python]
  --scope [provider|user|system]  Where to apply persistent configuration.
                                  [default: user]
  --providers [auto|all|manual]   Target selection mode for direct tool
                                  configuration.  [default: auto]
  --target [claude|copilot|codex|aider|cursor|openclaw]
                                  Tool target to configure when --providers
                                  manual is used.
  --profile TEXT                  Deployment profile name.  [default: default]
  -p, --port INTEGER              Persistent proxy port.  [default: 8787]
  --backend TEXT                  Proxy backend for the persistent runtime.
                                  [default: anthropic]
  --anyllm-provider TEXT          Provider for any-llm backends when --backend
                                  anyllm is used.
  --region TEXT                   Cloud region for Bedrock / Vertex style
                                  backends.
  --mode TEXT                     Proxy optimization mode.  [default: token]
  --memory                        Enable persistent memory in the proxy runtime.
  --telemetry                     Opt in to anonymous telemetry in the runtime
                                  (off by default).
  --no-telemetry                  Force anonymous telemetry off in the runtime
                                  (already the default).
  --image TEXT                    Docker image to use when runtime=docker or
                                  preset=persistent-docker.  [default:
                                  ghcr.io/chopratejas/headroom:latest]
  -?, --help                      Show this message and exit.
```

### `headroom install apply`

```bash
headroom install apply --preset persistent-service --providers auto
headroom install apply --preset persistent-task --providers manual --target claude --target codex
headroom install apply --preset persistent-docker --scope user
```

| Option | Default | Meaning |
|---|---|---|
| `--preset` | `persistent-service` | Lifecycle preset: `persistent-service`, `persistent-task`, or `persistent-docker` |
| `--runtime` | `python` | Runtime used for service/task installs: `python` or `docker` |
| `--scope` | `user` | Config scope: `provider`, `user`, or `system` |
| `--providers` | `auto` | Target selection mode: `auto`, `all`, or `manual` |
| `--target` | repeatable | Tool target used with `--providers manual` |
| `--profile` | `default` | Deployment profile name |
| `--port`, `-p` | `8787` | Persistent proxy port |
| `--backend` | `anthropic` | Backend for the managed runtime |
| `--anyllm-provider` | unset | Provider name used with `--backend anyllm` |
| `--region` | unset | Cloud region override |
| `--mode` | `token` | Proxy optimization mode |
| `--memory` | off | Enable persistent memory in the managed runtime |
| `--telemetry` | off | Opt in to anonymous telemetry (off by default) |
| `--no-telemetry` | off | Force anonymous telemetry off (already the default) |
| `--image` | `ghcr.io/chopratejas/headroom:latest` | Docker image for Docker-backed installs |

`apply` stores a manifest under
`${HEADROOM_WORKSPACE_DIR}/deploy/<profile>/manifest.json` (default
`~/.headroom/deploy/<profile>/manifest.json`), applies managed tool
configuration, starts the chosen runtime, and waits for `readyz`.

Docker-native host wrappers expose a narrower `headroom install` subset for `persistent-docker` only: `apply`, `status`, `start`, `stop`, `restart`, and `remove`. Those wrapper flows preserve the same port and manifest behavior, but they intentionally reject `persistent-service`, `persistent-task`, and provider mutation flags like `--scope`, `--providers`, and `--target`.

### `headroom install status`

```bash
headroom install status
headroom install status --profile default
```

Shows the stored profile, preset, runtime, supervisor kind, scope, port, runtime status, readiness, and backend from `/health`.

### `headroom install start`

```bash
headroom install start
headroom install start --profile default
```

Starts a previously installed deployment profile without reapplying mutations.

### `headroom install stop`

```bash
headroom install stop
```

Stops the managed runtime for an installed deployment profile.

### `headroom install restart`

```bash
headroom install restart
```

Stops and starts the selected deployment profile.

### `headroom install remove`

```bash
headroom install remove
```

Stops the runtime, removes installed supervisor artifacts, reverts managed configuration changes, and deletes the stored manifest.

See also: [Persistent Installs](persistent-installs.md)

## `headroom wrap`

Wrap external coding tools so their traffic flows through Headroom.

### Shared semantics

- `--port`, when available, defaults to `8787`
- `--no-proxy` skips proxy startup and assumes an existing proxy
- `--learn` enables live traffic learning
- `-v`, `--verbose` means **verbose output**
- Hidden `--prepare-only` exists for internal Docker-native bridge flows and is intentionally omitted from normal usage

### Concurrent runs: isolated by default, `--shared` to opt out

`headroom wrap` runs in **isolated mode by default**: every run is fully
independent, so two coding agents launched side by side never share a proxy
or state. `headroom wrap --shared <tool>` (a **group-level** flag — it goes
before the tool name; also honored as `HEADROOM_ISOLATED=0`) restores the
legacy behavior: the run reuses the proxy already listening on the shared
port, and reads/writes the shared workspace (`~/.headroom` — savings
ledger, memory DB, TOIN telemetry, logs). Shared mode is what enables
cross-agent memory and combined savings; `--isolated` exists as the
explicit spelling of the default.

In isolated mode, each run gets:

- a fresh per-run workspace created under
  `~/.headroom/runs/run-<timestamp>-<pid>-<rand>` and exported as
  `HEADROOM_WORKSPACE_DIR`, so the proxy, the wrapped agent, and any MCP
  children it spawns all inherit it. This carries the run-specific state:
  savings, TOIN telemetry, logs, and — via a per-run `HEADROOM_MEMORY_DB_PATH`
  — its own `memory.db`, so `--memory` truly isolates even for two runs in
  the same project. Run dirs are garbage-collected on a later launch once
  they are both older than 7 days **and** their creating process has exited
  (a paused long-running session is never pruned out from under itself);
- a dedicated proxy instance started on the first free port **above**
  `--port`. An already-running proxy is never reused, the base port is left
  reserved for the shared proxy, and the launcher confirms it actually owns
  the port it bound (retrying higher if a concurrent isolated launch wins
  the race), so a run never silently routes through another run's proxy;
- **persistent, cross-run state stays shared** (pinned to the real
  `~/.headroom` via `HEADROOM_SHARED_WORKSPACE_DIR`): the read-mostly config
  root, managed `rtk`/`lean-ctx` binaries, the cached license, Copilot auth
  saved by `headroom copilot-auth login`, and the MCP install ledger used by
  `headroom unwrap`. These would be wrong to duplicate into a throwaway run
  dir, so isolation deliberately keeps them on the shared root.

```bash
# Terminal 1 and 2 — fully independent proxies, savings, memory, logs:
headroom wrap claude
headroom wrap codex

# Legacy behavior: one proxy, one workspace, cross-agent memory:
headroom wrap --shared claude
headroom wrap --shared codex
```

For a tool that records the proxy port in its own on-disk config (Codex and
Grok register the `headroom` retrieve MCP server; OMP writes `models.yml`),
the dedicated port is reapplied after the proxy binds, so retrieval and
inference target the run's actual proxy rather than the reserved base port.
**Known limitation — concurrent runs of the *same* agent.** Some agents keep
their endpoint (and the Headroom MCP registration) in a single shared file
their own processes re-read: `~/.omp/agent/models.yml`,
`$CODEX_HOME/config.toml`, `~/.grok/config.toml`. Isolation deliberately
does **not** relocate those config homes, because they also hold durable
state — Codex keeps its transcripts and history under `$CODEX_HOME/sessions`,
so a per-run home would hide wrapped sessions from `codex resume` and let
run-directory GC delete them. Losing transcripts is worse than the race it
would prevent.

Retrieval is no longer affected by that. An isolated run registers a
**port-agnostic** `headroom` MCP entry — every concurrent run writes the
identical entry, so there is nothing left to overwrite in the shared file and
nothing pointing at a dead port after a run exits. The real port reaches the
server another way: it is a stdio server spawned as a child of the agent, and
`headroom mcp serve --proxy-url` reads `HEADROOM_PROXY_URL`, which the
launcher exports per run. Shared (`--shared`) runs keep the explicit pin,
since there is only one proxy to name.

What remains: config that records the port for **model routing** rather than
for MCP — OMP's `~/.omp/agent/models.yml` — still holds a single value, so
during overlapping runs the later launch wins there. An isolated `wrap omp`
does at least clean up after itself: it holds the override for the session and
releases it on exit, handing it to a still-live peer when one remains and
otherwise restoring the pre-wrap file, so a per-run port never outlives the run
that owned it. `wrap grok-build` does the same for `~/.grok/config.toml`.
A `--shared` run registers as an owner too — so an isolated peer hands routing
back rather than restoring over it — but keeps the original write-and-leave
contract, since its port is durable. Codex is unaffected (its endpoint is passed
per-process via `--config` overrides). To make two same-agent runs fully
independent including that, give them separate config homes yourself
(`CODEX_HOME=... headroom wrap codex`). `wrap claude` records the actual
bound port in its `.headroom_wrap_marker.json`, so the SessionStart self-heal
hook does not mistake a live run on 8788 for a dead one.

`--prepare-only` is exempt from isolation: it starts no proxy, and its
stdout is machine-readable (`scripts/install.sh` pipes it into `openclaw
config set --strict-json`). Isolation diagnostics go to stderr for the same
reason.

One project-local file cannot hold two different routes: if you run two
isolated `wrap claude` sessions **from the same directory**, both write
`.claude/settings.local.json`, and Claude re-reads it for each new
conversation. Headroom keeps this safe rather than silent — the second
launch prints a notice, never records the first run's URL as the project's
"original" value, and an exiting run leaves a still-live peer's routing
untouched. The marker holds a *stack* of live owners, and every
read-modify-write on it is serialized through an interprocess lock
(`.headroom_wrap_marker.lock`) — without it, two sessions starting at the
same instant read the same stack, each append themselves, and the later
write drops the other live owner, whose exit would then reset routing out
from under a peer that is still running. The lock is best-effort: if it
cannot be taken, the update still proceeds rather than blocking a launch.
The lock file itself lives under `<shared-workspace>/locks/`, keyed by a
digest of the resolved settings path — never in your project, since nothing
ever deletes a lock file. Exiting pops only that run's own `(pid, key)`
entry, so a concurrent run owning a different endpoint key
(`ANTHROPIC_VERTEX_BASE_URL`, the Foundry key) keeps its own record — that
holds for the crash-handover and marker-clearing paths too, and each run
inherits the project's original URL from the newest live owner of *its own*
key rather than whichever owner happens to be newest overall.
For genuinely independent Claude routing, run each agent from its own
working directory. (`headroom unwrap claude` still always wins.)

**Nested wraps do not chain proxies.** Running `headroom wrap claude` from
inside an already-wrapped session inherits the parent's
`ANTHROPIC_BASE_URL`, which points at the parent run's port — isolation
gives every run its own port, so it is never equal to the nested run's own
port. Headroom probes such loopback URLs and ignores any listener that
self-identifies as `headroom-proxy` on `/health`, so requests pass through
one pipeline rather than two. A genuine local gateway (LiteLLM and friends)
is still adopted as the upstream, per issue #1353. The same check covers
Foundry and Vertex mode, which route through their own endpoint variables
(`ANTHROPIC_FOUNDRY_BASE_URL`, `ANTHROPIC_VERTEX_BASE_URL`,
`VERTEX_TARGET_API_URL`) — a nested wrap inherits whichever one its parent
set, so all of them are probed the same way. The parent's *resolved upstream*
travels with it (`HEADROOM_PARENT_ANTHROPIC_UPSTREAM`), so a nested wrap keeps
routing through a user gateway the parent was configured with instead of
falling back to `api.anthropic.com`.

`wrap openclaw` is exempt from isolation alongside `selfheal`: it is a durable
installer that hands off to `openclaw gateway start`, whose `autoStart` spawns
a proxy of its own — that proxy belongs on shared state, not on a run
directory this wrapper never records and GC could later delete.

`--memory --no-proxy` reconciles memory with the proxy it reuses. That proxy
serves retrieval from its own store, so keeping this run pinned to an isolated
database would give the session two conflicting memory views. Headroom adopts
the database a persistent deployment recorded when there is one, and otherwise
simply drops the per-run pin so every consumer falls back to the same default
the proxy itself resolved (`{cwd}/.headroom/memory.db`) — substituting a
workspace path would invent a third store rather than reconcile. That
reconciliation happens **before** any memory setup, not just before the proxy
is contacted: the Claude and Codex flows both import native memories while
preparing the session, and settling the database afterwards would leave those
memories in the throwaway run database where neither the proxy nor the MCP
server would ever see them.

The exemptions above also apply when a wrap is nested inside an
already-isolated agent: `selfheal`, `openclaw` and `--prepare-only` return to
shared state rather than inheriting the parent's run directory.

The same applies to anything DURABLE an isolated agent starts. `headroom
install apply` can be run from inside a wrap, and the runtime it launches —
the detached agent, or the Docker container — outlives that wrap and is never
recorded as a run owner. Both are handed a de-isolated environment, so a
permanent deployment never lands on a run directory GC deletes after seven
quiet days, and the container keeps its own mounted workspace instead of
having it overridden by an inherited host path that does not exist inside it.

With `--no-proxy` an isolated run deliberately attaches to the shared
proxy; its client marker is registered in the shared workspace so the
wrapper that owns that proxy still sees it and will not shut the proxy down
underneath the run. Similarly, `wrap copilot` only adopts a running proxy's
backend when it will actually reuse that proxy.

Trade-offs of the isolated default: it severs cross-agent memory for the
run (each isolated run has its own `memory.db`), per-run savings are not
merged into the shared ledger, the isolated proxy shuts down with the run,
and a persistent deployment listening on the shared port is only picked up
by `--shared` runs. The hidden `wrap selfheal` maintenance subcommand is
exempt from isolation (it runs from a SessionStart hook and must not create
run dirs).

### `headroom wrap claude`

```bash
headroom wrap claude
headroom wrap claude --resume <session-id>
headroom wrap claude --port 9999
```

| Option / arg | Default | Meaning |
|---|---|---|
| `--port`, `-p` | `8787` | Proxy port |
| `--no-rtk` | off | Skip `rtk` installation and hook registration |
| `--no-proxy` | off | Reuse an existing proxy |
| `--learn` | off | Enable live traffic learning |
| `--verbose`, `-v` | off | Verbose output |
| `claude_args...` | passthrough | Additional Claude Code arguments |

Requires the `claude` binary on the host.

### `headroom wrap codex`

```bash
headroom wrap codex
headroom wrap codex -- "fix the bug"
headroom wrap codex --backend anyllm --anyllm-provider groq
```

| Option / arg | Default | Meaning |
|---|---|---|
| `--port`, `-p` | `8787` | Proxy port |
| `--no-rtk` | off | Skip `rtk` installation and `AGENTS.md` injection |
| `--no-proxy` | off | Reuse an existing proxy |
| `--learn` | off | Enable live traffic learning |
| `--backend` | unset | Proxy backend override |
| `--anyllm-provider` | unset | `anyllm` provider override |
| `--region` | unset | Cloud region override |
| `--verbose`, `-v` | off | Verbose output |
| `codex_args...` | passthrough | Additional Codex CLI arguments |

Requires the `codex` binary on the host.

### `headroom wrap copilot`

```bash
headroom wrap copilot -- --model claude-sonnet-4-20250514
headroom wrap copilot --backend anyllm --anyllm-provider groq -- --model gpt-4o
```

| Option / arg | Default | Meaning |
|---|---|---|
| `--port`, `-p` | `8787` | Proxy port |
| `--no-rtk` | off | Skip `rtk` installation and GitHub Copilot instructions injection |
| `--no-proxy` | off | Reuse an existing proxy |
| `--learn` | off | Enable live traffic learning |
| `--backend` | unset | Proxy backend override |
| `--anyllm-provider` | unset | `anyllm` provider override |
| `--region` | unset | Cloud region override |
| `--provider-type` | `auto` | Force Copilot BYOK provider type (`anthropic` or `openai`) |
| `--wire-api` | unset | OpenAI wire API override for OpenAI-style backends |
| `--verbose`, `-v` | off | Verbose output |
| `copilot_args...` | passthrough | Additional Copilot CLI arguments |

Requires the `copilot` binary on the host. When a matching persistent deployment exists on the requested port, `wrap copilot` reuses or recovers it before falling back to an ephemeral proxy.

### `headroom wrap aider`

```bash
headroom wrap aider
headroom wrap aider -- --model gpt-4o
headroom wrap aider --backend litellm-vertex --region us-central1
```

| Option / arg | Default | Meaning |
|---|---|---|
| `--port`, `-p` | `8787` | Proxy port |
| `--no-rtk` | off | Skip `rtk` installation and `CONVENTIONS.md` injection |
| `--no-proxy` | off | Reuse an existing proxy |
| `--learn` | off | Enable live traffic learning |
| `--backend` | unset | Proxy backend override |
| `--anyllm-provider` | unset | `anyllm` provider override |
| `--region` | unset | Cloud region override |
| `--verbose`, `-v` | off | Verbose output |
| `aider_args...` | passthrough | Additional Aider arguments |

Requires the `aider` binary on the host.

### `headroom wrap cursor`

```bash
headroom wrap cursor
headroom wrap cursor --port 9999
headroom wrap cursor --no-rtk
```

| Option | Default | Meaning |
|---|---|---|
| `--port`, `-p` | `8787` | Proxy port |
| `--no-rtk` | off | Skip `rtk` installation and `.cursorrules` injection |
| `--no-proxy` | off | Reuse an existing proxy |
| `--learn` | off | Enable live traffic learning |
| `--verbose`, `-v` | off | Verbose output |

This command prints Cursor configuration instructions and waits while the proxy stays up. It does **not** launch Cursor directly.

### `headroom wrap openclaw`

```bash
headroom wrap openclaw
headroom wrap openclaw --plugin-path ./plugins/openclaw
```

| Option | Default | Meaning |
|---|---|---|
| `--plugin-path` | unset | Local plugin source directory |
| `--plugin-spec` | `headroom-ai/openclaw` | NPM plugin spec |
| `--skip-build` | off | Skip local `npm install` / build steps |
| `--copy` | off | Copy plugin instead of linked install |
| `--proxy-port` | `8787` | Headroom proxy port |
| `--startup-timeout-ms` | `20000` | Proxy startup timeout |
| `--gateway-provider-id` | repeatable | OpenClaw provider IDs routed through Headroom |
| `--python-path` | unset | Python launcher override |
| `--no-auto-start` | off | Disable plugin auto-start behavior |
| `--no-restart` | off | Do not restart the OpenClaw gateway |
| `--verbose`, `-v` | off | Verbose output |

Requires the `openclaw` binary on the host, and local-source mode may also require `npm`. In Docker-native mode, the installed host wrapper drives the host `openclaw` CLI while the plugin auto-starts the host `headroom` wrapper from `PATH`.

## `headroom unwrap`

Undo durable wrapping for supported tools.

### `headroom unwrap openclaw`

```bash
headroom unwrap openclaw
headroom unwrap openclaw --no-restart
```

| Option | Default | Meaning |
|---|---|---|
| `--no-restart` | off | Do not restart the OpenClaw gateway |
| `--verbose`, `-v` | off | Verbose output |

This disables the Headroom OpenClaw plugin and restores the legacy context engine slot.

## Docker-native parity matrix

This matrix compares the **Python CLI contract** to the Docker-native host wrapper added in this branch.

Legend:

- **native in container** — the command runs entirely inside the Headroom container
- **host-bridged** — Headroom runs in Docker, but the wrapped external tool still runs on the host

| Command path | Python CLI | Docker-native wrapper | Parity |
|---|---|---|---|
| `headroom proxy` | native | native in container | full |
| `headroom learn` | native | native in container | full |
| `headroom perf` | native | native in container | full |
| `headroom evals memory` | native | native in container | full |
| `headroom evals memory-v2` | native | native in container | full |
| `headroom memory ...` | native (when memory deps are available) | native in container | full |
| `headroom mcp install` | native | native in container | full |
| `headroom mcp uninstall` | native | native in container | full |
| `headroom mcp status` | native | native in container | full |
| `headroom mcp serve` | native | native in container | full |
| `headroom install apply|status|start|stop|restart|remove` | native | Docker-native wrapper for `persistent-docker`; compose remains an alternative | partial |
| `headroom wrap claude` | native | host-bridged | partial |
| `headroom wrap copilot` | native | not implemented in Docker-native wrapper | none |
| `headroom wrap codex` | native | host-bridged | partial |
| `headroom wrap aider` | native | host-bridged | partial |
| `headroom wrap cursor` | native | host-bridged | partial |
| `headroom wrap openclaw` | native | host-bridged | partial |
| `headroom unwrap openclaw` | native | host-bridged | partial |

For the Docker-native execution model itself, see [Docker-Native Install](docker-install.md). For persistent service/task/docker lifecycle management, see [Persistent Installs](persistent-installs.md).

## Hidden and compatibility-only command paths

These exist in code but are intentionally excluded from normal user docs:

- `headroom memory-eval`
- `headroom memory-eval-v2`
- hidden internal `--prepare-only` flags on `wrap` subcommands

If you are documenting operational behavior or debugging internal wrapper flows, refer to the implementation in `headroom/cli/wrap.py`.

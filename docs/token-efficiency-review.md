# Token-Efficiency Code Review

A full-codebase review focused on one question: **where does Headroom itself spend tokens it doesn't need to?** Findings are grouped by theme and ranked by estimated impact. Token estimates use ~3.7 chars/token (hex hashes ~2 chars/token).

The short version: the compression algorithms are in good shape, but the *fixed text Headroom adds* — tool schemas, retrieval instructions, compression markers — costs roughly **3,000–4,000 tokens per request** in a fully-enabled session, and several bugs cause output to be duplicated or even grow. Most fixes below are small, lossless edits.

---

## A. Per-request injection overhead (largest, compounding)

### A1. Memory tool schemas: ~2,600 tokens replayed on every request
`headroom/memory/tools.py:25-280`, `headroom/proxy/memory_tool_adapter.py:73-225`

The full `MEMORY_TOOLS` JSON is ~9,640 chars ≈ **2,605 tokens** (`memory_save` alone: 793). Descriptions carry 5-bullet importance rubrics, DO/DO NOT lists, and "Search strategies 1/2/3" prose. `apply_session_sticky_memory_tools` (`proxy/helpers.py:2014-2100`) makes this sticky for the whole session — correct for prefix caching, which means the fix is to make the bytes *smaller*, not intermittent.

The OpenAI variant in `memory_tool_adapter.py:236-241` already proves the examples collapse to one line with no behavior change. Additional cuts:
- `extracted_entities` / `extracted_relationships` (~180 tok of nested schema) → separate tool registered only when graph storage is on, or free-form `object`.
- `reason` params on `memory_update`/`memory_delete`: **keep the fields** — they are consumed in production (`MemoryHandler._execute_update` records `reason` in edit history and forwards it into audit metadata, `proxy/memory_handler.py:1359-1393`). The saving here is limited to tightening the description strings, not removing the parameters.

**Target: ~120 tok/tool → saves ~2,000 tokens/request.** One-time cache-busting edit, permanent payback.

### A2. Tool dedup misses `mcp__*__`-prefixed names → double injection
`headroom/proxy/helpers.py:2339-2355`, `proxy/tool_name_policy.py:8-22`

The "don't double up" guard compares verbatim names, but `headroom wrap` registers the MCP server whose tools surface as `mcp__headroom__headroom_retrieve` — never equal to `headroom_retrieve`. The proxy then appends a second near-identical tool (~129 tok duplicated, plus model ambiguity). Same bug applies to sticky memory tools vs `mcp__headroom-memory__memory_save` — potentially the entire ~2,600-token block **twice**. `config.py:269-282` already has `_tool_name_aliases` for exactly this; it just isn't used here.

**Fix:** normalize only the known Headroom server labels (`mcp__headroom__*`, `mcp__headroom-memory__*`) before the dedup check. Do **not** strip arbitrary `mcp__<server>__` prefixes or reuse `_tool_name_aliases` directly — that would conflate tools from unrelated MCP servers whose final component happens to match (e.g. an existing `mcp__other-memory__memory_save` would wrongly suppress Headroom's own `memory_save`, which targets a different store). `_tool_name_aliases` is built for broad exclusion matching, not ownership-sensitive dedup.

### A3. CCR retrieval tool description: 3 verbatim copies, ~129 tok each
`headroom/ccr/tool_injection.py:42-110`

The same 478-char description is duplicated across OpenAI and Anthropic branches; the Google branch already ships a shorter working version. A terse shared constant — description `"Get the original uncompressed content for a hash shown in a compression marker."`, no param-level description — is ~61 tok. Sticky-on, so **~68 tok saved on every turn** after first compression, and one constant prevents drift.

### A4. MCP server tool descriptions: ~300 tok resident every request
`headroom/ccr/mcp_server.py:614-705`

`headroom_compress` (82 tok) + param (26) + `headroom_retrieve` (64) + `headroom_stats` (36) + optional `headroom_read` (94). Tighter rewrites keep behavior at roughly half the cost, e.g.:
- compress: `"Compress large text (tool output, files, logs) to save context. Returns compressed text + a retrieval hash."` (−53 tok)
- retrieve: `"Fetch the original content behind a compression marker's hash=."` (−48)
- stats: `"Session compression stats: counts, tokens saved, cost."` (−22)

**~145 tok/request recovered** — ~30k tokens over a 200-turn session. Also unify wording with A3 into one shared constant (dedup between MCP and proxy injection is handled at `tool_injection.py:346-350`, but the two descriptions have already drifted).

### A5. Memory recall block: ~190 tok of framing in the *uncached* live zone
`headroom/proxy/memory_handler.py:894-909`

The READ-ONLY preamble (~120 tok) + ID-usage trailer (~71 tok) wrap a payload capped at 1,024 tokens and often much smaller — and they're appended to the latest user turn, so they are **never prefix-cached**. Move the framing to the system prompt once per session (cacheable) and reduce the per-turn block to `<recall>` + entries, or compress the framing to one ~25-token line. **Saves ~165–180 uncached tok/request.**

### A6. Duplicated inline memory instruction on the Responses/WebSocket path
`headroom/proxy/handlers/openai.py:6745-6759`

A ~150-token `## Memory` instruction block duplicates guidance already in the tool descriptions. Keep the one novel clause ("search memory before searching files") as a sentence in `memory_search`'s description; delete the rest.

### A7. CCR system-instructions block (opt-in, but broken-by-design when on)
`headroom/ccr/tool_injection.py:113-144`

~158 tok, of which the `**Available hashes:**` list (~60 tok) is redundant with inline markers **and** mutates turn-to-turn — a cache-busting change in the system prompt, the exact hazard `output_steering.py` warns about. Cut to one byte-stable line: `"Compressed tool output carries a hash; call headroom_retrieve(hash) for the original."` (~20 tok).

Separate gap on the same path: when the system message has **structured** (list) content, `inject_into_system_message` hits the `else` branch at `:397-399` and appends the message unchanged — no instructions are injected at all. So shortening the string alone doesn't cover that case; structured-content insertion (appending a text block to the list) needs to be implemented alongside it.

---

## B. Marker & hash format (cross-cutting, many small × many)

### B1. `<headroom:tool_digest sha256="...">` is emitted but never read
`headroom/transforms/smart_crusher.py:1293-1294`, `:1326-1327`; built in `utils.py:127-135`

~16–18 tokens appended to every crushed message. Grep shows **zero production consumers** of `extract_markers`/`tool_digest` outside `utils.py` and its tests — pure metadata the LLM is charged for and nothing uses. On a 40-tool-result transcript that's ~650 wasted tokens. Carry the hash in `TransformResult.markers_inserted` instead of message content.

### B2. Five marker grammars; converge on one terse format
Emitters: `kompress_compressor.py:1526-1531`, `:1914-1919`; `kompress_remote.py:135-140`; `code_compressor.py:1268-1273`; `config_compressor.py:160`, `:231`; `read_lifecycle.py:502-513`; `read_maturation.py:311-314`; `log_compressor.rs:969-974`; `search_compressor.rs:341`.

Typical marker: `[1200 items compressed to 40 (from 3100 source lines). Retrieve more: hash=<24hex>]` ≈ 27 tok. SmartCrusher's Rust path already ships `<<ccr:{12hex} {n}_rows_offloaded>>` ≈ 10 tok — same information, 2.7× cheaper. With 15–50 markers live in context, prose markers cost 400–900 tokens per turn.

- Converge bracket markers on the `<<ccr:...>>` shape (e.g. `<<ccr:<hash> 1200→40>>`, keeping the store's 24-hex keys — see B3). The retrieval-tool description explains semantics once; restating "Retrieve more: hash=" per marker is redundant.
- `tool_injection.py:182-217` already parses 5 formats — consolidation also simplifies the scanner.
- **Caution:** `"Retrieve original: hash="` is load-bearing for `content_router.py:4976/6009/6064`, `compression_units.py:110`, `parser.py:30`, `read_maturation.py:282` — change all matchers in lockstep.

### B3. Hash keys cost ~12 tokens each; shrink the *encoding*, not the entropy
`headroom/cache/compression_store.py:325` and all Python emitters (`hexdigest()[:24]`)

Hex tokenizes ~2 chars/token, so a 24-hex key ≈ 12 tok per marker. **Do not truncate to 12 hex:** 96 bits was a deliberate collision-resistance choice (documented at `compression_store.py:303-305`) — at 48 bits an adversary crafting content needs only ~2^24 candidates to find a colliding pair, after which storing the second value overwrites the first and an older marker retrieves the wrong original. The Rust path's 12-hex keys (`crusher.rs:1154-1163`) are an existing compatibility exception, not a precedent to generalize.

The token saving is still available without losing entropy: re-encode the same 96 bits in base32/base64url (96 bits = 16 base64url chars ≈ 6–8 tok vs 12) — `parse_tool_call`/marker regexes would need to accept the wider alphabet in lockstep. Alternatively, keep hex and accept the cost; this is the lowest-priority marker item.

### B4. `Expires in {N}m.` is dead weight and factually wrong
`headroom/transforms/code_compressor.py:1266-1273`

~6 tok/marker the model can't act on (no clock; TTL measured from store time, replayed across turns). Worse: it says `5m` (`ccr_ttl: int = 300` at `code_compressor.py:529`) while the real store TTL is 1800s (`config.py:562`). Drop the clause.

### B5. `CCRConfig.marker_template` is dead config
`headroom/config.py:571-578`

Zero consumers — every compressor hardcodes its own f-string, which is *why* there are five grammars. Either wire all emitters through it (the enabling refactor for B2–B4, making terseness a single-knob change) or delete it.

### B6. Read-lifecycle/maturation markers are full English sentences
`headroom/transforms/read_lifecycle.py:502-513`, `read_maturation.py:311-314`

~35–50 tok each ("was modified after this read — re-read the file for current content..."), dozens per long session (~750 tok in a 40-Read session). The re-read advice is identical every time — state it once in the retrieval tool description; compress markers to ~12–18 tok (`[stale path/to/file.py <<ccr:hash>>]`). Same lockstep-matcher caution as B2.

### B7. Per-sub-array `_ccr_dropped` sentinel
`smart_crusher.py:79`; minted at `crusher.rs:913`, `:557-570`

`{"_ccr_dropped": "<<ccr:a1b2c3d4e5f6 42_rows_offloaded>>"}` ≈ 22 tok, minted **per sub-array** (`json_offload.rs:25`) — 8 crushed arrays ≈ 176 tok. `_rows_offloaded` restates what the key means. Shorten the constants (parsers scan by prefix + hex, `smart_crusher.py:87-100`), or hoist to one sentinel per document listing all hashes.

---

## C. Per-compressor emission waste

### C1. Search results repeat the file path on every match line
`headroom/transforms/search_compressor.py:129` (`group_by_file: bool = False`), `search_compressor.rs:586-598`

Default mode re-emits the path per match (×5/file) plus once more in the footer — ~1,000 tokens of pure redundancy in a large Grep result at default caps. The grouped `--heading` mode exists but is only enabled in token mode (`proxy/server.py:899`). **Flip the default**, or auto-group when a file has ≥2 matches.

### C2. Log compressor: double annotation, no error dedup, over-eager context
`headroom/transforms/log_compressor.py` (+ Rust mirror)

- **Double footer** (`:486` + `log_compressor.rs:969-974`): `[137 lines omitted: 3 ERROR, 12 WARN, 122 INFO]` immediately followed by `[200 lines compressed to 63. Retrieve more: hash=...]` ≈ 48 tok restating the same arithmetic. Merge to one ~18-tok line.
- **Misleading counts** (`:474-486`): the level breakdown counts *all* lines, not omitted ones — `[300 lines omitted: 12 ERROR ...]` advertises errors still present above. Subtract selected counts; drop the never-actionable INFO term.
- **No dedup on errors/fails** (`:374-384`): `_dedupe_similar` applies only to warnings. The most repetitive real-world content — the same assertion across N parametrized tests — is kept verbatim up to `max_errors=10`. Dedup with a `×N` suffix (strictly more informative, ~4 tok). Same for stack traces: first 3 kept with no equality check; hash normalized frames, emit `[same trace ×3]`.
- **Context expansion around everything** (`:445-459`): `error_context_lines` expands ±3 around errors *and* warnings, summaries, and stack-trace lines. pytest's `====` banners all match `summary_patterns`, dragging in up to 120 low-value neighbors against the `max_total_lines=100` budget. Restrict to ERROR/FAIL lines; make the window asymmetric (1 before, 2 after).

### C3. Diff output repeats each path 4× per file
`crates/headroom-core/src/transforms/diff_compressor.rs:1101-1129`

`diff --git a/p b/p` + `--- a/p` + `+++ b/p` = 4 path copies × `max_files=20` = ~80 repetitions of framing a model doesn't need to *read* a diff. For unchanged-name modifications emit a single path heading; keep the full git triple only for renames/creates/deletes.

### C4. Code compressor details
`headroom/transforms/code_compressor.py`

- **Dead `pass` after every truncated body** (`:1840-1850` + Rust `:1659-1670`): emitted unconditionally when lines are omitted, but only needed when the kept body is empty. ~4 tok × every compressed function (~240 tok on a 60-function module) and misleading about control flow. Guard on `not kept_lines and not docstring_text`.
- **`calls:` list in omitted-comments** (`:2422-2436`): ~15–25 tok/function, mostly `len`/`isinstance`/`logger.debug` noise. Cap at 3, filter builtins and callees appearing in >50% of functions, drop `+N more`.
- **Summary redundancy** (`compression_summary.py:108-115`): `"N tokens compressed. 5 bodies compressed: authenticate(), ..."` — duplicate verb, `()` suffix costs ~1 tok/name, `(+3 more)` ~5 tok. Emit bare comma-joined names.

### C5. `collapse_runs` can make output *larger*
`headroom/transforms/lossless_compaction.py:109-113`

The `... (repeated N times)` marker (~22 chars) is emitted for any run ≥ 2 with no per-fold net-win guard (verified: a 16-char input becomes 56 chars). Only the *global* `_smaller` check saves it, so on inputs where long runs win, short runs ride along as pure loss. `fold_repeated_blocks` (`:183`) already has the per-fold guard — add the same to `collapse_runs`, and shorten both markers (`…x5`; `…12@-40`) with their co-located regexes.

### C6. Cross-turn dedup pointers can exceed the replaced span
`headroom/transforms/cross_turn_dedup.py:121-143`, floors at `:40-41`

Verified: a 45-char span becomes a 49-char pointer; `chars_removed` goes negative and is reported as savings. The `↑` glyph costs 2–3 tokens alone and `{anchor!r}` quoting fragments tokenization. Add an explicit `len(ptr) < len(span_text)` acceptance check, drop `↑`/`!r`, raise `DEFAULT_MIN_CHARS` from 40 to ~120.

### C7. Universal compressor's per-span banner
`headroom/compression/universal.py:199-221`, `:331-345`

` ...[compressed]... ` (20 chars, padded) is emitted per non-structural span >50 chars — 40 spans ≈ 200 tok of identical framing, and at the 51-char threshold the banner is ~40% of the span. Use `…`, raise the per-span threshold to ~200 chars.

### C8. Config compressor never strips trailing inline comments
`headroom/transforms/config_compressor.py:47-51`, `:246-260`

All comment regexes are line-anchored, so `replicas: 3  # bumped for load test` — the dominant comment form in k8s/CI/pyproject files — survives untouched despite being CCR-recoverable. Extend `_strip_comment_lines` to trim trailing ` #…` with the existing quote/block-scalar conservatism.

### C9. Spreadsheet ingest: CRLF, phantom rows, empty columns
`headroom/transforms/spreadsheet_ingest.py:21-27`

Verified output carries `\r\n` per row (excel dialect; ~1 wasted token/row plus a dangling `\r`), the docstring's "dropping fully empty trailing rows" is **not implemented** (openpyxl read-only mode over-reports dimensions → hundreds of `,,,,` lines), and trailing empty columns are never trimmed. Pass `lineterminator="\n"` and compute a true bounding box — 20–40% reduction on typical exports, zero fidelity loss.

### C10. HTML extractor defaults maximize output
`headroom/transforms/html_extractor.py:60-71`

`include_links=True` renders every anchor as `[text](long-tracked-url)` — URLs are the most token-dense text there is — and `favor_recall=True` is documented in-line as "more content, may include some noise". Expect 25–50% savings on link-heavy pages with both off.

**But don't change the defaults silently:** the router forwards only `HTMLExtractionResult.extracted` (`content_router.py:413-419`, `:3276-3295`) and the original HTML is *not* stored in CCR, so whatever extraction drops is unrecoverable. On docs/search/index pages the destination URL often *is* the payload, and anchor text alone is ambiguous. Safe versions of this win: (a) keep defaults, expose the lean settings as an opt-in profile; (b) store the pre-extraction HTML (or at least the link map) in CCR first, then flip the defaults; (c) lossless middle ground — keep links but strip query strings/tracking params and collapse same-origin hrefs to paths.

### C11. Lossless-fold gates skip the highest-frequency payloads
`headroom/transforms/content_router.py:1659`, `:4506`, `:5521`, `:5568`; `smart_crusher.py:174`

Agent transcripts are dominated by 200–600-char bash/grep outputs, and the `< 200` gates skip them entirely even though `compact_lossless` is pure-stdlib, microsecond-fast, and self-verifying (returns input if not smaller). Lower the lossless gates to ~80 chars and `min_chars_for_block_compression` to ~200; split `min_tokens_to_crush` into separate lossy/lossless thresholds so the lossless CSV-schema fold isn't blocked by the lossy gate.

### C12. Search folds pick one axis, never compose
`headroom/transforms/lossless_compaction.py:459-471`

`search_heading` (repeated file) and `search_dir_heading` (repeated dir) compete; the winner is whichever is smaller *alone*, but real `grep -rn` output has both kinds of repetition. Add the composition as a third candidate with the paired inverse — the existing round-trip check makes it safe.

---

## D. Serialization & payload hygiene (free, lossless)

### D1. `json.dumps` without compact separators
- `headroom/transforms/smart_crusher.py:660`, `:715` — un-does the Rust side's compact serialization on the audit-safe splice path
- `headroom/transforms/content_detector.py:274`
- `headroom/transforms/tabular_ingest.py:179`
- `headroom/transforms/config_compressor.py:215`

Default separators add a space per comma/colon (~240 chars on a 15×8 result; `", "` often costs a token `","` doesn't). The repo already knows the fix (`utils.py:70`). Add `separators=(",", ":"), ensure_ascii=False` at all four sites.

### D2. MCP tool results: `indent=2` + telemetry fields
`headroom/ccr/mcp_server.py:778, 833, 898, 998-1013`

Every model-facing MCP result is pretty-printed (+20–30% on nested `headroom_stats`), lives in context forever. The module already uses compact separators for its *internal* stats file (`:212`). Also: `original_item_count`/`compressed_item_count`/`retrieval_count` in retrieve payloads (`:475-481`) are telemetry, not content — move to the log line; and the compress-result `note` (`:456`, ~28 tok) restates the `hash` field and the retrieve tool description — delete.

### D3. Retrieval-miss error prose
`headroom/ccr/mcp_server.py:512-535`

~103 tok explaining internal TTL policy the model can't act on, with the same advice repeated three times across `error` + `hint` + the expired branch. Collapse both branches to: `"Not found or expired — re-read the file / re-run the command that produced it."` (~20 tok).

### D4. Steering sentinel tags
`headroom/proxy/output_verbosity_policy.py:7-8`

`<headroom_output_shaping>` wrapper ≈ 16 tok of the ~75-tok level-2 block; a short sentinel saves ~10 tok/request. Batch with any other steering edit (the file itself warns these strings are cache-busting). Also: levels are documented as cumulative but implemented as independent full strings — fine, but the comment is wrong.

---

## E. `headroom learn` output (compounds across every future session)

`headroom/learn/writer.py:89-107`, `:162-193`; `headroom/learn/analyzer.py:473`, `:854`

The learned block lands in `CLAUDE.local.md`/`AGENTS.md` — loaded into **every request of every future session** — and:

1. **No cap** on sections, bullets, or bytes; every `--apply` unions new + all prior sections forever. Add a rendered-block cap (~1,500 tok), evicting carried sections by ascending `estimated_tokens_saved` (already sorted).
2. **Dedup is exact-string on LLM-free-written headings** — `Environment`, `Environment Rules`, and `Environment Setup` accumulate side by side. Normalize headings before comparison and give `_SYSTEM_PROMPT` a closed heading vocabulary.
3. **The `on {now}` date line rewrites the file on every run** even when rules are identical — byte churn that busts downstream prefix caches for no semantic change. Move provenance to a sidecar; drop the per-section `*~N tokens/session saved*` annotations (~8 tok each — the analyzer's own guesses, meaningless to the consuming agent).

---

## Dead code to remove

- `CCRConfig.marker_template` (`config.py:571-578`) — zero consumers (see B5).
- `summarize_dropped_items` (`compression_summary.py:20-77`) — only test callers; its would-be consumer is the dead template above. Delete, or fix its format before ever wiring it in.
- `tool_digest` marker machinery in prompt content (see B1) — keep the hash out-of-band.

## Verified clean (no action)

- `headroom/tools.json` is a CLI-binary registry (difft/scc/ast-grep), not LLM-facing.
- `headroom/lean_ctx/` is a binary downloader; `hooks.py` is an abstract base; `mcp_registry/` writes client config only — none inject prompt text.
- Sticky-on tool injection (PR-B7) is the right cache tradeoff — don't make it conditional; make it smaller.
- MCP-vs-proxy retrieve dedup at `tool_injection.py:346-350` works when names match (but see A2 for the prefixed-name miss).

## Suggested order of attack

1. **A1 + A3 + A4** — tool-schema shrink behind shared constants (~2,200 tok/request, one cache-busting edit).
2. **A2** — alias-aware dedup (bug fix; prevents paying A1 twice).
3. **B5 → B2/B4/B6** — single marker template, then terse format in one lockstep change with all matchers (B3's hash re-encoding is optional and last).
4. **D1, C9, C4-pass, B1** — free lossless one-liners.
5. **C1, C2, C3** — search/log/diff format changes (highest-frequency tool outputs).
6. **A5, E** — memory framing to system prompt; learn-block cap + dedup.
7. **C5, C6** — net-win guards (correctness).
8. Remaining C/D items opportunistically.

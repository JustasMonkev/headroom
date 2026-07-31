# Token-Efficiency Code Review

A full-codebase review focused on one question: **where does Headroom itself spend tokens it doesn't need to?** Findings are grouped by theme and ranked by estimated impact. Token estimates use ~3.7 chars/token (hex hashes ~2 chars/token).

The short version: the compression algorithms are in good shape, but the *fixed text Headroom adds* — tool schemas, retrieval instructions, compression markers — costs roughly **3,000–4,000 tokens per request** in a fully-enabled session, and several bugs cause output to be duplicated or even grow. Most fixes below are small, lossless edits.

---

## Measured outcome

The findings below were implemented and then measured against the merge-base on identical inputs with a real tokenizer (`o200k_base`). **The estimates in this document were made with a ~3.7 chars/token heuristic and run systematically high — treat every number below the fold as an estimate, and these as the measurements.**

| | measured |
|---|---|
| Fixed per-request injection overhead (Anthropic, memory + CCR) | **2,535 → 1,064 tok (−58%)** |
| — of which prefix-cacheable (tool schemas + system prompt) | 2,276 → 881 |
| — of which **uncached** live zone (recall block) | 259 → 183 |
| Whole-transcript token savings, 120-turn | 192,881 → 177,617 |
| Whole-transcript token savings, 200-turn | 321,466 → 296,026 |
| Overall ratio | 27.3% → **33.1%** saved |
| Latency, realistic 120-turn request | 1.01× (no meaningful change) |
| Peak memory, 200-turn / 442k tokens | 27.78 → 28.38 MB (+2.1%) |

Three corrections that matter more than the totals:

1. **The headline baseline was wrong.** The real fixed overhead is 2,535 tok/request, not "3,000–4,000".
2. **Most of the saving is prefix-cached.** 1,395 of the 1,471 saved tokens sit in the sticky prefix, so on a cache hit their marginal value is roughly 140 full-price-equivalent tokens — not 1,395. A5's uncached saving is the one that bills in full every turn, and it is ~41 tok, not the ~165–180 claimed.
3. **Section C largely does not reach the payloads it targets.** On a realistic 18-message coding transcript, `headroom.compress()` produced *identical* output before and after (69,285 → 47,227 both trees): the lossless fold wins first and returns before the improved lossy compressors are consulted. The section-C work pays off only where the router actually selects those paths (e.g. LOG: 1,911 → 1,136, −41%).

Measurement also caught four defects in the implementation itself, all since fixed: a narrowed log window that dropped the *reason* a test failed, an unfiltered marker list that re-injected the retrieval tool on every frozen turn (busting the tools cache segment), a quadratic mutating-input regex (771 ms on a 64 KB blob), and a routing pre-empt that made C11's own target workload worse. Two threshold changes were reverted outright after measuring as net losses (`_MIN_SPAN_TO_COMPRESS` 50→100, `DEFAULT_MIN_CHARS` 40→120).

Full test suite after all fixes: **107 failures on both the merge-base and the implemented tree — identical sets**, all environmental (blocked model downloads, AWS credentials, two order-dependent tests).

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

The token saving is still available without losing entropy: re-encode the same 96 bits in base32/base64url (96 bits = 16 base64url chars ≈ 6–8 tok vs 12). But the migration surface is wider than the marker regexes — every hex-only consumer must move in lockstep: `CompressionStore.store`'s `explicit_hash` validator rejects non-hex keys outright (`compression_store.py:311`), SmartCrusher's Rust→Python mirror scanner consumes only hex characters (`smart_crusher.py:1081-1112`), plus `parse_tool_call`, the marker collectors, and the Rust emitters — with compatibility handling for hex markers already live in stored sessions. Given that breadth, keeping hex and accepting the cost is reasonable; this is the lowest-priority marker item either way.

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

`{"_ccr_dropped": "<<ccr:a1b2c3d4e5f6 42_rows_offloaded>>"}` ≈ 22 tok, minted **per sub-array** (`json_offload.rs:25`) — 8 crushed arrays ≈ 176 tok. `_rows_offloaded` restates what the key means. Shorten the constants, or hoist to one sentinel per document listing all hashes.

**Correction — the parser does *not* scan the sentinel key by prefix.** `is_ccr_sentinel()` checks exact membership of `CCR_SENTINEL_KEY`, and `strip_ccr_sentinels()` relies on that check. Shortening `_ccr_dropped` without updating that consumer would expose the metadata object as a normal row in compressed arrays, breaking downstream uniform-schema iteration. This needs a lockstep change on both sides *plus* backward compatibility for old sentinels already present in live conversations (accept both the old and new key during the transition). That raises the cost well above the ~176 tok it recovers, which is why it is deferred rather than implemented.

---

## C. Per-compressor emission waste

### C1. Search results repeat the file path on every match line
`headroom/transforms/search_compressor.py:129` (`group_by_file: bool = False`), `search_compressor.rs:586-598`

Default mode re-emits the path per match (×5/file) plus once more in the footer — ~1,000 tokens of pure redundancy in a large Grep result at default caps. The grouped `--heading` mode exists but is only enabled in token mode (`proxy/server.py:899`). **Flip the default**, or auto-group when a file has ≥2 matches.

> **Outcome:** auto-group at ≥2 shipped and is confirmed correct — grouping trades N−1 path repetitions for one heading, so it is already ahead at N=2 and never behind. `o200k_base`: 20 files × 2 matches 500 → 380, one file × 2 matches 16 → 12, 40 files × 1 match unchanged (correctly no-ops). A floor of 3 buys nothing. The headline "324 → 204 (−37%)" was overstated; the real range is −18% to −29%, 0% when every file has a single match.

### C2. Log compressor: double annotation, no error dedup, over-eager context
`headroom/transforms/log_compressor.py` (+ Rust mirror)

- **Double footer** (`:486` + `log_compressor.rs:969-974`): `[137 lines omitted: 3 ERROR, 12 WARN, 122 INFO]` immediately followed by `[200 lines compressed to 63. Retrieve more: hash=...]` ≈ 48 tok restating the same arithmetic. Merge to one ~18-tok line.
- **Misleading counts** (`:474-486`): the level breakdown counts *all* lines, not omitted ones — `[300 lines omitted: 12 ERROR ...]` advertises errors still present above. Subtract selected counts; drop the never-actionable INFO term.
- **No dedup on errors/fails** (`:374-384`): `_dedupe_similar` applies only to warnings; the same assertion across N parametrized tests is kept verbatim up to `max_errors=10`. **But don't extend `_dedupe_similar` wholesale** — its similarity normalization would collapse failures that differ only in IDs, values, or paths into one `×N` entry, hiding *which* inputs failed. Use **byte-identical** dedup for errors/fails (first occurrence + `×N`), or keep each distinct normalized suffix. Stack traces are safer: hash the exact frame list and emit `[same trace ×3]` only for byte-identical traces.
- **Context expansion around everything** (`:445-459`): `error_context_lines` expands ±3 around errors *and* warnings, summaries, and stack-trace lines. pytest's `====` banners all match `summary_patterns`, dragging in up to 120 low-value neighbors against the `max_total_lines=100` budget. Restrict to ERROR/FAIL lines; make the window asymmetric (1 before, 2 after).

> **Outcome:** shipped, but it cost accuracy until paired with a second change. The narrowed window dropped the assertion text on pytest runs (load-bearing signals 2/4 → 1/4), because the level classifier never matches `AssertionError:` (no word boundary inside the name) or `E   assert 4 == 5`, so those lines were surviving only as context. Failure detail is now *selected on its own merit* (`is_failure_detail`), and `^_{3,}` was added to the summary patterns so pytest's `____ test_name ____` header still attributes it. Signals 3/4, tokens 303 (merge-base) → 197 — a smaller win than the 109 the bare narrowing produced, and the right one.

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

> **Outcome:** the acceptance check and the marker trim shipped; the floor raise was **measured as a loss and reverted to 40**. On 600 blocks the raise removed *fewer* chars (11,329 → 0) for *more* CPU (10.9 ms → 37.1 ms) — folding less leaves a larger corpus for the anchor matcher. With the acceptance check in place the floor is not load-bearing for safety: dropped to 1 over 11-char spans it performs 0 folds and grows nothing.

### C7. Universal compressor's per-span banner
`headroom/compression/universal.py:199-221`, `:331-345`

` ...[compressed]... ` (20 chars, padded) is emitted per non-structural span >50 chars — 40 spans ≈ 200 tok of identical framing, and at the 51-char threshold the banner is ~40% of the span. Shorten the banner and raise the per-span threshold to ~200 chars.

> **Outcome:** the banner shortened to `…[c]…` and a per-span net-win guard was added; the threshold raise was **measured as a loss and reverted to 50**. Swept with the marker held constant (`cl100k_base`, Kompress off so the fallback path is observable), 50 wins on every fixture and 100 is a total no-op on a 600-line log: 20,415 vs 29,999 tokens. Real log spans after entropy preservation sit in the 50–100 char band. The "banner is ~40% of the span" risk is what the net-win guard now answers, so the floor no longer has to.

**Do not replace it with a bare `…`.** This is the `_simple_compress` fallback path: it adds no CCR hash and no other recovery marker, so the banner is the *only* signal that bytes were deliberately removed. A bare ellipsis is indistinguishable from ellipses already present in source text, and an agent would read the truncated span as exact content. Keep a short but unambiguous marker.

### C8. Config compressor never strips trailing inline comments
`headroom/transforms/config_compressor.py:47-51`, `:246-260`

All comment regexes are line-anchored, so `replicas: 3  # bumped for load test` — the dominant comment form in k8s/CI/pyproject files — survives untouched despite being CCR-recoverable. **Caveat: this is not a regex-level change.** The compressor has block-scalar/multiline gates but no quote-aware inline handling, so a naive trailing-`#` trim would truncate real data like `command: "echo foo # keep"`. Doing this safely requires a flavor-aware lexer (track quote state before the `#`) or a parse-equivalence check (reparse the trimmed document and compare structures) — skip any line, or the whole file, where equivalence can't be proven.

### C9. Spreadsheet ingest: CRLF, phantom rows, empty columns
`headroom/transforms/spreadsheet_ingest.py:21-27`

Verified output carries `\r\n` per row (excel dialect; ~1 wasted token/row plus a dangling `\r`), the docstring's "dropping fully empty trailing rows" is **not implemented** (openpyxl read-only mode over-reports dimensions → hundreds of `,,,,` lines), and trailing empty columns are never trimmed. Pass `lineterminator="\n"` and compute a true bounding box — 20–40% reduction on typical exports, zero fidelity loss.

### C10. HTML extractor defaults maximize output
`headroom/transforms/html_extractor.py:60-71`

`include_links=True` renders every anchor as `[text](long-tracked-url)` — URLs are the most token-dense text there is — and `favor_recall=True` is documented in-line as "more content, may include some noise". Expect 25–50% savings on link-heavy pages with both off.

**But don't change the defaults silently:** the router forwards only `HTMLExtractionResult.extracted` (`content_router.py:413-419`, `:3276-3295`) and the original HTML is *not* stored in CCR, so whatever extraction drops is unrecoverable. On docs/search/index pages the destination URL often *is* the payload, and anchor text alone is ambiguous. Safe versions of this win: (a) keep defaults, expose the lean settings as an opt-in profile; (b) store the pre-extraction HTML (or at least the link map) in CCR first, then flip the defaults; (c) lossless middle ground — keep links but strip query strings/tracking params and collapse same-origin hrefs to paths.

### C11. Lossless-fold gates skip the highest-frequency payloads
`headroom/transforms/content_router.py:1659`, `:4506`, `:5521`, `:5568`; `smart_crusher.py:174`

**The original premise here was wrong twice over, and the measured payoff is negligible.**

First, the `< 200` checks do *not* skip 200–600-char outputs — they admit that entire range. The separate `min_chars_for_block_compression` default of 500 skips only the 200–499 portion; 500–600-char blocks already proceed. So lowering the lossless gates to 80 targets the **80–199** bucket, not the "200–600-char outputs" this item claimed.

Second, that bucket barely matters. Measured across 5 real Claude Code transcripts (604 `tool_result` blocks, 301,158 tokens): **95.7% of tool-result tokens live in ≥500-char blocks**. The 80–199 bucket is 30% of blocks but only **2.2% of tokens**, of which 1.6% actually fold — **24 tokens recovered on a 301k-token transcript**.

*Implemented anyway* (`LOSSLESS_FOLD_MIN_CHARS = 80` replacing both `< 200` literals): `compact_lossless` is pure-stdlib, self-verifying (returns its input when not smaller) and costs ~23 µs/block, so it is free. It is simply not a meaningful saving, and the lossy floor is deliberately left at 500. The `min_tokens_to_crush` lossy/lossless split was not done.

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

## F. Additional findings (round 2)

A second pass surfaced 11 more opportunities. The first two are larger than the marker/JSON micro-optimizations above. Items are tagged **[verified]** (claims checked against the code in this review) or **[reported]** (plausible, but confirm the cited behavior before implementing).

### Token & cost

#### F1. Auto-enable Anthropic Tool Search for large toolsets — P0 [verified]
`headroom/proxy/handlers/anthropic.py:2398-2440`, `proxy/helpers.py:2826`, `:2887`, `:2995`

The deferral machinery already exists (`inject_tool_search_deferral`, min-tools threshold `_TOOL_SEARCH_MIN_TOOLS = 12`, plus an OpenAI variant) and the code itself cites the stakes: ~135 tool defs ≈ **28k tokens on every request** (`helpers.py:2826`). But the Anthropic path only activates behind the opt-in `HEADROOM_TOOL_SEARCH` env var. The injection is deterministic (prompt-cache-safe) and already scoped to first-party Anthropic. Make it default-on (`auto`) with provider/model gating and an explicit opt-out — potentially thousands to tens of thousands of input tokens per request for MCP-heavy setups.

#### F2. Wire the Claude thinking compactor into the Anthropic handler — P0 [verified]
`headroom/transforms/thinking_compactor.py:50`, `:97`

`compact_thinking_to_text()` exists, is self-tested in-module, and has **zero production callers** — grep finds it only in its own file. The module's own measurements report ~688 tokens per historical Sonnet 4.6 thinking block and ~995 for Opus 4.6, re-billed on every subsequent turn for models where `bills_prior_thinking()` is true. Wire it into the Anthropic handler preserving the latest turn (`keep_last_turns=1`). Saves ~688–995 tokens per eligible historical thinking block per turn.

**Keep the explicit opt-in.** Gating on `bills_prior_thinking(model)` alone would silently enable a *lossy* transform for every eligible conversation: `compact_thinking_to_text()` converts signed thinking into a generated text summary that can omit decisions or context still needed in later turns. The billing predicate establishes only that compaction *could* save tokens, not that the user accepted the accuracy tradeoff. The module documents `HEADROOM_THINKING_COMPACT` as the intended call-site gate — require both.

*Implemented:* wired into the Anthropic handler before `PRE_SEND`, double-gated on the model predicate **and** `HEADROOM_THINKING_COMPACT`. Measured: a 674-token thinking block yields 580 tokens saved per historical block per turn (39.2% of that conversation).

#### F3. Compact completed tool-call *inputs*, not only outputs — P1 [implemented]
In-place compression currently targets tool outputs; historical tool-call **arguments** (Write payloads, apply_patch bodies, shell heredocs, SQL/query strings) stay verbatim in context forever. Once the matching result has completed and aged past the read-protection window, large arguments could be replaced with a reversible CCR reference while preserving the call ID and name. Worth hundreds to thousands of tokens in coding sessions.

**Mutating tool inputs must be excluded.** For `Write`, `apply_patch`, SQL mutations and shell heredocs a successful result is often just an acknowledgement, so the historical argument is the *sole* exact record of what changed — and a CCR entry expires (default 1,800s). Once the file or database changes again, neither the current source nor the expired marker can reconstruct the earlier mutation. Restrict compaction to reproducible/read-only inputs, or give mutating arguments durable storage.

*Implemented:* `transforms/tool_input_compactor.py` — a pre-processing pass in `ContentRouter.apply` (alongside read-lifecycle) that replaces completed, large (≥800 chars), non-recent, non-frozen tool-call arguments with `{"_ccr": "[tool input elided. Retrieve original: hash=…]"}` in both OpenAI and Anthropic wire shapes, storing originals in the CCR store. Opt-in via `HEADROOM_COMPACT_TOOL_INPUTS=1` / `ProxyConfig.compact_tool_inputs` while validated in pilots.

*Review fixes applied on top:*
- **Fails closed on storage failure.** `_store_original` returns `None` when the store is missing, raises, or returns a falsy hash; the tool call is then left byte-identical. Previously the exception was swallowed, a hash was still returned, and the caller replaced the arguments with a marker no entry backed — silently unrecoverable.
- **Mutating inputs are never compacted** (`is_mutating_tool_input()`): name denylist, `mcp__server__leaf` handling, mutating-verb prefixes as an MCP safety net, SQL DML/DDL, shell heredocs, write-redirection and in-place edits (`->` deliberately excluded so search patterns still compact).
- **Tool-input markers now drive CCR injection.** `CCRToolInjector.scan_for_markers` only scans message text and tool-result content, so on the first compaction of a session `detected_hashes` was empty and both handlers skipped the sticky `headroom_retrieve` injection — stranding the stored original. `merge_pipeline_ccr_hashes()` threads `TransformResult.markers_inserted` into the injection decision.
- **Disabled in lossless mode.** With `lossless=True` the proxy sets `ccr_inject_tool=False`, so compaction would have replaced inputs with markers while suppressing the retrieval tool. Gated off in `ContentRouter.__init__`'s lossless normalization (load-bearing on any construction path) and again in `server.py`.

#### F4. Memory rendering: UUIDs, content echo, optional metadata — P1 [implemented]
`headroom/proxy/memory_handler.py:1267`, `:1967`

Passive recall repeats a full UUID per memory row solely so it can later be edited — render shorter aliases and resolve them server-side in `memory_update`/`memory_delete`. Verified: `memory_save` results echo back the first 100 chars of the content the model just wrote (`:1267`) — pure round-trip waste, drop it. Make search scores/extracted-entities in results optional. Consistent savings on every memory-enabled request; combines with A1/A5.

**Scope the content-echo cut to the save response only.** The preview at `:1967` belongs to `_list_all_memories()` and is the identifying payload of `/memories/all` — dropping it would leave each listed memory represented by an ordinal with neither its UUID nor its content, making the listing unusable. Only `:1267` is round-trip waste.

**Render-order aliases (`m1`, `m2`, …) are not safely implementable.** In-memory, first-seen-order maps are empty after a proxy restart and differ between workers, while `[m1]` references persist in the client's conversation — so a later `memory_update`/`memory_delete` can fail *or mutate a different memory than the model meant*. An alias must be derivable from the memory's own identity, not from the order it happened to be rendered in.

*Implemented:* aliases are `m:` + the first 8 chars of the memory's own ID, resolved by strict prefix lookup against the backend — any worker resolves them without shared state, and zero or ≥2 matches raise rather than touching a record. The in-memory maps were removed entirely rather than demoted to a cache. Two rows in one block that would collide both render full IDs. The `memory_save` content echo is gone (`_list_all_memories()` content verified intact and pinned by a test), search `score` is behind an opt-in `include_scores` param — exposed in all five provider schemas and honoured by the adapter's executor — and empty `entities` lists are omitted.

#### F5. Separate native output controls from instruction steering — P1 [implemented]
On OpenAI models that support it, apply `text.verbosity=low` and reduced reasoning effort to mechanical continuations *natively*, without also appending the input-token steering paragraph (D4) — and don't force reduced verbosity onto fresh user questions. Lowers output/reasoning cost without spending steering tokens on requests where the native knob suffices.

*Implemented:* in `shape_openai_responses_request`, requests with native output controls (gpt-5 family, or a client-sent `text.verbosity`) never get the steering paragraph (keeping `instructions` byte-stable), and `text.verbosity` is now set/lowered only on mechanical continuations — new user asks and error continuations keep the client's verbosity.

### Latency & memory

#### F6. Request-specific state mutated on the shared `ContentRouter` [verified]
`headroom/transforms/content_router.py:1899`, `:2090`, `:4304`, `:4517`, comment at `:1968-1969`

`_runtime_target_ratio`, `_runtime_kompress_model`, `_tool_call_args`, and read-protection sets are instance fields mutated per request, while routers are reused and compression runs in a thread pool — concurrent requests can overwrite each other's routing context (wrong target ratio, wrong tool-args map). The comment at `:1968` acknowledges the pattern rather than fixing it. Pass an immutable per-request context object through the call chain instead.

#### F7. Compression caches: byte-weighting, expiry sweeping, single-flight [corrected]
`headroom/cache/compression_cache.py:112`, `:171`; `cache/compression_store.py:693`

Contrary to the round-2 report, both caches **are** entry-bounded (LRU at 10,000 entries; store capped at 1,000 with an eviction heap). The real gaps: bounds are entry-count, not byte-weighted (10k large originals can pin significant memory); TTL expiry is lazy (expired entries linger until queried or LRU-evicted); and there is no single-flight, so identical concurrent requests each run the same compression. Add byte-weighted accounting, a periodic sweep, and per-key in-flight futures.

#### F8. Quadratic read-lifecycle scans [reported]
`headroom/transforms/read_lifecycle.py`

Each tool operation reportedly rescans all messages to find its index, then each read scans all edits and later reads — O(n²) on long transcripts. Capture message indices during the initial traversal and classify files in reverse order using latest-edit / read-coverage state.

#### F9. Copy-on-write and token-count deltas in the pipeline [reported] [skipped]
`headroom/transforms/pipeline.py` (`CompressionPipeline.apply`)

**Corrected reference:** the round-2 report cited `headroom/pipeline.py`, which is the extension-event API and performs none of this work. The tokenization, deep copy and transform execution live in `headroom/transforms/pipeline.py`.

The pipeline does a full tokenization, a full deep copy, per-transform counts/copies, then another exact full count. Pass the baseline count into transforms and have them return exact deltas for changed slots; keep full recounts as a sampled validation path.

#### F10. Unconditional whole-payload serialization [reported]
The Responses path serializes the entire request before transformation and again after, even with debug metrics off or when nothing changed. Reuse the HTTP body bytes, skip the second serialization for unchanged payloads, and sample byte metrics.

### Measurement defect

#### F11. OpenAI tool-schema savings are double-counted when both layers run [verified]
`headroom/proxy/handlers/openai.py:2393-2397`, `:2434-2438`

Layer 1 (schema compaction) adds `count(original) − count(L1)` to `tokens_saved`; Layer 2 (description truncation) then adds `count(original) − count(L2)` — both diffs are taken against the *original* `payload`, so the L1 reduction is counted twice. Doesn't change billed tokens, but inflates reported savings and can distort optimization decisions. Fix: measure each stage against the preceding stage, or compute `original − final` once.

**Round-2 implementation order:** F1 (auto tool deferral) → F2 (thinking compactor wiring) → F6 (router concurrency) → F7 (cache hardening) → F3 (tool-input compaction) → F8–F10 (scans/serialization) → F11 alongside any savings-reporting work.

---

## Dead code to remove

- `CCRConfig.marker_template` (`config.py:571-578`) — zero consumers (see B5). *Removed.*
- `summarize_dropped_items` (`compression_summary.py:20-77`) — only test callers; its would-be consumer is the dead template above. Delete, or fix its format before ever wiring it in. *Removed in round 3, along with its private helpers and the two LLM-eval test files that existed only to exercise it.*
- `tool_digest` marker machinery in prompt content (see B1) — keep the hash out-of-band. *Done.*

---

## G. Round 3 — remaining items closed out (2026-07-31)

A follow-up pass implemented what was still open, re-measured with the bundled
`o200k_base` tokenizer, and ran a Haiku sub-agent review (three lenses:
injected text, compressor emissions, hygiene/perf) for anything the first two
rounds missed. Status of everything that was left:

### Implemented

- **B6-lite — read-lifecycle/maturation marker prose tersened.**
  `read_lifecycle.py`, `read_maturation.py`. The full-sentence markers were cut
  to terse forms that keep the load-bearing `Retrieve original: hash=` anchor
  (so no matcher moved) and keep the one piece of advice that changes model
  behavior — a STALE read must be *re-read*, not retrieved. Measured per
  marker: stale 57 → 50, superseded 56 → 45, maturation 52 → 45 tok
  (−7…−11 each; the path + 24-hex hash dominate what remains). Verified: the
  `tool_injection` scanner, `parser.CCR_RETRIEVAL_MARKER_RE`, and the
  `compression_units` marker-preserving regex all match the new forms.

- **C3 — diff compressor drops the redundant `diff --git a/p b/p` header
  for plain modifications.** `diff_compressor.rs` `format_output`. The
  first-cut implementation dropped the `---`/`+++` pair instead; PR review
  correctly flagged that `diff --git` followed directly by `@@` is a patch
  fragment `git apply` rejects (`parse_diff` discards `index` lines, so the
  extended-header form can't be made whole). The header and the marker pair
  tokenize identically (16 tok each for a typical path — the header carries
  the path twice), so the fix keeps the `---`/`+++` pair and drops the
  header: same saving, and the output is now a *valid* unified diff —
  verified with `git apply --check` on untrimmed output. (When context
  trimming or hunk dropping fires, hunk headers go stale and strict
  applicability was already broken pre-C3; that is a pre-existing property
  of the compressor, not a regression.) Creates/deletes/renames/binary
  files, `/dev/null`, quoted paths, and prefix mismatches all keep the full
  git triple. Measured: −22 tok per plain-modified file, −440 tok on a
  20-file diff. Rust unit tests updated (`compressed_line_count` 129 → 121
  on the 8-file synthetic), the Python extension rebuilt, the 27
  `diff_compressor` parity fixtures re-recorded, and the Rust parity
  harness re-run: 27/27 matched.

- **D1/D2 stragglers (found by the Haiku hygiene reviewer).** Five
  model-facing CCR response paths still pretty-printed with `indent=2`:
  `ccr/response_handler.py` (×4) and the `/v1/retrieve/tool_call` endpoint in
  `proxy/server.py`. All now use compact separators; the success payloads also
  drop the `original_item_count`/`compressed_item_count` telemetry echo
  (`items_retrieved` still travels out-of-band on `CCRToolResult`, and the
  endpoint's caller-facing `data` field keeps the full dict — only the
  model-facing `tool_result.content` slimmed). Measured: −16 tok per
  retrieval, −12 per miss. The evals-only `indent=2` sites the reviewer also
  flagged are not model-facing in production and were left alone.

- **F7 — cache hardening (the parts that pay).**
  `compression_cache.py`: byte-weighted LRU bound (`max_bytes`, default
  64 MB) alongside the 10k-entry bound, and opportunistic pruning of the
  previously unbounded `_first_seen` map (entries older than the defer TTL
  can never answer "defer" again). `compression_store.py`: byte-weighted
  eviction bound (`max_bytes`, default 256 MB) enforced in
  `_evict_if_needed`. Four PR-review hardenings on top of the first cut,
  each with a regression test: sizes are UTF-8 bytes, not code points
  (`len(str)` undercounts CJK/emoji payloads 3-4×; cached per entry so the
  encode is paid once); the bound is enforced on *stored + pending* so the
  store is back under `max_bytes` after the insert, not one entry over; an
  empty eviction heap over a populated backend (restart with the SQLite
  backend, or another worker's writes) is reseeded via `_rebuild_heap()`
  instead of silently skipping eviction; and the byte total rides the
  existing `_clean_expired` `items()` pass instead of a second
  full-backend scan per insert. Defaults sit far above the measured 28 MB
  peak of a 200-turn session: these are guard rails for pathological
  sessions, not tuners.
  Two F7 sub-items were **not** done, deliberately: the "periodic sweep" was
  already effectively present (`_clean_expired` runs on every new-key store;
  expired entries linger only in fully idle sessions, where they cost nothing
  until process exit), and **single-flight** requires restructuring the
  per-request compression call sites around shared futures — cross-request
  identical-content races are rare within a per-session cache, so the
  complexity is not currently justified. Revisit if profiling shows duplicate
  concurrent compression.

### Verified still-deferred (unchanged from rounds 1–2, re-confirmed)

- **B2/B3/B7 marker-grammar convergence** — the lockstep surface (five
  emitters across Python and Rust, the count-parsing regexes in
  `tool_injection`, parity fixtures, and live-session backward compatibility)
  still outweighs the per-marker delta; the B6-lite trim above captured the
  cheap share of the win without moving any matcher.
- **C8** (needs a quote-aware lexer), **C10** (unrecoverable without storing
  pre-extraction HTML), **F9** (wrong-file premise; skipped) — as documented.

### Haiku sub-agent review outcome

Three Haiku reviewers swept the codebase against this document. Net-new
findings: the five `indent=2` stragglers above (confirmed and fixed). The
compressor-emissions reviewer independently re-derived B6 (already queued);
everything else it checked matched this document's implemented/deferred
status. No false positives survived revalidation — the one telemetry-echo
claim was scoped down after checking that `/v1/retrieve`'s caller-facing
JSON contract must keep its fields.

### Round-2 review follow-ups (same PR)

A second reviewer pass on the fixes surfaced four more defects, all fixed
with regression tests:

- **Byte bound on growing re-stores.** The duplicate-store fast path
  skipped `_evict_if_needed`, so re-storing an existing hash with a larger
  compressed payload could keep the store above `max_bytes` with no new key
  ever arriving to trigger eviction. Growing replacements now
  delete-then-evict-then-set (the bound sees the store without the old
  entry and cannot evict the key being replaced); byte-identical duplicate
  re-stores — the common mirror-bridge pattern — still skip the scan.
- **Heap coverage by key, not cardinality.** Another worker deleting one
  row and inserting another leaves the backend count unchanged while the
  local heap lacks the new live key. Coverage is now verified against the
  live-key list the expiry pass already produces.
- **Maturation replay no longer clobbers stale markers.** The
  lifecycle-marker guard now runs before the matured-replay branch: a
  matured file that is edited later keeps read_lifecycle's stale marker
  ("re-read for current content") instead of being overwritten by the
  recorded maturation marker, which only advertises the pre-edit original.
- **Lone-surrogate-safe model-facing JSON.** `ensure_ascii=False` would
  emit a lone surrogate accepted from JSON input literally, crashing the
  continuation request's UTF-8 serialization. All five compact-JSON
  retrieval sites now go through `model_facing_json()`, which probes with
  an encode and falls back to ASCII escaping for exactly that case —
  normal CJK/emoji stays on the cheap unescaped path.

### Round-3 test status

`diff`/`cache`/`CCR`/`read-lifecycle`/`response-handler`/`token-headroom`
suites: 398 passed, 3 skipped. Rust: `headroom-core` diff tests 23/23,
parity harness diff_compressor 27/27 matched (kompress/ccr/cache_aligner
fixtures skip in this environment — no HF model cache — same as merge-base).

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

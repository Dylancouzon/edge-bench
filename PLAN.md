# edge-bench — Qdrant Edge vs Docker vs Cloud

A quick, honest comparison of `create collection → upload → query` across three real
Qdrant deployment shapes, plus one control cell. **Not an official benchmark** — the
goal is a clear, reproducible feel for the latency/throughput profile of each option.

## Targets (4 cells)

| Cell | Engine | Hardware | Network path |
|---|---|---|---|
| **Edge / Pi** | `qdrant-edge-py` (in-process) | Raspberry Pi 4B (ARM64) | none |
| **Edge / laptop** (control) | `qdrant-edge-py` (in-process) | laptop (x86/ARM) | none |
| **Docker / laptop** | Qdrant server in Docker | laptop | localhost loopback |
| **Cloud** | Qdrant Cloud (free tier) | cloud node | internet RTT |

The **Edge-on-laptop** control is the key to interpretation: comparing it against
Edge-on-Pi isolates *hardware* (Pi vs laptop), and comparing it against Docker-on-laptop
isolates *architecture* (embedded vs server) on identical silicon.

## What each number really measures (read before charting)

The cells differ on three axes at once — **hardware, network path, architecture** — so a
flat "fastest wins" chart would mislead. We measure and label each axis:

- **Edge** has *zero* network cost but is bound by local CPU.
- **Docker** is fast CPU + cheap loopback.
- **Cloud** latency is dominated by your internet round-trip, not Qdrant.

The story is *why Edge exists* (no network), not "X beats Y." Every reported number is
tagged with what it includes.

## Fairness controls

- **Identical inputs everywhere:** same 10k texts → same FastEmbed model
  (`BAAI/bge-small-en-v1.5`, 384-dim) → same vectors, same point count, same Cosine
  distance, same batch size (256), same query set (~1000 queries), `limit=10`.
- **Embedding is a separate, measured stage** from DB ops. On-device FastEmbed runs on
  the Pi (the real "embedded AI" story) and on the laptop; we report embedding
  throughput per machine but it does **not** pollute the upload/query DB numbers.
- **Force HNSW on for all server cells.** Default `indexing_threshold` is 20000, so at
  10k points Qdrant server would silently fall back to exact/brute-force search. We set
  `indexing_threshold` low (e.g. 1) and **wait until indexed** (collection status green /
  `indexed_vectors_count` reached) before querying. On Edge we call `optimize()` +
  `flush()` after upload. This makes all cells compare the *same* query path. (We can
  also run an exact-search variant if we want both.)
- **Single-threaded latency, separate throughput run.** Per-query latency is measured
  one-at-a-time (clean p50/p95/p99); QPS is a separate timed loop.

## Metrics (per cell)

1. **Embedding:** texts/sec, total time (Pi vs laptop hardware signal).
2. **Collection/shard create:** time.
3. **Upload:** total time, throughput (vec/s), per-batch latency.
4. **Query:** latency p50/p95/p99, QPS, `limit=10`.
5. **Resources (Pi especially):** peak RSS, CPU% (background sampler / psutil).

Output: one comparison table + short notes per number on what it includes.

---

## Phase 1 — Get the Pi online (never connected before)

Pi 4B is 64-bit capable. We need **64-bit Raspberry Pi OS** for the prebuilt ARM64 wheel.

1. **Flash headless** with Raspberry Pi Imager (needs an SD reader on the Mac):
   - OS: *Raspberry Pi OS Lite (64-bit)* (under "Raspberry Pi OS (other)").
   - Open the settings/⚙ before writing and set: hostname (e.g. `edgepi`), **enable SSH**
     (password or key), username + password, and **WiFi SSID + password + country**.
   - Write to the microSD.
2. **Boot:** insert card, power the Pi, wait ~1–2 min.
3. **Connect from the Mac** (macOS resolves `.local` natively):
   ```
   ssh <user>@edgepi.local
   ```
   Fallback if `.local` won't resolve: find the IP from your router, or `ping edgepi.local`,
   then `ssh <user>@<ip>`. Last resort: Ethernet straight into the Mac + Internet Sharing.
4. **Verify the environment on the Pi:**
   ```
   uname -m            # expect: aarch64   (if armv7l, OS is 32-bit — reflash 64-bit)
   cat /etc/os-release # expect Bookworm (glibc 2.36, satisfies wheel's glibc >= 2.28)
   ldd --version
   free -h             # RAM ceiling
   ```
5. **Python + deps** (Bookworm enforces PEP 668 — use a venv):
   ```
   sudo apt update && sudo apt install -y python3-venv python3-pip
   python3 -m venv ~/edge-bench-venv && source ~/edge-bench-venv/bin/activate
   pip install qdrant-edge-py fastembed
   ```
6. **Smoke test** (confirms the ARM64 wheel loads + a shard creates + FastEmbed runs).

## Phase 2 — Stand up the environments

- **Edge / Pi:** benchmark script runs *on the Pi*; writes `results-edge-pi.json`; `scp` back.
- **Edge / laptop:** same script, run locally.
- **Docker / laptop:** `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`; script runs on laptop.
- **Cloud:** create a free-tier cluster at cloud.qdrant.io; script uses cluster URL + API key.

## Phase 3 — One harness, two backends

A shared driver owns the dataset, the operations, and the metrics. A thin adapter swaps
the engine underneath so every cell runs *identical logic*:

- `EdgeBackend` → `qdrant_edge` (`EdgeShard.create` → `update(UpdateOperation.upsert_points(...))`
  → `query(QueryRequest(Query.Nearest(...)))` → `optimize`/`flush`).
- `ServerBackend` → `qdrant_client` (`QdrantClient`), used for **both** Docker and Cloud,
  differing only by connection params (url/api_key).

The Edge and `qdrant-client` APIs are **not** drop-in identical, so the adapter normalizes
create / upsert / query / count to one interface.

**Corpus:** a fixed 10k-row slice of a small public text dataset (default: `ag_news`),
pinned for reproducibility; swappable.

## Phase 4 — Run, collect, report

Run each cell, collect per-cell JSON, merge into one table with the axis labels above.

---

## Open risks / gotchas

- **32-bit OS** on the Pi → no wheel. Must be `aarch64` (verified in Phase 1.4).
- **FastEmbed on Pi** pulls `onnxruntime` (has aarch64 wheels) and downloads the model on
  first run (Pi needs internet). Embedding 10k texts on a Pi 4 is slow but fine at this size.
- **10k < indexing_threshold (20000):** addressed by forcing `indexing_threshold` low and
  waiting for indexing, so server cells don't silently brute-force.
- **Edge is beta (v0.7.2), API may shift** — pin the version.
- **Cloud numbers are network-bound** — label clearly; not a Qdrant-compute measurement.

## Decisions locked

- Pi 4 Model B · real FastEmbed embeddings · Edge-on-laptop control included · 10k × 384-dim.

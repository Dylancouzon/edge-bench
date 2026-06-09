# Agent brief — build & run the Qdrant edge-bench comparison

You are picking up a project in the `edge-bench` repo. **Read `PLAN.md` first** — it has the
full design. This brief gives you (a) verified facts about Qdrant Edge that are PAST YOUR
TRAINING CUTOFF (do not rely on memory — these are checked against current docs/PyPI), and
(b) the concrete task list. The human (Dylan) is the decision-maker: pause and check with
him before anything irreversible or cost-incurring (e.g. creating the Cloud cluster).

## Goal

A quick, **non-official** comparison of `create collection → upload → query` for Qdrant
across 4 deployment cells. Same data, same operations everywhere; an adapter swaps the
engine underneath. Produce one comparison table + honest notes on what each number means.

## The 4 cells

| Cell | Engine | Where it runs | Network |
|---|---|---|---|
| Edge / Pi | `qdrant-edge-py` (in-process) | on a Raspberry Pi 4B (ARM64) | none |
| Edge / laptop (control) | `qdrant-edge-py` (in-process) | laptop | none |
| Docker / laptop | Qdrant server in Docker | laptop | localhost loopback |
| Cloud | Qdrant Cloud (free tier) | laptop → cloud | internet RTT |

The cells differ on **hardware, network path, and architecture at once** — so do NOT make a
naive "fastest wins" chart. Edge has zero network cost but is CPU-bound; Docker is fast
CPU + cheap loopback; Cloud is dominated by internet round-trip. Label every number with
what it includes. Edge-on-laptop vs Edge-on-Pi isolates hardware; Edge-on-laptop vs
Docker-on-laptop isolates architecture.

## VERIFIED FACTS — Qdrant Edge (recent release; trust these, not your memory)

- **Install:** `pip install qdrant-edge-py` — **pin `qdrant-edge-py==0.7.2`** (beta, API may
  shift). Import name is `qdrant_edge`. Apache-2.0, no signup/license key.
- **It is an in-process embedded library** ("SQLite for vector search"): NO server, NO
  REST/gRPC, NO background optimizer. The core object is an `EdgeShard`. It is **NOT** the
  same API as `qdrant-client` — you cannot reuse `QdrantClient` calls against it.
- **Runs on the Pi via prebuilt ARM64 wheels** (`manylinux_2_28_aarch64`): requires a
  **64-bit OS (`aarch64`)** and **glibc ≥ 2.28**. 32-bit OS has no wheel — won't work.
- **Full create → upsert → query loop (use this exact API):**
  ```python
  from pathlib import Path
  from qdrant_edge import (
      Distance, EdgeConfig, EdgeVectorParams, EdgeShard,
      Point, UpdateOperation, Query, QueryRequest,
  )

  VEC = "vec"
  cfg = EdgeConfig(vectors={VEC: EdgeVectorParams(size=384, distance=Distance.Cosine)})
  Path(shard_dir).mkdir(parents=True, exist_ok=True)
  shard = EdgeShard.create(shard_dir, cfg)            # or EdgeShard.load(shard_dir)

  shard.update(UpdateOperation.upsert_points([
      Point(id=1, vector={VEC: [0.1, 0.2, 0.3, 0.4]}, payload={"k": "v"}),
  ]))
  shard.flush()      # persist to disk
  shard.optimize()   # build index / GC  -> call before timing queries

  res = shard.query(QueryRequest(
      query=Query.Nearest([0.2, 0.1, 0.9, 0.7], using=VEC),
      limit=10, with_vector=False, with_payload=True,
  ))
  ```
- Other `EdgeShard` methods: `load`, `retrieve`, `scroll`, `count`, snapshots. Supports
  dense + sparse + BM25. **On-device embeddings via FastEmbed:**
  `pip install fastembed qdrant-edge-py`.
- **No official Edge performance numbers exist** — our measurements are the contribution.
- If any of the above fails at runtime, verify against the live docs
  (`docs.qdrant.tech/documentation/edge/`) and PyPI rather than guessing the API.

## Server / Cloud cells (standard Qdrant)

- Docker: `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`
- Client: `qdrant-client` — `QdrantClient("localhost", port=6333)` for Docker;
  `QdrantClient(url=<cloud-url>, api_key=<key>)` for Cloud. Use **`client.query_points(...)`**
  (current API), not the deprecated `.search(...)`.
- `create_collection(..., vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))`.

## Fairness controls (non-negotiable)

- **Identical inputs across all cells:** same 10k texts → same FastEmbed model
  `BAAI/bge-small-en-v1.5` (384-dim) → same vectors, same Cosine distance, same batch size
  **256**, same query set (**~1000 queries**), `limit=10`.
- **Embedding is timed as a SEPARATE stage** from DB ops and must NOT be counted in
  upload/query numbers. Embedding runs on the Pi for the Edge/Pi cell and on the laptop for
  the other three (vectors are deterministic from the same model, so effectively identical).
  Report embedding throughput per machine — it's its own (hardware) result.
- **Force HNSW indexing for the server cells.** Default `indexing_threshold` is 20000, so a
  10k-vector collection would silently fall back to brute-force exact search. Lower it
  (e.g. `optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1)`; verify exact
  param against the installed client version) and **poll until `indexed_vectors_count`
  reaches the total** before timing queries. On Edge, call `optimize()` + `flush()` after
  upload. Goal: every cell exercises the same query path.
- **Latency single-threaded; throughput separate.** Measure per-query latency one-at-a-time
  for clean p50/p95/p99; measure QPS in a separate timed loop.

## Locked decisions

Pi 4 Model B · real FastEmbed embeddings (`bge-small-en-v1.5`, 384-dim) · Edge-on-laptop
control included · 10k × 384-dim · corpus = pinned 10k slice of `ag_news` (swappable).

---

## Tasks

### Phase 1 — Get the Pi online (human + you)
The human is flashing a microSD now. **Guide him**, then take over via SSH:
1. Confirm he flashed **Raspberry Pi OS Lite (64-bit)** with Raspberry Pi Imager, and that
   in the Imager ⚙ settings he set: hostname `edgepi`, **SSH enabled**, a username +
   password, and **WiFi SSID + password + country**. (You may use `rpi-imager --cli` to
   automate the flash if you have it and prefer to.)
2. Have him boot the Pi; then connect: `ssh <user>@edgepi.local` (macOS resolves `.local`).
   Fallback: find the IP from the router and `ssh <user>@<ip>`.
3. Verify the environment (must be `aarch64`):
   ```
   uname -m && cat /etc/os-release | head -2 && ldd --version | head -1 && free -h
   ```
   If `armv7l`, stop — the OS is 32-bit and must be reflashed 64-bit.
4. Install deps (Bookworm enforces PEP 668 — use a venv):
   ```
   sudo apt update && sudo apt install -y python3-venv python3-pip
   python3 -m venv ~/edge-bench-venv && source ~/edge-bench-venv/bin/activate
   pip install "qdrant-edge-py==0.7.2" fastembed psutil
   ```
5. Smoke test: create an `EdgeShard`, upsert one point, query it, and run one FastEmbed
   embedding — confirm the ARM64 wheel + ONNX runtime load. (FastEmbed downloads the model
   on first run; the Pi needs internet.)

### Phase 2 — Stand up environments
- Edge/Pi: code runs on the Pi; writes `results/edge-pi.json`; `scp` back to the repo.
- Edge/laptop & Docker/laptop: run locally. Start Docker with the command above.
- Cloud: **ask the human** to create a free-tier cluster at cloud.qdrant.io and provide the
  URL + API key (store in `.env`, never commit). Do not create paid resources.

### Phase 3 — Build the harness
- A shared driver owns: corpus loading, the FastEmbed embedding stage, the operation
  sequence (create → upload → query), the timing/metrics, and JSON output.
- A `Backend` adapter with two implementations behind one interface
  (`create_collection`, `upsert(points)`, `query(vector, limit)`, `count`, `finalize/index`):
  `EdgeBackend` (`qdrant_edge`) and `ServerBackend` (`qdrant_client`, used for both Docker
  and Cloud via different connection params).
- Suggested layout: `bench/driver.py`, `bench/backends.py`, `bench/embed.py`,
  `bench/metrics.py`, `run.py`, `requirements.txt`, `.env.example`, `results/`.
- **Verify the harness end-to-end on the Docker cell first** (fast feedback loop) before
  touching the Pi or Cloud.

### Phase 4 — Run, collect, report
For each cell capture: embedding texts/sec; collection-create time; upload total + vec/s +
per-batch latency; query p50/p95/p99 + QPS (`limit=10`); and on the Pi peak RSS + CPU%.
Merge into one comparison table + a short write-up that labels what each number includes
(especially the network component for Cloud). Write results under `results/` and a summary
in `RESULTS.md`.

## Deliverables
Working harness, per-cell `results/*.json`, a merged comparison table, and `RESULTS.md`
with the honest interpretation. Pin all versions. Don't commit secrets.

# edge-bench

A quick, **non-official** comparison of Qdrant across four deployment shapes, testing the
basics: create a collection, upload vectors, query. See **[PLAN.md](PLAN.md)** for the
design and **[RESULTS.md](RESULTS.md)** for the numbers.

## The cells

| Cell | Engine | Where | Network |
|---|---|---|---|
| **Edge / Pi** | `qdrant-edge-py` (in-process) | Raspberry Pi 4B | none |
| **Edge / laptop** | `qdrant-edge-py` (in-process) | laptop | none |
| **Docker / laptop** | Qdrant server in Docker | laptop | localhost loopback |
| **Cloud** | Qdrant Cloud (free tier) | laptop → cloud | internet RTT |

They differ on **hardware, network path, and architecture at once** — so the table isn't a
single leaderboard. Edge-laptop vs Edge-Pi isolates *hardware*; Edge-laptop vs Docker-laptop
isolates *architecture* (in-process vs client/server) on identical silicon; Cloud shows the
real cost of a network round-trip.

## How it works

`run.py` runs one cell end-to-end with identical logic; only the backend differs
(`bench/backends.py` adapts Qdrant Edge's `EdgeShard` API and `qdrant-client` to one
interface). Embedding (FastEmbed `BAAI/bge-small-en-v1.5`, 384-dim) is a **separate timed
stage** — it's a hardware signal and is excluded from the DB upload/query numbers. All cells
embed the same 10k AG-News texts (`bench/corpus.py`).

Fairness controls: identical vectors everywhere; server cells force HNSW indexing
(`indexing_threshold=1`) and wait until indexed so they don't silently brute-force a 10k
collection; query latency is single-threaded (clean percentiles), throughput is a separate loop.

## Reproduce

```bash
# Laptop cells (Python 3.12 venv — onnxruntime has no 3.14 wheels yet)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-server.txt   # qdrant-client + edge + fastembed

# Docker cell (isolated container on alt ports)
docker run -d --name qdrant-bench -p 6543:6333 -p 6544:6334 qdrant/qdrant
.venv/bin/python run.py --backend server --name docker-laptop \
    --url http://localhost:6543 --embed-batch 32 --out results/docker-laptop.json

# Edge on the laptop
.venv/bin/python run.py --backend edge --name edge-laptop --embed-batch 32 \
    --out results/edge-laptop.json

# Cloud (creds in .env — see .env.example)
set -a; . ./.env; set +a
.venv/bin/python run.py --backend server --name cloud --url "$QDRANT_URL" \
    --api-key "$QDRANT_API_KEY" --embed-batch 32 --out results/cloud.json

# Edge on the Pi (64-bit Raspberry Pi OS; embed-batch 32 keeps memory ~700MB on 4GB)
#   deploy bench/, run.py, data/corpus_ag_news.txt to the Pi, then:
~/edge-bench-venv/bin/python run.py --backend edge --name edge-pi --embed-batch 32 \
    --out results/edge-pi.json

# Merge into RESULTS.md
.venv/bin/python summarize.py
```

> **Embedding batch size matters on the Pi.** At batch 256 the padded transformer
> activations OOM a 4GB Pi mid-embed; `--embed-batch 32` keeps peak RSS ~500–700MB and is
> also faster (less padding waste). The same batch is used everywhere for a fair embed comparison.

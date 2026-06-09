# Results — Qdrant Edge vs Docker vs Cloud

Dataset: **10000 docs × 384-dim** (FastEmbed `BAAI/bge-small-en-v1.5`), 1000 queries, limit=10, upsert batch 256.

> Not an official benchmark — a quick, honest comparison. The cells differ on **hardware, network path, and architecture at once**, so read each number with its context, not as a single leaderboard.

## Performance

| Cell | Hardware | Network | Embed docs/s | Create (s) | Upload vec/s | Query p50 (ms) | Query p95 (ms) | QPS | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|
| Edge / Pi | aarch64 | none (in-process) | 4.2 | 0.0048 | 8389.7 | 1.099 | 1.339 | 916.4 | 840.8 |
| Edge / laptop | arm64 | none (in-process) | 188.2 | 0.0964 | 80891.5 | 0.072 | 0.097 | 14154.9 | 853.3 |
| Docker / laptop | arm64 | localhost loopback | 186.1 | 0.0869 | 5778.9 | 1.968 | 2.968 | 451.2 | 884.8 |
| Cloud | arm64 | internet RTT | 167.9 | 0.6448 | 199.8 | 102.802 | 112.139 | 9.7 | 879.2 |

## What each number means

- **Embed docs/s** — on-device FastEmbed throughput; a *hardware* signal (Pi CPU vs laptop CPU), measured separately and excluded from the DB numbers below.
- **Upload / Query** — pure DB-engine work on identical vectors.
- **Edge cells have zero network cost** (in-process); **Docker** adds localhost loopback; **Cloud** is dominated by internet round-trip, not Qdrant compute.
- **Edge-laptop vs Edge-Pi** isolates hardware. **Edge-laptop vs Docker-laptop** isolates architecture (in-process vs client/server) on identical silicon.

## Takeaways

- **Architecture (same Mac): in-process beats client/server by ~27× on query latency** — Edge 0.072 ms vs Docker 1.968 ms. The cost is the network/serialization layer, not the engine.
- **Hardware (same in-process engine): the laptop is ~15× faster than the Pi** on query latency (0.072 ms vs 1.099 ms) — the raw silicon gap.
- **Headline: Edge on the Pi out-queries Docker on the laptop** (1.099 ms / 916.4 QPS vs 1.968 ms / 451.2 QPS) — despite far weaker hardware. Removing the network layer matters more than CPU for serving queries.
- **Cloud query latency (102.802 ms) is ~1428× Edge's** — almost entirely the internet round-trip to the region, not Qdrant compute. Great for managed scale; not for latency-critical local inference.
- **On the Pi, embedding is the bottleneck, not the database**: FastEmbed runs at 4.2/s (~40 min for 10k) vs the laptop's 188.2/s, while Edge upload/query stay fast. For embedded AI, plan around on-device embedding cost.

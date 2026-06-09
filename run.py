"""Run one benchmark cell (edge or server) and write a results JSON.

Identical logic across cells; only the backend differs. Stages are timed separately:
embedding (hardware signal) is NOT counted in the DB upload/query numbers.

Examples:
  # Edge, on the Pi or laptop
  python run.py --backend edge --name edge-pi --out results/edge-pi.json

  # Docker on laptop
  python run.py --backend server --name docker-laptop \
      --url http://localhost:6333 --out results/docker-laptop.json

  # Cloud
  python run.py --backend server --name cloud --url "$QDRANT_URL" \
      --api-key "$QDRANT_API_KEY" --out results/cloud.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from bench import backends, corpus, embed, metrics


def host_info() -> dict:
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["edge", "server"], required=True)
    ap.add_argument("--name", required=True, help="cell label, e.g. edge-pi")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--queries", type=int, default=1_000)
    ap.add_argument("--batch", type=int, default=256, help="upsert batch size")
    ap.add_argument("--embed-batch", type=int, default=embed.DEFAULT_EMBED_BATCH,
                    help="embedding batch size (small to bound memory on the Pi)")
    ap.add_argument("--embed-threads", type=int, default=None,
                    help="onnxruntime intra-op threads for embedding")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--dim", type=int, default=384)
    # edge
    ap.add_argument("--shard-dir", default="/tmp/edge-bench-shard")
    # server
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--collection", default="edge_bench")
    ap.add_argument("--prefer-grpc", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = {
        "cell": args.name,
        "backend": args.backend,
        "host": host_info(),
        "params": {
            "n": args.n,
            "queries": args.queries,
            "batch": args.batch,
            "limit": args.limit,
            "dim": args.dim,
        },
    }

    # --- 1. corpus (identical text everywhere) ---
    docs = corpus.load_corpus(args.n)
    queries = corpus.load_queries(args.n, args.queries)

    # --- 2. embedding (separate timed stage; excluded from DB numbers) ---
    emb = embed.Embedder(threads=args.embed_threads)
    doc_vecs, doc_embed_s = emb.embed(docs, batch_size=args.embed_batch)
    q_vecs, q_embed_s = emb.embed(queries, batch_size=args.embed_batch)
    result["embedding"] = {
        "model": embed.MODEL_NAME,
        "embed_batch": args.embed_batch,
        "doc_embed_s": round(doc_embed_s, 3),
        "docs_per_s": round(args.n / doc_embed_s, 1),
        "query_embed_s": round(q_embed_s, 3),
    }
    print(f"[{args.name}] embedded {args.n} docs in {doc_embed_s:.1f}s "
          f"({args.n / doc_embed_s:.0f}/s)")

    # --- 3. backend ---
    if args.backend == "edge":
        be = backends.EdgeBackend(args.shard_dir, dim=args.dim)
    else:
        be = backends.ServerBackend(
            url=args.url, api_key=args.api_key, collection=args.collection,
            dim=args.dim, prefer_grpc=args.prefer_grpc,
        )

    # --- 4. create collection ---
    t = time.perf_counter()
    be.create_collection()
    result["create_s"] = round(time.perf_counter() - t, 4)

    # --- 5. upload (batched) + 6. index, under resource sampling ---
    ids = list(range(args.n))
    payloads = [{"text": docs[i][:120]} for i in range(args.n)]
    batch_lat: list[float] = []
    with metrics.ResourceSampler() as rs_up:
        t0 = time.perf_counter()
        for s in range(0, args.n, args.batch):
            e = min(s + args.batch, args.n)
            bt = time.perf_counter()
            be.upsert(ids[s:e], doc_vecs[s:e], payloads[s:e])
            batch_lat.append((time.perf_counter() - bt) * 1000)
        upload_s = time.perf_counter() - t0
        ti = time.perf_counter()
        indexed = be.ensure_indexed(args.n)
        index_s = time.perf_counter() - ti
    result["upload"] = {
        "total_s": round(upload_s, 3),
        "vectors_per_s": round(args.n / upload_s, 1),
        "batch_ms": metrics.percentiles(batch_lat),
        "index_s": round(index_s, 3),
        "indexed_ok": bool(indexed),
    }
    result["resources_upload"] = rs_up.summary()
    result["count_after"] = be.count()
    print(f"[{args.name}] uploaded {args.n} in {upload_s:.1f}s "
          f"({args.n / upload_s:.0f}/s), index {index_s:.1f}s, count={result['count_after']}")

    # --- 7. query latency (single-threaded) ---
    lat: list[float] = []
    with metrics.ResourceSampler() as rs_q:
        for i in range(args.queries):
            qv = q_vecs[i % len(q_vecs)]
            t = time.perf_counter()
            be.query(qv, args.limit)
            lat.append((time.perf_counter() - t) * 1000)
    result["query_latency_ms"] = metrics.percentiles(lat)
    result["resources_query"] = rs_q.summary()

    # --- 8. query throughput (QPS) ---
    t0 = time.perf_counter()
    for i in range(args.queries):
        be.query(q_vecs[i % len(q_vecs)], args.limit)
    result["query_throughput_qps"] = round(args.queries / (time.perf_counter() - t0), 1)

    be.close()
    print(f"[{args.name}] query p50={result['query_latency_ms']['p50_ms']}ms "
          f"p95={result['query_latency_ms']['p95_ms']}ms "
          f"qps={result['query_throughput_qps']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[{args.name}] wrote {out}")


if __name__ == "__main__":
    main()

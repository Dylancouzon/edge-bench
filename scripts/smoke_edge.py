"""Smoke test: validate the real qdrant-edge API + FastEmbed on THIS machine.

Run on the Pi (and the laptop) before trusting EdgeBackend. The Edge API is beta and
may differ from the docs, so this prints the actual module surface and exercises the
full create -> upsert -> optimize -> query loop, then a FastEmbed embedding.
"""
import tempfile
import time

import qdrant_edge as qe

print("qdrant_edge version:", getattr(qe, "__version__", "unknown"))
print("public symbols:", sorted(x for x in dir(qe) if not x.startswith("_")))

from qdrant_edge import (  # noqa: E402
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    Point,
    Query,
    QueryRequest,
    UpdateOperation,
)

VEC = "vec"
shard_dir = tempfile.mkdtemp(prefix="edge-smoke-")
cfg = EdgeConfig(vectors={VEC: EdgeVectorParams(size=4, distance=Distance.Cosine)})
shard = EdgeShard.create(shard_dir, cfg)

shard.update(
    UpdateOperation.upsert_points(
        [
            Point(id=1, vector={VEC: [0.1, 0.2, 0.3, 0.4]}, payload={"k": "a"}),
            Point(id=2, vector={VEC: [0.9, 0.8, 0.7, 0.6]}, payload={"k": "b"}),
        ]
    )
)
shard.flush()
shard.optimize()

res = shard.query(
    QueryRequest(
        query=Query.Nearest([0.1, 0.2, 0.3, 0.4], using=VEC),
        limit=2,
        with_vector=False,
        with_payload=True,
    )
)
print("query result type:", type(res))
print("query result:", res)
try:
    from qdrant_edge import CountRequest

    print("count:", shard.count(CountRequest()))
except Exception as e:  # noqa: BLE001 — probing the real signature
    print("count() probe:", type(e).__name__, e)

from fastembed import TextEmbedding  # noqa: E402

t0 = time.perf_counter()
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
load_s = time.perf_counter() - t0
t1 = time.perf_counter()
vecs = list(model.embed(["hello world", "vector search on the edge"]))
embed_s = time.perf_counter() - t1
print(f"fastembed: dim={len(vecs[0])} load={load_s:.2f}s embed2={embed_s:.2f}s")
print("SMOKE_OK")

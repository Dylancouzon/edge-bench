"""Backend adapters: one interface over Qdrant Edge (in-process) and Qdrant server.

Both expose: create_collection() -> upsert(ids, vectors, payloads) ->
ensure_indexed(total) -> query(vector, limit) -> count() -> close().

API confirmed against qdrant-edge-py 0.7.2 and qdrant-client.
"""
from __future__ import annotations

import os
import shutil
import time

VEC = "vec"  # single named vector used everywhere


class EdgeBackend:
    kind = "edge"

    def __init__(self, shard_dir: str, dim: int, distance: str = "cosine"):
        from qdrant_edge import Distance, EdgeConfig, EdgeShard, EdgeVectorParams

        dist = getattr(Distance, {"cosine": "Cosine", "dot": "Dot", "euclid": "Euclid"}[distance])
        self._EdgeShard = EdgeShard
        self._cfg = EdgeConfig(vectors={VEC: EdgeVectorParams(size=dim, distance=dist)})
        self._shard_dir = shard_dir
        self.shard = None

    def create_collection(self) -> None:
        if os.path.exists(self._shard_dir):
            shutil.rmtree(self._shard_dir)
        os.makedirs(self._shard_dir, exist_ok=True)
        self.shard = self._EdgeShard.create(self._shard_dir, self._cfg)

    def upsert(self, ids, vectors, payloads) -> None:
        from qdrant_edge import Point, UpdateOperation

        pts = [
            Point(id=int(i), vector={VEC: _tolist(v)}, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.shard.update(UpdateOperation.upsert_points(pts))

    def ensure_indexed(self, total: int) -> bool:
        # In-process: persist + build indexes / GC. Synchronous.
        self.shard.flush()
        self.shard.optimize()
        return True

    def query(self, vector, limit: int):
        from qdrant_edge import Query, QueryRequest

        res = self.shard.query(
            QueryRequest(
                query=Query.Nearest(_tolist(vector), using=VEC),
                limit=limit,
                with_vector=False,
                with_payload=False,
            )
        )
        return [(p.id, p.score) for p in res]

    def count(self) -> int:
        from qdrant_edge import CountRequest

        return self.shard.count(CountRequest())

    def close(self) -> None:
        if self.shard is not None:
            self.shard.flush()


class ServerBackend:
    """Standard Qdrant server (Docker or Cloud) via qdrant-client."""

    kind = "server"

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        collection: str = "edge_bench",
        dim: int = 384,
        distance: str = "cosine",
        force_index: bool = True,
        prefer_grpc: bool = False,
    ):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc, timeout=120)
        self.collection = collection
        self.dim = dim
        self.distance = getattr(
            models.Distance, {"cosine": "COSINE", "dot": "DOT", "euclid": "EUCLID"}[distance]
        )
        self.force_index = force_index

    def create_collection(self) -> None:
        m = self.models
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        # Force HNSW even below the default 20k indexing_threshold so every cell
        # exercises the same (indexed) query path.
        opt = m.OptimizersConfigDiff(indexing_threshold=1) if self.force_index else None
        self.client.create_collection(
            self.collection,
            vectors_config=m.VectorParams(size=self.dim, distance=self.distance),
            optimizers_config=opt,
        )

    def upsert(self, ids, vectors, payloads) -> None:
        m = self.models
        pts = [
            m.PointStruct(id=int(i), vector=_tolist(v), payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(self.collection, points=pts, wait=True)

    def ensure_indexed(self, total: int, timeout_s: float = 180.0) -> bool:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            info = self.client.get_collection(self.collection)
            if (info.indexed_vectors_count or 0) >= total:
                return True
            time.sleep(0.5)
        return False

    def query(self, vector, limit: int):
        res = self.client.query_points(
            self.collection, query=_tolist(vector), limit=limit, with_payload=False
        ).points
        return [(p.id, p.score) for p in res]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def close(self) -> None:
        self.client.close()


def _tolist(v):
    return v.tolist() if hasattr(v, "tolist") else list(v)

"""FastEmbed embedding stage.

Timed SEPARATELY from DB operations — embedding throughput is its own (hardware)
result and must not be counted in upload/query numbers. The model is deterministic,
so the vectors are effectively identical across machines.
"""
from __future__ import annotations

import time

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim
DIM = 384
# Peak embedding memory scales with batch_size x padded_seq_len x hidden x layers.
# A 4GB Pi OOMs at batch 256 (multi-GB activation tensors), so keep this small and
# identical across cells (embedding is a separate, fairly-measured stage anyway).
DEFAULT_EMBED_BATCH = 32


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, threads: int | None = None):
        # NOTE: construction downloads the model on first run (cached afterwards).
        # We construct outside the timed region so model download/load is excluded.
        self.model_name = model_name
        kwargs = {}
        if threads is not None:
            kwargs["threads"] = threads
        self.model = TextEmbedding(model_name=model_name, **kwargs)

    def embed(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_EMBED_BATCH,
    ) -> tuple[np.ndarray, float]:
        """Return (vectors[N, DIM] float32, elapsed_seconds).

        Uses FastEmbed's default in-process path (onnxruntime multi-threaded), which
        is fast and does not fork extra model copies. Memory is bounded by keeping
        batch_size small rather than by forcing the slow multiprocessing pool path.
        """
        t0 = time.perf_counter()
        vecs = list(self.model.embed(texts, batch_size=batch_size))
        elapsed = time.perf_counter() - t0
        arr = np.asarray(vecs, dtype=np.float32)
        return arr, elapsed

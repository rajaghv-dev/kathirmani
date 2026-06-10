"""NvidiaEmbeddingPlugin — the visual+text embedding model behind the
`EmbeddingPlugin` contract (master plan Phase 8 / spec/11).

Wraps **nvidia/C-RADIOv4-H** (transformers, PyTorch) for image/frame embeddings
*and* a text-embedding path used to embed event/observation `text_repr` and the
search query. The real model load is **lazy and optional**: if transformers or
the weights are missing — or anything else goes wrong — `infer()` transparently
falls back to `fake_infer()`, a deterministic hash-seeded vector. That keeps the
indexer, the worker, the 10-point plugin test (A11) and CI runnable with no GPU /
no weights / no DB.

Task: `embedding`  (contract: clip|text|image|event -> vector + metadata).
infer(request): one of {text}, {image}, {clip_path}  ->
  {vector: list[float](768), dim: 768, model_name: str, faked: bool}

DIM NOTE (spec/11 + db/schema.sql): the `embeddings` column is `vector(768)`.
C-RADIOv4-H's native summary/feature dim is NOT 768 (RADIO summary features are
~1280/3072 depending on the head). Rather than migrate the column (which would
drop the pgvector index and has no single "native" dim — it varies by head), we
keep the table at 768 and map every real RADIO feature vector down to it with a
proper **Gaussian random-projection head** (`_project`, Johnson–Lindenstrauss):
a fixed N(0, 1/target) matrix, deterministic per (source_dim → target_dim) shape,
that *approximately preserves cosine distance* — unlike the old truncate/pad which
discarded ~40% of the features. Identical inputs still map to identical vectors
(projection is linear), so the fake-path determinism contract holds. The target
dim is config-driven (`vector_dim` in configs/models.yaml, default EMBED_DIM=768).
To move to RADIO's native dim instead, set `vector_dim` + add a column migration;
the projection becomes a no-op when source_dim == target_dim.
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any

# --- repo imports (no editable install): put repo root + model-plugins on path -
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "model-plugins")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base.plugin import Health, ModelPlugin, PluginConfig      # noqa: E402

# The db `embeddings` column is vector(768); all vectors are coerced to this.
EMBED_DIM = 768
DEFAULT_MODEL_ID = "nvidia/C-RADIOv4-H"

# Fixed random-projection matrices, one per (source_dim, target_dim) shape. Built
# once, deterministic from the shape (see `_projection_matrix`), shared process-wide.
_PROJECTION_CACHE: dict[tuple[int, int], Any] = {}


def default_config(profile: str = "nvidia_gb10_retail_balanced") -> PluginConfig:
    """A PluginConfig matching the `embedding` task of the active profile, so the
    indexer/worker can construct the plugin without a config loader (mirrors
    configs/models.yaml: plugin nvidia_embedding, runtime transformers)."""
    return PluginConfig(
        task="embedding",
        plugin="nvidia_embedding",
        model_id=DEFAULT_MODEL_ID,
        runtime="transformers",
        endpoint="local",
        profile=profile,
        params={"dim": EMBED_DIM},        # = configs/models.yaml embedding.vector_dim
    )


class NvidiaEmbeddingPlugin(ModelPlugin):
    """C-RADIOv4-H visual+text embedder for the `embedding` task; fake-infer fallback."""

    def __init__(self, config: PluginConfig | None = None) -> None:
        super().__init__(config or default_config())
        self._model = None                       # lazily-loaded transformers model
        self._processor = None
        self._loaded = False
        self._load_error = ""
        self._dim = int(self.config.params.get("dim", EMBED_DIM))
        # lightweight metric counters (mirrored to prometheus in infer())
        self._n_requests = 0
        self._n_fake = 0
        self._last_latency_ms = 0.0

    # ---- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Try to load C-RADIOv4-H. Idempotent and *non-fatal*: on any failure we
        record the reason and stay in fake-infer mode (no GPU/weights needed)."""
        if self._loaded or self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoImageProcessor  # heavy; lazy
            mid = self.config.model_id
            self._model = AutoModel.from_pretrained(mid, trust_remote_code=True)
            try:
                self._processor = AutoImageProcessor.from_pretrained(
                    mid, trust_remote_code=True)
            except Exception:
                self._processor = None            # text-only path still works
            self._loaded = True
            self._load_error = ""
        except Exception as e:                    # transformers/weights/GPU absent
            self._model = None
            self._loaded = False
            self._load_error = f"{type(e).__name__}: {e}"

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        try:                                      # free GPU memory if torch present
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def health(self) -> Health:
        if self._loaded and self._model is not None:
            return Health(ok=True, detail=f"{self.config.model_id} loaded")
        detail = "fake-infer mode" + (f" ({self._load_error})" if self._load_error else "")
        return Health(ok=True, detail=detail)     # fake-infer keeps the worker usable

    def metrics(self) -> dict[str, float]:
        return {
            "requests_total": float(self._n_requests),
            "fake_infer_total": float(self._n_fake),
            "last_latency_ms": float(self._last_latency_ms),
            "model_loaded": 1.0 if self._loaded else 0.0,
            "dim": float(self._dim),
        }

    # ---- inference ---------------------------------------------------------
    def fake_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Deterministic, GPU-free stub (A11 #3). Hash the request's content into a
        reproducible unit-norm 768-d vector. Same text/image/clip -> same vector, so
        cosine similarity is meaningful (identical inputs score 1.0) and tests are
        stable. Different inputs -> different, near-orthogonal vectors."""
        seed = self._seed_of(request)
        vec = self._hash_vector(seed, self._dim)
        return self._package(vec, faked=True)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Embed one of {text}, {image} (PIL/path), {clip_path}. Falls back to
        fake_infer if the model isn't loaded or inference raises. Always returns
        {vector, dim, model_name, faked}."""
        self._n_requests += 1
        start = time.perf_counter()
        if not self._loaded:
            self.load()                           # lazy, non-fatal
        try:
            if self._model is None:
                raise RuntimeError("model unavailable")
            vec = self._embed_real(request)
            out = self._package(vec, faked=False)
        except Exception as e:
            out = self.fake_infer(request)
            out["error"] = f"fallback: {type(e).__name__}: {e}"
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        out["latency_ms"] = self._last_latency_ms
        self._emit_prometheus(out, faked=out.get("faked", False))
        return out

    # ---- real C-RADIO path (only exercised when weights are present) -------
    def _embed_real(self, request: dict[str, Any]) -> list[float]:
        """Run C-RADIOv4-H. RADIO is image-native; text comes via its CLIP-style
        text path when exposed. Returns a raw feature vector (any dim); `_to_dim`
        coerces it to EMBED_DIM. Kept defensive — model APIs vary by revision."""
        import torch

        text = request.get("text")
        if text is not None and self._has_text_head():
            feats = self._radio_text(text)
        else:
            image = self._load_image(request)
            inputs = (self._processor(images=image, return_tensors="pt")
                      if self._processor else None)
            with torch.no_grad():
                out = (self._model(**inputs) if inputs is not None
                       else self._model(image))
            # RADIO returns (summary, spatial) or an object with .summary/.pooler
            summary = getattr(out, "summary", None)
            if summary is None and isinstance(out, (tuple, list)):
                summary = out[0]
            if summary is None:
                summary = getattr(out, "pooler_output", None)
            feats = summary.flatten().float().cpu().tolist()
        return self._to_dim([float(x) for x in feats])

    def _has_text_head(self) -> bool:
        return any(hasattr(self._model, a) for a in ("encode_text", "get_text_features"))

    def _radio_text(self, text: str):
        if hasattr(self._model, "encode_text"):
            t = self._model.encode_text(text)
        else:
            t = self._model.get_text_features(text)
        return t.flatten().float().cpu().tolist()

    def _load_image(self, request: dict[str, Any]):
        """Resolve a PIL image from {image} (PIL or path) or the first frame of
        {clip_path}. Only called on the real path."""
        from PIL import Image

        image = request.get("image")
        if image is not None:
            return image if hasattr(image, "size") else Image.open(image).convert("RGB")
        clip = request.get("clip_path")
        if clip:
            try:                                  # grab first frame via PyAV
                import av
                with av.open(clip) as container:
                    for frame in container.decode(video=0):
                        return frame.to_image()
            except Exception:
                return Image.open(clip).convert("RGB")  # maybe it's an image file
        raise RuntimeError("no text/image/clip_path in request")

    # ---- dim coercion + deterministic fake vector --------------------------
    def _to_dim(self, vec: list[float]) -> list[float]:
        """Coerce an arbitrary-length feature vector to EMBED_DIM so it fits the db
        `vector(N)` column. When the source dim already matches, just normalise;
        otherwise apply the Gaussian random-projection head (`_project`), which
        approximately preserves cosine distance (JL lemma) instead of the old
        lossy truncate/pad. Always L2-normalised so cosine == dot product."""
        n = self._dim
        if len(vec) == n:
            return self._l2norm(vec)
        projected = self._project(vec, n)
        return self._l2norm(projected)

    @staticmethod
    def _projection_matrix(source_dim: int, target_dim: int):
        """A fixed (target_dim x source_dim) Gaussian random-projection matrix,
        deterministic per shape and cached, with entries ~ N(0, 1/target_dim).
        Deterministic ⇒ the same source vector always maps to the same output, so
        cosine similarity is preserved across calls and the fake-path determinism
        contract still holds. JL guarantees pairwise distances are ~preserved."""
        key = (source_dim, target_dim)
        mat = _PROJECTION_CACHE.get(key)
        if mat is None:
            import numpy as np
            # Seed from the shape only — stable across processes/runs (no RNG state
            # leak), so projections are reproducible everywhere.
            rng = np.random.default_rng(seed=(source_dim * 1_000_003 + target_dim))
            mat = rng.standard_normal((target_dim, source_dim)).astype("float64")
            mat /= math.sqrt(target_dim)
            _PROJECTION_CACHE[key] = mat
        return mat

    def _project(self, vec: list[float], target_dim: int) -> list[float]:
        """Random-project `vec` to `target_dim`. Falls back to truncate/pad only if
        numpy is somehow unavailable (base dep, so this is belt-and-suspenders) —
        the coercion must never raise and break indexing."""
        try:
            import numpy as np
            mat = self._projection_matrix(len(vec), target_dim)
            out = mat @ np.asarray(vec, dtype="float64")
            return [float(x) for x in out]
        except Exception:                          # numpy missing → degrade, don't crash
            if len(vec) > target_dim:
                return vec[:target_dim]
            return vec + [0.0] * (target_dim - len(vec))

    @staticmethod
    def _seed_of(request: dict[str, Any]) -> str:
        """Stable content key for the fake path: prefer text, then clip_path, then
        a repr of the image, then the whole request."""
        if request.get("text") is not None:
            return "text:" + str(request["text"])
        if request.get("clip_path"):
            return "clip:" + str(request["clip_path"])
        if request.get("image") is not None:
            img = request["image"]
            return "image:" + str(getattr(img, "tobytes", lambda: repr(img))() if
                                  hasattr(img, "tobytes") else img)
        return "req:" + repr(sorted(request.items()))

    @staticmethod
    def _hash_vector(seed: str, dim: int) -> list[float]:
        """Deterministic unit-norm vector from `seed`. Expand a SHA stream into
        `dim` floats in [-1, 1) so distinct seeds are near-orthogonal."""
        out: list[float] = []
        counter = 0
        while len(out) < dim:
            h = hashlib.sha256(f"{seed}|{counter}".encode()).digest()
            for i in range(0, len(h), 2):
                if len(out) >= dim:
                    break
                u = int.from_bytes(h[i:i + 2], "big") / 65535.0   # [0, 1]
                out.append(u * 2.0 - 1.0)                          # [-1, 1)
            counter += 1
        return NvidiaEmbeddingPlugin._l2norm(out)

    @staticmethod
    def _l2norm(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _package(self, vec: list[float], faked: bool) -> dict[str, Any]:
        return {
            "vector": vec,
            "dim": len(vec),
            "model_name": self.config.model_id,
            "faked": faked,
            "error": None,
        }

    # ---- prometheus mirror (best-effort) -----------------------------------
    def _emit_prometheus(self, out: dict[str, Any], faked: bool) -> None:
        if faked:
            self._n_fake += 1
        try:
            from base import metrics as M
            lbl = M.labels(self.config)
            status = "fake" if faked else ("error" if out.get("error") else "ok")
            M.model_requests_total.labels(**lbl, status=status).inc()
            M.model_latency_ms.labels(**lbl).observe(out.get("latency_ms", 0.0))
            if out.get("error") and not faked:
                M.model_errors_total.labels(**lbl, error_type="infer").inc()
        except Exception:
            pass                                  # metrics never break inference

"""NvidiaEmbeddingPlugin — must work with no GPU / no weights / no DB.

Covers the fake-infer essentials: a deterministic, unit-norm, 768-dim vector that
matches the db `embeddings` vector(768) column, plus the lifecycle methods.
"""
from __future__ import annotations

import math

from ai_workers.embedding_worker.plugin import EMBED_DIM, NvidiaEmbeddingPlugin, default_config


def test_fake_infer_is_768_dim_and_unit_norm():
    p = NvidiaEmbeddingPlugin()
    out = p.fake_infer({"text": "a person picked an item and walked away"})
    assert out["dim"] == EMBED_DIM == 768          # must match vector(768)
    assert len(out["vector"]) == 768
    assert out["faked"] is True
    assert out["model_name"] == "nvidia/C-RADIOv4-H"
    norm = math.sqrt(sum(v * v for v in out["vector"]))
    assert abs(norm - 1.0) < 1e-6                   # l2-normalized


def test_fake_infer_is_deterministic_and_content_sensitive():
    p = NvidiaEmbeddingPlugin()
    a1 = p.fake_infer({"text": "person walked away"})["vector"]
    a2 = p.fake_infer({"text": "person walked away"})["vector"]
    b = p.fake_infer({"text": "camera is blocked"})["vector"]
    assert a1 == a2                                 # same text -> same vector
    # identical inputs => cosine 1.0; different inputs => clearly lower
    cos_same = sum(x * y for x, y in zip(a1, a2))
    cos_diff = sum(x * y for x, y in zip(a1, b))
    assert cos_same > 0.999
    assert cos_diff < 0.5                           # near-orthogonal


def test_projection_head_coerces_native_dim_and_is_unit_norm():
    """RADIO emits a non-768 feature vector; the projection head maps it to a
    unit-norm 768-d vector that fits the db column."""
    p = NvidiaEmbeddingPlugin()
    native = [float((i * 7) % 13) - 6.0 for i in range(1280)]   # ~RADIO summary dim
    out = p._to_dim(native)
    assert len(out) == 768
    norm = math.sqrt(sum(v * v for v in out))
    assert abs(norm - 1.0) < 1e-6


def test_projection_head_is_deterministic():
    """Same source vector -> same projected vector across calls and instances, so
    cosine similarity is stable (the fake-path determinism contract relies on this)."""
    a = NvidiaEmbeddingPlugin()._to_dim([float(i % 5) for i in range(1280)])
    b = NvidiaEmbeddingPlugin()._to_dim([float(i % 5) for i in range(1280)])
    assert a == b
    cos = sum(x * y for x, y in zip(a, b))
    assert cos > 0.999


def test_projection_head_approximately_preserves_cosine():
    """Johnson-Lindenstrauss: a near-duplicate source pair stays far more similar
    after projection than an unrelated pair — what truncate/pad could not guarantee."""
    p = NvidiaEmbeddingPlugin()
    base = [math.sin(i * 0.013) for i in range(1280)]
    near = [v + 0.01 * math.cos(i * 0.31) for i, v in enumerate(base)]   # tiny perturbation
    far = [math.cos(i * 0.077) for i in range(1280)]
    pb, pn, pf = p._to_dim(base), p._to_dim(near), p._to_dim(far)
    cos_near = sum(x * y for x, y in zip(pb, pn))
    cos_far = sum(x * y for x, y in zip(pb, pf))
    assert cos_near > cos_far
    assert cos_near > 0.9                            # near-duplicate stays close


def test_infer_falls_back_to_fake_without_model():
    p = NvidiaEmbeddingPlugin()
    out = p.infer({"text": "anything"})             # no weights -> fake path
    assert out["dim"] == 768
    assert out["faked"] is True
    assert "latency_ms" in out


def test_text_clip_image_paths_all_produce_768():
    p = NvidiaEmbeddingPlugin()
    for req in ({"text": "x"}, {"clip_path": "/nope.mp4"}, {"image": "/nope.png"}):
        out = p.infer(req)
        assert len(out["vector"]) == 768


def test_lifecycle_and_health_metrics():
    p = NvidiaEmbeddingPlugin()
    p.load()                                        # non-fatal even without weights
    h = p.health()
    assert h.ok is True                             # fake-infer mode is still healthy
    m = p.metrics()
    assert set(m) >= {"requests_total", "fake_infer_total", "model_loaded", "dim"}
    assert m["dim"] == 768.0
    p.infer({"text": "y"})
    assert p.metrics()["requests_total"] >= 1.0
    p.unload()
    assert p.metrics()["model_loaded"] == 0.0


def test_default_config_matches_embedding_task():
    cfg = default_config()
    assert cfg.task == "embedding"
    assert cfg.plugin == "nvidia_embedding"
    assert cfg.runtime == "transformers"
    assert cfg.model_id == "nvidia/C-RADIOv4-H"

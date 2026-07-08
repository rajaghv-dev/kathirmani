"""VLM clip-reasoning plugins behind the `VLMClipReasoningPlugin` contract
(spec/11, schemas.py task `vlm_clip_reasoning`).

Two concrete plugins ship here; the active profile picks one (host.py):

  * ``NvidiaVlmPlugin`` (plugin name ``nvidia_openai_compatible_vlm``) — wraps
    ``nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1`` via transformers. The real model
    load is **lazy and optional**: missing transformers / weights / GPU -> the
    plugin stays in fake-infer mode (A11 #3) so the worker + the 10-point test run
    on any box.
  * ``QwenBaselinePlugin`` (plugin name ``vlm_qwen_baseline``) — the comparison-arm
    baseline (configs/models.yaml `research_qwen_baseline`); thin wrapper that may
    import ``src/kathirmani/qwen_vl.py`` lazily, else falls back to fake.

Task: ``vlm_clip_reasoning`` (contract: clip_path + prompt + hypothesis ->
structured JSON). ``infer(request)`` input ``{clip_path, prompt, event|hypothesis}``
-> a **VLMVerification** dict (§8.4) wrapped with a ``model_run`` block:

    infer -> {"verification": {<VLMVerification>}, "raw_output": "...",
              "parse_success": bool, "model_run": {ttft_ms, output_tokens,
              tokens_per_sec, latency_ms, faked, error, ...}}
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

from model_plugins.base.plugin import Health, ModelPlugin, PluginConfig      # noqa: E402
from model_plugins.base.schemas import VLMVerification                       # noqa: E402

from .prompts import DEFAULT_PACK, render
from .parser import parse_verification

# VLM weights location (configs/models.yaml model_id, fetched into models/).
NVIDIA_VLM_ID = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
NVIDIA_VLM_PATH = _REPO_ROOT / "models" / "Llama-3.1-Nemotron-Nano-VL-8B-V1"


def _patch_tied_weights_compat() -> None:
    """transformers 5.9 reads `model.all_tied_weights_keys` (a {target:source} dict)
    that it normally sets in its own init path — which this model's custom __init__
    bypasses, raising AttributeError on load. Add a per-instance default (empty dict,
    settable) so normal models are unaffected and this one loads. Idempotent."""
    from transformers.modeling_utils import PreTrainedModel as P
    if getattr(P, "_kathir_tied_shim", False):
        return
    KEY = "all_tied_weights_keys"

    def _get(self):
        d = self.__dict__.get(KEY)
        if d is None:
            d = self.__dict__[KEY] = {}
        return d

    def _set(self, v):
        self.__dict__[KEY] = v

    if not isinstance(getattr(P, KEY, None), property):
        setattr(P, KEY, property(_get, _set))
    P._kathir_tied_shim = True


# --------------------------------------------------------------------------
# Shared base: prompt rendering, packaging, fake_infer, metrics mirror.
# --------------------------------------------------------------------------
class _VlmPluginBase(ModelPlugin):
    """Common machinery for the VLM plugins (the only difference is the real
    ``_run_model`` path; both share fake_infer, packaging, and metrics)."""

    def __init__(self, config: PluginConfig) -> None:
        super().__init__(config)
        self._model = None
        self._processor = None
        self._loaded = False
        self._load_error = ""
        self._prompt_pack = str(config.params.get("prompt_pack", DEFAULT_PACK))
        self._max_output_tokens = int(config.params.get("max_output_tokens", 512))
        # lightweight counters (also mirrored to prometheus in infer()).
        self._n_requests = 0
        self._n_fake = 0
        self._n_parse_ok = 0
        self._last_latency_ms = 0.0

    # ---- prompt -----------------------------------------------------------
    def build_prompt(self, request: dict[str, Any]) -> str:
        """Use the caller-supplied prompt if present, else render the configured
        pack against the event/hypothesis (so the worker need not know prompts)."""
        if request.get("prompt"):
            return str(request["prompt"])
        event = request.get("event") or request.get("hypothesis") or request
        return render(event, self._prompt_pack)

    # ---- lifecycle (overridden by subclasses for the real load) -----------
    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def health(self) -> Health:
        if self._loaded and self._model is not None:
            return Health(ok=True, detail=f"{self.config.model_id} loaded")
        detail = "fake-infer mode" + (f" ({self._load_error})" if self._load_error else "")
        return Health(ok=True, detail=detail)             # fake mode is still healthy

    def metrics(self) -> dict[str, float]:
        return {
            "requests_total": float(self._n_requests),
            "fake_infer_total": float(self._n_fake),
            "parse_ok_total": float(self._n_parse_ok),
            "last_latency_ms": float(self._last_latency_ms),
            "model_loaded": 1.0 if self._loaded else 0.0,
        }

    # ---- inference --------------------------------------------------------
    def fake_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Deterministic, schema-valid VLMVerification (A11 #3). Returns an
        'unclear' triage verdict referencing the flagged hypothesis so the
        end-to-end path runs without a GPU/weights."""
        event = request.get("event") or request.get("hypothesis") or request
        etype = (event.get("event_type") if isinstance(event, dict) else "") or "concern"
        v = VLMVerification(
            verdict="unclear",
            confidence=0.5,
            observed_actions=["a person is present near merchandise"],
            missing_evidence=["a clear view of the person's hands and any bag"],
            recommended_next_step="review the billing-counter footage for this window",
            structured_event_type=str(etype),
            explanation=("Fake-infer stub: no VLM weights loaded, so this is a "
                         f"placeholder verification for hypothesis '{etype}'."),
        )
        raw = "{}"                                        # represents 'no real output'
        return self._package(v, raw, parse_success=True, request=request, faked=True)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run VLM verification on ``request['clip_path']``. Falls back to
        fake_infer if the model isn't loaded, the clip is missing, or inference
        raises. Always returns the packaged dict (never throws)."""
        self._n_requests += 1
        start = time.perf_counter()
        if not self._loaded:
            self.load()                                   # lazy, non-fatal
        clip_path = request.get("clip_path") or request.get("evidence_path") or ""
        try:
            if self._model is None or not clip_path or not Path(clip_path).exists():
                raise RuntimeError("model unavailable or clip missing")
            prompt = self.build_prompt(request)
            raw, ttft_ms, n_tokens = self._run_model(clip_path, prompt)
            verification, raw_text, parse_ok = parse_verification(raw)
            out = self._package_dict(verification, raw_text, parse_ok, request,
                                     faked=False, ttft_ms=ttft_ms, output_tokens=n_tokens)
        except Exception as e:
            out = self.fake_infer(request)
            out["model_run"]["error"] = f"fallback: {type(e).__name__}: {e}"
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        out["model_run"]["latency_ms"] = round(self._last_latency_ms, 3)
        # tokens/sec from the decode portion (latency minus ttft), best-effort.
        mr = out["model_run"]
        decode_ms = max(self._last_latency_ms - (mr.get("ttft_ms") or 0.0), 1e-3)
        if mr.get("output_tokens"):
            mr["tokens_per_sec"] = round(mr["output_tokens"] / (decode_ms / 1000.0), 3)
        self._emit_prometheus(out)
        return out

    def _run_model(self, clip_path: str, prompt: str) -> tuple[str, float, int]:
        """Real inference -> (raw_text, ttft_ms, output_tokens). Subclass hook."""
        raise NotImplementedError

    # ---- packaging --------------------------------------------------------
    def _package(self, verification: VLMVerification, raw: str, parse_success: bool,
                 request: dict[str, Any], faked: bool) -> dict[str, Any]:
        return self._package_dict(asdict(verification), raw, parse_success, request,
                                  faked=faked, ttft_ms=0.0, output_tokens=0)

    def _package_dict(self, verification: dict[str, Any], raw: str, parse_success: bool,
                      request: dict[str, Any], faked: bool, ttft_ms: float,
                      output_tokens: int) -> dict[str, Any]:
        if parse_success:
            self._n_parse_ok += 1
        return {
            "verification": verification,
            "raw_output": raw,
            "parse_success": parse_success,
            "prompt_version": self._prompt_pack,
            "event_id": (request.get("event_id")
                         or (request.get("event") or {}).get("event_id")),
            "clip_path": request.get("clip_path") or request.get("evidence_path"),
            "model_run": {
                "model_profile_name": self.config.profile,
                "model_id": self.config.model_id,
                "task": self.config.task,
                "runtime": self.config.runtime,
                "faked": faked,
                "parse_success": parse_success,
                "ttft_ms": round(float(ttft_ms), 3),
                "output_tokens": int(output_tokens),
                "tokens_per_sec": 0.0,
                "error": None,
            },
        }

    # ---- prometheus mirror (best-effort) ----------------------------------
    def _emit_prometheus(self, out: dict[str, Any]) -> None:
        mr = out["model_run"]
        if mr.get("faked"):
            self._n_fake += 1
        try:
            from model_plugins.base import metrics as M
            lbl = M.labels(self.config)
            status = "fake" if mr.get("faked") else ("error" if mr.get("error") else "ok")
            M.model_requests_total.labels(**lbl, status=status).inc()
            M.model_latency_ms.labels(**lbl).observe(mr["latency_ms"])
            M.model_clips_processed_total.labels(**lbl).inc()
            if mr.get("ttft_ms"):
                M.model_ttft_ms.labels(**lbl).observe(mr["ttft_ms"])
            if mr.get("output_tokens"):
                M.model_output_tokens_total.labels(**lbl).inc(mr["output_tokens"])
            if mr.get("tokens_per_sec"):
                M.model_tokens_per_second.labels(**lbl).set(mr["tokens_per_sec"])
            M.model_json_parse_total.labels(
                **lbl, result="ok" if out["parse_success"] else "fail").inc()
            M.model_verdict_total.labels(
                **lbl, verdict=out["verification"]["verdict"]).inc()
            if mr.get("error") and not mr.get("faked"):
                M.model_errors_total.labels(**lbl, error_type="infer").inc()
        except Exception:
            pass                                          # metrics never break inference


# --------------------------------------------------------------------------
# NvidiaVlmPlugin — the production default plugin.
# --------------------------------------------------------------------------
def nvidia_default_config(profile: str = "nvidia_gb10_retail_balanced") -> PluginConfig:
    """PluginConfig matching the `vlm_clip_reasoning` task of the default profile."""
    return PluginConfig(
        task="vlm_clip_reasoning",
        plugin="nvidia_openai_compatible_vlm",
        model_id=NVIDIA_VLM_ID,
        runtime="transformers",
        endpoint="local",
        profile=profile,
        params={"prompt_pack": "retail_loss_v1", "max_output_tokens": 512,
                "max_frames": 16, "temperature": 0.1, "weights": str(NVIDIA_VLM_PATH)},
    )


class NvidiaVlmPlugin(_VlmPluginBase):
    """`nvidia_openai_compatible_vlm` — Nemotron-Nano-VL via transformers."""

    def __init__(self, config: PluginConfig | None = None) -> None:
        super().__init__(config or nvidia_default_config())
        self._weights = Path(self.config.params.get("weights", NVIDIA_VLM_PATH))
        self._tokenizer = None
        self._device = "cpu"
        self._dtype = None

    def load(self) -> None:
        """Load Nemotron-Nano-VL via its `trust_remote_code` API (see the model's
        examples.py): AutoModel(bf16, device_map) + AutoTokenizer + AutoImageProcessor.
        Idempotent + non-fatal — any failure leaves the plugin in fake-infer mode."""
        if self._loaded or self._model is not None:
            return
        try:
            if not self._weights.exists():
                raise FileNotFoundError(f"weights not found: {self._weights}")
            import torch
            from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
            _patch_tied_weights_compat()                  # transformers 5.9 ↔ model custom code
            cuda = torch.cuda.is_available()
            self._device = "cuda" if cuda else "cpu"
            self._dtype = torch.bfloat16 if cuda else torch.float32
            self._model = AutoModel.from_pretrained(
                str(self._weights), dtype=self._dtype, low_cpu_mem_usage=True,
                trust_remote_code=True, local_files_only=True,
                device_map=self._device,
                attn_implementation="eager").eval()  # model lacks FlashAttention-2
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._weights), trust_remote_code=True, local_files_only=True)
            self._processor = AutoImageProcessor.from_pretrained(
                str(self._weights), device=self._device, trust_remote_code=True,
                local_files_only=True)
            self._loaded = True
            self._load_error = ""
        except Exception as e:                            # transformers/weights/GPU absent
            self._model = self._processor = self._tokenizer = None
            self._loaded = False
            self._load_error = f"{type(e).__name__}: {e}"

    def _sample_frames(self, clip_path: str, n: int):
        """Sample up to `n` evenly-spaced RGB frames from a clip as PIL Images
        (an image is returned as-is). PyAV decode; preload bundled FFmpeg if present."""
        from PIL import Image
        if clip_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            return [Image.open(clip_path).convert("RGB")]
        try:
            from kathirmani.ffmpeg_preload import preload_av_ffmpeg
            preload_av_ffmpeg()
        except Exception:
            pass
        import av
        frames = []
        with av.open(clip_path) as c:
            vs = c.streams.video[0]
            total = vs.frames or 0
            # decode and keep every k-th frame to land ~n samples
            step = max(1, (total // n)) if total else 1
            for i, fr in enumerate(c.decode(vs)):
                if i % step == 0:
                    frames.append(Image.fromarray(fr.to_ndarray(format="rgb24")))
                    if len(frames) >= n:
                        break
        return frames or None

    def _run_model(self, clip_path: str, prompt: str) -> tuple[str, float, int]:
        """Real Nemotron-Nano-VL path (image VLM → sample clip frames as multi-image).
        Builds `<image-k>` tags + the prompt, runs `model.chat(**image_features)`."""
        n = int(self.config.params.get("frame_sample", 4))
        frames = self._sample_frames(clip_path, n)
        if not frames:
            raise RuntimeError("no frames decoded from clip")
        # image tags must match the number of images (see examples.py).
        if len(frames) == 1:
            question = "<image>\n" + prompt
        else:
            tags = "".join(f"<image-{i+1}>: <image>\n" for i in range(len(frames)))
            question = tags + prompt
        image_features = self._processor(frames if len(frames) > 1 else frames[0])
        gen = dict(max_new_tokens=self._max_output_tokens, do_sample=False,
                   repetition_penalty=1.3,               # curb small-VLM looping → valid JSON
                   eos_token_id=self._tokenizer.eos_token_id)
        t0 = time.perf_counter()
        raw = self._model.chat(tokenizer=self._tokenizer, question=question,
                               generation_config=gen, **image_features)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        text = raw if isinstance(raw, str) else (raw[0] if isinstance(raw, tuple) else str(raw))
        try:
            n_tokens = len(self._tokenizer(text).input_ids)
        except Exception:
            n_tokens = max(1, len(text.split()))
        return text, ttft_ms, n_tokens


# --------------------------------------------------------------------------
# QwenBaselinePlugin — comparison-only baseline (research_qwen_baseline).
# --------------------------------------------------------------------------
def qwen_default_config(profile: str = "research_qwen_baseline") -> PluginConfig:
    return PluginConfig(
        task="vlm_clip_reasoning",
        plugin="vlm_qwen_baseline",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        runtime="transformers",
        endpoint="local",
        profile=profile,
        params={"prompt_pack": "retail_loss_v1", "max_output_tokens": 256},
    )


class QwenBaselinePlugin(_VlmPluginBase):
    """`vlm_qwen_baseline` — wraps src/kathirmani/qwen_vl.py (lazy); fake fallback."""

    def __init__(self, config: PluginConfig | None = None) -> None:
        super().__init__(config or qwen_default_config())

    def load(self) -> None:
        if self._loaded or self._model is not None:
            return
        try:
            from kathirmani.qwen_vl import load_qwen_model      # heavy/lazy import
            self._model, self._processor = load_qwen_model()
            self._loaded = True
            self._load_error = ""
        except Exception as e:                              # marlin/weights/GPU absent
            self._model = None
            self._processor = None
            self._loaded = False
            self._load_error = f"{type(e).__name__}: {e}"

    def _run_model(self, clip_path: str, prompt: str) -> tuple[str, float, int]:
        """Run one query via the existing Qwen path. Single-prompt verification
        (not the full QWEN_QUERIES battery — that is the analysis pipeline)."""
        from kathirmani.qwen_vl import run_qwen_query
        t0 = time.perf_counter()
        text = run_qwen_query(self._model, self._processor, clip_path, prompt)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        return text, ttft_ms, max(1, len(text.split()))

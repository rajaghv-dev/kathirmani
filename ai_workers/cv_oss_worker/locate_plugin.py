"""NvidiaLocateAnythingDetector — NVIDIA LocateAnything-3B behind DetectionPlugin.

Eagle-Embodied's grounding model (https://github.com/NVlabs/Eagle/tree/main/Embodied,
HF `nvidia/LocateAnything-3B`): open-vocabulary localization — image + text
categories → labelled boxes as ``label<box> x1, y1, x2, y2 </box>`` with
coordinates normalized to [0, 1000].

⚠ LICENSE (spec/11 §Licensing): the model is released for **non-commercial use
only** ("Commercial use is not permitted, except by NVIDIA and its affiliates")
and is built on Qwen2.5-3B (Qwen *Research* License). It therefore lives in a
`non_default: true` research profile — a comparison arm like the Qwen baseline,
NOT a production default.

Subclasses CvOssDetector, so it inherits the whole worker contract unchanged:
the 1 fps + motion frame gate (spec/16), per-frame tracker updates, per-subject
event dedup, fake_infer fallback, metrics. Only load() + per-frame detection
differ.

STATUS (verified 2026-07-09 on transformers 5.7): with the `_shim_remote_code`
compat patches the checkpoint LOADS and runs end-to-end, but generation is
degenerate (all-`<null>` in slow mode, repeated zero-area boxes in hybrid) —
the checkpoint hard-pins `transformers==4.57.1` and the 4.57↔5.x drift goes
beyond signature shims. Real inference therefore needs a dedicated env with
the pinned deps (or an NVIDIA remote-code update); until then the plugin
yields no boxes / falls back to fake, and the worker keeps running."""
from __future__ import annotations

import re
from pathlib import Path

from model_plugins.base.plugin import PluginConfig
from model_plugins.base.schemas import Detection

from .plugin import DEFAULT_CLASSES, CvOssDetector, _REPO_ROOT

MODEL_ID = "nvidia/LocateAnything-3B"
LOCAL_DIR = _REPO_ROOT / "models" / "LocateAnything-3B"

# Boxes come as "<box><x1><y1><x2><y2></box>" (Eagle-Embodied worker) or
# "<box> x1, y1, x2, y2 </box>" (model-card text form) — coords in [0, 1000],
# optionally preceded by a label token.
_BOX_RE = re.compile(
    r"([\w\- ]+?)?\s*<box>\s*"
    r"(?:<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>"
    r"|(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+))\s*</box>")


def parse_locate_boxes(text: str, model_name: str, frame_time: float,
                       min_conf: float = 0.5) -> list[Detection]:
    """Parse LocateAnything output text into normalized [x, y, w, h] Detections.

    The model emits no per-box confidence in text mode; boxes get a fixed
    `min_conf` so downstream thresholds still apply."""
    dets: list[Detection] = []
    for m in _BOX_RE.finditer(text):
        label = (m.group(1) or "object").strip().lower()
        coords = [g for g in m.groups()[1:] if g is not None]
        x1, y1, x2, y2 = (int(c) / 1000.0 for c in coords[:4])
        if x2 <= x1 or y2 <= y1:
            continue
        dets.append(Detection(
            label=label, confidence=min_conf,
            bbox=[round(x1, 5), round(y1, 5), round(x2 - x1, 5), round(y2 - y1, 5)],
            frame_time_sec=frame_time, model_name=model_name))
    return dets


def default_config(profile: str = "research_locate_anything") -> PluginConfig:
    return PluginConfig(
        task="detection",
        plugin="nvidia_locate_anything",
        model_id=MODEL_ID,
        runtime="transformers",
        endpoint="local",
        profile=profile,
        params={"classes": DEFAULT_CLASSES},
    )


class NvidiaLocateAnythingDetector(CvOssDetector):
    """CvOssDetector with LocateAnything-3B as the per-frame detector."""

    def __init__(self, config: PluginConfig | None = None):
        super().__init__(config or default_config())
        self._processor = None

    def load(self) -> None:
        """Lazy + non-fatal: transformers trust_remote_code from the local
        weights dir (falls back to the HF id). Any failure → fake mode."""
        self._loaded = True
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            # The remote code imports decord (video reader) unconditionally,
            # but our path is image-only and decord has no linux-aarch64 wheel.
            # Satisfy transformers' import check with an inert stub.
            import importlib.machinery
            import sys
            import types
            if "decord" not in sys.modules:
                try:
                    import decord  # noqa: F401
                except ImportError:
                    stub = types.ModuleType("decord")
                    stub.__spec__ = importlib.machinery.ModuleSpec("decord", None)
                    sys.modules["decord"] = stub
            src = str(LOCAL_DIR) if LOCAL_DIR.exists() else MODEL_ID
            self._processor = AutoProcessor.from_pretrained(
                src, trust_remote_code=True)
            self._shim_remote_code(src)
            self._model = AutoModel.from_pretrained(
                src, torch_dtype="auto", trust_remote_code=True)
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
        except Exception as e:                      # stays runnable without it
            self._model = None
            self._load_error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _shim_remote_code(src: str) -> None:
        """The checkpoint's remote code targets transformers 4.57; on 5.x the
        base `_check_and_adjust_attn_implementation` grew extra kwargs the
        overrides don't accept. Re-point the overrides at a kwargs-tolerant
        version (preserving their 'magi' special case)."""
        from transformers import PreTrainedModel
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # custom __init__ paths bypass the 5.x all_tied_weights_keys setup —
        # same fix the Nemotron plugin needed.
        from ai_workers.vlm_worker.plugin import _patch_tied_weights_compat
        _patch_tied_weights_compat()

        model_cls = get_class_from_dynamic_module(
            "modeling_locateanything.LocateAnythingForConditionalGeneration", src)

        base_impl = PreTrainedModel._check_and_adjust_attn_implementation

        def patched(self, attn_implementation, *args, **kwargs):
            if attn_implementation == "magi":
                return "magi"
            return base_impl(self, attn_implementation, *args, **kwargs)

        import sys
        for name, mod in list(sys.modules.items()):
            if "transformers_modules" not in name:
                continue
            for obj in vars(mod).values():
                # the dynamic modules re-export transformers' PreTrainedModel;
                # only patch the checkpoint's OWN subclasses, never the base.
                if (isinstance(obj, type) and obj is not PreTrainedModel
                        and issubclass(obj, PreTrainedModel)
                        and "_check_and_adjust_attn_implementation" in vars(obj)):
                    obj._check_and_adjust_attn_implementation = patched
                # 4.x declared tied weights as a list; 5.x wants {target: source}.
                # This checkpoint ties lm_head to the input embeddings
                # (tie_word_embeddings: true), the standard causal-LM tying.
                tk = vars(obj).get("_tied_weights_keys") if isinstance(obj, type) else None
                if isinstance(tk, list):
                    obj._tied_weights_keys = {
                        k: "model.embed_tokens.weight" for k in tk}
        assert model_cls is not None

        # transformers 5.x dropped the legacy KV-cache tuple converters the
        # 4.x-era decoding loop round-trips through — restore them.
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "to_legacy_cache"):
            DynamicCache.to_legacy_cache = lambda self: tuple(
                (layer.keys, layer.values) for layer in self.layers)
        if not hasattr(DynamicCache, "from_legacy_cache"):
            DynamicCache.from_legacy_cache = classmethod(
                lambda cls, past=None:
                cls(ddp_cache_data=past) if past is not None else cls())

        # transformers 5.x folded rope_theta/rope_scaling into rope_parameters;
        # the 4.57-era remote code still reads the old attributes.
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
        if not hasattr(Qwen2Config, "rope_theta"):
            Qwen2Config.rope_theta = property(
                lambda self: (getattr(self, "rope_parameters", None) or {})
                .get("rope_theta", 1000000.0))
        if not hasattr(Qwen2Config, "rope_scaling"):
            Qwen2Config.rope_scaling = property(
                lambda self: (getattr(self, "rope_parameters", None) or {})
                .get("rope_scaling"))

    def _detect_frame(self, rgb, t: float) -> list[Detection]:
        """One gated frame → LocateAnything grounding → Detections.

        Mirrors the Eagle-Embodied worker: categories joined by ``</c>`` in the
        'Locate all the instances…' template, image via the ``<image-N>``
        placeholder, boxes decoded from the newly generated tokens only."""
        import torch
        from PIL import Image

        prompt = ("Locate all the instances that matches the following "
                  "description: " + "</c>".join(self._classes) + ".")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            text = prompt
        if "<image-" not in text:
            text = f"<image-0>{text}"

        img = Image.fromarray(rgb)
        inputs = self._processor(images=[img], text=text, return_tensors="pt")
        inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=2048, use_cache=True,
                                       tokenizer=self._processor.tokenizer)
        if isinstance(out, tuple):                 # (text, stats) in some modes
            out = out[0]
        if not isinstance(out, str):               # token ids → decode new tail
            out = self._processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=False)[0]
        return parse_locate_boxes(out, self.config.model_id, t)

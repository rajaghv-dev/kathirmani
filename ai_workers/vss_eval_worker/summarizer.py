"""NemotronSummaryPlugin — the `nvidia_summary` plugin behind the
`SummarizationPlugin` contract (master plan A2 / spec/11 §"SummarizationPlugin").

Wraps **nvidia/NVIDIA-Nemotron-3-Nano** (the reasoning LLM that turns
clips/events/time-ranges into a timestamped, incident-aware summary). The real
model is reached via **Ollama** (preferred on this GB10 box — GGUF, portable) or
**transformers** as a fallback, and the load is **lazy and optional**: if the
runtime / weights / GPU are missing — or anything raises — `infer()` transparently
falls back to `fake_infer()`, a deterministic text summary built purely from the
request. That keeps `lvs.py`, the worker, the A11 plugin test, and CI runnable
with NO GPU / NO weights / NO network.

Task: `summarization` (contract: clips|events|time_range -> timestamped_summary).

infer(request) ->
  {
    "summary": str,          # natural-language summary text
    "model_name": str,
    "faked": bool,
    "latency_ms": float,
    "error": str | None,
  }

The request is a small dict the staged pipeline (lvs.py) assembles at each level:
  {
    "level": "clip" | "5min" | "hour" | "report",
    "instruction": str,          # what to produce at this level
    "items": list[str],          # the lower-level text inputs to fold up
    "context": dict,             # store_id / camera / time_range / mode, optional
  }

VSS NOTE (spec/10 + spec/11): VSS is a *reference* for the capability shape, never
a runtime dependency (`allow_vss_runtime_dependency: false`). Nothing here imports
or calls VSS; the hierarchical fold is implemented in OSS + the NVIDIA summarizer.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any


from model_plugins.base.plugin import Health, ModelPlugin, PluginConfig      # noqa: E402

# Free/OSS NVIDIA summarizer/critic LLM (spec/11 selected-models table). The
# Ollama tag is the GGUF served via `ollama run`; transformers uses the HF id.
DEFAULT_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano"
DEFAULT_OLLAMA_TAG = "nemotron3-nano"
DEFAULT_RUNTIME = "ollama"


def default_config(profile: str = "nvidia_gb10_retail_balanced") -> PluginConfig:
    """A PluginConfig matching the `summarization` task of the active profile, so
    the pipeline/worker can construct the plugin without a config loader (mirrors
    configs/models.yaml: plugin nvidia_summary, runtime ollama)."""
    return PluginConfig(
        task="summarization",
        plugin="nvidia_summary",
        model_id=DEFAULT_MODEL_ID,
        runtime=DEFAULT_RUNTIME,
        endpoint="local",
        profile=profile,
        params={"ollama_tag": DEFAULT_OLLAMA_TAG, "max_output_tokens": 512},
    )


class NemotronSummaryPlugin(ModelPlugin):
    """Nemotron-3-Nano summarizer for the `summarization` task; fake-infer fallback."""

    def __init__(self, config: PluginConfig | None = None) -> None:
        super().__init__(config or default_config())
        self._client = None                # ollama client / transformers pipeline
        self._mode = ""                    # "ollama" | "transformers" | ""
        self._loaded = False
        self._load_error = ""
        self._tag = str(self.config.params.get("ollama_tag", DEFAULT_OLLAMA_TAG))
        self._max_tokens = int(self.config.params.get("max_output_tokens", 512))
        # lightweight counters (mirrored to prometheus in infer()).
        self._n_requests = 0
        self._n_fake = 0
        self._last_latency_ms = 0.0

    # ---- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Try Ollama first (default runtime on this box), then transformers.
        Idempotent and *non-fatal*: on any failure we record the reason and stay
        in fake-infer mode (no GPU/weights/network needed)."""
        if self._loaded:
            return
        # 1) Ollama (preferred): GGUF served locally, portable on aarch64/Blackwell.
        if self.config.runtime == "ollama":
            try:
                import ollama                       # optional dependency
                self._client = ollama
                self._mode = "ollama"
                self._loaded = True
                self._load_error = ""
                return
            except Exception as e:
                self._load_error = f"ollama: {type(e).__name__}: {e}"
        # 2) transformers fallback (heavy; only if weights are present).
        try:
            from transformers import pipeline       # heavy; lazy
            self._client = pipeline(
                "text-generation", model=self.config.model_id)
            self._mode = "transformers"
            self._loaded = True
            self._load_error = ""
        except Exception as e:
            self._client = None
            self._mode = ""
            self._loaded = False
            self._load_error = (self._load_error + " | " if self._load_error else "") \
                + f"transformers: {type(e).__name__}: {e}"

    def unload(self) -> None:
        self._client = None
        self._mode = ""
        self._loaded = False
        try:                                          # free GPU memory if torch present
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def health(self) -> Health:
        if self._loaded and self._client is not None:
            return Health(ok=True, detail=f"{self.config.model_id} via {self._mode}")
        detail = "fake-infer mode" + (f" ({self._load_error})" if self._load_error else "")
        return Health(ok=True, detail=detail)         # fake-infer keeps lvs.py usable

    def metrics(self) -> dict[str, float]:
        return {
            "requests_total": float(self._n_requests),
            "fake_infer_total": float(self._n_fake),
            "last_latency_ms": float(self._last_latency_ms),
            "model_loaded": 1.0 if self._loaded else 0.0,
        }

    # ---- inference ---------------------------------------------------------
    def fake_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Deterministic, GPU-free summary text (A11 #3). Builds a stable summary
        purely from the request so tests are reproducible: same inputs -> same
        text. It is real English (not a hash) so the staged fold stays legible,
        with a short stable digest suffix to guarantee determinism is testable."""
        level = str(request.get("level", "clip"))
        items = [str(x) for x in (request.get("items") or [])]
        ctx = request.get("context") or {}
        digest = hashlib.sha256(
            ("|".join(items) + f"#{level}").encode()).hexdigest()[:8]

        n = len(items)
        if n == 0:
            body = "No activity recorded in this interval."
        elif level == "clip":
            # one clip's caption: fold its event lines into a sentence
            body = "; ".join(items[:4])
            if n > 4:
                body += f"; (+{n - 4} more)"
        else:
            # fold lower-level summaries: lead with count + the salient lines
            lead = {"5min": "5-minute window", "hour": "1-hour window",
                    "report": "incident-aware report"}.get(level, level)
            head = items[0] if items else ""
            body = f"{lead}: {n} sub-summaries. " + head
            if n > 1:
                body += f" Other notable: {items[1]}"

        scope = []
        if ctx.get("camera"):
            scope.append(f"camera={ctx['camera']}")
        if ctx.get("time_range"):
            tr = ctx["time_range"]
            scope.append(f"{tr.get('start', '?')}..{tr.get('end', '?')}")
        if ctx.get("summary_mode"):
            scope.append(f"mode={ctx['summary_mode']}")
        scope_str = (" [" + ", ".join(scope) + "]") if scope else ""

        summary = f"{body}{scope_str} (sum:{digest})"
        return self._package(summary, faked=True)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Summarize one fold level. Falls back to fake_infer if the model isn't
        loaded or inference raises. Always returns {summary, model_name, faked,
        latency_ms, error}."""
        self._n_requests += 1
        start = time.perf_counter()
        if not self._loaded:
            self.load()                               # lazy, non-fatal
        try:
            if self._client is None:
                raise RuntimeError("summarizer unavailable")
            summary = self._summarize_real(request)
            out = self._package(summary, faked=False)
        except Exception as e:
            out = self.fake_infer(request)
            out["error"] = f"fallback: {type(e).__name__}: {e}"
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        out["latency_ms"] = round(self._last_latency_ms, 3)
        self._emit_prometheus(out, faked=out.get("faked", False))
        return out

    # ---- real Nemotron path (only when a runtime+weights are present) ------
    def _summarize_real(self, request: dict[str, Any]) -> str:
        """Build the prompt and run the configured runtime. Defensive — runtime
        APIs vary; any failure bubbles up to the fake fallback in infer()."""
        prompt = self._build_prompt(request)
        if self._mode == "ollama":
            resp = self._client.generate(
                model=self._tag, prompt=prompt,
                options={"num_predict": self._max_tokens})
            return (resp.get("response") or "").strip()
        if self._mode == "transformers":
            out = self._client(prompt, max_new_tokens=self._max_tokens,
                               return_full_text=False)
            return (out[0].get("generated_text") or "").strip()
        raise RuntimeError(f"unknown mode {self._mode!r}")

    @staticmethod
    def _build_prompt(request: dict[str, Any]) -> str:
        """Render the level instruction + items into a single LLM prompt. Kept
        plain so swapping the runtime never changes the business prompt (A1.3)."""
        level = str(request.get("level", "clip"))
        instruction = str(request.get("instruction")
                          or _DEFAULT_INSTRUCTIONS.get(level, "Summarize the following."))
        items = [str(x) for x in (request.get("items") or [])]
        ctx = request.get("context") or {}
        ctx_line = ", ".join(f"{k}={v}" for k, v in ctx.items() if not isinstance(v, (dict, list)))
        joined = "\n".join(f"- {it}" for it in items) or "- (no inputs)"
        return (
            "You are a retail loss-prevention video analyst.\n"
            f"Context: {ctx_line}\n"
            f"Task: {instruction}\n"
            "Inputs:\n"
            f"{joined}\n"
            "Respond with a concise, factual summary; cite times where present."
        )

    # ---- packaging + metrics ----------------------------------------------
    def _package(self, summary: str, faked: bool) -> dict[str, Any]:
        return {
            "summary": summary,
            "model_name": self.config.model_id,
            "faked": faked,
            "latency_ms": self._last_latency_ms,
            "error": None,
        }

    def _emit_prometheus(self, out: dict[str, Any], faked: bool) -> None:
        if faked:
            self._n_fake += 1
        try:
            from model_plugins.base import metrics as M
            lbl = M.labels(self.config)
            status = "fake" if faked else ("error" if out.get("error") else "ok")
            M.model_requests_total.labels(**lbl, status=status).inc()
            M.model_latency_ms.labels(**lbl).observe(out.get("latency_ms", 0.0))
            if out.get("error") and not faked:
                M.model_errors_total.labels(**lbl, error_type="infer").inc()
        except Exception:
            pass                                      # metrics never break inference


# Default per-level instructions (the §A4.2 staged fold). The pipeline may
# override `instruction`, but these keep the summarizer self-describing.
_DEFAULT_INSTRUCTIONS = {
    "clip": "Caption this 10-second clip from its detector/rule events.",
    "5min": "Summarize this 5-minute window from its per-clip captions.",
    "hour": "Summarize this 1-hour window from its 5-minute summaries.",
    "report": ("Write an incident-aware report from the hourly summaries: what "
               "happened, when, and what a reviewer should check."),
}

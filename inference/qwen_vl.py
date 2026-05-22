"""Qwen2.5-VL-7B-Instruct spatial analysis pipeline.
Complements Marlin-2B (temporal) with frame-level visual understanding.
"""
import threading
from pathlib import Path

import torch

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_LOCAL_PATH = Path(__file__).parent.parent / "models" / "Qwen2.5-VL-7B-Instruct"

# Loss prevention and store operations queries for Qwen2.5-VL
# Professional / neutral language — retail industry standard terms
QWEN_QUERIES = [
    # Merchandise security
    "Count the customers visible and describe what each person is doing with their hands and any bags they carry.",
    "Is any customer placing merchandise into a personal bag, pocket, or clothing while at the shelf — not into a store basket?",
    "Are any customers moving toward the store exit while carrying unpaid merchandise without a billing bag or receipt?",
    "Is there a group of customers clustered together near merchandise in a way that could obstruct the camera view of the shelf?",
    "Describe any customer spending an unusually long time handling a single product without placing it in a basket.",
    # Distraction and diversion
    "Is there any commotion, argument, or distraction happening near the billing counter or merchandise display?",
    "Are any customers engaging billing staff in extended conversation while another person is near the merchandise?",
    # Inventory and operations
    "Are there any empty shelf gaps or sections that appear to need restocking?",
    "Is there any merchandise on the floor, misplaced, or fallen from shelves?",
    "How many customers are at or waiting at the billing counter? Estimate queue length.",
    # Customer welfare
    "Is any child present in the store without an accompanying adult nearby?",
    "Is any customer appearing distressed, confused, or in need of staff assistance?",
    # Scene context
    "Describe the overall store activity level: how many staff and customers visible, and what is the general level of activity?",
]


def load_qwen_model(model_path: Path | None = None) -> tuple[object, object]:
    """Load Qwen2.5-VL-7B-Instruct model and processor."""
    resolved = model_path or QWEN_LOCAL_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Qwen2.5-VL model not found at {resolved}. "
            f"Run: python download_model.py --model qwen-vl"
        )
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(resolved),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(str(resolved), local_files_only=True)
    print(f"[qwen_vl] Loaded Qwen2.5-VL-7B from {resolved}")
    return model, processor


def _sanitise_video_kwargs(kwargs: dict) -> dict:
    """Flatten any list-wrapped scalars that qwen-vl-utils returns for fps."""
    clean = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], (int, float)):
            clean[k] = v[0]
        else:
            clean[k] = v
    return clean


def run_qwen_query(model, processor, video_path: str, query: str) -> str:
    """Run a single spatial query on a video clip using Qwen2.5-VL."""
    from qwen_vl_utils import process_vision_info
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "max_pixels": 200704, "nframes": 8},
            {"type": "text", "text": query},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    video_kwargs = _sanitise_video_kwargs(video_kwargs)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        **video_kwargs,
    ).to("cuda")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def analyze_video(model, processor, video_path: Path, label: str, gpu_lock: threading.Lock) -> dict:
    """Run all QWEN_QUERIES on a video. Serializes GPU calls via gpu_lock."""
    from .metrics import QWEN_QUERIES_DONE, QWEN_ANSWERS_OK

    results = {}
    queries_done = 0
    answers_ok = 0

    for query in QWEN_QUERIES:
        with gpu_lock:
            try:
                answer = run_qwen_query(model, processor, str(video_path), query)
                results[query] = {"answer": answer, "ok": True}
                answers_ok += 1
            except Exception as e:
                results[query] = {"answer": "", "ok": False, "error": str(e)[:200]}
        queries_done += 1

    QWEN_QUERIES_DONE.labels(video=label).set(queries_done)
    QWEN_ANSWERS_OK.labels(video=label).set(answers_ok)

    return {
        "label": label,
        "model": "Qwen2.5-VL-7B-Instruct",
        "queries": results,
    }

#!/usr/bin/env python3
"""AI image generation for *design assets* — Google "Nano Banana" + OpenAI gpt-image-1.

Scope: this is **design-time tooling** (infographics, mockups, hero art), NOT the
production video-inference pipeline. The NVIDIA-only model policy (spec/11) governs the
inference plugins; design-asset generation is a separate concern, so Gemini/OpenAI are
fine here and never touch the task-contract plugin layer.

Providers
  - gemini : Google Gemini 2.5 Flash Image ("Nano Banana"). Text->image AND image
             editing/fusion (pass --ref images). SDK: google-genai. Key: GEMINI_API_KEY
             (or GOOGLE_API_KEY).
  - openai : OpenAI gpt-image-1. Text->image, or edit when --ref given. SDK: openai.
             Key: OPENAI_API_KEY.

Security (spec/10): keys come from the environment ONLY — never hard-coded, never logged.
Copy .env.example -> .env and fill them; .env is gitignored.

Usage
  python design/ai_images.py --provider gemini --prompt "a clean retail-store top-down
      diagram, NVIDIA-green accents, 5 ceiling cameras" --out store_topdown
  python design/ai_images.py --provider both --prompt-file design/prompts/hero.txt --out hero
  # edit/fuse existing figures (Nano Banana is strong at this):
  python design/ai_images.py --provider gemini --ref design/figures/platform_architecture.png \
      --prompt "redraw as a glossy 3D isometric infographic, same labels" --out arch_iso

Deps: pip install -r requirements/design.txt   (or: make setup-design)
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "design" / "figures" / "generated"

GEMINI_DEFAULT = "gemini-2.5-flash-image"   # "Nano Banana"; -pro-image-preview = Nano Banana Pro
OPENAI_DEFAULT = "gpt-image-1"


def _die(msg: str, code: int = 2) -> None:
    print(f"ai_images: {msg}", file=sys.stderr)
    sys.exit(code)


def _env_key(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


# --------------------------------------------------------------------------- gemini
def gen_gemini(prompt: str, out: Path, refs: list[Path], model: str) -> Path:
    key = _env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not key:
        _die("set GEMINI_API_KEY (or GOOGLE_API_KEY) in your env / .env")
    try:
        from google import genai          # pip install google-genai
    except ImportError:
        _die("google-genai not installed — `pip install -r requirements/design.txt`")
    from PIL import Image

    client = genai.Client(api_key=key)
    contents: list = [prompt]
    for r in refs:
        contents.append(Image.open(r))    # editing / multi-image fusion
    resp = client.models.generate_content(model=model, contents=contents)

    parts = resp.candidates[0].content.parts if resp.candidates else []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            out.write_bytes(inline.data)
            return out
    # no image came back — surface any text the model returned (e.g. a refusal)
    txt = " ".join(getattr(p, "text", "") or "" for p in parts).strip()
    _die(f"Gemini returned no image. Model said: {txt or '(nothing)'}", code=1)


# --------------------------------------------------------------------------- openai
def gen_openai(prompt: str, out: Path, refs: list[Path], model: str, size: str) -> Path:
    if not _env_key("OPENAI_API_KEY"):
        _die("set OPENAI_API_KEY in your env / .env")
    try:
        from openai import OpenAI          # pip install openai
    except ImportError:
        _die("openai not installed — `pip install -r requirements/design.txt`")

    client = OpenAI()
    if refs:
        files = [open(r, "rb") for r in refs]
        try:
            result = client.images.edit(model=model, image=files, prompt=prompt, size=size)
        finally:
            for f in files:
                f.close()
    else:
        result = client.images.generate(model=model, prompt=prompt, size=size)

    b64 = result.data[0].b64_json         # gpt-image-1 always returns b64
    out.write_bytes(base64.b64decode(b64))
    return out


# --------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate design images via Nano Banana / gpt-image-1.")
    ap.add_argument("--provider", choices=["gemini", "openai", "both"], default="gemini")
    ap.add_argument("--prompt", help="text prompt")
    ap.add_argument("--prompt-file", type=Path, help="read the prompt from a file")
    ap.add_argument("--out", required=True, help="output basename (no extension)")
    ap.add_argument("--ref", action="append", type=Path, default=[],
                    help="input image(s) to edit/fuse; repeatable")
    ap.add_argument("--size", default="1536x1024", help="openai size (1024x1024|1536x1024|1024x1536|auto)")
    ap.add_argument("--gemini-model", default=GEMINI_DEFAULT)
    ap.add_argument("--openai-model", default=OPENAI_DEFAULT)
    args = ap.parse_args(argv)

    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text().strip()
    if not prompt:
        _die("provide --prompt or --prompt-file")
    for r in args.ref:
        if not r.exists():
            _die(f"ref image not found: {r}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    providers = ["gemini", "openai"] if args.provider == "both" else [args.provider]
    written: list[Path] = []
    for p in providers:
        out = OUT_DIR / f"{args.out}__{p}.png"
        if p == "gemini":
            gen_gemini(prompt, out, args.ref, args.gemini_model)
        else:
            gen_openai(prompt, out, args.ref, args.openai_model, args.size)
        print(f"wrote {out.relative_to(ROOT)}")
        written.append(out)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())

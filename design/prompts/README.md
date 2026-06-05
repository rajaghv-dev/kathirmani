# Ready-to-run image prompts

Plain-text prompts for `design/ai_images.py` (Google **Nano Banana** + OpenAI
**gpt-image-1**). Each file is self-contained — it embeds the palette/style so it works
standalone. Grounded in the platform spec and `design/INFOGRAPHIC-BRIEF.md`.

```bash
make setup-design                      # one-time: installs google-genai + openai + pillow
# key via env / .env only:  GEMINI_API_KEY=…   or   OPENAI_API_KEY=…

# generate from a prompt file:
.venv/bin/python design/ai_images.py --provider gemini \
    --prompt-file design/prompts/cascade_explainer.txt --out cascade

# both providers at once (compare):
.venv/bin/python design/ai_images.py --provider both \
    --prompt-file design/prompts/platform_architecture.txt --out arch

# redraw an existing matplotlib figure as polished art (Nano Banana is great at this):
.venv/bin/python design/ai_images.py --provider gemini \
    --ref design/figures/data_ingestion_layer.png \
    --prompt-file design/prompts/data_ingestion_layer.txt --out ingest_polished
```

Outputs land in `design/figures/generated/<out>__<provider>.png` (gitignored).

| File | What it makes |
|------|---------------|
| `operator_one_pager.txt` | vertical one-pager for a store owner |
| `platform_architecture.txt` | end-to-end architecture flow |
| `data_ingestion_layer.txt` | RTSP → model-feed ingestion lane |
| `cascade_explainer.txt` | 3-step cheapest-gate-first cascade |
| `efficiency_cards.txt` | 6 cost/efficiency stat cards |
| `store_topdown.txt` | top-down store with 5 cameras + zones |
| `multiscale_reasoning.txt` | subtle actions over time (why pg-cataloged spans) |

Palette (in every prompt): NVIDIA green `#76B900`, ink `#1A1A1A`, slate `#3B4252`,
muted `#8A8F98`, off-white card `#F5F7F2`, white bg, optional alert-red `#E5484D`.

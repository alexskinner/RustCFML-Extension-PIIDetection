# Models

The ONNX graphs the extension runs, committed here so that installing it needs
nothing from a third-party service: by default the engine fetches the model
from **this repository** (raw GitHub content for the files in this directory,
release assets for the larger sizes). Hugging Face is an option, not a
requirement — see [docs/configuration.md](../docs/configuration.md).

| directory / asset | parameters | ONNX size | F1 (Nemotron-PII test) | where |
|---|---|---|---|---|
| `ettin-17m-nemotron-pii/` | 17M | 68 MB | 94.21 | in this directory (the default) |
| `ettin-32m-nemotron-pii.onnx` | 32M | 129 MB | 95.73 | release asset `models-v1` |
| `ettin-68m-nemotron-pii.onnx` | 68M | 274 MB | 96.27 | release asset `models-v1` |

The two larger graphs exceed GitHub's 100 MB per-file limit for ordinary
commits, so they are attached to the `models-v1` release instead; their
`config.json` and `tokenizer.json` are identical to the 17m ones apart from
the hidden size, and the extension fetches them from the same release.

## Attribution

These graphs are **exports, not our work**. Each was produced from the
published PyTorch checkpoint by [`scripts/export-onnx.py`](../scripts/export-onnx.py)
(float32, opset 17; verified against PyTorch in onnxruntime before being
committed) and is otherwise unmodified. Credit belongs to:

* **ettin-\*-nemotron-pii** — Kalyan KS, 2026. MIT licence.
  https://huggingface.co/kalyan-ks/ettin-17m-nemotron-pii (also `-32m-`, `-68m-`).
  The model card is copied beside each export as `MODEL_CARD.md`; it carries
  the licence declaration, the evaluation figures and the per-entity scores.
  ```
  @misc{ettin-17m-pii-2026,
    title = {ettin-17m-nemotron-pii-2026: PII Detection Model},
    author = {Kalyan KS}, year = {2026}, publisher = {Hugging Face},
    url = {https://huggingface.co/kalyan-ks/ettin-17m-nemotron-pii} }
  ```
* **Ettin encoders** (the base models) — Johns Hopkins University, Center for
  Language and Speech Processing. MIT licence.
  https://huggingface.co/jhu-clsp/ettin-encoder-17m — Weller et al., 2025,
  *Seq vs Seq: An Open Suite of Paired Encoders and Decoders*.
* **Nemotron-PII** (the fine-tuning data) — NVIDIA. CC BY 4.0.
  https://huggingface.co/datasets/nvidia/Nemotron-PII

The MIT licence of the model and of its base, and the attribution the dataset
licence asks for, are satisfied by keeping this file and the model card with
the graphs. The extension's own code is MIT OR Apache-2.0 — see
[COPYRIGHT](../COPYRIGHT).

## Reproducing an export

```bash
python3 -m venv venv && venv/bin/pip install torch transformers onnx onnxruntime huggingface_hub
venv/bin/python scripts/export-onnx.py 17m     # → models/ettin-17m-nemotron-pii/
venv/bin/python scripts/export-onnx.py 68m
```

The script prints the SHA-256 of what it wrote; the ones in `src/config.rs`
pin the committed graphs so a download is verified byte for byte.

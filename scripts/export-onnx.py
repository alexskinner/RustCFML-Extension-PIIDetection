#!/usr/bin/env python3
"""Export an ettin-*-nemotron-pii checkpoint to the ONNX graph this extension runs.

    python3 -m venv venv && venv/bin/pip install torch transformers onnx onnxruntime
    venv/bin/python scripts/export-onnx.py 17m            # → models/ettin-17m-nemotron-pii/
    venv/bin/python scripts/export-onnx.py 68m --out /opt/models/pii

What it does, and why each choice matters:

* Loads the model in FLOAT32 and eager attention. transformers 5 keeps the
  checkpoint's bfloat16, and an export in bfloat16 puts bfloat16 constants in
  the attention mask that neither onnxruntime nor tract will run.
* Uses the TorchScript exporter at opset 17 with dynamic batch/sequence axes.
  tract runs this graph with the batch pinned to 1 and the sequence symbolic.
* Verifies the graph against PyTorch in onnxruntime at two lengths before
  writing anything else, and prints the SHA-256 — the value to put in
  `onnxSha256` (or in `config.rs` for the default model).

The output directory holds everything the extension needs: config.json,
tokenizer.json, onnx/model.onnx, plus the model's README (its licence and
card). Point `modelPath` at it, or upload onnx/model.onnx somewhere and set
`onnxUrl`.
"""
import argparse, hashlib, os, shutil, sys

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("size", choices=["17m", "32m", "68m"], help="which ettin checkpoint")
ap.add_argument("--repo", help="override the Hub repository id")
ap.add_argument("--out", help="output directory (default models/<repo name>)")
ap.add_argument("--opset", type=int, default=17)
args = ap.parse_args()

repo = args.repo or f"kalyan-ks/ettin-{args.size}-nemotron-pii"
out = args.out or os.path.join("models", repo.split("/")[-1])
os.makedirs(os.path.join(out, "onnx"), exist_ok=True)

print(f"loading {repo}")
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForTokenClassification.from_pretrained(
    repo, attn_implementation="eager", dtype=torch.float32
).eval()
print(f"  {sum(p.numel() for p in model.parameters()):,} parameters, {next(model.parameters()).dtype}")

tok.save_pretrained(out)
model.config.save_pretrained(out)
# The card travels with the weights.
try:
    from huggingface_hub import hf_hub_download
    for f in ("README.md", "LICENSE"):
        try:
            shutil.copy(hf_hub_download(repo, f), os.path.join(out, f))
        except Exception:
            pass
except ImportError:
    pass

sample = "Kalyan KS is from India. His email id is kalyan.ks@yahoo.com"
enc = tok(sample, return_tensors="pt")
with torch.no_grad():
    ref = model(**enc).logits

onnx_path = os.path.join(out, "onnx", "model.onnx")
torch.onnx.export(
    model,
    (enc["input_ids"], enc["attention_mask"]),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    },
    opset_version=args.opset,
    dynamo=False,
)
print(f"exported {onnx_path} ({os.path.getsize(onnx_path):,} bytes)")

import onnxruntime as ort

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
worst = 0.0
for text in (sample, "The quarterly results look strong. " * 40 + "Finally email sarah@example.com."):
    e = tok(text, return_tensors="pt")
    with torch.no_grad():
        r = model(**e).logits.numpy()
    g = sess.run(None, {"input_ids": e["input_ids"].numpy(), "attention_mask": e["attention_mask"].numpy()})[0]
    worst = max(worst, float(np.abs(g - r).max()))
    print(f"  verified at {e['input_ids'].shape[1]} tokens, max |onnx - torch| = {float(np.abs(g - r).max()):.2e}")
if worst > 1e-3:
    print("ERROR: the ONNX graph does not reproduce PyTorch", file=sys.stderr)
    sys.exit(1)

h = hashlib.sha256()
with open(onnx_path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(f"sha256 {h.hexdigest()}")
print(f"\nDone. Set modelPath = \"{os.path.abspath(out)}\" — or host onnx/model.onnx and set onnxUrl + onnxSha256.")

# RustCFML PII detection extension

Find personal data in text and mask it — names, emails, phone numbers,
addresses, card numbers, identifiers, 55 types in all — **in-process**, with
no API to call and nothing leaving the machine. Everything in it is MIT or
Apache-2.0 licensed, the model included.

```cfml
spans = piiDetect( "Contact John Smith at john@example.com or +44 7700 900123." );
// [ { type: "first_name",   text: "John",             start: 9,  end: 12, length: 4,  score: 0.995 },
//   { type: "last_name",    text: "Smith",            start: 14, end: 18, length: 5,  score: 0.393 },
//   { type: "email",        text: "john@example.com", start: 23, end: 38, length: 16, score: 0.981 },
//   { type: "phone_number", text: "44 7700 900123",   start: 44, end: 57, length: 14, score: 0.831 } ]

piiMask( "Contact John Smith at john@example.com or +44 7700 900123." );
// Contact [FIRST_NAME] [LAST_NAME] at [EMAIL] or +[PHONE_NUMBER].

piiMask( "Visa card 4532 7153 3790 1241", { types = "credit_debit_card", maskChar = "*", keepLast = 4 } );
// Visa card **** **** **** 1241
```

The detector is [ettin-17m-nemotron-pii](https://huggingface.co/kalyan-ks/ettin-17m-nemotron-pii)
by Kalyan KS (MIT): a 17M-parameter ModernBERT token classifier fine-tuned on
NVIDIA's Nemotron-PII, 94.2 F1 on its test set. It runs as ONNX with
[tract](https://github.com/sonos/tract): pure Rust, CPU, 2–3 ms a sentence,
~90 MB resident. The 32M and 68M sizes are a one-line switch. Positions are
exact character offsets, so spans can be masked in place without string
matching.

## Install

Download the `.rcx` for your platform from
[Releases](https://github.com/alexskinner/RustCFML-Extension-PIIDetection/releases), then:

```bash
rustcfml ext install pii-0.1.0.rcx    # --user, or --dir ./extensions
```

Extensions load once at process start, so **restart the engine** afterwards.
`rustcfml ext list` shows what loaded. The first start fetches the model
(68 MB, from this repository) into `~/.rustcfml/models/pii/` and says so on
stderr; after that it loads from disk in about 200 ms. No configuration is
needed for that. Air-gapped? Copy `models/ettin-17m-nemotron-pii/` from a
checkout and set `modelPath`.

## Functions

| | |
|---|---|
| `piiDetect( text [, options ] )` | array of `{ type, text, start, end, length, score }` spans |
| `piiMask( text [, options ] )` | the text with spans replaced: `[TYPE]`, a template, or a mask character |
| `piiTypes()` | the 55 entity types |
| `piiIsReady()`, `piiLoad()`, `piiSettings()` | load state, explicit warm-up / retry, effective settings |

`options` takes `types` (only these), `minScore`, and for masking
`replacement` / `maskChar` / `keepLast`. **[docs/functions.md](docs/functions.md)**
has every key and the exact output shapes.

## Configure and tune

Optional, server-level, in the `.cfconfig.json` the engine starts with:

```json
{ "extensions": { "settings": { "pii": {
    "model":     "ettin-68m-nemotron-pii",
    "source":    "github",
    "modelPath": "",
    "preload":   true
} } } }
```

| key | default | |
|---|---|---|
| `model` | `ettin-17m-nemotron-pii` | or `ettin-32m-…` (95.7 F1) / `ettin-68m-…` (96.3 F1), bigger and slower |
| `source` | `github` | fetch the small files from this repository, or `huggingface` for the model's own Hub repo |
| `modelPath` | — | a directory already holding the model: nothing is downloaded |
| `windowTokens`, `chunkOverlap` | `512`, `32` | how long texts are chunked |
| `mergeGap` | `1` | characters between same-type pieces that still make one span |
| `preload`, `autoDownload`, `cacheDir` | `true`, `true`, `~/.rustcfml/models/pii` | |

**[docs/configuration.md](docs/configuration.md)** covers each key, what the
download verifies, what happens when a load fails, and a **tuning** section
with what the three model sizes actually do differently on real sentences,
when to use `minScore`, and how to bring a model of your own.

## How it works

Text is tokenised with the model's own tokenizer (`[CLS] … [SEP]` per
window), run through the ONNX graph, and each token's label is the argmax of
the logits. Labels are `B-`/`I-` tagged; following the model author's own
post-processing, the prefix is ignored and consecutive tokens of one type that
touch or are a character apart become a span, trimmed of whitespace. Texts
longer than the window are processed in overlapping windows, each token
taking the prediction from the window where it sits furthest from an edge, so
a span straddling a boundary is neither split nor lost. Offsets are tracked
through the tokenizer back to the original characters.

Inference is re-entrant: one loaded model serves every request thread with no
lock on the call path.

## Try it

[`examples/redact`](examples/redact) exercises every function against
sentences whose answers are known, and is what CI runs:

```bash
rustcfml ext build .
rustcfml ext install pii-0.1.0.rcx --dir examples/redact/extensions
cd examples/redact && rustcfml selftest.cfm
```

## Build from source

Clone this next to a RustCFML checkout — the ABI wrapper is referenced as
`../RustCFML/crates/rustcfml-module`:

```bash
git clone https://github.com/RustCFML/RustCFML.git
git clone https://github.com/alexskinner/RustCFML-Extension-PIIDetection.git
cd RustCFML-Extension-PIIDetection
PII_MODEL_DIR=$PWD/models/ettin-17m-nemotron-pii cargo test   # with the real-model tests
rustcfml ext build .        # produces pii-<version>.rcx
```

Everything is pure Rust — tract for inference, `tokenizers` without its C
regex engine, rustls for the download — so there is no system library to find
at load time and the same source builds on every platform the engine ships for.
The only RustCFML dependency is the extension ABI wrapper.

The ONNX graphs in [`models/`](models/) are exports of the published PyTorch
weights, made by [`scripts/export-onnx.py`](scripts/export-onnx.py) and
verified against PyTorch before being committed. The script reproduces them,
and exports a model of your own the same way.

## Credit and licensing

All of the detection is the model's. **ettin-\*-nemotron-pii** is by
[Kalyan KS](https://huggingface.co/kalyan-ks), MIT licensed, built on the
[Ettin encoders](https://huggingface.co/jhu-clsp/ettin-encoder-17m) (Johns
Hopkins CLSP, MIT) and fine-tuned on [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII)
(NVIDIA, CC BY 4.0). The ONNX exports are redistributed here under the
model's MIT licence with the model card beside each; **[models/README.md](models/README.md)**
has the full attribution and the citation the authors ask for.

This extension's own code is dual-licensed MIT or Apache-2.0. A built `.rcx`
statically links its Rust dependencies; their notices are in
[THIRD-PARTY.txt](THIRD-PARTY.txt), which is packaged inside every archive.
See [COPYRIGHT](COPYRIGHT) for the whole picture.

# Configuration

Extensions load once per process, before anything compiles, so the settings
block is **server-level**: in the `.cfconfig.json` the engine starts with
(`--cfconfig`, the `CFCONFIG` env var, or the file beside the webroot). A block
in a per-application `.cfconfig.json` or `Application.cfc` cannot reach it.

```json
{
  "extensions": {
    "settings": {
      "pii": {
        "model":        "ettin-17m-nemotron-pii",
        "source":       "github",
        "modelPath":    "",
        "cacheDir":     "~/.rustcfml/models/pii",
        "autoDownload": true,
        "preload":      true,
        "windowTokens": 512,
        "chunkOverlap": 32,
        "mergeGap":     1
      }
    }
  }
}
```

Every key is optional. With no block at all, the 17M model is fetched from
this repository into `~/.rustcfml/models/pii/` the first time the engine
starts and loaded at start-up.

## Where the model comes from

| key | default | |
|---|---|---|
| `model` | `ettin-17m-nemotron-pii` | which model: `ettin-17m-nemotron-pii`, `ettin-32m-nemotron-pii` or `ettin-68m-nemotron-pii` (see [tuning](#tuning)). Any other name is a model you supply yourself, via `modelPath` or `onnxUrl`. |
| `source` | `github` | where `config.json` and `tokenizer.json` are fetched from: **`github`**, this repository's `models/` directory; or **`huggingface`**, the model's own Hub repository. The ONNX graph always comes from `onnxUrl`, because the Hub repositories publish PyTorch weights only. |
| `modelPath` | — | a directory that already holds `config.json`, `tokenizer.json` and `onnx/model.onnx`. When set nothing is downloaded and `cacheDir` is unused. The way to run air-gapped: copy `models/ettin-17m-nemotron-pii/` from a checkout and point at it. |
| `cacheDir` | `~/.rustcfml/models/pii` | where downloads go; one subdirectory per model |
| `autoDownload` | `true` | fetch missing files. Off, a missing file is an error that lists what is missing and where it was expected. |
| `onnxUrl` | per model | where the graph is fetched from. Pinned, with its SHA-256, for the three published models; required for any other. |
| `onnxSha256` | per model | expected digest of the graph; verified after download when set |
| `modelUrl` | `…/models/<model>` | base URL for the small files with `source = "github"` — a mirror, or a fork's branch |
| `hfRepo`, `hfRevision`, `hfEndpoint`, `token` | `kalyan-ks/<model>`, `main`, `https://huggingface.co`, `$HF_TOKEN` | the Hub repository with `source = "huggingface"`; the token is sent to the Hub only |

### What the download does

Each file is fetched to `<cacheDir>/<model>/<file>.part` and renamed into place
only when complete — a half-fetched `model.onnx` is never mistaken for a whole
one on the next start. The ONNX graph is checked against the SHA-256 of the
export it came from (Hub files, when that source is used, against the digest
the Hub publishes). The model card is fetched beside the weights. A line is
written to stderr saying what was fetched and from where.

A load that fails — offline, a typo in `model`, a corrupt file — is reported
on stderr at start-up (with `preload`) and then by every `pii*` call, and is
**not retried per request**: a server must not attempt a 68 MB download on
every call because the first one was offline. `piiLoad()` retries;
`piiSettings().error` has the reason.

## Loading

| key | default | |
|---|---|---|
| `preload` | `true` | load at process start. Off, the first call pays the load (~150–200 ms from disk; the download too if the cache is cold). `piiLoad()` warms it explicitly either way. |

The loaded 17M model is ~90 MB resident (weights plus the optimised plan). One
instance serves every request thread at once — inference is re-entrant and
nothing on the call path takes a lock. A sentence takes 2–3 ms on one core, a
full 512-token window about 65 ms.

## Tuning

The defaults are the published model and its author's post-processing. These
are the levers, in the order they are worth reaching for.

### Pick a size

Three sizes of the same model are published; all three are exported here and
share one label set and one tokenizer. Figures are the authors' on the
Nemotron-PII test set (10k samples); the right-hand columns are from running
all three on the same sentences in this extension.

| `model` | parameters | download | F1 | precision | recall | what we saw |
|---|---|---|---|---|---|---|
| `ettin-17m-nemotron-pii` (default) | 17M | 68 MB | 94.21 | 94.48 | 93.93 | fastest; the most false positives in plain prose (a stray `to` as `email` at 0.5–0.8 once a document is a few hundred tokens long); UK postcodes fragmented; a card number with spaces found only when it looks real |
| `ettin-32m-nemotron-pii` | 32M | 129 MB | 95.73 | 95.96 | 95.49 | clean on prose; postcodes whole; names and cities more confident |
| `ettin-68m-nemotron-pii` | 68M | 274 MB | 96.27 | 96.35 | 96.19 | cleanest; the best boundaries on addresses; ~4× the 17M's inference time |

Switching is one setting and a restart:

```json
{ "extensions": { "settings": { "pii": { "model": "ettin-68m-nemotron-pii" } } } }
```

The two larger graphs are fetched from this repository's `models-v1` release
(they are over GitHub's 100 MB in-tree limit); the 17M one from the tree.

The data the models were trained on is US-centric synthetic text. Expect UK
and EU identifiers (NHS numbers, National Insurance numbers, IBANs written
with spaces) to be found inconsistently or labelled as a neighbouring type.
Measure on your own traffic before relying on a type — the model card beside
each export lists per-type precision and recall, and `occupation`, `time`,
`state`, `city` and `company_name` are the weak ones on every size.

### `minScore`

`piiDetect( text, { minScore = 0.5 } )`. The score is the weakest token in the
span. On the 17M model the false positives on ordinary prose sit at 0.5–0.8
while real emails, phones and names are mostly above 0.9 — but a correct
`last_name` can score 0.4 (`Smith` above), so a threshold trades a few real
spans for fewer false ones. It is a per-call option rather than a setting so
that a masking pass (where a false positive is cheap and a miss is not) and a
reporting pass can use different values on the same model.

### `types`

Restrict a call to the types that matter: `{ types = "email,phone_number" }`.
Cheaper than filtering afterwards only in that it is less to carry; the model
always runs the full label set.

### `windowTokens` and `chunkOverlap`

| key | default | |
|---|---|---|
| `windowTokens` | `512` | tokens per model pass, including the two special tokens. The model's context is 7,999; a bigger window gives attention more context at a cost that grows with the square of the window on the global-attention layers. |
| `chunkOverlap` | `32` | tokens shared between consecutive windows of a text longer than one window. Each token takes its prediction from the window where it sits furthest from an edge, so a span straddling a boundary is neither split nor lost. |

Texts shorter than the window are a single pass and these do not apply. Raise
the window for documents where an entity's meaning depends on text far away
from it; lower it (`128`, overlap `16`) to cap per-request latency on a server
where a request may carry a very long document.

### `mergeGap`

| key | default | |
|---|---|---|
| `mergeGap` | `1` | characters allowed between two pieces of the same type for them to be one span. `1` is the model author's rule (a space). `0` requires pieces to touch. Larger values join, for example, `4111 1111 1111 1111` when the model labels the groups separately — at the price of joining two genuinely separate entities of one type that happen to be close. |

### A model of your own

Any ModernBERT / BERT-style token classifier exported with
`input_ids` + `attention_mask` → `logits` works. Export it with
`scripts/export-onnx.py` (or the same recipe: float32, eager attention, opset
17), put `config.json`, `tokenizer.json` and `onnx/model.onnx` in a directory,
and set `model` to a name of your choosing plus `modelPath` to the directory.
Labels may be `B-`/`I-` prefixed or plain; `O` is the outside label.

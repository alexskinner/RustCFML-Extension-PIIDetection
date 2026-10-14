# RustCFML PII extension — working notes

A `.rcx` extension that detects and masks personal data in text with a local
ONNX model. See `README.md` for usage, `docs/functions.md` for the API and
`docs/configuration.md` for the settings block.

## Build

Clone next to a RustCFML checkout — `Cargo.toml` points at
`../RustCFML/crates/rustcfml-module`.

```bash
cargo test                      # pure-Rust tests: spans, masking, chunking, config
PII_MODEL_DIR=$PWD/models/ettin-17m-nemotron-pii cargo test   # + the real-model tests
rustcfml ext build .            # produces pii-<version>.rcx
./scripts/gen-licenses.sh       # after ANY dependency change; commit the result
```

The real-model tests PRINT that they were skipped when `PII_MODEL_DIR` is
unset; they never pass vacuously. CI sets it to the in-tree model. A full
local run is:

```bash
rustcfml ext install pii-0.1.0.rcx --dir examples/redact/extensions
( cd examples/redact && rustcfml selftest.cfm )     # fetches the model on first run
```

`Cargo.toml` turns `debug-assertions` OFF for `tract-core` in dev builds: its
debug-only per-node shape check is wrong at one attention-mask `Where` in the
exported ModernBERT graph (declared `[1,1,seq,seq]`, actual `[1,1,1,seq]`,
broadcast later). Release builds skip the check and match onnxruntime to 1e-5;
without the override `cargo test` cannot load the model.

## The one invariant

**The only RustCFML dependency is `rustcfml-module`.** The ABI wrapper is a C
boundary over opaque handles; reaching past it couples this extension to the
engine's Rust ABI and means it must be rebuilt in lockstep.

## Rules that are not stylistic

- **Everything must stay permissively licensed.** RustCFML is MIT. The model
  is MIT (ettin-*-nemotron-pii), its base is MIT, its data CC BY 4.0, and the
  attribution lives in `models/README.md` + the model card beside each export.
  Check a replacement model's licence BEFORE building on it — a non-commercial
  model was built in and torn out once already. The ONNX exports are committed
  in-tree (17m) or as release assets (32m/68m, over GitHub's 100 MB limit);
  `src/config.rs` pins their SHA-256s. They are not inside the `.rcx`.
- **GitHub is the default download source, Hugging Face the option.** The Hub
  repos publish PyTorch weights only, so the graph always comes from this repo
  (`onnxUrl`); `source = "huggingface"` only changes where config/tokenizer come
  from.
- **Positions are characters, never bytes.** `tokenizers` offsets come from
  `encode_char_offsets`; `spans::mask` and `spans::slice` index `chars()`. The
  CFML edge is 1-based `start` + `length` so `mid(text, start, length)` is the
  span. A byte offset leaking out shows up on the first accented name.
- **Expected spans in tests come from onnxruntime, not from this crate.** A
  divergence means our inference or decoding is wrong, not the model. The
  export script verifies ONNX against PyTorch; the tests verify tract against
  those onnxruntime answers. Score boundary EQUALITY, never containment
  — an over-extended span passes a `contains` check and still corrupts masked
  text.
- **Labels are BIO but the prefix is IGNORED when grouping**, as in the model
  author's own post-processing: the model emits `B-` on every sub-word piece
  of one entity, so honouring it splits emails in two. Same-type pieces merge
  when within `mergeGap` (1) characters; spans are whitespace-trimmed.
- **Long text is chunked, 512 tokens with a 32-token overlap**, `[CLS]`/`[SEP]`
  added PER WINDOW, and each token takes the prediction from the window where
  it sits furthest from an edge. `engine::windows` is unit-tested; the
  straddling-span test in `tests/real_model.rs` is the end-to-end check.
- **The 17M model false-positives on plain prose** (`to` as `email` at 0.5–0.8
  once a text is a few hundred tokens) and is confused by synthetic filler.
  Fixtures use varied natural sentences and assert the real span is present
  and whole, not that nothing else fires. 32m/68m are clean; documented in
  docs/configuration.md#tuning.
- **The exported graph fails in tract at sequence length 2** (the mask
  collapses a dimension). A real window is never shorter than 3 tokens; the
  load-time probe uses 3.
- **A failed load is remembered**, not retried per request: an offline server
  must not attempt a 42 MB download on every call. `piiLoad()` retries.
- **Pure Rust only.** tract, `tokenizers` with `fancy-regex` (not `onig`),
  `ureq` with rustls. No C library, no OpenSSL, nothing to find at load time —
  that is what makes one source build on every platform the engine ships for.
- **THIRD-PARTY.txt is generated and committed**, and CI fails if it is stale.

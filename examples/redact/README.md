# Example: detect and mask personal data

Runs the extension end to end against sentences whose answers are known
(taken from onnxruntime on the same model), exercising every function.

```bash
rustcfml ext install ../../pii-0.1.0.rcx --dir extensions
rustcfml selftest.cfm
```

`selftest.cfm` is also what CI runs, so it asserts rather than just printing:
it exits non-zero if anything is wrong, and prints `all checks passed` when
everything is right.

`.cfconfig.json` here points `modelPath` at the checkout's own copy of the
model, so the run needs no network. Remove that key to exercise the download
path instead (the model is then fetched from this repository on GitHub into
`~/.rustcfml/models/pii/`). The settings block is server-level — it is read
once when the engine starts, which is why it sits beside the page rather than
in an `Application.cfc`.

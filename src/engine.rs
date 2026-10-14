//! The model: tokenizer + ONNX graph, loaded once, run per call.
//!
//! `Engine` is `Send + Sync` and every method takes `&self`, so one instance
//! serves every request thread at once: tract's `run` and the tokenizer's
//! `encode` are both re-entrant. No locking happens on the call path.

use std::path::Path;
use std::sync::Arc;

use tokenizers::Tokenizer;
use tract_onnx::prelude::*;

use crate::spans::{self, Span, TokenPrediction};

pub struct Engine {
    tokenizer: Tokenizer,
    model: Arc<TypedRunnableModel>,
    /// `id2label`, by index: `O`, `B-email`, `I-email`, …
    labels: Vec<String>,
    /// The entity type of each label with its `B-`/`I-` prefix removed;
    /// `None` for `O`. Index-aligned with `labels`.
    types: Vec<Option<String>>,
    /// The distinct entity types, sorted — what `piiTypes()` returns.
    type_names: Vec<String>,
    /// Tokens per window, INCLUDING the two special tokens.
    window: usize,
    overlap: usize,
    merge_gap: usize,
    cls: u32,
    sep: u32,
    ids_input: usize,
    mask_input: usize,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("labels", &self.labels.len())
            .field("types", &self.type_names.len())
            .field("window", &self.window)
            .finish()
    }
}

#[derive(serde::Deserialize)]
struct ModelConfig {
    id2label: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
}

/// How the engine is to be loaded. Separate from `config::Config` so the
/// tests (and anyone embedding this crate) can load from a directory without
/// going through the settings block.
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    pub window_tokens: usize,
    pub chunk_overlap: usize,
    pub merge_gap: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            window_tokens: crate::config::DEFAULT_WINDOW,
            chunk_overlap: crate::config::DEFAULT_OVERLAP,
            merge_gap: crate::config::DEFAULT_MERGE_GAP,
        }
    }
}

impl Engine {
    /// Load from a directory holding `config.json`, `tokenizer.json` and
    /// `onnx/model.onnx`.
    pub fn load(dir: &Path, opts: LoadOptions) -> Result<Engine, String> {
        let cfg: ModelConfig =
            serde_json::from_slice(&crate::fetch::read(&dir.join("config.json"))?)
                .map_err(|e| format!("pii: config.json: {e}"))?;
        let mut labels: Vec<(usize, String)> = cfg
            .id2label
            .iter()
            .map(|(k, v)| {
                k.parse::<usize>()
                    .map(|i| (i, v.clone()))
                    .map_err(|_| format!("pii: config.json: id2label key [{k}] is not a number"))
            })
            .collect::<Result<_, _>>()?;
        labels.sort_by_key(|(i, _)| *i);
        for (expect, (i, _)) in labels.iter().enumerate() {
            if *i != expect {
                return Err(format!(
                    "pii: config.json: id2label is missing index {expect}"
                ));
            }
        }
        let labels: Vec<String> = labels.into_iter().map(|(_, l)| l).collect();
        if !labels.iter().any(|l| l == "O") {
            return Err("pii: config.json: id2label has no `O` (outside) label".into());
        }
        let types: Vec<Option<String>> = labels.iter().map(|l| entity_type(l)).collect();
        let mut type_names: Vec<String> = types.iter().flatten().cloned().collect();
        type_names.sort();
        type_names.dedup();

        let max_pos = cfg.max_position_embeddings.unwrap_or(512);
        if opts.window_tokens > max_pos {
            return Err(format!(
                "pii: windowTokens ({}) exceeds the model's {max_pos}-token context",
                opts.window_tokens
            ));
        }
        if opts.chunk_overlap + 2 >= opts.window_tokens || opts.window_tokens < 16 {
            return Err(format!(
                "pii: chunkOverlap ({}) must be smaller than windowTokens ({}) less the two special tokens",
                opts.chunk_overlap, opts.window_tokens
            ));
        }

        let tok_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("pii: {}: {e}", tok_path.display()))?;
        let special = |name: &str| {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| format!("pii: the tokenizer has no {name} token"))
        };
        let cls = special("[CLS]")?;
        let sep = special("[SEP]")?;

        let onnx_path = dir.join(crate::config::ONNX_FILE);
        let graph = tract_onnx::onnx()
            .model_for_path(&onnx_path)
            .map_err(|e| format!("pii: {}: {e}", onnx_path.display()))?;

        let input_names: Vec<String> = graph
            .inputs
            .iter()
            .map(|o| graph.node(o.node).name.to_ascii_lowercase())
            .collect();
        let find_input = |want: &str| {
            input_names.iter().position(|n| n == want).ok_or_else(|| {
                format!("pii: the graph has no input named {want}; it has {input_names:?}")
            })
        };
        let ids_input = find_input("input_ids")?;
        let mask_input = find_input("attention_mask")?;
        if graph.inputs.len() != 2 || graph.outputs.len() != 1 {
            return Err(format!(
                "pii: the graph has {} inputs and {} outputs; this extension knows \
                 input_ids + attention_mask → logits",
                graph.inputs.len(),
                graph.outputs.len()
            ));
        }

        // The batch dimension is FIXED at 1 and the sequence left symbolic.
        // The exporter builds ModernBERT's attention masks with a Flatten over
        // [batch, seq], whose size tract cannot express as a product of two
        // symbols; with the batch known it is a plain multiple of `seq`, and
        // one plan then serves every length.
        let seq = graph.symbols.sym("seq");
        let fact =
            |_: usize| InferenceFact::dt_shape(i64::datum_type(), tvec![1.to_dim(), seq.to_dim()]);
        let model = graph
            .with_input_fact(ids_input, fact(ids_input))
            .and_then(|g| g.with_input_fact(mask_input, fact(mask_input)))
            .map_err(|e| format!("pii: {}: {e}", onnx_path.display()))?
            .into_optimized()
            .map_err(|e| format!("pii: optimising {}: {e}", onnx_path.display()))?
            .into_runnable()
            .map_err(|e| format!("pii: {}: {e}", onnx_path.display()))?;

        let engine = Engine {
            tokenizer,
            model,
            labels,
            types,
            type_names,
            window: opts.window_tokens,
            overlap: opts.chunk_overlap,
            merge_gap: opts.merge_gap,
            cls,
            sep,
            ids_input,
            mask_input,
        };
        // One probe run: the shape of the output is checked here, at load,
        // rather than on the first request. Three tokens, not two: the
        // exported sliding-window mask collapses a dimension at length 2,
        // and a real window is never shorter than [CLS] + 1 token + [SEP].
        let probe = engine.run_window(&[cls, sep, sep])?;
        if probe.len() != 3 * engine.labels.len() {
            return Err(format!(
                "pii: the graph produced {} values for 3 tokens; expected 3 × {} labels",
                probe.len(),
                engine.labels.len()
            ));
        }
        Ok(engine)
    }

    /// The entity types this model emits, sorted.
    pub fn types(&self) -> &[String] {
        &self.type_names
    }

    pub fn window_tokens(&self) -> usize {
        self.window
    }

    /// Run one window (special tokens included) and return the logits,
    /// `tokens × labels`, row-major.
    fn run_window(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let n = ids.len();
        let ids_t = Tensor::from_shape(
            &[1, n],
            &ids.iter().map(|&i| i as i64).collect::<Vec<i64>>(),
        )
        .map_err(|e| format!("pii: {e}"))?;
        let mask_t =
            Tensor::from_shape(&[1, n], &vec![1i64; n]).map_err(|e| format!("pii: {e}"))?;
        let mut inputs: TVec<TValue> = tvec![ids_t.clone().into(), ids_t.into()];
        inputs[self.ids_input] = inputs[self.ids_input].clone();
        inputs[self.mask_input] = mask_t.into();
        let out = self
            .model
            .run(inputs)
            .map_err(|e| format!("pii: inference failed: {e}"))?;
        let view = out[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| format!("pii: logits: {e}"))?;
        Ok(view.iter().copied().collect())
    }

    /// Detect everything in `text`.
    pub fn detect(&self, text: &str) -> Result<Vec<Span>, String> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        // Special tokens are added PER WINDOW below, not here, so that every
        // window the model sees is a well-formed `[CLS] … [SEP]` sequence.
        let enc = self
            .tokenizer
            .encode_char_offsets(text, false)
            .map_err(|e| format!("pii: tokenising: {e}"))?;
        let ids = enc.get_ids();
        let offsets = enc.get_offsets();
        let n = ids.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Per token: the prediction from whichever window saw it furthest
        // from an edge, where attention has the most context on both sides.
        let mut best_margin = vec![usize::MAX; n];
        let mut preds: Vec<TokenPrediction> = offsets
            .iter()
            .map(|&(s, e)| TokenPrediction {
                start: s,
                end: e,
                label: 0,
                prob: 0.0,
            })
            .collect();

        let n_labels = self.labels.len();
        let body = self.window - 2;
        for (ws, we) in windows(n, body, self.overlap) {
            let len = we - ws;
            let mut window_ids = Vec::with_capacity(len + 2);
            window_ids.push(self.cls);
            window_ids.extend_from_slice(&ids[ws..we]);
            window_ids.push(self.sep);
            let logits = self.run_window(&window_ids)?;
            if logits.len() != (len + 2) * n_labels {
                return Err(format!(
                    "pii: logits are {} values for {} tokens × {n_labels} labels",
                    logits.len(),
                    len + 2
                ));
            }
            for j in 0..len {
                let i = ws + j;
                let margin = j.min(len - 1 - j);
                if best_margin[i] != usize::MAX && margin <= best_margin[i] {
                    continue;
                }
                best_margin[i] = margin;
                // Row j+1: row 0 is [CLS].
                let row = &logits[(j + 1) * n_labels..(j + 2) * n_labels];
                let (label, prob) = argmax_softmax(row);
                preds[i].label = label;
                preds[i].prob = prob;
            }
        }

        Ok(spans::group(text, &preds, &self.types, self.merge_gap))
    }
}

/// `B-email` → `email`, `I-email` → `email`, `email` → `email`, `O` → None.
fn entity_type(label: &str) -> Option<String> {
    if label == "O" {
        return None;
    }
    let t = label
        .strip_prefix("B-")
        .or_else(|| label.strip_prefix("I-"))
        .or_else(|| label.strip_prefix("S-"))
        .or_else(|| label.strip_prefix("E-"))
        .unwrap_or(label);
    Some(t.to_ascii_lowercase())
}

/// `[start, end)` token windows covering `n` tokens, `size` wide, sharing
/// `overlap` tokens with the previous one. The last window is pulled back so
/// it ends exactly at `n` and is never shorter than the others unless the
/// whole text is.
pub fn windows(n: usize, size: usize, overlap: usize) -> Vec<(usize, usize)> {
    if n <= size {
        return vec![(0, n)];
    }
    let stride = size - overlap;
    let mut out = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + size).min(n);
        let start_adj = end.saturating_sub(size);
        out.push((start_adj, end));
        if end == n {
            break;
        }
        start += stride;
    }
    out
}

fn argmax_softmax(row: &[f32]) -> (usize, f32) {
    let mut am = 0usize;
    let mut mx = f32::NEG_INFINITY;
    for (j, v) in row.iter().enumerate() {
        if *v > mx {
            mx = *v;
            am = j;
        }
    }
    let mut z = 0f32;
    for v in row {
        z += (*v - mx).exp();
    }
    (am, 1.0 / z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_text_is_one_window() {
        assert_eq!(windows(35, 510, 32), vec![(0, 35)]);
        assert_eq!(windows(510, 510, 32), vec![(0, 510)]);
    }

    #[test]
    fn long_texts_are_covered_with_overlap_and_the_last_window_is_full_width() {
        let w = windows(1200, 510, 32);
        assert_eq!(w[0], (0, 510));
        assert_eq!(w[1], (478, 988));
        assert_eq!(*w.last().unwrap(), (690, 1200));
        for pair in w.windows(2) {
            assert!(pair[0].1 - pair[1].0 >= 32, "{pair:?}");
        }
    }

    #[test]
    fn one_token_past_the_window_still_gets_two_full_windows() {
        assert_eq!(windows(511, 510, 32), vec![(0, 510), (1, 511)]);
    }

    #[test]
    fn bio_prefixes_are_stripped_and_o_is_none() {
        assert_eq!(entity_type("B-email").as_deref(), Some("email"));
        assert_eq!(
            entity_type("I-phone_number").as_deref(),
            Some("phone_number")
        );
        assert_eq!(entity_type("PERSON").as_deref(), Some("person"));
        assert_eq!(entity_type("O"), None);
    }

    #[test]
    fn argmax_softmax_reports_the_winning_probability() {
        let (i, p) = argmax_softmax(&[0.0, 2.0, 0.0]);
        assert_eq!(i, 1);
        let want = 2f32.exp() / (2f32.exp() + 2.0);
        assert!((p - want).abs() < 1e-6);
    }
}

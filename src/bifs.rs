//! The CFML functions.
//!
//! Positions in the structs returned here are CFML positions: `start` is the
//! 1-based character index and `length` the character count, so
//! `mid( text, span.start, span.length )` is exactly `span.text`. `end` is the
//! 1-based index of the last character, inclusive.

use rustcfml_module::{abi::ty, Ctx, Error, Result, Value};

use crate::spans::{self, MaskStyle, Span};
use crate::state;

fn text_arg(args: &[Value<'_>], func: &str) -> Result<String> {
    match args.first() {
        Some(v) if !v.is_null() => Ok(v.to_string()),
        _ => Err(Error::expression(format!(
            "{func}: argument 1 [text] is required"
        ))),
    }
}

/// `types` / `minScore`, shared by detect and mask.
struct Filter {
    types: Option<Vec<String>>,
    min_score: f32,
}

impl Filter {
    fn from(opts: Option<Value<'_>>, func: &str, labels: &[String]) -> Result<Filter> {
        let mut f = Filter {
            types: None,
            min_score: 0.0,
        };
        let Some(o) = opts else { return Ok(f) };
        if o.is_null() {
            return Ok(f);
        }
        if o.kind() != ty::STRUCT {
            return Err(Error::expression(format!(
                "{func}: argument 2 [options] must be a struct"
            )));
        }
        let t = o.key("types");
        if !t.is_null() {
            let mut wanted: Vec<String> = if t.kind() == ty::ARRAY {
                t.to_vec()?.into_iter().map(|v| v.to_string()).collect()
            } else {
                t.to_string().split(',').map(str::to_string).collect()
            };
            wanted = wanted
                .into_iter()
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            for w in &wanted {
                if !labels.iter().any(|l| l == w) {
                    return Err(Error::expression(format!(
                        "{func}: [{w}] is not a PII type this model knows; piiTypes() lists the {} it does",
                        labels.len()
                    )));
                }
            }
            f.types = Some(wanted);
        }
        let m = o.key("minScore");
        if !m.is_null() {
            let v = m.as_f64().map_err(|_| {
                Error::expression(format!("{func}: options.minScore must be a number"))
            })?;
            if !(0.0..=1.0).contains(&v) {
                return Err(Error::expression(format!(
                    "{func}: options.minScore must be between 0 and 1"
                )));
            }
            f.min_score = v as f32;
        }
        Ok(f)
    }

    fn apply(&self, spans: Vec<Span>) -> Vec<Span> {
        spans
            .into_iter()
            .filter(|s| s.score >= self.min_score)
            .filter(|s| match &self.types {
                Some(t) => t.iter().any(|w| w == &s.label),
                None => true,
            })
            .collect()
    }
}

fn span_struct<'a>(ctx: &'a Ctx, text: &str, s: &Span) -> Result<Value<'a>> {
    let out = ctx.strukt();
    out.put("type", ctx.string(&s.label))?;
    out.put("text", ctx.string(spans::slice(text, s)))?;
    out.put("start", ctx.int(s.start as i64 + 1))?;
    out.put("end", ctx.int(s.end as i64))?;
    out.put("length", ctx.int((s.end - s.start) as i64))?;
    out.put("score", ctx.double(round4(s.score)))?;
    Ok(out)
}

fn span_array<'a>(ctx: &'a Ctx, text: &str, spans: &[Span]) -> Result<Value<'a>> {
    let arr = ctx.array_with_capacity(spans.len());
    for s in spans {
        arr.push(span_struct(ctx, text, s)?)?;
    }
    Ok(arr)
}

fn round4(v: f32) -> f64 {
    (v as f64 * 10_000.0).round() / 10_000.0
}

fn detect_spans(text: &str) -> Result<(Vec<Span>, std::sync::Arc<crate::engine::Engine>)> {
    let engine = state::engine().map_err(Error::new)?;
    let spans = engine.detect(text).map_err(Error::new)?;
    Ok((spans, engine))
}

/// `piiDetect( text [, { types, minScore } ] )` — an array of entity spans.
pub fn detect<'a>(ctx: &'a Ctx, args: &[Value<'a>]) -> Result<Value<'a>> {
    const F: &str = "piiDetect";
    let text = text_arg(args, F)?;
    let (spans, engine) = detect_spans(&text)?;
    let filter = Filter::from(args.get(1).copied(), F, engine.types())?;
    span_array(ctx, &text, &filter.apply(spans))
}

/// `piiMask( text [, { replacement, maskChar, keepLast, types, minScore } ] )`
/// — the text with every detected span replaced.
pub fn mask<'a>(ctx: &'a Ctx, args: &[Value<'a>]) -> Result<Value<'a>> {
    const F: &str = "piiMask";
    let text = text_arg(args, F)?;
    let (spans, engine) = detect_spans(&text)?;
    let opts = args.get(1).copied().filter(|v| !v.is_null());
    let filter = Filter::from(opts, F, engine.types())?;

    let style = match opts {
        Some(o) => {
            let mc = o.key("maskChar");
            if !mc.is_null() && !mc.to_string().is_empty() {
                let keep = o.key("keepLast");
                let keep_last = if keep.is_null() {
                    0
                } else {
                    keep.as_i64().ok().filter(|n| *n >= 0).ok_or_else(|| {
                        Error::expression(format!("{F}: options.keepLast must be a whole number"))
                    })? as usize
                };
                MaskStyle::Char {
                    ch: mc.to_string(),
                    keep_last,
                }
            } else {
                let r = o.key("replacement");
                if r.is_null() {
                    MaskStyle::Template("[{TYPE}]".into())
                } else {
                    MaskStyle::Template(r.to_string())
                }
            }
        }
        None => MaskStyle::Template("[{TYPE}]".into()),
    };
    let masked = spans::mask(&text, &filter.apply(spans), &style).map_err(Error::new)?;
    Ok(ctx.string(masked))
}

/// `piiTypes()` — the entity types the model emits, sorted.
pub fn types<'a>(ctx: &'a Ctx, _args: &[Value<'a>]) -> Result<Value<'a>> {
    let engine = state::engine().map_err(Error::new)?;
    let arr = ctx.array_with_capacity(engine.types().len());
    for t in engine.types() {
        arr.push(ctx.string(t))?;
    }
    Ok(arr)
}

/// `piiIsReady()` — whether the model is loaded. Never loads it.
pub fn is_ready<'a>(ctx: &'a Ctx, _args: &[Value<'a>]) -> Result<Value<'a>> {
    Ok(ctx.bool(state::ready()))
}

/// `piiLoad()` — load now (downloading if needed), retrying a failed load.
/// Returns what `piiSettings()` returns.
pub fn load<'a>(ctx: &'a Ctx, args: &[Value<'a>]) -> Result<Value<'a>> {
    state::load().map_err(Error::new)?;
    settings(ctx, args)
}

/// `piiSettings()` — the effective configuration and load state.
pub fn settings<'a>(ctx: &'a Ctx, _args: &[Value<'a>]) -> Result<Value<'a>> {
    let cfg = state::config();
    let out = ctx.strukt();
    out.put("model", ctx.string(&cfg.model))?;
    out.put("source", ctx.string(cfg.source.name()))?;
    out.put("modelUrl", ctx.string(&cfg.model_url))?;
    out.put("hfRepo", ctx.string(&cfg.hf_repo))?;
    out.put("onnxUrl", ctx.string(&cfg.onnx_url))?;
    out.put(
        "modelDir",
        ctx.string(cfg.model_dir().display().to_string()),
    )?;
    out.put("cacheDir", ctx.string(cfg.cache_dir.display().to_string()))?;
    out.put("autoDownload", ctx.bool(cfg.auto_download))?;
    out.put("preload", ctx.bool(cfg.preload))?;
    out.put("windowTokens", ctx.int(cfg.window_tokens as i64))?;
    out.put("chunkOverlap", ctx.int(cfg.chunk_overlap as i64))?;
    out.put("mergeGap", ctx.int(cfg.merge_gap as i64))?;
    out.put("ready", ctx.bool(state::ready()))?;
    match state::last_error() {
        Some(e) => out.put("error", ctx.string(e))?,
        None => out.put("error", ctx.string(""))?,
    }
    if let Some(ms) = state::load_millis() {
        out.put("loadMillis", ctx.int(ms as i64))?;
    }
    if state::ready() {
        if let Ok(e) = state::engine() {
            out.put("typeCount", ctx.int(e.types().len() as i64))?;
        }
    }
    Ok(out)
}

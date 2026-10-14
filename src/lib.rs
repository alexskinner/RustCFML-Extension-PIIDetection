//! # RustCFML PII detection extension
//!
//! Finds personal data in text — names, emails, phone numbers, addresses,
//! identifiers, 55 types in all — and masks it, in-process, with no API to
//! call and nothing leaving the machine.
//!
//! ```cfml
//! spans = piiDetect( "Contact John Smith at john@example.com" );
//! // [ { type: "first_name", text: "John",  start: 9,  end: 12, length: 4,  score: 0.995 },
//! //   { type: "last_name",  text: "Smith", start: 14, end: 18, length: 5,  score: 0.393 },
//! //   { type: "email", text: "john@example.com", start: 23, end: 38, length: 16, score: 0.981 } ]
//!
//! piiMask( "Contact John Smith at john@example.com" );
//! // "Contact [FIRST_NAME] [LAST_NAME] at [EMAIL]"
//! ```
//!
//! The detector is [ettin-17m-nemotron-pii](https://huggingface.co/kalyan-ks/ettin-17m-nemotron-pii)
//! (MIT), a 17M-parameter ModernBERT token classifier fine-tuned on NVIDIA's
//! Nemotron-PII, run here as ONNX with [tract](https://github.com/sonos/tract)
//! — pure Rust, CPU, a few milliseconds per paragraph. The graph is exported
//! from the published weights by `scripts/export-onnx.py`; the files are
//! fetched on first use (see `config.rs`), never bundled.
//!
//! ## What is here
//!
//! * `config.rs` — the settings block and where the model lives.
//! * `fetch.rs` — downloading it, verified and atomic.
//! * `engine.rs` — tokenizer + graph; chunking long texts into windows;
//!   decoding the BIO labels.
//! * `spans.rs` — token predictions → character spans → masked text.
//! * `state.rs` — once-per-process load, lazily or at start-up.
//! * `bifs.rs` — the six CFML functions.
//!
//! The only RustCFML dependency is `rustcfml-module`, the extension ABI
//! wrapper. No engine internals.

pub mod config;
pub mod engine;
pub mod fetch;
pub mod spans;
pub mod state;

mod bifs;

use rustcfml_module::{module, Ctx, Error, Result, Value};

fn on_load(_ctx: &Ctx, settings: Value) -> Result<()> {
    let cfg = config::Config::from_settings(settings).map_err(|e| {
        eprintln!("{e}");
        Error::new(e)
    })?;
    let preload = cfg.preload;
    state::set_config(cfg);
    if preload {
        // A failure here is REMEMBERED and reported by the first call, not
        // fatal to the load: an offline box should still have piiSettings()
        // and piiLoad() to diagnose and retry with, and the rest of the
        // application should not lose every pii* function over it.
        if let Err(e) = state::load() {
            eprintln!("{e}");
        }
    }
    Ok(())
}

module! {
    name: "pii",
    version: "0.1.0",
    bifs: {
        "piiDetect"   => bifs::detect,
        "piiMask"     => bifs::mask,
        "piiTypes"    => bifs::types,
        "piiIsReady"  => bifs::is_ready,
        "piiLoad"     => bifs::load,
        "piiSettings" => bifs::settings,
    },
    on_load: on_load,
}

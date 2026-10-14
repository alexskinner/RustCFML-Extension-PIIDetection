//! Process-wide state: the settings read at load, and the engine once it is up.
//!
//! The engine loads at most once per process, lazily or at start-up as
//! `preload` says. A failed load is remembered and reported by every call
//! until `piiLoad()` retries it — a server must not re-attempt a 68 MB
//! download on every request because the first one was offline.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::config::Config;
use crate::engine::{Engine, LoadOptions};

static CONFIG: OnceLock<Config> = OnceLock::new();
static ENGINE: Mutex<Slot> = Mutex::new(Slot::Unloaded);

enum Slot {
    Unloaded,
    Ready(Arc<Engine>, u128),
    Failed(String),
}

pub fn set_config(cfg: Config) {
    // Extensions load once; a second call would be the engine's bug, and the
    // first configuration is the one every later decision was made from.
    let _ = CONFIG.set(cfg);
}

pub fn config() -> &'static Config {
    CONFIG.get_or_init(Config::defaults)
}

/// The engine, loading it (and fetching the model) if this is the first call.
pub fn engine() -> Result<Arc<Engine>, String> {
    let mut slot = ENGINE.lock().unwrap_or_else(|p| p.into_inner());
    match &*slot {
        Slot::Ready(e, _) => return Ok(e.clone()),
        Slot::Failed(msg) => return Err(format!("{msg} (call piiLoad() to retry)")),
        Slot::Unloaded => {}
    }
    match load_now() {
        Ok((e, ms)) => {
            *slot = Slot::Ready(e.clone(), ms);
            Ok(e)
        }
        Err(msg) => {
            *slot = Slot::Failed(msg.clone());
            Err(msg)
        }
    }
}

/// Load unconditionally, replacing a failed attempt. Returns the engine and
/// how long the load took in milliseconds (0 when it was already up).
pub fn load() -> Result<(Arc<Engine>, u128), String> {
    let mut slot = ENGINE.lock().unwrap_or_else(|p| p.into_inner());
    if let Slot::Ready(e, ms) = &*slot {
        return Ok((e.clone(), *ms));
    }
    match load_now() {
        Ok((e, ms)) => {
            *slot = Slot::Ready(e.clone(), ms);
            Ok((e, ms))
        }
        Err(msg) => {
            *slot = Slot::Failed(msg.clone());
            Err(msg)
        }
    }
}

fn load_now() -> Result<(Arc<Engine>, u128), String> {
    let cfg = config();
    let t0 = Instant::now();
    let fetched = crate::fetch::ensure_present(cfg)?;
    if !fetched.is_empty() {
        eprintln!(
            "pii: fetched {} from {} into {}",
            fetched.join(", "),
            cfg.source_page(),
            cfg.model_dir().display()
        );
    }
    let engine = Engine::load(
        &cfg.model_dir(),
        LoadOptions {
            window_tokens: cfg.window_tokens,
            chunk_overlap: cfg.chunk_overlap,
            merge_gap: cfg.merge_gap,
        },
    )?;
    Ok((Arc::new(engine), t0.elapsed().as_millis()))
}

/// Up, without side effects.
pub fn ready() -> bool {
    matches!(
        &*ENGINE.lock().unwrap_or_else(|p| p.into_inner()),
        Slot::Ready(..)
    )
}

/// The remembered failure, if the last load failed.
pub fn last_error() -> Option<String> {
    match &*ENGINE.lock().unwrap_or_else(|p| p.into_inner()) {
        Slot::Failed(m) => Some(m.clone()),
        _ => None,
    }
}

/// Milliseconds the successful load took, if it has happened.
pub fn load_millis() -> Option<u128> {
    match &*ENGINE.lock().unwrap_or_else(|p| p.into_inner()) {
        Slot::Ready(_, ms) => Some(*ms),
        _ => None,
    }
}

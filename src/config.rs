//! Settings, read once at load from the server-level `.cfconfig.json`:
//!
//! ```json
//! { "extensions": { "settings": { "pii": {
//!     "model":        "ettin-17m-nemotron-pii",
//!     "source":       "github",
//!     "modelPath":    "/opt/models/pii",
//!     "cacheDir":     "~/.rustcfml/models/pii",
//!     "autoDownload": true,
//!     "preload":      true,
//!     "windowTokens": 512,
//!     "chunkOverlap": 32,
//!     "mergeGap":     1
//! } } } }
//! ```
//!
//! Every key is optional; the defaults fetch the 17M model from this
//! repository into the user's cache on first use. Extensions load once per
//! process, before any template compiles, so these are server settings — a
//! per-application block cannot reach them.

use std::path::{Path, PathBuf};

use rustcfml_module::Value;

pub const DEFAULT_MODEL: &str = "ettin-17m-nemotron-pii";
pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";

/// This repository, where the exported graphs live (see `models/README.md`).
pub const REPO_RAW: &str =
    "https://raw.githubusercontent.com/alexskinner/RustCFML-Extension-PIIDetection/main/models";
pub const REPO_RELEASE: &str =
    "https://github.com/alexskinner/RustCFML-Extension-PIIDetection/releases/download/models-v1";

/// Tokens per window, including the `[CLS]`/`[SEP]` the tokenizer adds. The
/// model accepts up to 7,999, but attention cost grows with the square of the
/// window on the global layers and the training sequences were far shorter;
/// 512 is where a paragraph fits whole and a long document still runs in
/// bounded time per window.
pub const DEFAULT_WINDOW: usize = 512;
pub const DEFAULT_OVERLAP: usize = 32;
/// Characters between two same-type pieces that are still one entity — the
/// model card's merge rule (`next.start <= current.end + 1`).
pub const DEFAULT_MERGE_GAP: usize = 1;

pub const ONNX_FILE: &str = "onnx/model.onnx";
pub const HUB_FILES: [&str; 3] = ["config.json", "tokenizer.json", "tokenizer_config.json"];

/// The graphs this repository publishes, with the SHA-256 of each export so a
/// download is verified byte for byte (GitHub serves no digest header).
///
/// The 17M graph is committed in-tree; the two larger ones exceed GitHub's
/// 100 MB per-file limit and are release assets instead. Their small files
/// (`config.json`, `tokenizer.json`) are in-tree for all three.
pub struct KnownModel {
    pub name: &'static str,
    pub onnx_url: &'static str,
    pub sha256: &'static str,
    /// The Hugging Face repository the export came from.
    pub hf_repo: &'static str,
}

pub const KNOWN_MODELS: [KnownModel; 3] = [
    KnownModel {
        name: "ettin-17m-nemotron-pii",
        onnx_url: "https://raw.githubusercontent.com/alexskinner/RustCFML-Extension-PIIDetection/main/models/ettin-17m-nemotron-pii/onnx/model.onnx",
        sha256: "765e0c5afb4fc13e00331a167b3dd1759dc697c28c0e5a1692f4a6a837c2d7a5",
        hf_repo: "kalyan-ks/ettin-17m-nemotron-pii",
    },
    KnownModel {
        name: "ettin-32m-nemotron-pii",
        onnx_url: "https://github.com/alexskinner/RustCFML-Extension-PIIDetection/releases/download/models-v1/ettin-32m-nemotron-pii.onnx",
        sha256: "88cb2ddb7167e9b272c14c8105d1120776b3720d3600796b8c56251c747d1ef3",
        hf_repo: "kalyan-ks/ettin-32m-nemotron-pii",
    },
    KnownModel {
        name: "ettin-68m-nemotron-pii",
        onnx_url: "https://github.com/alexskinner/RustCFML-Extension-PIIDetection/releases/download/models-v1/ettin-68m-nemotron-pii.onnx",
        sha256: "a2490d291028d186a98380aeb12d410540a027415254db5e373d827a894fa55c",
        hf_repo: "kalyan-ks/ettin-68m-nemotron-pii",
    },
];

pub fn known(name: &str) -> Option<&'static KnownModel> {
    KNOWN_MODELS
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case(name))
}

/// Where `config.json` and `tokenizer.json` come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// This repository's `models/<name>/` directory.
    GitHub,
    /// The model's own Hugging Face repository.
    HuggingFace,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::GitHub => "github",
            Source::HuggingFace => "huggingface",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// A name from `KNOWN_MODELS`, or any directory name under `cacheDir`
    /// when `modelPath`/`modelUrl`/`hfRepo` say where its files are.
    pub model: String,
    pub source: Source,
    /// Base URL for the small files when `source` is GitHub; files are
    /// appended. Defaults to this repository's `models/<model>`.
    pub model_url: String,
    /// Hugging Face repository id when `source` is HuggingFace.
    pub hf_repo: String,
    pub hf_revision: String,
    pub hf_endpoint: String,
    /// Bearer token for a gated or private Hub repository.
    pub token: Option<String>,
    /// Where the ONNX graph is fetched from.
    pub onnx_url: String,
    /// Expected SHA-256 of the ONNX file, hex; verified after download when set.
    pub onnx_sha256: Option<String>,
    /// A directory already holding the model files. When set, nothing is
    /// downloaded and `cache_dir` is unused.
    pub model_path: Option<PathBuf>,
    /// Where downloaded models live; one subdirectory per model.
    pub cache_dir: PathBuf,
    pub auto_download: bool,
    /// Load (and if needed download) at process start rather than on the first
    /// call. The first request otherwise pays the load, which is ~150 ms from
    /// disk and a 68 MB fetch when the cache is cold.
    pub preload: bool,
    pub window_tokens: usize,
    /// Tokens shared between consecutive windows of a long text.
    pub chunk_overlap: usize,
    pub merge_gap: usize,
}

impl Config {
    pub fn defaults() -> Config {
        let k = known(DEFAULT_MODEL).expect("the default model is known");
        Config {
            model: DEFAULT_MODEL.to_string(),
            source: Source::GitHub,
            model_url: format!("{REPO_RAW}/{DEFAULT_MODEL}"),
            hf_repo: k.hf_repo.to_string(),
            hf_revision: "main".to_string(),
            hf_endpoint: DEFAULT_HF_ENDPOINT.to_string(),
            token: None,
            onnx_url: k.onnx_url.to_string(),
            onnx_sha256: Some(k.sha256.to_string()),
            model_path: None,
            cache_dir: default_cache_dir(),
            auto_download: true,
            preload: true,
            window_tokens: DEFAULT_WINDOW,
            chunk_overlap: DEFAULT_OVERLAP,
            merge_gap: DEFAULT_MERGE_GAP,
        }
    }

    /// From the `settings.pii` struct the engine hands `on_load`. Unknown keys
    /// are ignored; a malformed value is an error that names the key, so a
    /// typo in `.cfconfig.json` is reported at start-up rather than silently
    /// becoming the default.
    pub fn from_settings(settings: Value<'_>) -> Result<Config, String> {
        let mut cfg = Config::defaults();
        let text = |k: &str| -> Option<String> {
            let v = settings.key(k);
            if v.is_null() {
                return None;
            }
            let s = v.to_string();
            let s = s.trim();
            (!s.is_empty()).then(|| s.to_string())
        };
        let flag = |k: &str| -> Option<bool> {
            let v = settings.key(k);
            (!v.is_null()).then(|| v.is_true())
        };
        let number = |k: &str| -> Result<Option<usize>, String> {
            match text(k) {
                None => Ok(None),
                Some(s) => s
                    .parse::<usize>()
                    .map(Some)
                    .map_err(|_| format!("pii: {k} must be a whole number, not [{s}]")),
            }
        };

        if let Some(m) = text("model") {
            cfg.model = m.clone();
            cfg.model_url = format!("{REPO_RAW}/{m}");
            match known(&m) {
                Some(k) => {
                    cfg.hf_repo = k.hf_repo.to_string();
                    cfg.onnx_url = k.onnx_url.to_string();
                    cfg.onnx_sha256 = Some(k.sha256.to_string());
                }
                None => {
                    // An unknown model has no pinned graph: the caller must
                    // say where its files are (modelPath, or modelUrl +
                    // onnxUrl, or hfRepo + onnxUrl).
                    cfg.hf_repo = m;
                    cfg.onnx_url = String::new();
                    cfg.onnx_sha256 = None;
                }
            }
        }
        if let Some(s) = text("source") {
            cfg.source = match s.to_ascii_lowercase().as_str() {
                "github" | "repo" | "repository" => Source::GitHub,
                "huggingface" | "hf" | "hub" => Source::HuggingFace,
                other => {
                    return Err(format!(
                        "pii: source must be \"github\" or \"huggingface\", not [{other}]"
                    ))
                }
            };
        }
        if let Some(u) = text("modelUrl") {
            cfg.model_url = u.trim_end_matches('/').to_string();
        }
        if let Some(r) = text("hfRepo") {
            cfg.hf_repo = r;
        }
        if let Some(r) = text("hfRevision") {
            cfg.hf_revision = r;
        }
        if let Some(e) = text("hfEndpoint") {
            cfg.hf_endpoint = e.trim_end_matches('/').to_string();
        }
        cfg.token =
            text("token").or_else(|| std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()));
        if let Some(u) = text("onnxUrl") {
            cfg.onnx_url = u;
            cfg.onnx_sha256 = None;
        }
        if let Some(h) = text("onnxSha256") {
            let h = h.trim().to_ascii_lowercase();
            if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("pii: onnxSha256 must be 64 hex characters".into());
            }
            cfg.onnx_sha256 = Some(h);
        }
        cfg.model_path = text("modelPath").map(|p| expand_tilde(&p));
        if let Some(c) = text("cacheDir") {
            cfg.cache_dir = expand_tilde(&c);
        }
        if let Some(b) = flag("autoDownload") {
            cfg.auto_download = b;
        }
        if let Some(b) = flag("preload") {
            cfg.preload = b;
        }
        if let Some(w) = number("windowTokens")? {
            cfg.window_tokens = w;
        }
        if let Some(o) = number("chunkOverlap")? {
            cfg.chunk_overlap = o;
        }
        if let Some(g) = number("mergeGap")? {
            cfg.merge_gap = g;
        }
        if cfg.window_tokens < 16 {
            return Err(format!(
                "pii: windowTokens ({}) is too small to hold a sentence",
                cfg.window_tokens
            ));
        }
        if cfg.chunk_overlap + 2 >= cfg.window_tokens {
            return Err(format!(
                "pii: chunkOverlap ({}) must be smaller than windowTokens ({}) less the two special tokens",
                cfg.chunk_overlap, cfg.window_tokens
            ));
        }
        Ok(cfg)
    }

    /// The directory the model files are read from.
    pub fn model_dir(&self) -> PathBuf {
        match &self.model_path {
            Some(p) => p.clone(),
            None => self.cache_dir.join(self.model.replace('/', "--")),
        }
    }

    /// The files the engine needs, relative to `model_dir()`, that are absent.
    pub fn missing_files(&self) -> Vec<&'static str> {
        let dir = self.model_dir();
        HUB_FILES
            .iter()
            .copied()
            .chain(std::iter::once(ONNX_FILE))
            .filter(|f| !dir.join(f).is_file())
            .collect()
    }

    /// The model card, fetched beside the weights so its licence and figures
    /// travel with them. Best effort.
    pub fn companion_file(&self) -> &'static str {
        match self.source {
            Source::GitHub => "MODEL_CARD.md",
            Source::HuggingFace => "README.md",
        }
    }

    /// Where a small file is fetched from.
    pub fn file_url(&self, file: &str) -> String {
        match self.source {
            Source::GitHub => format!("{}/{file}", self.model_url),
            Source::HuggingFace => format!(
                "{}/{}/resolve/{}/{file}",
                self.hf_endpoint, self.hf_repo, self.hf_revision
            ),
        }
    }

    /// Whether `url` is on the Hub, i.e. whether the bearer token applies.
    pub fn is_hub_url(&self, url: &str) -> bool {
        url.starts_with(&self.hf_endpoint)
    }

    pub fn source_page(&self) -> String {
        match self.source {
            Source::GitHub => self.model_url.clone(),
            Source::HuggingFace => format!("{}/{}", self.hf_endpoint, self.hf_repo),
        }
    }
}

pub fn default_cache_dir() -> PathBuf {
    expand_tilde("~/.rustcfml/models/pii")
}

/// `~/x` → `$HOME/x`. Anything else is returned as given, so a relative path
/// stays relative to the working directory, as it would for any other setting.
pub fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        if let Some(home) = home_dir() {
            return home;
        }
    }
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .filter(|p| Path::new(p).is_absolute())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tilde_expands_only_at_the_front() {
        let home = home_dir().expect("HOME");
        assert_eq!(expand_tilde("~/a/b"), home.join("a/b"));
        assert_eq!(expand_tilde("/opt/~/x"), PathBuf::from("/opt/~/x"));
        assert_eq!(expand_tilde("models"), PathBuf::from("models"));
    }

    #[test]
    fn the_cache_subdirectory_is_the_model_name() {
        let cfg = Config::defaults();
        assert!(cfg.model_dir().ends_with("pii/ettin-17m-nemotron-pii"));
    }

    #[test]
    fn a_model_path_wins_over_the_cache() {
        let mut cfg = Config::defaults();
        cfg.model_path = Some(PathBuf::from("/opt/pii"));
        assert_eq!(cfg.model_dir(), PathBuf::from("/opt/pii"));
    }

    #[test]
    fn github_is_the_default_source_and_the_hub_the_alternative() {
        let cfg = Config::defaults();
        assert_eq!(cfg.source, Source::GitHub);
        assert_eq!(
            cfg.file_url("tokenizer.json"),
            "https://raw.githubusercontent.com/alexskinner/RustCFML-Extension-PIIDetection/main/models/ettin-17m-nemotron-pii/tokenizer.json"
        );
        assert!(cfg
            .onnx_url
            .ends_with("/models/ettin-17m-nemotron-pii/onnx/model.onnx"));
        let mut hf = cfg.clone();
        hf.source = Source::HuggingFace;
        assert_eq!(
            hf.file_url("tokenizer.json"),
            "https://huggingface.co/kalyan-ks/ettin-17m-nemotron-pii/resolve/main/tokenizer.json"
        );
        assert!(hf.is_hub_url(&hf.file_url("config.json")));
        assert!(!hf.is_hub_url(&hf.onnx_url));
    }

    #[test]
    fn every_known_model_has_a_real_digest_and_a_github_url() {
        for m in &KNOWN_MODELS {
            assert_eq!(m.sha256.len(), 64, "{}", m.name);
            assert!(m.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            assert!(m.onnx_url.contains("github"), "{}", m.name);
            assert!(known(m.name).is_some());
        }
        assert!(known("ETTIN-68M-NEMOTRON-PII").is_some());
        assert!(known("something-else").is_none());
    }
}

//! Fetching the model files.
//!
//! Nothing is bundled: 68 MB has no place inside an extension archive, and
//! keeping the weights out of the `.rcx` keeps the archive's own licensing
//! simple. On first use `config.json` and `tokenizer.json` are fetched from
//! this repository (or, as an option, the model's Hub repository) and the ONNX
//! graph from `onnxUrl`, each written atomically — a half-fetched `model.onnx` must never be mistaken for a whole
//! one on the next start — and verified: the graph against the pinned digest of
//! the export it came from, Hub files against the SHA-256 the Hub publishes.

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::config::{Config, ONNX_FILE};

const USER_AGENT: &str = concat!("rustcfml-pii/", env!("CARGO_PKG_VERSION"));

/// Make sure every file the model needs is present in `cfg.model_dir()`,
/// downloading what is missing. Returns the list of files fetched, so the
/// caller can say so — a 68 MB download at start-up deserves a log line.
pub fn ensure_present(cfg: &Config) -> Result<Vec<String>, String> {
    let dir = cfg.model_dir();
    let missing = cfg.missing_files();
    if missing.is_empty() {
        return Ok(Vec::new());
    }
    if !cfg.auto_download {
        return Err(format!(
            "pii: the model directory {} is missing {} and autoDownload is off. \
             Put the files there (the repository's models/ directory has them; \
             scripts/export-onnx.py produces onnx/model.onnx from the published weights), \
             point modelPath at a directory that has them, or set \
             extensions.settings.pii.autoDownload = true.",
            dir.display(),
            missing.join(", ")
        ));
    }
    if missing.contains(&ONNX_FILE) && cfg.onnx_url.is_empty() {
        return Err(format!(
            "pii: {} is not in {} and there is nowhere to fetch it from: [{}] is not a model \
             this extension publishes, so it has no pinned ONNX graph. Export one with \
             scripts/export-onnx.py and set modelPath (or onnxUrl) to it.",
            ONNX_FILE,
            dir.display(),
            cfg.model
        ));
    }
    fs::create_dir_all(&dir).map_err(|e| format!("pii: cannot create {}: {e}", dir.display()))?;

    let agent = ureq::AgentBuilder::new()
        .user_agent(USER_AGENT)
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .build();

    let mut fetched = Vec::new();
    for file in missing {
        if file == ONNX_FILE {
            download(
                &agent,
                cfg,
                &dir,
                file,
                &cfg.onnx_url,
                cfg.onnx_sha256.clone(),
            )?;
        } else {
            let url = cfg.file_url(file);
            // The Hub publishes a digest for its files; GitHub does not. A
            // small JSON file that fails to parse fails loudly at load anyway.
            let digest = if cfg.is_hub_url(&url) {
                linked_digest(cfg, &url)
            } else {
                None
            };
            download(&agent, cfg, &dir, file, &url, digest)?;
        }
        fetched.push(file.to_string());
    }
    // Best effort: the model card should sit beside the weights, but a source
    // without one is not a reason to refuse a model that loaded.
    let card = cfg.companion_file();
    if !dir.join("MODEL_CARD.md").is_file() {
        let url = cfg.file_url(card);
        if download(&agent, cfg, &dir, "MODEL_CARD.md", &url, None).is_ok() {
            fetched.push("MODEL_CARD.md".to_string());
        }
    }
    Ok(fetched)
}

fn download(
    agent: &ureq::Agent,
    cfg: &Config,
    dir: &Path,
    file: &str,
    url: &str,
    expected: Option<String>,
) -> Result<(), String> {
    let dest = dir.join(file);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("pii: cannot create {}: {e}", parent.display()))?;
    }

    let mut req = agent.get(url);
    // The bearer token is for the Hub; it must not leak to wherever the ONNX
    // graph is hosted.
    if let (Some(t), true) = (&cfg.token, cfg.is_hub_url(url)) {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = req
        .call()
        .map_err(|e| format!("pii: downloading {url}: {e}"))?;

    let tmp = dest.with_extension(match dest.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{ext}.part"),
        None => "part".to_string(),
    });
    let result = (|| -> Result<(), String> {
        let mut out = fs::File::create(&tmp)
            .map_err(|e| format!("pii: cannot write {}: {e}", tmp.display()))?;
        let mut hasher = Sha256::new();
        let mut reader = resp.into_reader();
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| format!("pii: reading {url}: {e}"))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n])
                .map_err(|e| format!("pii: writing {}: {e}", tmp.display()))?;
        }
        out.flush().map_err(|e| e.to_string())?;
        let got = hex(&hasher.finalize());
        if let Some(want) = &expected {
            if !want.eq_ignore_ascii_case(&got) {
                return Err(format!(
                    "pii: {file} did not verify — the Hub says sha256 {want}, the download was {got}"
                ));
            }
        }
        Ok(())
    })();

    match result {
        Ok(()) => fs::rename(&tmp, &dest)
            .map_err(|e| format!("pii: cannot move {} into place: {e}", dest.display())),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// The SHA-256 the Hub records for an LFS file. The Hub answers `resolve/`
/// with a redirect to its CDN and puts the content digest on THAT first
/// response (`X-Linked-Etag`), not on the CDN's; one un-followed HEAD reads it. `None` when the Hub is not a Hub (a plain mirror) or
/// the file is a small non-LFS blob, whose `ETag` is not a content digest.
fn linked_digest(cfg: &Config, url: &str) -> Option<String> {
    let agent = ureq::AgentBuilder::new()
        .user_agent(USER_AGENT)
        .redirects(0)
        .timeout(Duration::from_secs(30))
        .build();
    let mut req = agent.head(url);
    if let Some(t) = &cfg.token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        // A redirect with redirects(0) comes back as Ok(3xx) in ureq 2; an
        // actual transport error here is not fatal, the GET will report it.
        Err(ureq::Error::Status(_, r)) => r,
        Err(_) => return None,
    };
    let v = resp.header("x-linked-etag")?;
    let v = v.trim().trim_matches('"');
    (v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit())).then(|| v.to_string())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Read a whole file, with the path in the error.
pub fn read(path: &Path) -> Result<Vec<u8>, String> {
    let mut f = fs::File::open(path).map_err(|e| describe(path, e))?;
    let mut v = Vec::new();
    f.read_to_end(&mut v).map_err(|e| describe(path, e))?;
    Ok(v)
}

fn describe(path: &Path, e: io::Error) -> String {
    format!("pii: {}: {e}", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_is_lowercase_and_zero_padded() {
        assert_eq!(hex(&[0, 1, 0xab, 0xff]), "0001abff");
    }
}

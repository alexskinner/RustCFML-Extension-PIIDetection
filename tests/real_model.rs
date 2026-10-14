//! Against the real model. Needs `PII_MODEL_DIR` pointing at a directory with
//! `config.json`, `tokenizer.json` and `onnx/model.onnx` for
//! ettin-17m-nemotron-pii (`scripts/export-onnx.py 17m` produces one; CI runs
//! it). Without it these tests print that they were skipped — they do not pass
//! vacuously.
//!
//! Expected spans come from onnxruntime on the same inputs with the model
//! card's own post-processing, so a divergence here is in this crate's
//! inference or decoding, not in the model.

use std::path::PathBuf;

use rustcfml_pii::engine::{Engine, LoadOptions};
use rustcfml_pii::spans::{self, MaskStyle};

fn model_dir() -> Option<PathBuf> {
    match std::env::var_os("PII_MODEL_DIR") {
        Some(d) => Some(PathBuf::from(d)),
        None => {
            eprintln!("SKIPPED: set PII_MODEL_DIR to run the real-model tests");
            None
        }
    }
}

fn load(dir: &std::path::Path) -> Engine {
    Engine::load(dir, LoadOptions::default()).unwrap()
}

fn texts(engine: &Engine, text: &str) -> Vec<(String, String)> {
    engine
        .detect(text)
        .unwrap()
        .iter()
        .map(|s| (s.label.clone(), spans::slice(text, s)))
        .collect()
}

fn of_type(engine: &Engine, text: &str, label: &str) -> Vec<String> {
    engine
        .detect(text)
        .unwrap()
        .iter()
        .filter(|s| s.label == label)
        .map(|s| spans::slice(text, s))
        .collect()
}

fn pair(a: &str, b: &str) -> (String, String) {
    (a.to_string(), b.to_string())
}

#[test]
fn the_label_set_is_the_model_cards_55_types() {
    let Some(dir) = model_dir() else { return };
    let engine = load(&dir);
    assert_eq!(engine.types().len(), 55);
    assert_eq!(engine.types()[0], "account_number");
    assert_eq!(engine.types()[54], "vehicle_identifier");
    assert!(engine.types().contains(&"email".to_string()));
    assert_eq!(engine.window_tokens(), 512);
}

#[test]
fn entity_spans_match_onnxruntime_boundary_exactly() {
    let Some(dir) = model_dir() else { return };
    let engine = load(&dir);

    // The model card's own example and its documented output.
    assert_eq!(
        texts(
            &engine,
            "Kalyan KS is from India. His email id is kalyan.ks@yahoo.com"
        ),
        vec![
            pair("first_name", "Kalyan KS"),
            pair("country", "India"),
            pair("email", "kalyan.ks@yahoo.com"),
        ]
    );
    assert_eq!(
        texts(
            &engine,
            "Contact John Smith at john@example.com or +44 7700 900123."
        ),
        vec![
            pair("first_name", "John"),
            pair("last_name", "Smith"),
            pair("email", "john@example.com"),
            // The model leaves the `+` outside; that is its boundary, not ours.
            pair("phone_number", "44 7700 900123"),
        ]
    );
    assert_eq!(
        texts(&engine, "Sarah Whitfield called about the insurance claim"),
        vec![pair("first_name", "Sarah"), pair("last_name", "Whitfield")]
    );
    assert_eq!(
        texts(&engine, "Daniel's email is daniel@example.com and he lives at 12 Baker Street, London. Call 020 7946 0123."),
        vec![
            pair("first_name", "Daniel"),
            pair("email", "daniel@example.com"),
            pair("street_address", "12 Baker Street"),
            pair("state", "London"),
            pair("phone_number", "020 7946 0123"),
        ]
    );
    assert_eq!(
        of_type(
            &engine,
            "Payment was made with Visa card 4532 7153 3790 1241 expiring 09/28.",
            "credit_debit_card"
        ),
        vec!["4532 7153 3790 1241".to_string()]
    );
    assert_eq!(texts(&engine, "Hope the survey results help"), vec![]);
    assert_eq!(texts(&engine, ""), vec![]);
    assert_eq!(texts(&engine, "   \n  "), vec![]);
}

#[test]
fn scores_are_the_minimum_piece_probability_and_lie_in_range() {
    let Some(dir) = model_dir() else { return };
    let engine = load(&dir);
    let text = "Contact John Smith at john@example.com or +44 7700 900123.";
    let spans = engine.detect(text).unwrap();
    for s in &spans {
        assert!(s.score > 0.0 && s.score <= 1.0, "{s:?}");
    }
    // onnxruntime: John 0.995, Smith 0.393, the email's weakest piece 0.981.
    let by = |l: &str| spans.iter().find(|s| s.label == l).unwrap().score;
    assert!(
        (by("first_name") - 0.995).abs() < 0.01,
        "{}",
        by("first_name")
    );
    assert!(
        (by("last_name") - 0.393).abs() < 0.01,
        "{}",
        by("last_name")
    );
    assert!((by("email") - 0.981).abs() < 0.01, "{}", by("email"));
}

#[test]
fn non_ascii_text_keeps_exact_character_offsets() {
    let Some(dir) = model_dir() else { return };
    let engine = load(&dir);
    let text = "Zoë Müller lives in München and her email is zoe@example.de";
    assert_eq!(
        texts(&engine, text),
        vec![
            pair("first_name", "Zoë Müller"),
            pair("city", "München"),
            pair("email", "zoe@example.de"),
        ]
    );
    let spans = engine.detect(text).unwrap();
    for s in &spans {
        let sliced: String = text.chars().skip(s.start).take(s.end - s.start).collect();
        assert_eq!(sliced, spans::slice(text, s));
    }
    let masked = spans::mask(text, &spans, &MaskStyle::Template("[{TYPE}]".into())).unwrap();
    assert_eq!(
        masked,
        "[FIRST_NAME] lives in [CITY] and her email is [EMAIL]"
    );
}

const FILLER: [&str; 8] = [
    "The quarterly results look strong and the team is pleased with progress.",
    "Our next planning session covers the roadmap for the autumn release.",
    "Please review the attached summary before Thursday's meeting.",
    "The warehouse move is on schedule and the new shelving has arrived.",
    "Marketing will circulate the updated brand guidelines next week.",
    "Thanks to everyone who helped with the customer survey this month.",
    "The finance team has closed the books for the previous period.",
    "Training for the new ticketing system starts on Monday morning.",
];

/// `k` sentences of ordinary business text. Synthetic filler such as a
/// repeated word is NOT used: the model behaves strangely on text no human
/// would write, and a test built on it measures the fixture, not the engine.
fn fill(k: usize) -> String {
    let mut s = String::new();
    for i in 0..k {
        s.push_str(FILLER[i % FILLER.len()]);
        s.push(' ');
    }
    s
}

#[test]
fn a_text_longer_than_the_window_is_chunked_and_still_finds_the_end() {
    let Some(dir) = model_dir() else { return };
    let engine = load(&dir);
    // ~1,000 tokens: three 512-token windows.
    let text = fill(80) + "Finally, please send the invoice to sarah@example.com for the balance.";
    let emails = of_type(&engine, &text, "email");
    // The 17m model also flags a few "to"s in that much prose (onnxruntime
    // agrees, scores 0.55–0.76); the real address must be among them, whole.
    assert!(
        emails.contains(&"sarah@example.com".to_string()),
        "{emails:?}"
    );
    assert!(
        emails.iter().all(|e| e == "to" || e == "sarah@example.com"),
        "{emails:?}"
    );
}

#[test]
fn a_span_straddling_a_window_boundary_is_not_split_or_lost() {
    let Some(dir) = model_dir() else { return };
    // A small window so the boundary is cheap to reach: 64 tokens, overlap 16.
    let engine = Engine::load(
        &dir,
        LoadOptions {
            window_tokens: 64,
            chunk_overlap: 16,
            merge_gap: 1,
        },
    )
    .unwrap();
    let target = "You can reach Sarah Whitfield at sarah.whitfield@example.com today.";
    assert_eq!(
        of_type(&engine, target, "email"),
        vec!["sarah.whitfield@example.com".to_string()]
    );

    // Each extra filler sentence moves the target ~12 tokens; across these
    // it crosses the first window's end (62 body tokens) at every offset.
    for k in 2..=7 {
        let text = fill(k) + target + " " + &fill(3);
        let emails = of_type(&engine, &text, "email");
        let whole = emails.iter().any(|e| {
            e == "sarah.whitfield@example.com" || e == "sarah.whitfield@example.com today"
        });
        assert!(whole, "k={k}: {emails:?}");
        assert!(
            of_type(&engine, &text, "first_name").contains(&"Sarah".to_string()),
            "k={k}"
        );
    }
}

#[test]
fn window_and_overlap_limits_are_enforced_at_load() {
    let Some(dir) = model_dir() else { return };
    let err = Engine::load(
        &dir,
        LoadOptions {
            window_tokens: 64,
            chunk_overlap: 62,
            merge_gap: 1,
        },
    )
    .unwrap_err();
    assert!(err.contains("chunkOverlap"), "{err}");
    let err = Engine::load(
        &dir,
        LoadOptions {
            window_tokens: 20_000,
            chunk_overlap: 32,
            merge_gap: 1,
        },
    )
    .unwrap_err();
    assert!(err.contains("windowTokens"), "{err}");
}

#[test]
fn a_wider_merge_gap_joins_pieces_the_default_keeps_apart() {
    let Some(dir) = model_dir() else { return };
    let text = "Contact John Smith at john@example.com or +44 7700 900123.";
    let tight = load(&dir);
    assert_eq!(
        of_type(&tight, text, "first_name"),
        vec!["John".to_string()]
    );
    // first_name and last_name are different types: a wider gap never merges
    // them. It only matters within a type.
    let loose = Engine::load(
        &dir,
        LoadOptions {
            merge_gap: 10,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        of_type(&loose, text, "first_name"),
        vec!["John".to_string()]
    );
    assert_eq!(
        of_type(&loose, text, "last_name"),
        vec!["Smith".to_string()]
    );
}

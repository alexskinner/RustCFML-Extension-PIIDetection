//! Turning per-token predictions into character spans, and spans into masked
//! text. Pure functions over plain data, so they are testable without a model.
//!
//! Positions are CHARACTER offsets into the original text (not bytes, not
//! UTF-16 units): `start` is 0-based and `end` exclusive here; the CFML edge
//! converts to 1-based `start`/`length` so that `mid(text, start, length)`
//! returns exactly `text`.

/// One detected entity.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    /// The entity type, e.g. `email`, `first_name`.
    pub label: String,
    /// Character offset of the first character, 0-based.
    pub start: usize,
    /// Character offset one past the last character.
    pub end: usize,
    /// The lowest per-token probability inside the span, so the figure
    /// describes the whole span and not its most confident piece.
    pub score: f32,
}

/// What the model said about one token.
#[derive(Debug, Clone, Copy)]
pub struct TokenPrediction {
    /// Character offsets of the token in the original text.
    pub start: usize,
    pub end: usize,
    /// Index into the label set.
    pub label: usize,
    /// Softmax probability of that label.
    pub prob: f32,
}

/// Group tokens into spans, the way the model card's own post-processing does:
/// consecutive tokens of the same entity TYPE (the `B-`/`I-` prefix ignored)
/// become one span when the next starts within `merge_gap` characters of
/// where the previous ended; a token of another type or of no type ends it.
/// Leading and trailing whitespace is then trimmed, because a byte-level BPE
/// token carries the space before its word.
///
/// Ignoring the prefix is deliberate. The model frequently emits `B-` on
/// every sub-word piece of one entity (`j` `ohn` of `john@…`), so honouring it
/// would split entities; and two entities of one type with nothing between
/// them are, in the training data's annotation, one entity. `types[label]` is
/// `None` for the outside label.
pub fn group(
    text: &str,
    preds: &[TokenPrediction],
    types: &[Option<String>],
    merge_gap: usize,
) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
    let mut cur: Option<Span> = None;
    for p in preds {
        // A zero-width token (a special token, nothing of the original text)
        // carries no span and does not break one either.
        if p.end <= p.start {
            continue;
        }
        let Some(ty) = types.get(p.label).and_then(|t| t.as_deref()) else {
            if let Some(s) = cur.take() {
                out.push(s);
            }
            continue;
        };
        match cur.as_mut() {
            Some(s) if s.label == ty && p.start <= s.end + merge_gap => {
                // NFD can split an accented letter into two tokens that share
                // one offset range, so take the max rather than assuming order.
                s.end = s.end.max(p.end);
                s.score = s.score.min(p.prob);
            }
            _ => {
                if let Some(s) = cur.take() {
                    out.push(s);
                }
                cur = Some(Span {
                    label: ty.to_string(),
                    start: p.start,
                    end: p.end,
                    score: p.prob,
                });
            }
        }
    }
    if let Some(s) = cur {
        out.push(s);
    }
    trim_all(text, out)
}

/// Pull each span's edges in past whitespace. A span that was all whitespace
/// is dropped.
fn trim_all(text: &str, spans: Vec<Span>) -> Vec<Span> {
    let chars: Vec<char> = text.chars().collect();
    spans
        .into_iter()
        .filter_map(|mut s| {
            let end = s.end.min(chars.len());
            let mut a = s.start.min(end);
            let mut b = end;
            while a < b && chars[a].is_whitespace() {
                a += 1;
            }
            while b > a && chars[b - 1].is_whitespace() {
                b -= 1;
            }
            if a == b {
                return None;
            }
            s.start = a;
            s.end = b;
            Some(s)
        })
        .collect()
}

/// How `mask` replaces a span.
#[derive(Debug, Clone)]
pub enum MaskStyle {
    /// A template; `{type}` and `{TYPE}` expand to the label, `{text}` to the
    /// original. The default is `[{TYPE}]`.
    Template(String),
    /// Every character becomes `ch`, keeping the span's length, except the
    /// last `keep_last` characters, which stay visible (`**** **** **** 1111`).
    Char { ch: String, keep_last: usize },
}

/// Replace each span in `text`. Spans must be sorted and non-overlapping,
/// which `group` guarantees; anything else is a caller bug and is reported.
pub fn mask(text: &str, spans: &[Span], style: &MaskStyle) -> Result<String, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut pos = 0usize;
    for s in spans {
        if s.start < pos || s.end > chars.len() || s.end < s.start {
            return Err(format!(
                "pii: spans overlap or run past the text ({}..{} after {pos}, text is {} chars)",
                s.start,
                s.end,
                chars.len()
            ));
        }
        out.extend(&chars[pos..s.start]);
        let original: String = chars[s.start..s.end].iter().collect();
        match style {
            MaskStyle::Template(t) => out.push_str(
                &t.replace("{TYPE}", &s.label.to_uppercase())
                    .replace("{type}", &s.label)
                    .replace("{text}", &original),
            ),
            MaskStyle::Char { ch, keep_last } => {
                let n = s.end - s.start;
                let hidden = n.saturating_sub(*keep_last);
                for c in chars[s.start..s.end].iter().take(hidden) {
                    // Whitespace inside a span keeps its shape, so a masked card
                    // number still reads as four groups.
                    if c.is_whitespace() {
                        out.push(*c);
                    } else {
                        out.push_str(ch);
                    }
                }
                out.extend(&chars[s.start + hidden..s.end]);
            }
        }
        pos = s.end;
    }
    out.extend(&chars[pos..]);
    Ok(out)
}

/// The span's own text.
pub fn slice(text: &str, s: &Span) -> String {
    text.chars().skip(s.start).take(s.end - s.start).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // O, B-person, I-person, B-email, I-email
    fn types() -> Vec<Option<String>> {
        vec![
            None,
            Some("person".into()),
            Some("person".into()),
            Some("email".into()),
            Some("email".into()),
        ]
    }

    fn tp(start: usize, end: usize, label: usize, prob: f32) -> TokenPrediction {
        TokenPrediction {
            start,
            end,
            label,
            prob,
        }
    }

    #[test]
    fn pieces_of_one_entity_merge_whatever_their_prefix_and_the_space_is_trimmed() {
        // " j" B-email, "ohn" B-email, "@" I-email, "example" I-email, "." I, "com" I
        let text = "Mail john@example.com";
        let preds = [
            tp(4, 6, 3, 0.98),
            tp(6, 9, 3, 0.98),
            tp(9, 10, 4, 0.99),
            tp(10, 17, 4, 0.98),
            tp(17, 18, 4, 1.0),
            tp(18, 21, 4, 0.99),
        ];
        let spans = group(text, &preds, &types(), 1);
        assert_eq!(spans.len(), 1);
        assert_eq!(slice(text, &spans[0]), "john@example.com");
        assert_eq!((spans[0].start, spans[0].end), (5, 21));
        assert!(
            (spans[0].score - 0.98).abs() < 1e-6,
            "score is the MIN piece"
        );
    }

    #[test]
    fn a_type_change_or_an_outside_token_ends_the_span() {
        let text = "Ann Bob x mail";
        let preds = [
            tp(0, 3, 1, 0.9),
            tp(3, 7, 3, 0.9),
            tp(7, 9, 0, 0.9),
            tp(9, 14, 3, 0.9),
        ];
        let spans = group(text, &preds, &types(), 1);
        assert_eq!(spans.len(), 3);
        assert_eq!(slice(text, &spans[1]), "Bob");
        assert_eq!(spans[1].label, "email");
        assert_eq!(slice(text, &spans[2]), "mail");
    }

    #[test]
    fn same_type_pieces_further_apart_than_the_gap_are_two_spans() {
        let text = "Ann and Bob";
        let preds = [tp(0, 3, 1, 0.9), tp(7, 11, 1, 0.9)];
        assert_eq!(group(text, &preds, &types(), 1).len(), 2);
        assert_eq!(group(text, &preds, &types(), 4).len(), 1);
    }

    #[test]
    fn nfd_pieces_sharing_one_offset_range_do_not_corrupt_the_span() {
        // "Zoë": z | o | e | ◌̈  — the last two both report (2,3).
        let text = "Zoë";
        let preds = [
            tp(0, 1, 1, 0.9),
            tp(1, 2, 1, 0.9),
            tp(2, 3, 1, 0.9),
            tp(2, 3, 2, 0.8),
        ];
        let spans = group(text, &preds, &types(), 1);
        assert_eq!(spans.len(), 1);
        assert_eq!((spans[0].start, spans[0].end), (0, 3));
    }

    #[test]
    fn special_tokens_are_zero_width_and_ignored() {
        let text = "Ann";
        let preds = [tp(0, 0, 3, 0.9), tp(0, 3, 1, 0.9), tp(0, 0, 4, 0.3)];
        let spans = group(text, &preds, &types(), 1);
        assert_eq!(spans.len(), 1);
        assert_eq!(slice(text, &spans[0]), "Ann");
    }

    #[test]
    fn a_whitespace_only_span_is_dropped() {
        let text = "a  b";
        let preds = [tp(1, 3, 1, 0.9)];
        assert!(group(text, &preds, &types(), 1).is_empty());
    }

    fn span(label: &str, start: usize, end: usize) -> Span {
        Span {
            label: label.into(),
            start,
            end,
            score: 1.0,
        }
    }

    #[test]
    fn template_masking_is_boundary_exact() {
        let text = "Mail john@x.com now";
        let out = mask(
            text,
            &[span("email", 5, 15)],
            &MaskStyle::Template("[{TYPE}]".into()),
        )
        .unwrap();
        assert_eq!(out, "Mail [EMAIL] now");
        let out = mask(
            text,
            &[span("email", 5, 15)],
            &MaskStyle::Template("<{type}:{text}>".into()),
        )
        .unwrap();
        assert_eq!(out, "Mail <email:john@x.com> now");
    }

    #[test]
    fn character_masking_keeps_length_whitespace_and_a_visible_tail() {
        let text = "card 4532 7153 3790 1241 ok";
        let out = mask(
            text,
            &[span("credit_debit_card", 5, 24)],
            &MaskStyle::Char {
                ch: "*".into(),
                keep_last: 4,
            },
        )
        .unwrap();
        assert_eq!(out, "card **** **** **** 1241 ok");
        assert_eq!(out.chars().count(), text.chars().count());
    }

    #[test]
    fn masking_counts_characters_not_bytes() {
        let text = "Zoë Müller <zoe@example.de>";
        let out = mask(
            text,
            &[span("first_name", 0, 10)],
            &MaskStyle::Template("[{TYPE}]".into()),
        )
        .unwrap();
        assert_eq!(out, "[FIRST_NAME] <zoe@example.de>");
        assert_eq!(slice(text, &span("first_name", 0, 10)), "Zoë Müller");
    }

    #[test]
    fn overlapping_spans_are_an_error_not_a_corruption() {
        let err = mask(
            "abcdef",
            &[span("a", 0, 4), span("b", 2, 6)],
            &MaskStyle::Template("X".into()),
        );
        assert!(err.is_err());
    }
}

# Functions

Every function is available once the extension is installed and the engine
restarted. Positions are **CFML positions**: `start` is the 1-based character
index, `length` the character count, and `end` the 1-based index of the last
character, so `mid( text, span.start, span.length )` is exactly `span.text`.
Characters, not bytes — an accented name or an emoji earlier in the text does
not shift anything.

## piiDetect( text [, options ] )

Returns an array of entity spans, in text order, non-overlapping:

```cfml
spans = piiDetect( "Contact John Smith at john@example.com or +44 7700 900123." );
```

```
[ { type: "first_name",   text: "John",             start: 9,  end: 12, length: 4,  score: 0.995 },
  { type: "last_name",    text: "Smith",            start: 14, end: 18, length: 5,  score: 0.393 },
  { type: "email",        text: "john@example.com", start: 23, end: 38, length: 16, score: 0.981 },
  { type: "phone_number", text: "44 7700 900123",   start: 44, end: 57, length: 14, score: 0.831 } ]
```

| key | |
|---|---|
| `type` | one of the 55 types from `piiTypes()`, lower case with underscores |
| `text` | the span's text, exactly as it appears |
| `start`, `end`, `length` | character positions as above |
| `score` | the lowest per-token probability inside the span, so it describes the whole span and not its most confident piece; 0–1 |

`options` is a struct:

| key | |
|---|---|
| `types` | only these types — a list (`"email,phone_number"`) or an array, case-insensitive. An unknown type is an error that names it. |
| `minScore` | drop spans scoring below this, 0–1. See [tuning](configuration.md#tuning) before setting it. |

Spans are decoded the way the model's author does it: consecutive tokens of
one type become a span when they touch or are one character apart, and the
span is trimmed of whitespace. Names come back as `first_name` and
`last_name` separately, because that is how the model was trained.

## piiMask( text [, options ] )

The text with every detected span replaced. Boundary-exact: nothing outside a
span changes.

```cfml
piiMask( "Contact John Smith at john@example.com" );
// Contact [FIRST_NAME] [LAST_NAME] at [EMAIL]

piiMask( text, { replacement = "<{type}>" } );
// Contact <first_name> <last_name> at <email>

piiMask( "Visa card 4532 7153 3790 1241", { types = "credit_debit_card", maskChar = "*", keepLast = 4 } );
// Visa card **** **** **** 1241
```

| key | |
|---|---|
| `replacement` | a template; `{TYPE}` and `{type}` expand to the label in upper or lower case, `{text}` to the original. Default `[{TYPE}]`. |
| `maskChar` | replace every character of the span with this, keeping its length; whitespace inside the span is kept so a card number keeps its groups. Overrides `replacement`. |
| `keepLast` | with `maskChar`, leave this many trailing characters visible. |
| `types`, `minScore` | as for `piiDetect`. Spans not selected are left in place. |

## piiTypes()

The entity types the model emits, sorted: `account_number`, `age`, `api_key`,
`bank_routing_number`, … `vehicle_identifier`. 55 for the Ettin models; the
descriptions and per-type accuracy are in the model card beside each export
in `models/`.

## piiIsReady()

`true` once the model is loaded. Never loads it.

## piiLoad()

Load the model now — fetching it if it is not in the cache — and return what
`piiSettings()` returns. The place to warm a server from an `onApplicationStart`
when `preload` is off, and the way to retry after a failed load: a failure is
remembered and reported by every call until this succeeds.

## piiSettings()

The effective configuration and the load state:

```
{ model: "ettin-17m-nemotron-pii", source: "github",
  modelUrl: "https://raw.githubusercontent.com/alexskinner/RustCFML-Extension-PIIDetection/main/models/ettin-17m-nemotron-pii",
  hfRepo: "kalyan-ks/ettin-17m-nemotron-pii", onnxUrl: "…/onnx/model.onnx",
  modelDir: "/home/app/.rustcfml/models/pii/ettin-17m-nemotron-pii", cacheDir: "…",
  autoDownload: true, preload: true, windowTokens: 512, chunkOverlap: 32, mergeGap: 1,
  ready: true, error: "", loadMillis: 180, typeCount: 55 }
```

`error` is the message of the last failed load, empty otherwise.

## Errors

A missing or wrong argument is an expression error naming the function and
the argument. A model that cannot be loaded is an application error with the
reason and what to do: the directory looked in, the files missing, and the
setting to change. Nothing is ever approximated silently — an empty array means
the model found nothing, not that it could not run.

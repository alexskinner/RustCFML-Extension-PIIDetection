<cfscript>
// End-to-end check of the PII extension, and CI's gate for it. It ASSERTS
// rather than printing: a detector that quietly returned nothing would
// otherwise look like a passing example. Expected spans were taken from
// onnxruntime on the same model and sentences, with the model author's own
// post-processing.

failures = 0;
function check( required string what, required boolean ok, string detail = "" ) {
	writeOutput( ( arguments.ok ? "  ok   " : "  FAIL " ) & arguments.what
		& ( len( arguments.detail ) ? "  (" & arguments.detail & ")" : "" ) & chr( 10 ) );
	if ( !arguments.ok ) { variables.failures++; }
}
function spanList( required array spans ) {
	return arrayToList( arrayMap( arguments.spans, function( s ) { return s.type & "=" & s.text; } ), "|" );
}

writeOutput( "PII extension self-test" & chr( 10 ) );

// --- the model loads (fetching it if this is the first run) ----------------
try {
	info = piiLoad();
	check( "the model loads", info.ready, info.model & " from " & info.modelDir & " in " & ( info.loadMillis ?: "?" ) & " ms" );
} catch ( any e ) {
	check( "the model loads", false, e.message );
	writeOutput( chr( 10 ) & "Is the model there? See piiSettings() / docs/configuration.md." & chr( 10 ) );
	abort;
}
check( "piiIsReady() agrees", piiIsReady() );
check( "the settings report the model", piiSettings().model EQ "ettin-17m-nemotron-pii", piiSettings().model );

// --- the label set ----------------------------------------------------------
types = piiTypes();
check( "55 entity types", arrayLen( types ) EQ 55, arrayLen( types ) );
check( "and they are sorted", types[ 1 ] EQ "account_number" && types[ 55 ] EQ "vehicle_identifier" );
check( "email is among them", arrayFindNoCase( types, "email" ) GT 0 );

// --- detection ---------------------------------------------------------------
text  = "Contact John Smith at john@example.com or +44 7700 900123.";
spans = piiDetect( text );
check( "detects name, email and phone",
	spanList( spans ) EQ "first_name=John|last_name=Smith|email=john@example.com|phone_number=44 7700 900123", spanList( spans ) );
check( "positions are CFML positions: mid(text, start, length) is the span",
	arrayEvery( spans, function( s ) { return mid( text, s.start, s.length ) EQ s.text; } ) );
check( "end is the inclusive last character",
	arrayEvery( spans, function( s ) { return s.end EQ s.start + s.length - 1; } ) );
check( "scores lie in (0, 1]", arrayEvery( spans, function( s ) { return s.score GT 0 && s.score LTE 1; } ) );

card = piiDetect( "Kalyan KS is from India. His email id is kalyan.ks@yahoo.com" );
check( "the model card's own example gives its documented answer",
	spanList( card ) EQ "first_name=Kalyan KS|country=India|email=kalyan.ks@yahoo.com", spanList( card ) );

person = piiDetect( "Sarah Whitfield called about the insurance claim" );
check( "a multi-piece surname is one boundary-exact span", spanList( person ) EQ "first_name=Sarah|last_name=Whitfield", spanList( person ) );

visa = piiDetect( "Payment was made with Visa card 4532 7153 3790 1241 expiring 09/28.", { types = "credit_debit_card" } );
check( "a card number with spaces is one span", spanList( visa ) EQ "credit_debit_card=4532 7153 3790 1241", spanList( visa ) );

addr = piiDetect( "Daniel's email is daniel@example.com and he lives at 12 Baker Street, London. Call 020 7946 0123." );
check( "address, email and a UK phone number",
	spanList( addr ) EQ "first_name=Daniel|email=daniel@example.com|street_address=12 Baker Street|state=London|phone_number=020 7946 0123", spanList( addr ) );

check( "benign text finds nothing", arrayLen( piiDetect( "Hope the survey results help" ) ) EQ 0 );
check( "an empty string finds nothing", arrayLen( piiDetect( "" ) ) EQ 0 );

// --- options -----------------------------------------------------------------
only = piiDetect( text, { types = "phone_number" } );
check( "types filters to the listed types", spanList( only ) EQ "phone_number=44 7700 900123", spanList( only ) );
only = piiDetect( text, { types = [ "EMAIL" ] } );
check( "types accepts an array, case-insensitively", spanList( only ) EQ "email=john@example.com", spanList( only ) );
check( "minScore drops the weak span (Smith scores 0.39)",
	spanList( piiDetect( text, { minScore = 0.5 } ) ) EQ "first_name=John|email=john@example.com|phone_number=44 7700 900123",
	spanList( piiDetect( text, { minScore = 0.5 } ) ) );
bad = "";
try { piiDetect( text, { types = "ssn_number" } ); } catch ( any e ) { bad = e.message; }
check( "an unknown type is refused by name", findNoCase( "[ssn_number]", bad ) GT 0 && findNoCase( "piiTypes", bad ) GT 0, bad );

// --- masking -----------------------------------------------------------------
check( "default mask is [TYPE]", piiMask( text ) EQ "Contact [FIRST_NAME] [LAST_NAME] at [EMAIL] or +[PHONE_NUMBER].", piiMask( text ) );
check( "a replacement template sees {type}",
	piiMask( text, { replacement = "<{type}>" } ) EQ "Contact <first_name> <last_name> at <email> or +<phone_number>.", piiMask( text, { replacement = "<{type}>" } ) );
masked = piiMask( "Payment was made with Visa card 4532 7153 3790 1241 expiring 09/28.", { types = "credit_debit_card", maskChar = "*", keepLast = 4 } );
check( "maskChar keeps length and shape, keepLast shows the tail", masked EQ "Payment was made with Visa card **** **** **** 1241 expiring 09/28.", masked );
check( "masking only the listed types leaves the others",
	piiMask( text, { types = "email" } ) EQ "Contact John Smith at [EMAIL] or +44 7700 900123.", piiMask( text, { types = "email" } ) );

// --- non-ASCII text: positions are characters, not bytes --------------------
utf = "Zoë Müller lives in München and her email is zoe@example.de";
uspans = piiDetect( utf );
check( "accented names and cities come back whole",
	spanList( uspans ) EQ "first_name=Zoë Müller|city=München|email=zoe@example.de", spanList( uspans ) );
check( "and their offsets are exact", arrayEvery( uspans, function( s ) { return mid( utf, s.start, s.length ) EQ s.text; } ) );
check( "and the masked text is right", piiMask( utf ) EQ "[FIRST_NAME] lives in [CITY] and her email is [EMAIL]", piiMask( utf ) );

// --- long text is chunked over the 512-token window -------------------------
// Ordinary business sentences as filler, not a repeated word: the model
// behaves strangely on text no human would write. The 17M model also flags a
// few stray "to"s in this much prose (onnxruntime agrees); the real address
// must be among the email spans, whole.
filler = [
	"The quarterly results look strong and the team is pleased with progress.",
	"Our next planning session covers the roadmap for the autumn release.",
	"Please review the attached summary before Thursday's meeting.",
	"The warehouse move is on schedule and the new shelving has arrived.",
	"Marketing will circulate the updated brand guidelines next week.",
	"Thanks to everyone who helped with the customer survey this month.",
	"The finance team has closed the books for the previous period.",
	"Training for the new ticketing system starts on Monday morning."
];
function fill( required numeric k ) {
	out = "";
	for ( i = 0; i < arguments.k; i++ ) { out &= filler[ ( i % arrayLen( filler ) ) + 1 ] & " "; }
	return out;
}
long = fill( 80 ) & "Finally, please send the invoice to sarah@example.com for the balance.";   // ~1,000 tokens, three windows
emails = arrayMap( piiDetect( long, { types = "email" } ), function( s ) { return s.text; } );
check( "an email at the end of a long document is found whole", arrayFind( emails, "sarah@example.com" ) GT 0, arrayToList( emails ) );
check( "and nothing but the known noise fires", arrayEvery( emails, function( t ) { return t EQ "to" || t EQ "sarah@example.com"; } ), arrayToList( emails ) );

// --- errors are errors ---------------------------------------------------------
msg = "";
try { piiDetect(); } catch ( any e ) { msg = e.message; }
check( "a missing argument is reported", findNoCase( "[text]", msg ) GT 0, msg );

writeOutput( chr( 10 ) );
if ( failures ) {
	writeOutput( failures & " check(s) FAILED" & chr( 10 ) );
	throw( type = "SelfTestFailed", message = failures & " check(s) failed" );
}
writeOutput( "all checks passed" & chr( 10 ) );
</cfscript>

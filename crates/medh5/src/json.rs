//! JSON exactly as the format's documents have always been serialized.
//!
//! Three serializations are normative or load-bearing, and all three were
//! defined by Python's `json` module in the 1.x reference implementation:
//!
//! * **`/meta`** is `json.dumps(doc, ensure_ascii=False)` --- default separators
//!   `", "` and `": "`, members in insertion order.  Its bytes are hashed into
//!   `content_id` (§13.2), so two implementations that serialize one document
//!   differently give one sample two addresses.  [`dumps_meta`] reproduces it.
//! * **canonical JSON** (§5.1, §13.2): sorted keys, no insignificant whitespace,
//!   non-ASCII kept as UTF-8.  [`canonical`].
//! * **floats** are written in Python's `repr` form: the shortest digit string
//!   that round-trips, in positional notation unless the decimal exponent is
//!   below -4 or above 15, always with a `.0` when integral.  [`float_repr`].
//!
//! Everything is built on `serde_json::Value` with `preserve_order`, so an
//! object's members keep the order they were inserted in, as Python's do.

use serde_json::{Map, Value};

/// How to serialize: the subset of `json.dumps` options the format uses.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    /// Pretty-print with this many spaces per level.
    pub indent: Option<usize>,
    /// Sort object members by key.
    pub sort_keys: bool,
    /// Separator between items (`", "` by default, `","` when indenting).
    pub item_sep: &'static str,
    /// Separator between a key and its value.
    pub key_sep: &'static str,
    /// Escape every non-ASCII character as `\uXXXX`.
    pub ensure_ascii: bool,
}

impl Style {
    /// `json.dumps(v)`: Python's defaults.
    pub const PYTHON: Style =
        Style { indent: None, sort_keys: false, item_sep: ", ", key_sep: ": ", ensure_ascii: true };
    /// `json.dumps(v, ensure_ascii=False)`: the `/meta` serialization.
    pub const META: Style =
        Style { indent: None, sort_keys: false, item_sep: ", ", key_sep: ": ", ensure_ascii: false };
    /// Sorted keys, compact, UTF-8: the canonical form of §5.1 and §13.2.
    pub const CANONICAL: Style =
        Style { indent: None, sort_keys: true, item_sep: ",", key_sep: ":", ensure_ascii: false };
    /// `json.dumps(v, indent=2)`: what `--json` prints.
    pub const PRETTY: Style =
        Style { indent: Some(2), sort_keys: false, item_sep: ",", key_sep: ": ", ensure_ascii: true };

    /// The same style with a different indent.
    pub const fn with_indent(mut self, indent: Option<usize>) -> Style {
        self.indent = indent;
        if indent.is_some() {
            self.item_sep = ",";
        }
        self
    }

    /// The same style, sorting keys.
    pub const fn sorted(mut self) -> Style {
        self.sort_keys = true;
        self
    }
}

/// Serialize `value` in `style`.
pub fn dumps(value: &Value, style: Style) -> String {
    let mut out = String::new();
    write_value(&mut out, value, style, 0);
    out
}

/// `json.dumps(value, ensure_ascii=False)`, the `/meta` form.
pub fn dumps_meta(value: &Value) -> String {
    dumps(value, Style::META)
}

/// The canonical form: sorted keys, compact, UTF-8.
pub fn canonical(value: &Value) -> String {
    dumps(value, Style::CANONICAL)
}

/// `json.dumps(value, indent=2)`.
pub fn pretty(value: &Value) -> String {
    dumps(value, Style::PRETTY)
}

fn write_value(out: &mut String, value: &Value, style: Style, level: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&float_repr(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_string(out, s, style.ensure_ascii),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item_sep);
                }
                newline(out, style, level + 1);
                write_value(out, item, style, level + 1);
            }
            newline(out, style, level);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if style.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item_sep);
                }
                newline(out, style, level + 1);
                write_string(out, key, style.ensure_ascii);
                out.push_str(style.key_sep);
                write_value(out, item, style, level + 1);
            }
            newline(out, style, level);
            out.push('}');
        }
    }
}

fn newline(out: &mut String, style: Style, level: usize) {
    if let Some(indent) = style.indent {
        out.push('\n');
        for _ in 0..indent * level {
            out.push(' ');
        }
    }
}

pub(crate) fn write_string(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: shortest round-trip digits, `.0` when integral.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() };
    }
    let sci = format!("{:e}", value.abs());
    let (mantissa, exponent) = sci.split_once('e').expect("`{:e}` always has an exponent");
    let exponent: i32 = exponent.parse().expect("integer exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if digits == "0" {
        return format!("{sign}0.0");
    }
    let n = digits.len() as i32;
    // Position of the decimal point relative to the digit string: the value is
    // 0.<digits> x 10^decpt.
    let decpt = exponent + 1;
    if decpt <= -4 || decpt > 16 {
        let mut out = String::from(sign);
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", e.abs()));
        out
    } else if decpt <= 0 {
        format!("{sign}0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt < n {
        format!("{sign}{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    } else {
        format!("{sign}{}{}.0", digits, "0".repeat((decpt - n) as usize))
    }
}

/// Python's `repr(str)`.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c if is_unprintable(c) => {
                let cp = c as u32;
                if cp <= 0xff {
                    out.push_str(&format!("\\x{cp:02x}"));
                } else if cp <= 0xffff {
                    out.push_str(&format!("\\u{cp:04x}"));
                } else {
                    out.push_str(&format!("\\U{cp:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn is_unprintable(c: char) -> bool {
    let cp = c as u32;
    (0x80..0xa0).contains(&cp) || cp == 0xad || (0xd800..0xe000).contains(&cp)
}

/// Python's `repr()` of a JSON value: `'a'`, `None`, `True`, `[1, 2]`, `{'k': 1}`.
pub fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f.is_nan() {
                    "nan".to_string()
                } else if f.is_infinite() {
                    if f > 0.0 {
                        "inf".to_string()
                    } else {
                        "-inf".to_string()
                    }
                } else {
                    float_repr(f)
                }
            }
        }
        Value::String(s) => repr_str(s),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map.iter().map(|(k, v)| format!("{}: {}", repr_str(k), repr(v))).collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// Python's truthiness of a JSON value: `None`, `False`, `0`, `""`, `[]` and
/// `{}` are false.
pub fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python's `str()` of a JSON value: strings bare, everything else as `repr`.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => repr(other),
    }
}

/// `[‘a’, ‘b’]` as Python prints a list of strings.
pub fn repr_list<S: AsRef<str>>(items: &[S]) -> String {
    let inner: Vec<String> = items.iter().map(|s| repr_str(s.as_ref())).collect();
    format!("[{}]", inner.join(", "))
}

/// `(1, 2, 3)` as Python prints a tuple of integers.
pub fn repr_int_tuple<T: std::fmt::Display>(items: &[T]) -> String {
    match items.len() {
        1 => format!("({},)", items[0]),
        _ => {
            let inner: Vec<String> = items.iter().map(|i| i.to_string()).collect();
            format!("({})", inner.join(", "))
        }
    }
}

/// `[1, 2, 3]` as Python prints a list of integers.
pub fn repr_int_list<T: std::fmt::Display>(items: &[T]) -> String {
    let inner: Vec<String> = items.iter().map(|i| i.to_string()).collect();
    format!("[{}]", inner.join(", "))
}

/// A float as Python's `str()`/`repr()` prints it (`inf`, `nan` spelled lower).
pub fn py_float(value: f64) -> String {
    if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else {
        float_repr(value)
    }
}

/// `format(value, "g")`: six significant digits, exponent when out of range.
pub fn format_g(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0".to_string() } else { "0".to_string() };
    }
    let p = precision.max(1);
    let sci = format!("{:.*e}", p - 1, value);
    let (mantissa, exponent) = sci.split_once('e').unwrap();
    let exp: i32 = exponent.parse().unwrap();
    if exp < -4 || exp >= p as i32 {
        let mantissa = strip_trailing_zeros(mantissa);
        format!("{}e{}{:02}", mantissa, if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        strip_trailing_zeros(&format!("{:.*}", decimals, value))
    }
}

fn strip_trailing_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// An empty JSON object.
pub fn object() -> Map<String, Value> {
    Map::new()
}

/// A JSON number from an `f64`, or `null` when it is not finite.
pub fn num(value: f64) -> Value {
    serde_json::Number::from_f64(value).map(Value::Number).unwrap_or(Value::Null)
}

/// Parse a JSON document.
pub fn loads(text: &str) -> Result<Value, serde_json::Error> {
    serde_json::from_str(text)
}

/// A token JSON does not have, and where it was (1-based, in characters).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonFinite {
    pub token: &'static str,
    pub line: usize,
    pub column: usize,
}

impl std::fmt::Display for NonFinite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at line {} column {}", self.token, self.line, self.column)
    }
}

/// Parse JSON that 1.x may have written: `NaN`, `Infinity` and `-Infinity`,
/// which Python's `json.dumps` emits and JSON does not have, read as `null`.
///
/// A file is not rewritten by reading it, so the stored text keeps them; the
/// first one is returned for the validator to report (E004).  Anything else
/// that is not JSON is the strict parser's error.
pub fn loads_lenient(text: &str) -> Result<(Value, Option<NonFinite>), serde_json::Error> {
    match serde_json::from_str(text) {
        Ok(value) => Ok((value, None)),
        Err(err) => {
            let (patched, first) = null_non_finite(text);
            match first {
                Some(found) => serde_json::from_str(&patched).map(|v| (v, Some(found))).map_err(|_| err),
                None => Err(err),
            }
        }
    }
}

/// `text` with every non-finite token outside a string replaced by `null`.
fn null_non_finite(text: &str) -> (String, Option<NonFinite>) {
    const TOKENS: [&str; 3] = ["-Infinity", "Infinity", "NaN"];
    let mut out = String::with_capacity(text.len());
    let mut first = None;
    let (mut in_string, mut escaped) = (false, false);
    let (mut line, mut column) = (1, 1);
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if !in_string {
            if let Some(token) = TOKENS.iter().copied().find(|t| rest.starts_with(t)) {
                first.get_or_insert(NonFinite { token, line, column });
                out.push_str("null");
                rest = &rest[token.len()..];
                column += token.len();
                continue;
            }
        }
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
        }
        out.push(c);
        if c == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
        rest = &rest[c.len_utf8()..];
    }
    (out, first)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_print_as_python_repr() {
        for (value, expected) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (1.5, "1.5"),
            (0.1, "0.1"),
            (0.8, "0.8"),
            (-12.0, "-12.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.2345e-7, "1.2345e-07"),
            (2.5e-5, "2.5e-05"),
            (2.34567, "2.34567"),
            (1e100, "1e+100"),
            (0.33325, "0.33325"),
            (9.6, "9.6"),
        ] {
            assert_eq!(float_repr(value), expected, "{value}");
        }
    }

    #[test]
    fn meta_style_matches_python_defaults() {
        let v = json!({"a": 1, "b": [1.0, "x"], "c": {"é": null}});
        assert_eq!(dumps_meta(&v), r#"{"a": 1, "b": [1.0, "x"], "c": {"é": null}}"#);
        let escaped = format!("{{\"a\": 1, \"b\": [1.0, \"x\"], \"c\": {{\"{}u00e9\": null}}}}", '\\');
        assert_eq!(dumps(&v, Style::PYTHON), escaped);
        assert_eq!(canonical(&json!({"b": 1, "a": [2, 3]})), r#"{"a":[2,3],"b":1}"#);
        assert_eq!(
            pretty(&json!({"a": [], "b": {}, "c": [1]})),
            "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": [\n    1\n  ]\n}"
        );
    }

    #[test]
    fn strings_escape_like_python() {
        assert_eq!(dumps_meta(&json!("a\"b\\c\n\u{1}")), r#""a\"b\\c\n\u0001""#);
        let bs = '\\';
        assert_eq!(dumps(&json!("\u{1F600}"), Style::PYTHON), format!("\"{bs}ud83d{bs}ude00\""));
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("a"), "'a'");
    }

    #[test]
    fn format_g_matches_python() {
        assert_eq!(format_g(1e-4, 6), "0.0001");
        assert_eq!(format_g(0.000123456, 3), "0.000123");
        assert_eq!(format_g(1234567.0, 6), "1.23457e+06");
        assert_eq!(format_g(0.5, 6), "0.5");
        assert_eq!(format_g(2.0, 3), "2");
    }
}

#[cfg(test)]
mod non_finite_tests {
    use super::*;

    /// 1.x's `json.dumps` wrote these; JSON has none of them.
    #[test]
    fn s2_4_non_finite_tokens_read_as_null_and_are_located() {
        let text = "{\"a\": NaN, \"s\": \"NaN in a string\",\n \"b\": [-Infinity, Infinity, 1.5]}";
        assert!(loads(text).is_err());
        let (value, found) = loads_lenient(text).unwrap();
        assert_eq!(value["a"], Value::Null);
        assert_eq!(value["s"], Value::from("NaN in a string"));
        assert_eq!(value["b"], serde_json::json!([null, null, 1.5]));
        assert_eq!(found, Some(NonFinite { token: "NaN", line: 1, column: 7 }));
        assert_eq!(found.unwrap().to_string(), "NaN at line 1 column 7");
    }

    #[test]
    fn s2_4_valid_json_and_other_errors_are_untouched() {
        assert_eq!(loads_lenient("{\"a\": 1}").unwrap(), (serde_json::json!({"a": 1}), None));
        assert!(loads_lenient("{\"a\": nan}").is_err());
        assert!(loads_lenient("{\"a\": NaN,}").is_err());
        let escaped = "{\"q\": \"\\\"NaN\\\"\", \"n\": NaN}";
        let (value, found) = loads_lenient(escaped).unwrap();
        assert_eq!(value["q"], Value::from("\"NaN\""));
        assert_eq!(found.unwrap().column, 23);
    }
}

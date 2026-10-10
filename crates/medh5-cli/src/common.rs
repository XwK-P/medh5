//! What every command shares: output streams, JSON output, errors, tables.

use std::io::Write;
use std::path::PathBuf;

use clap::ArgMatches;
use serde_json::Value;

use medh5::json::{format_g, pretty, repr_str};
use medh5::Error;

/// Unix-conventional exit codes: success, a handled error, a usage error.
pub const EXIT_OK: i32 = 0;
pub const EXIT_ERROR: i32 = 1;
pub const EXIT_USAGE: i32 = 2;

/// What a frontend hosting the engine can run that the engine cannot.
///
/// The format converters (NIfTI, DICOM, DICOM SEG, RTSTRUCT, nnU-Net) and the
/// dataloader benchmark are Python integrations: they stay in the Python
/// package, and the command line hands them over to it.
pub trait Host {
    /// Run `medh5 convert COMMAND`; `None` when no Python package is
    /// reachable.  `argv` is the whole command line; `args` the parsed
    /// arguments of `COMMAND`, keyed as the 1.x parser named them.
    fn convert(
        &self,
        argv: &[String],
        command: &str,
        args: &Value,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Option<i32>;

    /// The dataloader throughput measurement, as `Measurement.to_json()`;
    /// `Err(reason)` when it cannot run here.
    fn throughput(&self, path: &str, patch: i64, workers: i64, annotation: Option<&str>) -> Result<Value, String>;
}

/// One command's view of the outside world.
pub struct Ctx<'a> {
    out: &'a mut dyn Write,
    err: &'a mut dyn Write,
    pub host: &'a dyn Host,
    /// Set when stdout went away (`medh5 info | head`): not an error.
    pub broken_pipe: bool,
}

impl<'a> Ctx<'a> {
    pub fn new(out: &'a mut dyn Write, err: &'a mut dyn Write, host: &'a dyn Host) -> Ctx<'a> {
        Ctx { out, err, host, broken_pipe: false }
    }

    /// `print(text)`.
    pub fn print(&mut self, text: impl AsRef<str>) {
        if self.broken_pipe {
            return;
        }
        let result = writeln!(self.out, "{}", text.as_ref());
        if let Err(e) = result {
            if e.kind() == std::io::ErrorKind::BrokenPipe {
                self.broken_pipe = true;
            }
        }
    }

    /// `print(text, file=sys.stderr)`.
    pub fn eprint(&mut self, text: impl AsRef<str>) {
        let _ = writeln!(self.err, "{}", text.as_ref());
        let _ = self.err.flush();
    }

    pub fn out(&mut self) -> &mut dyn Write {
        self.out
    }

    pub fn err(&mut self) -> &mut dyn Write {
        self.err
    }

    pub fn streams(&mut self) -> (&mut dyn Write, &mut dyn Write) {
        (self.out, self.err)
    }

    /// Print a JSON document, or nothing when the caller wants text output.
    pub fn emit(&mut self, payload: &Value, as_json: bool) {
        if as_json {
            self.print(pretty(payload));
        }
    }

    /// `medh5: message` on stderr, and the handled-error exit code.
    pub fn fail(&mut self, message: impl AsRef<str>) -> i32 {
        self.eprint(format!("medh5: {}", message.as_ref()));
        EXIT_ERROR
    }

    pub fn flush(&mut self) {
        let _ = self.out.flush();
        let _ = self.err.flush();
    }
}

/// A command's standard output as a [`Write`], for a document written as it
/// is produced: once the reader has gone it goes quiet, as [`Ctx::print`]
/// does, and any other failure is the command's error.
pub struct Stdout<'c, 'a>(pub &'c mut Ctx<'a>);

impl Write for Stdout<'_, '_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if !self.0.broken_pipe {
            if let Err(e) = self.0.out.write_all(buf) {
                if e.kind() != std::io::ErrorKind::BrokenPipe {
                    return Err(e);
                }
                self.0.broken_pipe = true;
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.0.broken_pipe {
            return Ok(());
        }
        self.0.out.flush()
    }
}

/// A command's result: an exit code, or an engine error to report.
pub type CmdResult = Result<i32, Error>;

/// What the 1.x CLI printed for an error that reached its top level.
///
/// A lookup failure (`KeyError`) printed its message as it was when it was a
/// sentence, and named the key otherwise; everything else printed `str(exc)`.
pub fn top_level_message(error: &Error) -> String {
    if error.is_lookup() {
        let text = error.message();
        if text.contains(' ') {
            return text.to_string();
        }
        return format!("no such entry: {}", repr_str(text));
    }
    error.to_string()
}

// -- argument access ---------------------------------------------------------

pub fn get_str<'m>(m: &'m ArgMatches, id: &str) -> Option<&'m str> {
    m.get_one::<String>(id).map(String::as_str)
}

pub fn req_str<'m>(m: &'m ArgMatches, id: &str) -> &'m str {
    get_str(m, id).unwrap_or_default()
}

pub fn get_strs(m: &ArgMatches, id: &str) -> Option<Vec<String>> {
    m.get_many::<String>(id).map(|v| v.cloned().collect())
}

pub fn get_paths(m: &ArgMatches, id: &str) -> Vec<String> {
    get_strs(m, id).unwrap_or_default()
}

pub fn get_i64(m: &ArgMatches, id: &str) -> Option<i64> {
    m.get_one::<i64>(id).copied()
}

pub fn get_f64(m: &ArgMatches, id: &str) -> Option<f64> {
    m.get_one::<f64>(id).copied()
}

pub fn flag(m: &ArgMatches, id: &str) -> bool {
    m.get_flag(id)
}

pub fn path_of(text: &str) -> PathBuf {
    PathBuf::from(text)
}

// -- text output -------------------------------------------------------------

/// `1.5 KiB`: bytes in binary units, as the 1.x CLI printed them.
pub fn human_bytes(n: f64) -> String {
    let mut n = n;
    for unit in ["B", "KiB", "MiB", "GiB"] {
        if n.abs() < 1024.0 || unit == "GiB" {
            return if unit == "B" { format!("{} B", n.trunc() as i64) } else { format!("{n:.1} {unit}") };
        }
        n /= 1024.0;
    }
    format!("{n:.1} GiB")
}

/// Indent every line of a block, for nesting a table under a heading.
pub fn indent(text: &str) -> String {
    text.lines().map(|line| format!("  {line}")).collect::<Vec<_>>().join("\n")
}

fn ljust(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if len >= width {
        text.to_string()
    } else {
        format!("{text}{}", " ".repeat(width - len))
    }
}

/// `format(text, "Ns")`: left-justified to `width` characters.
pub fn pad(text: &str, width: usize) -> String {
    ljust(text, width)
}

/// A minimal fixed-width table --- no dependency, predictable in a pipe.
pub fn table<S: AsRef<str>>(rows: &[Vec<String>], headers: &[S]) -> String {
    let widths: Vec<usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let mut w = h.as_ref().chars().count();
            for row in rows {
                w = w.max(row[i].chars().count());
            }
            w
        })
        .collect();
    let line = headers.iter().zip(&widths).map(|(h, w)| ljust(h.as_ref(), *w)).collect::<Vec<_>>().join("  ");
    let mut out = vec![line, widths.iter().map(|w| "-".repeat(*w)).collect::<Vec<_>>().join("  ")];
    for row in rows {
        out.push(row.iter().zip(&widths).map(|(c, w)| ljust(c, *w)).collect::<Vec<_>>().join("  "));
    }
    out.join("\n")
}

/// `f"{v:g}"`.
pub fn g(v: f64) -> String {
    format_g(v, 6)
}

/// `f"{v:.Ng}"`.
pub fn gp(v: f64, precision: usize) -> String {
    format_g(v, precision)
}

/// `f"{v:.N%}"` and, with `sign`, `f"{v:+.N%}"`.
pub fn percent(v: f64, precision: usize, sign: bool) -> String {
    let scaled = v * 100.0;
    if sign {
        format!("{scaled:+.precision$}%")
    } else {
        format!("{scaled:.precision$}%")
    }
}

/// `f"{n:,}"` for an integer.
pub fn thousands(n: i128) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

/// Python's `str()` of a JSON value, for a table cell.
pub fn cell(value: &Value) -> String {
    medh5::json::py_str(value)
}

/// `value or "-"` for a JSON value: falsy values print as a dash.
pub fn or_dash(value: &Value) -> String {
    if truthy(value) {
        cell(value)
    } else {
        "-".into()
    }
}

/// Python truthiness of a JSON value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `",".join(values)` over a JSON array of strings.
pub fn join_strs(value: &Value, sep: &str) -> String {
    match value {
        Value::Array(items) => items.iter().map(cell).collect::<Vec<_>>().join(sep),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_prints_what_1x_printed() {
        assert_eq!(human_bytes(0.0), "0 B");
        assert_eq!(human_bytes(1023.0), "1023 B");
        assert_eq!(human_bytes(512.0), "512 B");
        assert_eq!(human_bytes(1024.0), "1.0 KiB");
        assert_eq!(human_bytes(2048.0), "2.0 KiB");
        assert_eq!(human_bytes(5.0 * 1024.0 * 1024.0 * 1024.0), "5.0 GiB");
        assert_eq!(human_bytes(1536.0 * 1024.0), "1.5 MiB");
        assert_eq!(human_bytes(5.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0), "5120.0 GiB");
    }

    #[test]
    fn tables_pad_by_characters() {
        let rows = vec![vec!["§1".to_string(), "x".to_string()]];
        assert_eq!(table(&rows, &["a", "bb"]), "a   bb\n--  --\n§1  x ");
        let rows = vec![vec!["a".to_string(), "1".to_string()], vec!["bbbb".to_string(), "22".to_string()]];
        assert!(table(&rows, &["k", "v"]).starts_with("k     v"));
        assert_eq!(table(&[], &["k", "v"]).lines().next(), Some("k  v"));
    }

    #[test]
    fn number_formats_match_python() {
        assert_eq!(percent(0.1234, 1, true), "+12.3%");
        assert_eq!(percent(-0.5, 1, true), "-50.0%");
        assert_eq!(percent(0.75, 0, false), "75%");
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(thousands(12), "12");
    }
}

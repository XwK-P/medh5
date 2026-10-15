//! What every command shares: output streams, JSON output, errors, tables.

use std::io::Write;
use std::path::Path;

use clap::ArgMatches;
use serde_json::{json, Value};

use medh5::collection::open_collection;
use medh5::json::{format_g, pretty, repr_str};
use medh5::sample::{open_sample, Sample};
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

    /// The program and arguments that run this command line again, as a
    /// worker of a `--workers` run; `None` when the host cannot start one,
    /// and a `--workers` run then runs in one process and says so.
    fn worker_command(&self) -> Option<Vec<std::ffi::OsString>> {
        None
    }
}

/// One command's view of the outside world.
pub struct Ctx<'a> {
    out: &'a mut dyn Write,
    err: &'a mut dyn Write,
    pub host: &'a dyn Host,
    /// Set when stdout went away (`medh5 info | head`): not an error.
    pub broken_pipe: bool,
    /// The first failure to write stdout other than its reader going away
    /// (a full disk, a closed descriptor): the output is lost, so the command
    /// has failed whatever it computed.
    write_error: Option<std::io::Error>,
}

impl<'a> Ctx<'a> {
    pub fn new(out: &'a mut dyn Write, err: &'a mut dyn Write, host: &'a dyn Host) -> Ctx<'a> {
        Ctx { out, err, host, broken_pipe: false, write_error: None }
    }

    /// `print(text)`.  Once stdout has failed nothing more is written to it.
    pub fn print(&mut self, text: impl AsRef<str>) {
        if self.broken_pipe || self.write_error.is_some() {
            return;
        }
        let result = writeln!(self.out, "{}", text.as_ref());
        self.record(result);
    }

    /// Note how a write to stdout went.
    fn record(&mut self, result: std::io::Result<()>) {
        if let Err(e) = result {
            if e.kind() == std::io::ErrorKind::BrokenPipe {
                self.broken_pipe = true;
            } else if self.write_error.is_none() {
                self.write_error = Some(e);
            }
        }
    }

    /// `print(text, file=sys.stderr)`.  What stdout holds goes out first, so
    /// the two read in the order they were written where they meet (a
    /// terminal, `2>&1`) however stdout is buffered.
    pub fn eprint(&mut self, text: impl AsRef<str>) {
        self.flush_out();
        let _ = writeln!(self.err, "{}", text.as_ref());
        let _ = self.err.flush();
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

    /// `medh5: message` on stderr, and the handled-error exit code.  Every
    /// failure is said here, so here it is put in the command line's terms.
    pub fn fail(&mut self, message: impl AsRef<str>) -> i32 {
        self.eprint(format!("medh5: {}", in_cli_terms(message.as_ref())));
        EXIT_ERROR
    }

    /// `medh5: message` on stderr, and the usage-error exit code: for a
    /// combination of arguments the grammar cannot refuse by itself.
    pub fn usage(&mut self, message: impl AsRef<str>) -> i32 {
        self.eprint(format!("medh5: {}", message.as_ref()));
        EXIT_USAGE
    }

    /// A file of a batch that could not be processed: `medh5: why` on stderr
    /// as it happens, and the file's entry in the `--json` document.  The
    /// batch goes on, so the files before it keep their reports and the files
    /// after it are still processed; the command fails at the end.
    pub fn file_failed(&mut self, path: &str, error: &Error) -> Value {
        let message = in_cli_terms(&top_level_message(error));
        self.eprint(format!("medh5: {message}"));
        json!({"path": path, "ok": false, "error": message})
    }

    /// Hand stdout what it holds, unless it has already failed.
    fn flush_out(&mut self) {
        if !self.broken_pipe && self.write_error.is_none() {
            let result = self.out.flush();
            self.record(result);
        }
    }

    pub fn flush(&mut self) {
        self.flush_out();
        let _ = self.err.flush();
    }

    /// The exit code, once the output is flushed: the command's `code`, unless
    /// stdout could not take the output --- then the command has failed, and
    /// says so.  A reader that went away early (`medh5 validate ... | head`)
    /// is not a failure, and not a success either: the command's own code
    /// stands.
    pub fn finish(mut self, code: i32) -> i32 {
        self.flush();
        // Kept, not taken: stdout has failed, and the failure said on stderr
        // does not try it again.
        match self.write_error.as_ref().map(os_reason) {
            Some(reason) => self.fail(format!("cannot write output: {reason}")),
            None => code,
        }
    }
}

/// An OS error as Python words one with no file name to it:
/// `[Errno 28] No space left on device`.
fn os_reason(e: &std::io::Error) -> String {
    let full = e.to_string();
    match e.raw_os_error() {
        Some(code) => {
            let reason = full.strip_suffix(&format!(" (os error {code})")).unwrap_or(&full);
            format!("[Errno {code}] {reason}")
        }
        None => full,
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

/// What the engine says in the Python API's terms --- an option by its
/// keyword argument, a collection by the function that opens one --- and
/// what the command line says instead.
const CLI_TERMS: [(&str, &str); 4] = [
    ("pass rewrite_digests with a reason", "pass --rewrite-digests with --reason"),
    ("rebuild them with rebuild_index", "rebuild them with --rebuild-index"),
    ("`key` applies to collections", "--key applies to collections"),
    ("open it with open_collection()", "extract its samples with `medh5 unpack`"),
];

/// `text` in the command line's terms: `--rebuild-index` where the engine,
/// speaking to the Python API, says `rebuild_index`.
pub fn in_cli_terms(text: &str) -> String {
    CLI_TERMS.iter().fold(text.to_string(), |text, (api, cli)| text.replace(api, cli))
}

// -- a sample file, or a collection ------------------------------------------

/// The sample file `path`, for a command that works on one: a collection is
/// refused as [`sample_file_error`] words it.
pub fn sample_file(path: &str) -> medh5::Result<Sample> {
    open_sample(Path::new(path)).map_err(|e| sample_file_error(path, e))
}

/// `error`, from working on `path` as one sample file, as the command line
/// words it.  The engine refuses a collection in the Python API's terms
/// ("open it with open_collection()"); from the command line, the way to a
/// member is `medh5 unpack`.
pub fn sample_file_error(path: &str, error: Error) -> Error {
    match sample_file_refusal(path) {
        Some(refusal) => Error::invalid(refusal),
        None => error,
    }
}

/// Why `path` is not worked on as one sample file, as the command line words
/// it: a collection is named with the way to its members.  `None` for
/// anything else, whose own error says what is wrong.  Public for the Python
/// converters, which refuse a collection in the same words.
#[doc(hidden)]
pub fn sample_file_refusal(path: &str) -> Option<String> {
    let count = open_collection(Path::new(path)).and_then(|c| c.len()).ok()?;
    Some(format!(
        "{path} is a collection of {count} sample(s); this command works on one sample file --- extract them with \
         `medh5 unpack {path} -o DIR`"
    ))
}

/// `error`, from working on `path` as a collection, as the command line
/// words it: a sample file is named as one, not as a wrong `medh5_kind` with
/// the Python API's way to open it.
pub fn collection_file_error(path: &str, error: Error, instead: &str) -> Error {
    match open_sample(Path::new(path)) {
        Ok(_) => Error::invalid(format!("{path} is a sample file, not a collection; {instead}")),
        Err(_) => error,
    }
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

/// `format(text, "Ns")`: left-justified to `width` characters.
pub fn pad(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if len >= width {
        text.to_string()
    } else {
        format!("{text}{}", " ".repeat(width - len))
    }
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
    let line = headers.iter().zip(&widths).map(|(h, w)| pad(h.as_ref(), *w)).collect::<Vec<_>>().join("  ");
    let mut out = vec![line, widths.iter().map(|w| "-".repeat(*w)).collect::<Vec<_>>().join("  ")];
    for row in rows {
        out.push(row.iter().zip(&widths).map(|(c, w)| pad(c, *w)).collect::<Vec<_>>().join("  "));
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

/// Python truthiness of a JSON value: the engine's one implementation.
pub use medh5::pyval::truthy;

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

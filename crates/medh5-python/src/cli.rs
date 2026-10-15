//! `medh5.cli`: the native command line, run inside the Python package.
//!
//! The grammar, the output and the exit codes are the Rust CLI's
//! (`medh5-cli`); this module runs it against Python's `sys.stdout` and
//! `sys.stderr` --- looked up on every write, so redirection and test capture
//! see it --- and hands the commands only Python can run (the format
//! converters, the PyTorch dataloader benchmark) to a Python host object.

use std::ffi::OsString;
use std::io::Write;

use pyo3::exceptions::{PyBrokenPipeError, PyUnicodeEncodeError};
use pyo3::prelude::*;
use pyo3::types::PyString;
use serde_json::Value;

use crate::convert::{json_to_py, py_to_json};

/// `sys.stdout` or `sys.stderr`, as a byte sink.
///
/// Buffered: each `write` was a call into Python, and a document written in
/// fragments --- `writeln!`, a streamed `--json` --- made one per fragment.
/// A line goes when it ends where someone is watching the stream, as Python's
/// own buffering does; a block at a time elsewhere.  What is left when it is
/// dropped unflushed is dropped with it: the command line flushes what it
/// means to show ([`Write::flush`]), and leaves unflushed only what a stream
/// that failed could not take.
struct PyStream {
    name: &'static str,
    /// Written and not yet handed to Python, a UTF-8 sequence split across
    /// two writes among it.
    pending: Vec<u8>,
    /// Hand Python each line as it ends, rather than a block at a time.
    lines: bool,
}

/// The most a block-buffered stream holds before handing it to Python.
const BLOCK: usize = 64 * 1024;

/// A stream's exception as the I/O error the CLI judges: a reader that went
/// away is a broken pipe, anything else output that was lost.
fn io_error(py: Python<'_>, e: PyErr) -> std::io::Error {
    let kind = if e.is_instance_of::<PyBrokenPipeError>(py) {
        std::io::ErrorKind::BrokenPipe
    } else {
        std::io::ErrorKind::Other
    };
    std::io::Error::new(kind, e.value(py).to_string())
}

impl PyStream {
    /// `sys.<name>`, a line at a time where Python's own stream goes at least
    /// that often: stderr always, and a stream that is line-buffered (stdout
    /// at a terminal), writes through (`python -u`, `PYTHONUNBUFFERED`) or is
    /// a terminal.  Each is asked on its own, and one a stream cannot answer
    /// (`io.StringIO` has no `write_through`) is a no.
    fn new(py: Python<'_>, name: &'static str) -> Self {
        let lines = name == "stderr"
            || py.import("sys").and_then(|sys| sys.getattr(name)).is_ok_and(|stream| {
                let says = |answer: PyResult<Bound<'_, PyAny>>| answer.and_then(|a| a.is_truthy()).unwrap_or(false);
                says(stream.getattr("line_buffering"))
                    || says(stream.getattr("write_through"))
                    || says(stream.call_method0("isatty"))
            });
        PyStream { name, pending: Vec::new(), lines }
    }

    /// Hand Python what of the first `due` pending bytes is whole UTF-8; the
    /// rest of a sequence split across two writes waits for its end.
    fn send_pending(&mut self, due: usize) -> std::io::Result<()> {
        let valid = match std::str::from_utf8(&self.pending[..due]) {
            Ok(_) => due,
            Err(e) => e.valid_up_to(),
        };
        if valid == 0 {
            return Ok(());
        }
        let text = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        self.send(&text)
    }

    fn send(&self, text: &str) -> std::io::Result<()> {
        Python::attach(|py| {
            let sent = (|| -> PyResult<()> {
                let stream = py.import("sys")?.getattr(self.name)?;
                match stream.call_method1("write", (text,)) {
                    // A stream whose encoding cannot spell a character --- a
                    // redirected stdout on Windows writes the ANSI code page,
                    // which has no `├` --- gets it escaped (`├`) rather
                    // than losing the line, as `sys.stderr` already does.
                    Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) => {
                        let encoding: String = stream.getattr("encoding")?.extract()?;
                        let escaped = PyString::new(py, text)
                            .call_method1("encode", (encoding.as_str(), "backslashreplace"))?
                            .call_method1("decode", (encoding.as_str(),))?;
                        stream.call_method1("write", (escaped,)).map(drop)
                    }
                    written => written.map(drop),
                }
            })();
            sent.map_err(|e| io_error(py, e))
        })
    }
}

impl Write for PyStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.pending.extend_from_slice(buf);
        let due = if self.lines {
            self.pending.iter().rposition(|b| *b == b'\n').map_or(0, |end| end + 1)
        } else if self.pending.len() >= BLOCK {
            self.pending.len()
        } else {
            0
        };
        if due > 0 {
            self.send_pending(due)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if !self.pending.is_empty() {
            // Whatever is left, a sequence cut short replaced.
            let text = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            self.send(&text)?;
        }
        Python::attach(|py| {
            let flushed =
                (|| -> PyResult<()> { py.import("sys")?.getattr(self.name)?.call_method0("flush").map(drop) })();
            flushed.map_err(|e| io_error(py, e))
        })
    }
}

/// The Python package as the CLI's host: `host.convert(command, args)` and
/// `host.throughput(path, patch, workers, annotation)`.  The converters run
/// in this process, on the arguments the CLI parsed; the command line itself
/// is for a host that starts another one.
struct PyHost {
    host: Py<PyAny>,
}

impl medh5_cli::Host for PyHost {
    fn convert(
        &self,
        _argv: &[String],
        command: &str,
        args: &Value,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Option<i32> {
        // The converters print through `sys.stdout`/`sys.stderr` themselves;
        // flush the CLI's streams first so the two interleave in order.
        let _ = out.flush();
        let _ = err.flush();
        let result = Python::attach(|py| -> PyResult<Option<i32>> {
            let args = json_to_py(py, args)?;
            let found = self.host.bind(py).call_method1("convert", (command, args))?;
            if found.is_none() {
                return Ok(None);
            }
            found.extract::<i32>().map(Some)
        });
        match result {
            Ok(code) => code,
            Err(e) => {
                // An exception the converter did not handle is a traceback in
                // 1.x; keep it visible rather than swallowing it.
                Python::attach(|py| e.print(py));
                Some(medh5_cli::EXIT_ERROR)
            }
        }
    }

    fn throughput(&self, path: &str, patch: i64, workers: i64, annotation: Option<&str>) -> Result<Value, String> {
        Python::attach(|py| {
            let found = self
                .host
                .bind(py)
                .call_method1("throughput", (path, patch, workers, annotation))
                .map_err(|e| e.value(py).to_string())?;
            py_to_json(&found).map_err(|e| e.to_string())
        })
    }

    /// This interpreter, running `medh5.cli.main` --- the current directory
    /// off `sys.path` first, as the standalone binary's hand-off runs it, so
    /// a `medh5/` where the command runs is not imported in its place.
    fn worker_command(&self) -> Option<Vec<OsString>> {
        let executable: String = Python::attach(|py| py.import("sys")?.getattr("executable")?.extract()).ok()?;
        if executable.is_empty() {
            return None;
        }
        let code = "import sys; sys.path[:1] == [''] and sys.path.pop(0); from medh5.cli import main; \
                    raise SystemExit(main())";
        Some(vec![executable.into(), "-c".into(), code.into()])
    }
}

/// Run the command line on `argv` (without the program name); the exit code.
#[pyfunction]
fn cli_main(py: Python<'_>, argv: Vec<OsString>, host: Py<PyAny>) -> i32 {
    let host = PyHost { host };
    let mut out = PyStream::new(py, "stdout");
    let mut err = PyStream::new(py, "stderr");
    // `sys.argv` carries an argument that is not UTF-8 as surrogate escapes,
    // which no conversion to text accepts: taken as the operating system's
    // bytes, it is refused as the standalone binary refuses it.
    let argv = match medh5_cli::utf8_args(argv) {
        Ok(argv) => argv,
        Err(message) => {
            let _ = writeln!(err, "medh5: {message}");
            let _ = err.flush();
            return medh5_cli::EXIT_USAGE;
        }
    };
    py.detach(|| medh5_cli::run_with(&argv, &mut out, &mut err, &host))
}

/// The grammar as data: `{"options", "positionals", "commands"}`.
#[pyfunction]
fn cli_command_tree(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    json_to_py(py, &medh5_cli::command_tree())
}

/// What the command line prints for a failed lookup whose message is `text`:
/// the message when it is a sentence, the key named otherwise.
#[pyfunction]
fn cli_lookup_message(text: &str) -> String {
    medh5_cli::top_level_message(&medh5::Error::Key(text.to_string()))
}

/// Why `path` is not worked on as one sample file, as every native command
/// words it (a collection, with the way to its members); `None` otherwise.
#[pyfunction]
fn cli_sample_file_refusal(py: Python<'_>, path: String) -> Option<String> {
    py.detach(|| medh5_cli::sample_file_refusal(&path))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    m.add_function(wrap_pyfunction!(cli_sample_file_refusal, m)?)?;
    m.add_function(wrap_pyfunction!(cli_command_tree, m)?)?;
    m.add_function(wrap_pyfunction!(cli_lookup_message, m)?)?;
    m.add("EXIT_OK", medh5_cli::EXIT_OK)?;
    m.add("EXIT_ERROR", medh5_cli::EXIT_ERROR)?;
    m.add("EXIT_USAGE", medh5_cli::EXIT_USAGE)?;
    Ok(())
}

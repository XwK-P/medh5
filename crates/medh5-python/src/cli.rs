//! `medh5.cli`: the native command line, run inside the Python package.
//!
//! The grammar, the output and the exit codes are the Rust CLI's
//! (`medh5-cli`); this module runs it against Python's `sys.stdout` and
//! `sys.stderr` --- looked up on every write, so redirection and test capture
//! see it --- and hands the commands only Python can run (the format
//! converters, the PyTorch dataloader benchmark) to a Python host object.

use std::io::Write;

use pyo3::exceptions::PyBrokenPipeError;
use pyo3::prelude::*;
use serde_json::Value;

use crate::convert::{json_to_py, py_to_json};

/// `sys.stdout` or `sys.stderr`, as a byte sink.
struct PyStream {
    name: &'static str,
    /// Bytes of a UTF-8 sequence split across two writes.
    pending: Vec<u8>,
}

impl PyStream {
    fn new(name: &'static str) -> Self {
        PyStream { name, pending: Vec::new() }
    }

    fn send(&self, text: &str) -> std::io::Result<()> {
        Python::attach(|py| -> PyResult<()> {
            let stream = py.import("sys")?.getattr(self.name)?;
            stream.call_method1("write", (text,))?;
            Ok(())
        })
        .map_err(|e| {
            let broken = Python::attach(|py| e.is_instance_of::<PyBrokenPipeError>(py));
            let kind = if broken { std::io::ErrorKind::BrokenPipe } else { std::io::ErrorKind::Other };
            std::io::Error::new(kind, e.to_string())
        })
    }
}

impl Write for PyStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.pending.extend_from_slice(buf);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid > 0 {
            let text = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
            self.pending.drain(..valid);
            self.send(&text)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if !self.pending.is_empty() {
            let text = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            self.send(&text)?;
        }
        Python::attach(|py| -> PyResult<()> {
            py.import("sys")?.getattr(self.name)?.call_method0("flush")?;
            Ok(())
        })
        .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// The Python package as the CLI's host: `host.convert(argv, command, args)`
/// and `host.throughput(path, patch, workers, annotation)`.
struct PyHost {
    host: Py<PyAny>,
}

impl medh5_cli::Host for PyHost {
    fn convert(
        &self,
        argv: &[String],
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
            let found = self.host.bind(py).call_method1("convert", (argv.to_vec(), command, args))?;
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
}

/// Run the command line on `argv` (without the program name); the exit code.
#[pyfunction]
fn cli_main(py: Python<'_>, argv: Vec<String>, host: Py<PyAny>) -> i32 {
    let host = PyHost { host };
    let mut out = PyStream::new("stdout");
    let mut err = PyStream::new("stderr");
    py.detach(|| medh5_cli::run_with(&argv, &mut out, &mut err, &host))
}

/// The grammar as data: `{"options", "positionals", "commands"}`.
#[pyfunction]
fn cli_command_tree(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    json_to_py(py, &medh5_cli::command_tree())
}

/// `512 B`, `2.0 KiB`, ... as the CLI prints sizes.
#[pyfunction]
fn cli_human_bytes(n: f64) -> String {
    medh5_cli::common::human_bytes(n)
}

/// What the command line prints for a failed lookup whose message is `text`:
/// the message when it is a sentence, the key named otherwise.
#[pyfunction]
fn cli_lookup_message(text: &str) -> String {
    medh5_cli::common::top_level_message(&medh5::Error::Key(text.to_string()))
}

/// A plain-text table as the CLI prints one.
#[pyfunction]
fn cli_table(rows: Vec<Vec<Bound<'_, PyAny>>>, headers: Vec<String>) -> PyResult<String> {
    let rows: Vec<Vec<String>> = rows
        .iter()
        .map(|r| r.iter().map(|v| Ok(v.str()?.to_string())).collect::<PyResult<Vec<_>>>())
        .collect::<PyResult<_>>()?;
    Ok(medh5_cli::common::table(&rows, &headers))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    m.add_function(wrap_pyfunction!(cli_command_tree, m)?)?;
    m.add_function(wrap_pyfunction!(cli_human_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(cli_table, m)?)?;
    m.add_function(wrap_pyfunction!(cli_lookup_message, m)?)?;
    m.add("EXIT_OK", medh5_cli::EXIT_OK)?;
    m.add("EXIT_ERROR", medh5_cli::EXIT_ERROR)?;
    m.add("EXIT_USAGE", medh5_cli::EXIT_USAGE)?;
    Ok(())
}

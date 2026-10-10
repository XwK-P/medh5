//! The `medh5` command line.
//!
//! One native application over the format engine: the same grammar, output and
//! exit codes as the 1.x Python command line (0 success, 1 a handled error, 2
//! a usage error), whether it runs as the standalone `medh5` binary or inside
//! the Python package's console script.  The format converters are Python
//! integrations and run in the Python package (see [`Host`]).

use std::ffi::OsString;
use std::io::Write;
use std::process::{Command, Stdio};

use clap::error::ErrorKind;
use serde_json::{json, Map, Value};

use medh5::json::repr_str;

pub mod app;
mod bench;
mod clinical;
pub mod common;
mod conformance;
mod convert;
mod curation;
mod dataset;
mod inspect;
mod labels;
mod perf;
mod scrub;
mod seg;

use common::{top_level_message, Ctx};
pub use common::{Host, EXIT_ERROR, EXIT_OK, EXIT_USAGE};

/// The host of the standalone binary: hands the converters and the dataloader
/// benchmark to a Python interpreter that has the `medh5` package, when there
/// is one (`MEDH5_PYTHON`, else `python3`, or `python` on Windows).
pub struct NativeHost;

impl NativeHost {
    fn python() -> String {
        std::env::var("MEDH5_PYTHON").unwrap_or_else(|_| if cfg!(windows) { "python".into() } else { "python3".into() })
    }
}

/// What every program the binary hands to Python runs first.  `python -c`
/// and `python -m` put the current directory at the head of `sys.path`, so
/// a `json.py` --- or a `medh5/` --- where `medh5 convert` runs would be
/// imported, and run, in place of the real one; a downloaded study folder is
/// where converters run.  `-P` does this only from Python 3.11, and `-I`
/// would also drop the user's site-packages and `PYTHONPATH`.
const SAFE_PATH: &str = "import sys; sys.path[:1] == [''] and sys.path.pop(0)";

/// `python -c CODE`, the current directory off `sys.path` before anything
/// is imported; arguments added after it are `sys.argv[1:]`.
fn python_running(python: &str, code: &str) -> Command {
    let mut command = Command::new(python);
    command.arg("-c").arg(format!("{SAFE_PATH}; {code}"));
    command
}

impl Host for NativeHost {
    fn convert(
        &self,
        argv: &[String],
        _command: &str,
        _args: &Value,
        _out: &mut dyn Write,
        _err: &mut dyn Write,
    ) -> Option<i32> {
        let python = Self::python();
        let probe =
            python_running(&python, "import medh5.io, medh5.cli").stdout(Stdio::null()).stderr(Stdio::null()).status();
        if !matches!(probe, Ok(s) if s.success()) {
            return None;
        }
        let status = python_running(&python, "from medh5.cli import main; raise SystemExit(main(sys.argv[1:]))")
            .args(argv)
            .status()
            .ok()?;
        Some(status.code().unwrap_or(EXIT_ERROR))
    }

    fn throughput(&self, path: &str, patch: i64, workers: i64, annotation: Option<&str>) -> Result<Value, String> {
        measure_throughput(&Self::python(), path, patch, workers, annotation)
    }
}

/// The dataloader benchmark, run by `python`'s `medh5` package as the console
/// script runs it: `Measurement.to_json()`, or why it could not run there.
fn measure_throughput(
    python: &str,
    path: &str,
    patch: i64,
    workers: i64,
    annotation: Option<&str>,
) -> Result<Value, String> {
    const CODE: &str = "import json; from medh5.bench import throughput; print(json.dumps(throughput([sys.argv[1]], \
                        patch=int(sys.argv[2]), workers=int(sys.argv[3]), annotation=sys.argv[4] or None).to_json()))";
    let needs = format!(
        "the dataloader benchmark runs in the Python package, with PyTorch (pip install 'medh5[torch]'); {python} \
         could not run it (MEDH5_PYTHON names another interpreter)"
    );
    let ran = python_running(python, CODE)
        .args([path, &patch.to_string(), &workers.to_string(), annotation.unwrap_or("")])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("{needs}: {e}"))?;
    let last_line = |bytes: &[u8]| {
        String::from_utf8_lossy(bytes).lines().map(str::trim).filter(|l| !l.is_empty()).last().map(str::to_string)
    };
    if !ran.status.success() {
        // The last line of a traceback is the exception: `No module named 'torch'`.
        return Err(format!("{needs}: {}", last_line(&ran.stderr).unwrap_or_else(|| ran.status.to_string())));
    }
    let printed = last_line(&ran.stdout).unwrap_or_default();
    serde_json::from_str(&printed).map_err(|e| format!("{needs}: it printed no measurement ({e})"))
}

/// Run the command line on `args` (without the program name) against the
/// process's standard streams.
pub fn run(args: Vec<String>) -> i32 {
    let mut out = std::io::stdout().lock();
    let mut err = std::io::stderr().lock();
    run_with(&args, &mut out, &mut err, &NativeHost)
}

/// [`run`] on arguments as the operating system passed them, which need not
/// be UTF-8: one that is not is a usage error, said on stderr.
pub fn run_os(args: impl IntoIterator<Item = OsString>) -> i32 {
    match utf8_args(args) {
        Ok(args) => run(args),
        Err(message) => {
            let _ = writeln!(std::io::stderr(), "medh5: {message}");
            EXIT_USAGE
        }
    }
}

/// The arguments as text, or what to say about the first that is not UTF-8
/// --- a Latin-1 file name a shell glob expanded, say.  Every command takes
/// its paths as text, so such an argument is refused before anything runs
/// rather than ending the program in a panic.
pub fn utf8_args(args: impl IntoIterator<Item = OsString>) -> Result<Vec<String>, String> {
    args.into_iter()
        .map(|arg| {
            arg.into_string().map_err(|arg| format!("argument {} is not valid UTF-8", repr_str(&arg.to_string_lossy())))
        })
        .collect()
}

/// Run the command line on `args` (without the program name); the exit code.
pub fn run_with(args: &[String], out: &mut dyn Write, err: &mut dyn Write, host: &dyn Host) -> i32 {
    medh5::h5::init();
    let argv = std::iter::once("medh5".to_string()).chain(args.iter().cloned());
    let mut command = app::command();
    let matches = match command.try_get_matches_from_mut(argv) {
        Ok(m) => m,
        Err(e) => {
            let code = match e.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                    let _ = write!(out, "{}", e.render());
                    EXIT_OK
                }
                _ => {
                    let _ = write!(err, "{}", e.render());
                    EXIT_USAGE
                }
            };
            let _ = out.flush();
            let _ = err.flush();
            return code;
        }
    };
    let mut ctx = Ctx::new(out, err, host);
    let Some((name, sub)) = matches.subcommand() else {
        let help = command.render_help();
        ctx.print(help.to_string().trim_end());
        ctx.flush();
        return EXIT_USAGE;
    };
    let result = match name {
        "info" | "tree" | "validate" | "verify" | "fix" | "timeline" | "track" => {
            inspect::dispatch(name, sub, &mut ctx)
        }
        "seg" | "index" => seg::dispatch(name, sub, &mut ctx),
        "labels" => labels::dispatch(sub, &mut ctx),
        "pack" | "unpack" | "ls" | "prov" | "agree" | "scrub" | "splits" => curation::dispatch(name, sub, &mut ctx),
        "dataset" => dataset::dispatch(sub, &mut ctx),
        "convert" => convert::convert(args, sub, &mut ctx),
        "migrate" => convert::migrate(sub, &mut ctx),
        "recompress" | "bench" => perf::dispatch(name, sub, &mut ctx),
        "conformance" => conformance::dispatch(sub, &mut ctx),
        "clinical" | "task" | "cache" => clinical::dispatch(name, sub, &mut ctx),
        _ => Ok(EXIT_USAGE),
    };
    let code = match result {
        Ok(code) => code,
        Err(e) => {
            let message = top_level_message(&e);
            ctx.fail(message)
        }
    };
    ctx.flush();
    if ctx.broken_pipe {
        return EXIT_OK;
    }
    code
}

/// The grammar as data: `{"options": [...], "positionals": [...],
/// "commands": {name: {...}}}`, for tools that check documentation against it.
pub fn command_tree() -> Value {
    fn walk(cmd: &clap::Command) -> Value {
        let mut options = Vec::new();
        let mut positionals = Vec::new();
        for arg in cmd.get_arguments() {
            if arg.is_positional() {
                positionals.push(json!(arg.get_id().as_str()));
                continue;
            }
            for short in arg.get_short_and_visible_aliases().unwrap_or_default() {
                options.push(json!(format!("-{short}")));
            }
            for long in arg.get_long_and_visible_aliases().unwrap_or_default() {
                options.push(json!(format!("--{long}")));
            }
        }
        options.push(json!("-h"));
        options.push(json!("--help"));
        let mut commands = Map::new();
        for sub in cmd.get_subcommands() {
            commands.insert(sub.get_name().to_string(), walk(sub));
        }
        json!({"options": options, "positionals": positionals, "commands": commands})
    }
    let mut command = app::command();
    command.build();
    let mut tree = walk(&command);
    if let Value::Object(m) = &mut tree {
        if let Some(Value::Array(options)) = m.get_mut("options") {
            options.push(json!("--version"));
        }
    }
    tree
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn an_argument_that_is_not_utf8_is_a_usage_error_not_a_panic() {
        use std::os::unix::ffi::OsStringExt;

        let latin1 = || OsString::from_vec(b"caf\xe9.medh5".to_vec());
        let refused = utf8_args([OsString::from("info"), latin1()]).unwrap_err();
        assert_eq!(refused, "argument 'caf\u{fffd}.medh5' is not valid UTF-8");
        assert_eq!(run_os([OsString::from("info"), latin1()]), EXIT_USAGE);
        assert_eq!(utf8_args([OsString::from("info"), OsString::from("café.medh5")]).unwrap(), ["info", "café.medh5"]);
    }

    #[test]
    fn python_handed_a_program_imports_nothing_from_the_current_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("json.py"), "raise SystemExit(99)\n").unwrap();
        let python = NativeHost::python();
        let planted = |command: &mut Command| command.current_dir(dir.path()).env_remove("PYTHONSAFEPATH").status();
        // Without the guard the planted module is what runs: the attack is real here.
        let Ok(unguarded) = planted(Command::new(&python).args(["-c", "import json"])) else {
            return; // no Python to hand anything to
        };
        assert_eq!(unguarded.code(), Some(99));
        let guarded = planted(&mut python_running(&python, "import json; json.dumps([])")).unwrap();
        assert_eq!(guarded.code(), Some(0));
    }

    #[test]
    fn a_throughput_run_python_cannot_make_says_what_it_needs() {
        let why = measure_throughput("/nonexistent/python3", "case.medh5", 64, 0, None).unwrap_err();
        assert!(why.contains("pip install 'medh5[torch]'") && why.contains("MEDH5_PYTHON"), "{why}");
    }
}

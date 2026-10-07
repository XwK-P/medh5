//! The `medh5` command line.
//!
//! One native application over the format engine: the same grammar, output and
//! exit codes as the 1.x Python command line (0 success, 1 a handled error, 2
//! a usage error), whether it runs as the standalone `medh5` binary or inside
//! the Python package's console script.  The format converters are Python
//! integrations and run in the Python package (see [`Host`]).

use std::io::Write;

use clap::error::ErrorKind;
use serde_json::{json, Map, Value};

pub mod app;
mod bench;
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

/// How a run ended: its exit code, and whether the 1.x parser would have
/// ended it by raising `SystemExit` (`--help`, `--version`, a usage error)
/// rather than by returning the code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    pub code: i32,
    pub parser_exit: bool,
}

/// The host of the standalone binary: hands converters to a Python
/// interpreter that has the `medh5` package, when there is one.
pub struct NativeHost;

impl NativeHost {
    fn python() -> String {
        std::env::var("MEDH5_PYTHON").unwrap_or_else(|_| if cfg!(windows) { "python".into() } else { "python3".into() })
    }
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
        let probe = std::process::Command::new(&python)
            .args(["-c", "import medh5.io, medh5.cli"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        if !matches!(probe, Ok(s) if s.success()) {
            return None;
        }
        let status = std::process::Command::new(&python).args(["-m", "medh5.cli"]).args(argv).status().ok()?;
        Some(status.code().unwrap_or(EXIT_ERROR))
    }

    fn throughput(&self, _path: &str, _patch: i64, _workers: i64, _annotation: Option<&str>) -> Result<Value, String> {
        Err("the dataloader benchmark needs the Python package with PyTorch (pip install 'medh5[torch]')".into())
    }
}

/// Run the command line on `args` (without the program name) against the
/// process's standard streams.
pub fn run(args: Vec<String>) -> i32 {
    let mut out = std::io::stdout().lock();
    let mut err = std::io::stderr().lock();
    run_with(&args, &mut out, &mut err, &NativeHost).code
}

/// Run the command line on `args` (without the program name).
pub fn run_with(args: &[String], out: &mut dyn Write, err: &mut dyn Write, host: &dyn Host) -> Outcome {
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
            return Outcome { code, parser_exit: true };
        }
    };
    let mut ctx = Ctx::new(out, err, host);
    let Some((name, sub)) = matches.subcommand() else {
        let help = command.render_help();
        ctx.print(help.to_string().trim_end());
        ctx.flush();
        return Outcome { code: EXIT_USAGE, parser_exit: false };
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
        return Outcome { code: EXIT_OK, parser_exit: false };
    }
    Outcome { code, parser_exit: false }
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

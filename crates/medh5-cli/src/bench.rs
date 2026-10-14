//! `bench` --- reproduce the performance targets (§14).

use std::path::Path;

use clap::ArgMatches;
use serde_json::{json, Value};

use medh5::bench::{
    benchmark_file, many_class_measurement, report, synthetic_pair, synthetic_sample, Measurement, MANY_CLASSES,
};

use crate::common::*;

/// The synthetic samples `bench` writes when it is given none.
struct Synthetic {
    /// The sample every metric but the paired and many-class ones is timed on.
    sample: [usize; 3],
    /// Two visits related by a displacement field, for `paired_center_ms`.
    pair: [usize; 3],
    /// [`MANY_CLASSES`] classes, for `foreground_sample_many_ms`.
    many: [usize; 3],
}

/// The shapes the published numbers were measured on.
const PUBLISHED: Synthetic = Synthetic { sample: [192, 256, 256], pair: [64, 96, 96], many: [64, 128, 128] };

pub fn bench(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    bench_with(m, ctx, &PUBLISHED)
}

fn bench_with(m: &ArgMatches, ctx: &mut Ctx, synthetic: &Synthetic) -> CmdResult {
    // At least 1 each: the parser refuses less.
    let patch = get_i64(m, "patch").unwrap_or(64) as usize;
    let repeats = get_i64(m, "repeats").unwrap_or(20) as usize;
    let annotation = get_str(m, "annotation");
    let result = (|| -> CmdResult {
        // Progress goes to stderr: with `--json`, stdout is the document.
        let mut temporary = None;
        let (path, pair, many) = match get_str(m, "path") {
            Some(p) => (p.to_string(), None, None),
            None => {
                let dir = tempfile::Builder::new().prefix("medh5-bench-").tempdir()?;
                let shape = |s: &[usize; 3]| s.map(|n| n.to_string()).join("x");
                ctx.eprint(format!("writing a synthetic {} sample ...", shape(&synthetic.sample)));
                let path =
                    synthetic_sample(dir.path(), &synthetic.sample, 8, "training", true, 20260815, "bench.medh5")?;
                ctx.eprint("writing a synthetic two-visit sample ...");
                let pair = synthetic_pair(dir.path(), &synthetic.pair, "training", 20260815)?;
                ctx.eprint(format!("writing a synthetic {MANY_CLASSES}-class sample ..."));
                let many = synthetic_sample(
                    dir.path(),
                    &synthetic.many,
                    MANY_CLASSES,
                    "training",
                    true,
                    20260815,
                    "bench-many.medh5",
                )?;
                temporary = Some(dir);
                (path.to_string_lossy().into_owned(), Some(pair), Some(many))
            }
        };
        let mut measurements =
            benchmark_file(Path::new(&path), annotation, patch, repeats).map_err(|e| sample_file_error(&path, e))?;
        if let Some(pair) = &pair {
            measurements.extend(
                benchmark_file(pair, None, patch, repeats)?.into_iter().filter(|m| m.name == "paired_center_ms"),
            );
        }
        if let Some(many) = &many {
            measurements.push(many_class_measurement(many, patch, repeats)?);
        }
        if !flag(m, "no_throughput") {
            let workers = get_i64(m, "workers").unwrap_or(0);
            // The run is Python's, which prints to the same streams.
            ctx.flush();
            match ctx.host.throughput(&path, patch as i64, workers, annotation) {
                Ok(doc) => measurements.push(Measurement::from_json(&doc)?),
                Err(reason) => ctx.eprint(format!("skipping throughput: {reason}")),
            }
        }
        let ok = measurements.iter().all(Measurement::ok);
        if flag(m, "json") {
            let payload = json!({
                "path": path,
                "measurements": measurements.iter().map(Measurement::to_json).collect::<Vec<Value>>(),
                "ok": ok,
            });
            ctx.emit(&payload, true);
        } else {
            ctx.print(format!("\n{path}"));
            ctx.print(report(&measurements));
        }
        drop(temporary);
        Ok(if ok { EXIT_OK } else { EXIT_ERROR })
    })();
    match result {
        Err(e) if e.is_medh5() => Ok(ctx.fail(e.to_string())),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The synthetic samples at a fraction of their published size: the
    /// command is the same, and the Python suite ran it at full size.
    const SMALL: Synthetic = Synthetic { sample: [16, 32, 32], pair: [6, 8, 8], many: [8, 16, 16] };

    /// Progress lines on stdout ahead of the JSON made `--json` unparsable;
    /// with no file given, every row is measured --- the paired and the
    /// many-class ones on samples of their own.
    #[test]
    fn s14_bench_json_is_only_the_document() {
        medh5::h5::init();
        let argv = ["medh5", "bench", "--no-throughput", "--json", "--repeats", "1", "--patch", "8"];
        let matches = crate::app::command().try_get_matches_from(argv).unwrap();
        let (_, sub) = matches.subcommand().unwrap();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut ctx = Ctx::new(&mut out, &mut err, &crate::NativeHost);
        let code = bench_with(sub, &mut ctx, &SMALL).unwrap();
        let code = ctx.finish(code);
        let payload: Value = serde_json::from_slice(&out)
            .unwrap_or_else(|e| panic!("stdout is one JSON document: {e}; stderr: {}", String::from_utf8_lossy(&err)));
        let names: Vec<&str> =
            payload["measurements"].as_array().unwrap().iter().map(|m| m["name"].as_str().unwrap()).collect();
        for row in ["paired_center_ms", "foreground_sample_many_ms", "foreground_sample_ms", "meta_read_ms"] {
            assert!(names.contains(&row), "{row} not in {names:?}");
        }
        let err = String::from_utf8(err).unwrap();
        assert!(err.contains("writing a synthetic 16x32x32 sample"), "{err}");
        assert_eq!(code, if payload["ok"] == Value::Bool(true) { EXIT_OK } else { EXIT_ERROR });
    }
}

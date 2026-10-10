//! `bench` --- reproduce the performance targets (§14).

use std::path::Path;

use clap::ArgMatches;
use serde_json::{json, Value};

use medh5::bench::{
    benchmark_file, many_class_measurement, report, synthetic_many_class_sample, synthetic_pair, synthetic_sample,
    Measurement, MANY_CLASSES,
};

use crate::common::*;

pub fn bench(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let patch = get_i64(m, "patch").unwrap_or(64).max(1) as usize;
    let repeats = get_i64(m, "repeats").unwrap_or(20).max(1) as usize;
    let annotation = get_str(m, "annotation");
    let result = (|| -> CmdResult {
        // Progress goes to stderr: with `--json`, stdout is the document.
        let mut temporary = None;
        let (path, pair, many) = match get_str(m, "path") {
            Some(p) => (p.to_string(), None, None),
            None => {
                let dir = tempfile::Builder::new().prefix("medh5-bench-").tempdir()?;
                ctx.eprint("writing a synthetic 192x256x256 sample ...");
                let path =
                    synthetic_sample(dir.path(), &[192, 256, 256], 8, "training", true, 20260815, "bench.medh5")?;
                ctx.eprint("writing a synthetic two-visit sample ...");
                let pair = synthetic_pair(dir.path(), &[64, 96, 96], "training", 20260815)?;
                ctx.eprint(format!("writing a synthetic {MANY_CLASSES}-class sample ..."));
                let many = synthetic_many_class_sample(dir.path(), MANY_CLASSES)?;
                temporary = Some(dir);
                (path.to_string_lossy().into_owned(), Some(pair), Some(many))
            }
        };
        let mut measurements = benchmark_file(Path::new(&path), annotation, patch, repeats)?;
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

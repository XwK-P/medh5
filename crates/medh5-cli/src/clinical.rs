//! `clinical` (format 1.1), `task` and `cache` (the task-and-cache contract).
//!
//! Every answer is the engine's: what a cutoff admits is
//! `medh5::clinical::select`, what a task admits per row is
//! `medh5::companion::preflight`, and whether a cache may be used is
//! `medh5::companion::validate_cache` --- the same calls the Python package
//! makes, so the command line, Rust and Python agree by construction.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::ArgMatches;
use serde_json::{json, Value};

use medh5::clinical::model::{Event, Link, HOUR};
use medh5::clinical::select::SelectionPolicy;
use medh5::clinical::ClinicalRecords;
use medh5::collection::{open_any, AnyFile};
use medh5::companion::cache::{validate_cache_in, CacheCheck, CacheEntry, FeatureCache};
use medh5::companion::task::notes_of;
use medh5::companion::{preflight_each_in, write_preflight_noted, Admitted, Note, RowView, SourceAccess, TaskManifest};
use medh5::h5::file::same_file;
use medh5::json::{pretty, repr_str};
use medh5::sample::Sample;
use medh5::Error;

use crate::common::*;

pub fn dispatch(name: &str, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let result = match (name, m.subcommand()) {
        ("clinical", Some(("show", sub))) => show(sub, ctx),
        ("clinical", Some(("select", sub))) => select_cmd(sub, ctx),
        ("clinical", Some(("export", sub))) => export(sub, ctx),
        ("clinical", Some(("augment", sub))) => augment(sub, ctx),
        ("clinical", Some(("strip", sub))) => strip(sub, ctx),
        ("task", Some(("validate", sub))) => task_validate(sub, ctx),
        ("task", Some(("preflight", sub))) => task_preflight(sub, ctx),
        ("task", Some(("reconcile", sub))) => task_reconcile(sub, ctx),
        ("task", Some(("repin", sub))) => task_repin(sub, ctx),
        ("cache", Some(("validate", sub))) => cache_validate(sub, ctx),
        ("cache", Some(("info", sub))) => cache_info(sub, ctx),
        ("cache", Some(("ls", sub))) => cache_ls(sub, ctx),
        ("clinical", _) => return Ok(ctx.usage("usage: medh5 clinical {show|select|export|augment|strip}")),
        ("task", _) => return Ok(ctx.usage("usage: medh5 task {validate|preflight|reconcile|repin}")),
        _ => return Ok(ctx.usage("usage: medh5 cache {info|ls|validate}")),
    };
    match result {
        Err(e) if e.is_medh5() || matches!(e, Error::Io(_) | Error::Value(_) | Error::Key(_)) => {
            Ok(ctx.fail(top_level_message(&e)))
        }
        other => other,
    }
}

fn open_sample(m: &ArgMatches) -> medh5::Result<Sample> {
    let path = req_str(m, "path");
    match open_any(Path::new(path), get_str(m, "key"))? {
        AnyFile::Sample(s) => Ok(s),
        AnyFile::Collection(c) => {
            let mut keys = c.keys()?;
            keys.sort();
            Err(Error::invalid(format!(
                "{path} is a collection of {} sample(s); name one with --key (keys: {})",
                keys.len(),
                keys.into_iter().take(5).collect::<Vec<_>>().join(", ")
            )))
        }
    }
}

fn clinical_of(sample: &Sample, path: &str) -> medh5::Result<std::sync::Arc<medh5::clinical::Clinical>> {
    sample.clinical()?.cloned().ok_or_else(|| {
        Error::invalid(format!(
            "{path} does not declare the clinical profile (MEDH5 {})",
            sample.version().unwrap_or_default()
        ))
    })
}

/// `"-48 h"`, `"day 90 + 1 h"`: a time on the subject clock, for a table.
fn when(bounds: Option<medh5::clinical::model::Bounds>) -> String {
    let Some(b) = bounds else { return "?".into() };
    let one = |t: i64| -> String {
        let hours = t as f64 / HOUR as f64;
        if t % HOUR == 0 {
            format!("{} h", t / HOUR)
        } else {
            format!("{hours:.2} h")
        }
    };
    if b.lo == b.hi {
        one(b.lo)
    } else {
        format!("[{}, {}]", one(b.lo), one(b.hi))
    }
}

fn event_row(e: &Event) -> Vec<String> {
    let code = match (&e.code_system, &e.code) {
        (Some(s), Some(c)) => format!("{s}|{c}"),
        _ => "-".into(),
    };
    let value = match (e.value_num, &e.value_text) {
        (Some(v), _) => format!("{}{}", g(v), e.unit.as_deref().map(|u| format!(" {u}")).unwrap_or_default()),
        (None, Some(t)) => t.clone(),
        _ => "-".into(),
    };
    vec![
        e.event_id.clone(),
        e.record_id.clone(),
        e.kind.clone(),
        e.status.clone(),
        when(e.effective_start),
        when(e.available),
        code,
        value,
    ]
}

const EVENT_HEADERS: [&str; 8] = ["event", "record", "kind", "status", "effective", "available", "code", "value"];

fn show(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let sample = open_sample(m)?;
    let clinical = clinical_of(&sample, path)?;
    if flag(m, "json") {
        let mut out = clinical.summary();
        out["events_table"] = Value::Array(clinical.events.iter().map(Event::to_json).collect());
        out["links_table"] = Value::Array(clinical.links.iter().map(Link::to_json).collect());
        ctx.emit(&out, true);
        return Ok(EXIT_OK);
    }
    let summary = clinical.summary();
    let clock = &clinical.descriptor.clock;
    ctx.print(format!("{path}  (MEDH5 {}, clinical profile)", sample.version()?));
    ctx.print(format!(
        "  clock       {} ({}{})",
        clock.id,
        clock.reference,
        clock.origin_description.as_deref().map(|o| format!(": {o}")).unwrap_or_default()
    ));
    ctx.print(format!(
        "  records     {} events in {} records, {} documents, {} links; {} with unknown availability",
        summary["events"], summary["records"], summary["documents"], summary["links"], summary["unknown_availability"]
    ));
    ctx.print("\nevents");
    let rows: Vec<Vec<String>> = clinical.events.iter().map(event_row).collect();
    ctx.print(indent(&table(&rows, &EVENT_HEADERS)));
    if !clinical.documents().is_empty() {
        ctx.print("\ndocuments");
        let rows: Vec<Vec<String>> = clinical
            .documents()
            .iter()
            .map(|d| {
                vec![
                    d.document_id.clone(),
                    d.media_type.clone(),
                    d.language.clone().unwrap_or_else(|| "-".into()),
                    d.source_type.clone().unwrap_or_else(|| "-".into()),
                    thousands(d.n_bytes as i128),
                ]
            })
            .collect();
        ctx.print(indent(&table(&rows, &["document", "media type", "language", "source", "bytes"])));
    }
    if !clinical.links.is_empty() {
        ctx.print("\nlinks");
        let rows: Vec<Vec<String>> = clinical.links.iter().map(|l| vec![l.describe()]).collect();
        ctx.print(indent(&table(&rows, &["link"])));
    }
    Ok(EXIT_OK)
}

fn policy_of(m: &ArgMatches) -> medh5::Result<SelectionPolicy> {
    let mut policy = match get_str(m, "policy_file") {
        Some(file) => {
            let text = std::fs::read_to_string(file).map_err(|e| medh5::error::os_error(&e, file))?;
            let doc = medh5::json::loads(&text).map_err(|e| Error::Value(format!("{file} is not JSON: {e}")))?;
            SelectionPolicy::from_json(&doc)?
        }
        None => SelectionPolicy::strict(),
    };
    if let Some(selection) = get_str(m, "policy") {
        policy.selection = selection.to_string();
    }
    if let Some(w) = get_i64(m, "context_us") {
        policy.context_us = Some(w);
    }
    policy.check()?;
    Ok(policy)
}

fn select_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let sample = open_sample(m)?;
    let clinical = clinical_of(&sample, path)?;
    let cutoff = match (get_i64(m, "cutoff_us"), get_f64(m, "cutoff_hours")) {
        (Some(us), None) => us,
        (None, Some(h)) => medh5::clinical::hours(h)?,
        _ => return Ok(ctx.usage("give the cutoff once: --cutoff-us or --cutoff-hours")),
    };
    let policy = policy_of(m)?;
    let selection = clinical.select(cutoff, &policy)?;
    if flag(m, "json") {
        ctx.emit(&selection.to_json(&clinical.events), true);
        return Ok(EXIT_OK);
    }
    ctx.print(format!(
        "{path} at {} ({} us): {} under {}",
        when(Some(medh5::clinical::model::Bounds::exact(cutoff))),
        cutoff,
        selection.status,
        selection.policy
    ));
    let rows: Vec<Vec<String>> = selection.events.iter().map(|s| event_row(&clinical.events[s.index])).collect();
    ctx.print(format!("\nadmitted events ({})", rows.len()));
    ctx.print(indent(&table(&rows, &EVENT_HEADERS)));
    if !selection.payloads.is_empty() {
        let listed: Vec<String> = selection.payloads.iter().map(|(_, k, i)| format!("{k} {}", repr_str(i))).collect();
        ctx.print(format!("\nadmitted payloads: {}", listed.join(", ")));
    }
    if !selection.uncertain_records.is_empty() {
        ctx.print(format!(
            "\nuncertifiable: a later revision of {} has unknown or straddling availability",
            selection.uncertain_records.join(", ")
        ));
    }
    if !selection.excluded.is_empty() {
        let listed: Vec<String> = selection.excluded.iter().map(|(k, v)| format!("{k}={v}")).collect();
        ctx.print(format!("excluded: {}", listed.join(", ")));
    }
    Ok(EXIT_OK)
}

fn export(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    // The write truncates whatever `--out` names: the sample itself, named
    // again or through a link, became the JSON describing it, and exit 0.
    if let Some(out) = get_str(m, "out").filter(|out| same_file(Path::new(path), Path::new(out))) {
        return Err(Error::invalid(format!(
            "--out {} is {}, the file being exported; write the records to another file",
            repr_str(out),
            repr_str(path)
        )));
    }
    let sample = open_sample(m)?;
    let records = clinical_of(&sample, path)?.records()?;
    let text = pretty(&records.to_json()) + "\n";
    match get_str(m, "out") {
        Some(out) => {
            std::fs::write(out, text).map_err(|e| medh5::error::os_error(&e, out))?;
            ctx.print(format!(
                "wrote {} events, {} documents and {} links to {out}",
                records.events.len(),
                records.documents.len(),
                records.links.len()
            ));
        }
        None => ctx.print(text.trim_end()),
    }
    Ok(EXIT_OK)
}

fn augment(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = PathBuf::from(req_str(m, "path"));
    let file = req_str(m, "records");
    let text = std::fs::read_to_string(file).map_err(|e| medh5::error::os_error(&e, file))?;
    let doc = medh5::json::loads(&text).map_err(|e| Error::Value(format!("{file} is not JSON: {e}")))?;
    let records = ClinicalRecords::from_json(&doc)?;
    let out = get_str(m, "out").map(PathBuf::from);
    let report = medh5::clinical::augment::augment(&path, records, out.as_deref())?;
    if flag(m, "json") {
        ctx.emit(&report.to_json(), true);
        return Ok(EXIT_OK);
    }
    ctx.print(format!(
        "{}: MEDH5 {} -> {}; {} events, {} documents, {} links; {} payload digests unchanged",
        report.path,
        report.version_before,
        report.version_after,
        report.events,
        report.documents,
        report.links,
        report.unchanged_digests
    ));
    ctx.print(format!(
        "  content_id {} -> {}",
        report.content_id_before.as_deref().unwrap_or("-"),
        report.content_id_after.as_deref().unwrap_or("-")
    ));
    for line in &report.assumptions {
        ctx.print(format!("  unknown: {line}"));
    }
    Ok(EXIT_OK)
}

fn strip(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = PathBuf::from(req_str(m, "path"));
    let out = PathBuf::from(req_str(m, "out"));
    let report = medh5::clinical::augment::strip(&path, &out)?;
    if flag(m, "json") {
        ctx.emit(&report.to_json(), true);
        return Ok(EXIT_OK);
    }
    ctx.print(format!(
        "wrote the imaging projection to {} (MEDH5 {}): removed {} events, {} documents, {} links",
        medh5::pyval::py_path(&out),
        report.version_after,
        report.events_removed,
        report.documents_removed,
        report.links_removed
    ));
    Ok(EXIT_OK)
}

fn load_task(m: &ArgMatches) -> medh5::Result<(TaskManifest, Option<PathBuf>, Vec<Note>)> {
    let file = Path::new(req_str(m, "manifest"));
    let doc = TaskManifest::read_document(file)?;
    let manifest = TaskManifest::from_json(&doc)?;
    let base = get_str(m, "base").map(PathBuf::from).or_else(|| file.parent().map(Path::to_path_buf));
    Ok((manifest, base, notes_of(&doc)))
}

/// Where a task's sources may be opened from: confined to `--source-root`,
/// unconfined with `--any-source-path`, and otherwise unconfined with a note
/// on stderr naming the sources 3.0 will refuse (task-cache-1 §2).
fn source_access(
    m: &ArgMatches,
    manifest: &TaskManifest,
    base: Option<&Path>,
    ctx: &mut Ctx,
) -> medh5::Result<SourceAccess> {
    if let Some(root) = get_str(m, "source_root") {
        return SourceAccess::confined(base, Some(Path::new(root)));
    }
    if !flag(m, "any_source_path") {
        // A base that does not exist confines nothing to note: its sources
        // fail to open (T301) whatever 3.0 does.
        if let Ok(confined) = SourceAccess::confined(base, None) {
            let outside = manifest.sources_outside(&confined);
            if let Some(first) = outside.first() {
                ctx.eprint(format!(
                    "note: {} source(s) lie outside the manifest's directory ({}{}): medh5 3.0 confines a task's \
                     sources to its directory; pass --any-source-path to keep reading them, or --source-root DIR",
                    outside.len(),
                    repr_str(&first.uri),
                    if outside.len() > 1 { ", ..." } else { "" }
                ));
            }
        }
    }
    Ok(SourceAccess::new(base))
}

fn findings_text(findings: &[medh5::companion::Finding], ctx: &mut Ctx) {
    for f in findings {
        ctx.print(format!("  {}", f.line()));
    }
}

/// What the manifest should state and does not: printed after the findings,
/// and never a failure.
fn notes_text(notes: &[Note], ctx: &mut Ctx) {
    for n in notes {
        ctx.print(format!("  {}", n.line()));
    }
}

fn task_validate(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let file = req_str(m, "manifest");
    let loaded = TaskManifest::read_document(Path::new(file))
        .and_then(|doc| Ok((TaskManifest::from_json(&doc)?, notes_of(&doc))));
    let (manifest, notes) = match loaded {
        Ok(t) => t,
        Err(e) if e.code().is_some() => {
            let finding = medh5::companion::Finding::new(e.code().unwrap_or("T101"), "/", e.message());
            if flag(m, "json") {
                ctx.emit(&json!({"path": file, "ok": false, "findings": [finding.to_json()]}), true);
            } else {
                ctx.print(format!("{file}: FAILED"));
                findings_text(&[finding], ctx);
            }
            return Ok(EXIT_ERROR);
        }
        Err(e) => return Err(e),
    };
    let findings = manifest.validate();
    if flag(m, "json") {
        ctx.emit(
            &json!({
                "path": file,
                "ok": findings.is_empty(),
                "task_fingerprint": manifest.task_fingerprint(),
                "manifest_fingerprint": manifest.manifest_fingerprint(),
                "findings": findings.iter().map(medh5::companion::Finding::to_json).collect::<Vec<_>>(),
                "notes": notes.iter().map(Note::to_json).collect::<Vec<_>>(),
            }),
            true,
        );
    } else {
        ctx.print(format!(
            "{file}: {} ({} {} v{}, {} subjects, {} rows)",
            if findings.is_empty() { "OK" } else { "FAILED" },
            medh5::companion::task::SCHEMA,
            manifest.task_id,
            manifest.task_version,
            manifest.subjects.len(),
            manifest.rows.len()
        ));
        ctx.print(format!("  task fingerprint      {}", manifest.task_fingerprint()));
        ctx.print(format!("  manifest fingerprint  {}", manifest.manifest_fingerprint()));
        findings_text(&findings, ctx);
        notes_text(&notes, ctx);
    }
    Ok(if findings.is_empty() { EXIT_OK } else { EXIT_ERROR })
}

/// Both forms take the subjects one at a time (`preflight_each`): `--json`
/// writes each subject's history as soon as it is merged, and the summary
/// keeps one line per row --- neither holds every subject's records.
fn task_preflight(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let (manifest, base, notes) = load_task(m)?;
    let access = source_access(m, &manifest, base.as_deref(), ctx)?;
    let deep = flag(m, "deep");
    if flag(m, "json") {
        let ok = write_preflight_noted(&manifest, &access, deep, &notes, &mut Stdout(ctx))?;
        ctx.print("");
        return Ok(if ok { EXIT_OK } else { EXIT_ERROR });
    }
    let mut lines: Vec<Option<(String, Vec<String>)>> = vec![None; manifest.rows.len()];
    let done = preflight_each_in(&manifest, &access, deep, &mut |_, rows| {
        for (r, view) in rows {
            lines[r] = Some(row_line(&view));
        }
        Ok(())
    })?;
    let lines: Vec<(String, Vec<String>)> = lines
        .into_iter()
        .zip(&done.unclaimed)
        .map(|(line, blank)| line.or_else(|| blank.as_ref().map(row_line)).expect("every row"))
        .collect();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (status, _) in &lines {
        *counts.entry(status).or_default() += 1;
    }
    let ok = done.findings.is_empty();
    let counts: Vec<String> = counts.iter().map(|(k, v)| format!("{v} {k}")).collect();
    ctx.print(format!(
        "{}: {} ({} rows: {})",
        req_str(m, "manifest"),
        if ok { "OK" } else { "FAILED" },
        lines.len(),
        if counts.is_empty() { "none".into() } else { counts.join(", ") }
    ));
    findings_text(&done.findings, ctx);
    notes_text(&notes, ctx);
    let rows: Vec<Vec<String>> = lines.into_iter().map(|(_, line)| line).collect();
    ctx.print(String::new());
    ctx.print(table(&rows, &["row", "partition", "cutoff_us", "status", "events", "slots", "target", "why"]));
    Ok(if ok { EXIT_OK } else { EXIT_ERROR })
}

/// A row's status and its line of the summary table.
fn row_line(r: &RowView) -> (String, Vec<String>) {
    let slots: Vec<String> =
        r.slots.iter().map(|s| format!("{}={}", s.slot, s.image_id.as_deref().unwrap_or("-"))).collect();
    let line = vec![
        r.row_id.clone(),
        r.partition.clone().unwrap_or_else(|| "-".into()),
        r.cutoff_us.to_string(),
        r.status.clone(),
        r.selection.as_ref().map_or(0, |s| s.events.len()).to_string(),
        if slots.is_empty() { "-".into() } else { slots.join(" ") },
        r.target.status.clone(),
        if r.reasons.is_empty() { "-".into() } else { r.reasons.join("; ") },
    ];
    (r.status.clone(), line)
}

fn task_repin(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let (mut manifest, base, _) = load_task(m)?;
    let access = source_access(m, &manifest, base.as_deref(), ctx)?;
    let only: Option<std::collections::BTreeSet<String>> =
        m.get_many::<String>("source").map(|ids| ids.cloned().collect());
    let out = get_str(m, "out").unwrap_or_else(|| req_str(m, "manifest"));
    let dry_run = flag(m, "dry_run");
    if !dry_run {
        manifest.check_output(Path::new(out), &access).map_err(|e| Error::invalid(format!("--out {}", e.message())))?;
    }
    let report = manifest.repin(&access, only.as_ref())?;
    if !dry_run && !report.changed.is_empty() {
        manifest.save(Path::new(out))?;
    }
    if flag(m, "json") {
        let mut doc = report.to_json();
        doc["dry_run"] = json!(dry_run);
        ctx.emit(&doc, true);
        return Ok(EXIT_OK);
    }
    let verb = if dry_run { "would repin" } else { "repinned" };
    for r in &report.changed {
        ctx.print(format!("{verb} {}: {} -> {}", r.source_id, r.before, r.after));
    }
    ctx.print(format!(
        "{}: {} {}, {} unchanged{}",
        req_str(m, "manifest"),
        verb,
        report.changed.len(),
        report.unchanged.len(),
        if report.rows.is_empty() {
            String::new()
        } else {
            format!("; {} row fingerprint(s) change, so their patient-level cache entries go stale", report.rows.len())
        }
    ));
    Ok(EXIT_OK)
}

fn task_reconcile(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let (mut manifest, base, _) = load_task(m)?;
    let access = source_access(m, &manifest, base.as_deref(), ctx)?;
    // The save replaces whatever `--out` names: a source sample, named again
    // or through a link, became the manifest that pins it, and exit 0.
    let out = get_str(m, "out").unwrap_or_else(|| req_str(m, "manifest"));
    manifest.check_output(Path::new(out), &access).map_err(|e| Error::invalid(format!("--out {}", e.message())))?;
    let recorded = manifest.reconcile_all(&access)?;
    manifest.save(Path::new(out))?;
    ctx.print(format!("recorded {recorded} reconciled event(s); wrote {out}"));
    Ok(EXIT_OK)
}

/// What a cache holds, from its manifest: nothing is decoded.
fn cache_info(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let cache = FeatureCache::open(Path::new(req_str(m, "path")))?;
    let h = &cache.header;
    let sources: std::collections::BTreeSet<(&str, Option<&str>)> = cache
        .entries
        .iter()
        .flat_map(|e| &e.sources)
        .map(|s| (s.content_id.as_str(), s.sample_key.as_deref()))
        .collect();
    let info = json!({
        "path": cache.path.to_string_lossy(),
        "schema": medh5::companion::cache::SCHEMA,
        "level": h.level,
        "encoder": h.encoder,
        "preprocessing": h.preprocessing,
        "output": h.output,
        "task_fingerprint": h.task_fingerprint,
        "selection": h.selection,
        "fitted_on": h.fitted_on,
        "manifest_digest": cache.manifest_digest,
        "entries": cache.entries.len(),
        "sources": sources.len(),
    });
    if flag(m, "json") {
        ctx.emit(&info, true);
        return Ok(EXIT_OK);
    }
    let one = |v: &Value| match v {
        Value::Null => "-".to_string(),
        Value::String(s) => s.clone(),
        other => medh5::json::canonical(other),
    };
    ctx.print(format!(
        "{}: {} {}-level, {} entries over {} source versions",
        req_str(m, "path"),
        medh5::companion::cache::SCHEMA,
        h.level,
        cache.entries.len(),
        sources.len()
    ));
    for key in ["encoder", "preprocessing", "output", "task_fingerprint", "selection", "fitted_on", "manifest_digest"] {
        ctx.print(format!("  {key:<16}  {}", one(&info[key])));
    }
    Ok(EXIT_OK)
}

/// A cache's entries, as its manifest records them, optionally only those
/// reading one source version or of one row.
fn cache_ls(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let cache = FeatureCache::open(Path::new(req_str(m, "path")))?;
    let source = get_str(m, "source");
    let row = get_str(m, "row");
    let shown: Vec<&CacheEntry> = cache
        .entries
        .iter()
        .filter(|e| source.is_none_or(|c| e.sources.iter().any(|s| s.content_id == c)))
        .filter(|e| row.is_none_or(|r| e.row_id.as_deref() == Some(r)))
        .collect();
    if flag(m, "json") {
        ctx.emit(&Value::Array(shown.iter().map(|e| e.to_json()).collect()), true);
        return Ok(EXIT_OK);
    }
    let rows: Vec<Vec<String>> = shown
        .iter()
        .map(|e| {
            let what = match (&e.event_id, &e.row_id) {
                (Some(event), _) => format!("{event} {}", e.document_id.as_deref().unwrap_or("-")),
                (None, Some(row)) => format!("{row} @{}", e.cutoff_us.map_or("-".into(), |c| c.to_string())),
                (None, None) => "-".into(),
            };
            let pins: Vec<String> = e.sources.iter().map(|s| format!("{} {}", s.locator(), s.content_id)).collect();
            vec![e.entry_id.clone(), pins.join("; "), what, e.digest.clone()]
        })
        .collect();
    let what = if cache.header.level == "event" { "event document" } else { "row @cutoff" };
    ctx.print(table(&rows, &["entry", "sources", what, "digest"]));
    Ok(EXIT_OK)
}

fn cache_validate(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = Path::new(req_str(m, "path"));
    let mut base = get_str(m, "base").map(PathBuf::from);
    let task = match get_str(m, "task") {
        Some(file) => {
            let file = Path::new(file);
            let manifest = TaskManifest::load(file)?;
            let task_base = get_str(m, "task_base").map(PathBuf::from).or_else(|| file.parent().map(Path::to_path_buf));
            let access = source_access(m, &manifest, task_base.as_deref(), ctx)?;
            let admitted = Admitted::preflight_in(&manifest, &access, false)?;
            // An entry source the task pins is opened where the task says
            // (task-cache-1 §7.1); any other resolves by its own URI, which
            // copied the task's, relative to its base.
            base = base.or(task_base);
            Some((manifest, access, admitted))
        }
        None => None,
    };
    let base = base.or_else(|| path.parent().map(Path::to_path_buf));
    let entries = match get_str(m, "source_root") {
        Some(root) => SourceAccess::confined(base.as_deref(), Some(Path::new(root)))?,
        None => SourceAccess::new(base.as_deref()),
    };
    let mut check = CacheCheck::new(entries);
    if let Some((manifest, access, admitted)) = &task {
        check = check.with_task(manifest, access).with_admitted(admitted);
    }
    let report = validate_cache_in(path, &check)?;
    if flag(m, "json") {
        ctx.emit(&report.to_json(), true);
    } else {
        ctx.print(format!(
            "{}: {} ({} entries{}{})",
            report.path,
            if report.ok() { "OK" } else { "FAILED" },
            report.entries,
            if report.stale().is_empty() { String::new() } else { format!(", {} stale", report.stale().len()) },
            if report.corrupt().is_empty() { String::new() } else { format!(", {} corrupt", report.corrupt().len()) }
        ));
        findings_text(&report.findings, ctx);
    }
    Ok(if report.ok() { EXIT_OK } else { EXIT_ERROR })
}

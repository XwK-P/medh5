//! `labels show`, `labels registry list`, `labels check`.

use clap::ArgMatches;
use indexmap::IndexMap;
use serde_json::{json, Value};

use medh5::labels::{registry, VocabularyCheck};

use crate::common::*;

pub fn dispatch(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    match m.subcommand() {
        Some(("show", sub)) => show(sub, ctx),
        Some(("check", sub)) => check(sub, ctx),
        Some(("registry", sub)) => registry_cmd(sub, ctx),
        _ => Ok(ctx.usage("usage: medh5 labels {show|check|registry}")),
    }
}

fn show(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let label_set = match sample_file(path).and_then(|s| Ok(s.label_set()?.cloned())) {
        Ok(ls) => ls,
        Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
        Err(e) => return Err(e),
    };
    if flag(m, "json") {
        // No label set is `null`: `--json` output is always a document.
        ctx.emit(&label_set.map_or(Value::Null, |ls| ls.to_json(None)), true);
        return Ok(EXIT_OK);
    }
    let Some(label_set) = label_set else {
        ctx.print(format!("{path}: no label set"));
        return Ok(EXIT_OK);
    };
    let digest = label_set.sha256();
    ctx.print(format!(
        "{} v{}  form={}  sha256={}...  ({} classes)",
        label_set.id,
        label_set.version,
        label_set.form,
        &digest[..16.min(digest.len())],
        label_set.len()
    ));
    let rows: Vec<Vec<String>> = label_set
        .classes()
        .iter()
        .map(|c| {
            let parents = c.parents.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(",");
            let codes =
                c.codes.iter().map(|code| format!("{}:{}", code.system, code.code)).collect::<Vec<_>>().join(",");
            vec![
                c.id.to_string(),
                c.key.clone(),
                c.name.clone(),
                c.category.clone().filter(|v| !v.is_empty()).unwrap_or_else(|| "-".into()),
                if parents.is_empty() { "-".into() } else { parents },
                if codes.is_empty() { "-".into() } else { codes },
            ]
        })
        .collect();
    ctx.print(table(&rows, &["id", "key", "name", "category", "parents", "codes"]));
    Ok(EXIT_OK)
}

fn check(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let paths = get_paths(m, "paths");
    // Grouped as `dataset check`'s C101 groups (D25); only the failures are
    // worded as the command line words them.
    let mut found = VocabularyCheck::default();
    let mut failed = IndexMap::new();
    for path in &paths {
        let sample = sample_file(path);
        match sample.as_ref().map_err(Clone::clone).and_then(|s| s.label_set()) {
            Ok(label_set) => found.add(path.clone(), Ok(label_set)),
            Err(e) => {
                failed.insert(path.clone(), ctx.file_failed(path, &e));
                found.add(path.clone(), Err(e));
            }
        }
    }
    let ok = found.ok();
    if flag(m, "json") {
        let mut payload = found.to_json();
        // `files` gives each file an entry, a failure the one every batch
        // command gives it; `unreadable` keeps the same failures by path.
        payload["unreadable"] = failed.iter().map(|(path, f)| (path.clone(), f["error"].clone())).collect();
        payload["files"] = paths
            .iter()
            .map(|path| match failed.get(path) {
                Some(f) => f.clone(),
                None => json!({"path": path, "ok": true, "vocabulary": found.key_of(path)}),
            })
            .collect();
        ctx.emit(&payload, true);
    } else {
        let rows: Vec<Vec<String>> = paths
            .iter()
            .map(|path| vec![path.clone(), found.key_of(path).unwrap_or("(unreadable)").to_string()])
            .collect();
        ctx.print(table(&rows, &["file", "vocabulary"]));
        if found.groups.len() > 1 {
            ctx.print(format!(
                "\n{} distinct vocabularies across {} files; class ids are not comparable between them.",
                found.groups.len(),
                paths.len() - found.unreadable.len()
            ));
        }
    }
    Ok(if ok { EXIT_OK } else { EXIT_ERROR })
}

fn registry_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let Some(("list", sub)) = m.subcommand() else {
        return Ok(ctx.usage("usage: medh5 labels registry list"));
    };
    let described = registry::describe()?;
    if flag(sub, "json") {
        ctx.emit(&Value::Object(described), true);
        return Ok(EXIT_OK);
    }
    let rows: Vec<Vec<String>> = described
        .iter()
        .map(|(name, info)| {
            let sha = info["sha256"].as_str().unwrap_or_default();
            vec![
                name.clone(),
                cell(&info["version"]),
                cell(&info["classes"]),
                format!("{}...", &sha[..16.min(sha.len())]),
            ]
        })
        .collect();
    ctx.print(table(&rows, &["name", "version", "classes", "sha256"]));
    Ok(EXIT_OK)
}

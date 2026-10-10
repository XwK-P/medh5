//! `labels show`, `labels registry list`, `labels check`.

use std::path::Path;

use clap::ArgMatches;
use indexmap::IndexMap;
use serde_json::{json, Value};

use medh5::labels::registry;
use medh5::sample::open_sample;

use crate::common::*;

pub fn dispatch(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    match m.subcommand() {
        Some(("show", sub)) => show(sub, ctx),
        Some(("check", sub)) => check(sub, ctx),
        Some(("registry", sub)) => registry_cmd(sub, ctx),
        _ => Ok(ctx.fail("usage: medh5 labels {show|check|registry}")),
    }
}

fn show(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let label_set = match open_sample(Path::new(path)).and_then(|s| Ok(s.label_set()?.cloned())) {
        Ok(ls) => ls,
        Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
        Err(e) => return Err(e),
    };
    let Some(label_set) = label_set else {
        ctx.print(format!("{path}: no label set"));
        return Ok(EXIT_OK);
    };
    if flag(m, "json") {
        ctx.emit(&label_set.to_json(None), true);
        return Ok(EXIT_OK);
    }
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
    let mut seen: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut rows = Vec::new();
    for path in &paths {
        let label_set = match open_sample(Path::new(path)).and_then(|s| Ok(s.label_set()?.cloned())) {
            Ok(ls) => ls,
            Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
            Err(e) => return Err(e),
        };
        let key = match &label_set {
            Some(ls) if !ls.is_empty() || ls.declared_sha256().is_some() => {
                let digest = ls.sha256();
                format!("{}@{}#{}", ls.id, ls.version, &digest[..16.min(digest.len())])
            }
            _ => "<none>".to_string(),
        };
        seen.entry(key.clone()).or_default().push(path.clone());
        rows.push(vec![path.clone(), key]);
    }
    if flag(m, "json") {
        let vocabularies: serde_json::Map<String, Value> = seen.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
        ctx.emit(&json!({"vocabularies": vocabularies}), true);
    } else {
        ctx.print(table(&rows, &["file", "vocabulary"]));
        if seen.len() > 1 {
            ctx.print(format!(
                "\n{} distinct vocabularies across {} files; class ids are not comparable between them.",
                seen.len(),
                paths.len()
            ));
        }
    }
    Ok(if seen.len() <= 1 { EXIT_OK } else { EXIT_ERROR })
}

fn registry_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let Some(("list", sub)) = m.subcommand() else {
        return Ok(ctx.fail("usage: medh5 labels registry list"));
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

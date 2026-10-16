//! The NIfTI parity corpus (`tests/data/nifti-parity`), run through the
//! engine's converter: every `read_nifti` and `from_nifti` case must give
//! what medh5 2.1.0 gave (2.2.0 decision E3) --- the same geometry and
//! voxels, the same objects with the same digests, the same `/meta` but for
//! the version its software agent records, the same report but for the
//! paths it names, and the same refusals with the same words.
//!
//! The records are Python's: a refusal is named by the exception 2.1.0
//! raised, which the Python layer raises from the engine's error kind.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use medh5::convert::nifti::{from_nifti, read_nifti, ImportOptions, ReadOptions};
use medh5::convert::ConversionReport;
use medh5::h5::attrs;
use medh5::h5::ops::{visit, Node};
use medh5::labels::ClassKey;
use medh5::sample::writer::Annotated;
use medh5::Error;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

fn corpus() -> PathBuf {
    plain_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/nifti-parity"))
}

/// `path` made absolute and resolved, as Python's `Path.resolve()` spells
/// it: on Windows, with no `\\?\` prefix.
fn plain_path(path: PathBuf) -> PathBuf {
    let resolved = path.canonicalize().unwrap();
    let text = resolved.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) => PathBuf::from(rest),
        None => resolved,
    }
}

/// What `record.py` writes for a refusal.
fn refusal(err: &Error) -> Value {
    let (kind, message) = match err {
        Error::Validation { message, code } => (
            "MEDH5ValidationError",
            match code {
                Some(code) => format!("[{code}] {message}"),
                None => message.clone(),
            },
        ),
        Error::Io(m) if m.starts_with("[Errno 2]") => ("FileNotFoundError", m.clone()),
        Error::Io(m) => ("OSError", m.clone()),
        Error::Value(m) => ("ValueError", m.clone()),
        Error::Type(m) => ("TypeError", m.clone()),
        Error::Key(m) => ("KeyError", medh5::json::repr_str(m)),
        other => ("MEDH5Error", other.message().to_string()),
    };
    json!({"type": kind, "code": err.code(), "message": message})
}

fn options_of(case: &Value) -> &Map<String, Value> {
    case["options"].as_object().unwrap()
}

fn read_options(options: &Map<String, Value>) -> ReadOptions {
    let mut out = ReadOptions::default();
    for (key, value) in options {
        out = match key.as_str() {
            "coord_system" => out.with_coord_system(value.as_str().unwrap()),
            "transpose" => out.with_transpose(value.as_bool().unwrap()),
            "fourth_axis" => out.with_fourth_axis(value.as_str().unwrap()),
            "assume_geometry" => out.with_assume_geometry(value.as_bool().unwrap()),
            "plane_as_volume" => out.with_plane_as_volume(value.as_bool().unwrap()),
            other => panic!("unknown read option {other}"),
        };
    }
    out
}

fn pairs(value: &Value) -> Vec<(String, String)> {
    value.as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect()
}

fn import_options(case: &Value, root: &Path) -> ImportOptions {
    let named = |key: &str| -> Vec<(String, PathBuf)> {
        case[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_str().unwrap().into(), root.join(p[1].as_str().unwrap())))
            .collect()
    };
    let mut out = ImportOptions::default().with_masks(named("masks"));
    for (key, value) in options_of(case) {
        out = match key.as_str() {
            "coord_system" => out.with_coord_system(value.as_str().unwrap()),
            "transpose" => out.with_transpose(value.as_bool().unwrap()),
            "fourth_axis" => out.with_fourth_axis(value.as_str().unwrap()),
            "assume_geometry" => out.with_assume_geometry(value.as_bool().unwrap()),
            "plane_as_volume" => out.with_plane_as_volume(value.as_bool().unwrap()),
            "sample_id" => out.with_sample_id(value.as_str().unwrap()),
            "subject_id" => out.with_subject_id(value.as_str().unwrap()),
            "codec" => out.with_codec(value.as_str().unwrap()),
            "modalities" => out.with_modalities(pairs(value)),
            "value_units" => out.with_value_units(pairs(value)),
            "label_set" => {
                out.with_label_set(medh5::labels::registry::load_file(&root.join(value.as_str().unwrap())).unwrap())
            }
            "annotated_classes" => out.with_annotated_classes(match value {
                Value::String(s) if s == "all" => Annotated::All,
                Value::String(s) if s == "all_given" => Annotated::AllGiven,
                Value::Array(items) => Annotated::Classes(
                    items
                        .iter()
                        .map(|i| match i {
                            Value::String(s) => ClassKey::Key(s.clone()),
                            other => ClassKey::Id(other.as_i64().unwrap()),
                        })
                        .collect(),
                ),
                other => panic!("annotated_classes {other}"),
            }),
            other => panic!("unknown import option {other}"),
        };
    }
    out
}

/// An attribute as `record.py` writes it: a non-finite float as its `repr`.
fn attr_json(value: &attrs::AttrValue) -> Value {
    match value {
        attrs::AttrValue::Float(f) if !f.is_finite() => Value::String(medh5::json::py_float(*f)),
        other => other.to_json(),
    }
}

fn attributes(location: &medh5::hdf5::Location) -> Value {
    let mut names = attrs::names(location).unwrap();
    names.sort();
    let mut out = Map::new();
    for name in names {
        let value = attrs::read(location, &name).unwrap().unwrap();
        out.insert(name, attr_json(&value));
    }
    Value::Object(out)
}

/// What an import wrote, as `record.py`'s `sample_record` saw it.
fn sample_record(path: &Path) -> Value {
    let file = medh5::hdf5::File::open(path).unwrap();
    let mut objects = Map::new();
    let mut meta = Value::Null;
    visit(&file, &mut |name, node| {
        let entry = match node {
            Node::Group(group) => json!({"type": "group", "attrs": attributes(group)}),
            Node::Dataset(ds) => {
                let td = ds.dtype().unwrap().to_descriptor().unwrap();
                let dtype = attrs::numeric_dtype(&td).map_or("|O", |d| d.numpy_str());
                if name == "meta" {
                    let text = medh5::h5::data::read_strings(ds).unwrap().remove(0);
                    meta = serde_json::from_str(&text).unwrap();
                }
                json!({"type": "dataset", "attrs": attributes(ds), "dtype": dtype, "shape": ds.shape()})
            }
        };
        objects.insert(name.to_string(), entry);
        Ok(true)
    })
    .unwrap();
    let mut root = attributes(&file);
    root.as_object_mut().unwrap().remove("created");
    json!({"root": root, "meta": meta, "objects": objects})
}

/// The voxels as `record.py`'s `array_record` digests them.
fn array_record(read: &medh5::convert::nifti::NiftiRead) -> Value {
    let voxels = &read.voxels;
    let little: Vec<u8> = if voxels.big_endian && voxels.element.has_byte_order() {
        let unit = voxels.element.swap_unit();
        voxels.bytes.chunks(unit).flat_map(|c| c.iter().rev().copied().collect::<Vec<_>>()).collect()
    } else {
        voxels.bytes.clone()
    };
    json!({
        "dtype": voxels.numpy_descr(),
        "shape": voxels.shape,
        "sha256": hex::encode(Sha256::digest(&little)),
    })
}

/// One case, as `record.py`'s `outcome` records it.
fn outcome(case: &Value, root: &Path, out: &Path) -> Value {
    let mut record = Map::new();
    let result: Result<(), Error> = match case["call"].as_str().unwrap() {
        "read_nifti" => {
            read_nifti(&root.join(case["path"].as_str().unwrap()), &read_options(options_of(case))).map(|read| {
                record
                    .insert("read".into(), json!({"geometry": read.geometry.to_json(), "array": array_record(&read)}));
            })
        }
        "from_nifti" => {
            let images: Vec<(String, PathBuf)> = case["images"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (p[0].as_str().unwrap().into(), root.join(p[1].as_str().unwrap())))
                .collect();
            let target = out.join(case["out"].as_str().unwrap());
            let mut report = ConversionReport::new("from-nifti", "");
            let done = from_nifti(&images, &target, &import_options(case, root), &mut report);
            done.map(|()| {
                record.insert("report".into(), report.to_json());
                record.insert("sample".into(), sample_record(&target));
            })
        }
        other => panic!("not an engine call yet: {other}"),
    };
    match result {
        Ok(()) => {
            record.insert("outcome".into(), json!("ok"));
        }
        Err(err) => {
            record.insert("outcome".into(), json!("refused"));
            record.insert("error".into(), refusal(&err));
        }
    }
    let mut files: Vec<String> = Vec::new();
    for entry in walk(out) {
        files.push(entry.strip_prefix(out).unwrap().to_string_lossy().replace('\\', "/"));
    }
    files.sort();
    record.insert("files".into(), json!(files));
    relative(Value::Object(record), root, out)
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(walk(&path));
        }
        out.push(path);
    }
    out
}

/// Paths relative to the corpus and the case's directory, however a
/// message spells them: as given, with `/`, or in a `repr`.
fn relative(value: Value, root: &Path, out: &Path) -> Value {
    match value {
        Value::Object(map) => Value::Object(map.into_iter().map(|(k, v)| (k, relative(v, root, out))).collect()),
        Value::Array(items) => Value::Array(items.into_iter().map(|v| relative(v, root, out)).collect()),
        Value::String(mut s) => {
            for (base, token) in [(out, "{out}"), (root, "{corpus}")] {
                let native = base.to_string_lossy();
                for spelling in [native.replace('\\', "\\\\"), native.to_string(), native.replace('\\', "/")] {
                    s = s.replace(&spelling, token);
                }
            }
            Value::String(s)
        }
        other => other,
    }
}

/// Within a `{corpus}` or `{out}` path, Windows separators as `/` (a path in
/// a `repr` has each one doubled).
fn slashes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = ["{corpus}", "{out}"].iter().filter_map(|t| rest.find(t)).min() {
        out.push_str(&rest[..at]);
        let tail = &rest[at..];
        let end = tail.find(|c: char| c == '\'' || c == '"' || c.is_whitespace() || c == ',').unwrap_or(tail.len());
        out.push_str(&tail[..end].replace("\\\\", "/").replace('\\', "/"));
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

/// A record without what parity does not compare (E3).
fn masked(record: &Value) -> Value {
    let mut out = record.as_object().unwrap().clone();
    for key in ["id", "generator", "nibabel", "numpy"] {
        out.remove(key);
    }
    if let Some(Value::Object(report)) = out.get_mut("report") {
        report.remove("source");
        report.remove("outputs");
    }
    if let Some(Value::Object(sample)) = out.get_mut("sample") {
        if let Some(Value::Object(root)) = sample.get_mut("root") {
            root.remove("generator");
            root.remove("content_id");
        }
        if let Some(agents) = sample.get_mut("meta").and_then(|m| m.pointer_mut("/provenance/agents")) {
            for agent in agents.as_array_mut().unwrap() {
                if agent["type"] == "software" && agent["name"] == "medh5" {
                    agent["version"] = json!("*");
                }
            }
        }
    }
    with_slashes(Value::Object(out))
}

/// Every string in `value` through [`slashes`].
fn with_slashes(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(map.into_iter().map(|(k, v)| (k, with_slashes(v))).collect()),
        Value::Array(items) => Value::Array(items.into_iter().map(with_slashes).collect()),
        Value::String(s) => Value::String(slashes(&s)),
        other => other,
    }
}

/// The differences E3 allows beyond its two masks, by case: a Python
/// exception the engine's error kind cannot name.
fn allowed(expected: &mut Value, got: &mut Value) {
    if expected["error"]["type"] == "HeaderDataError" && got["error"]["type"] == "ValueError" {
        expected["error"]["type"] = json!("ValueError");
    }
}

/// Floats compared as values, so `2` and `2.0` agree as Python's `==` has it.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        _ => a == b,
    }
}

#[test]
fn e3_the_engine_matches_2_1_0() {
    let root = corpus();
    let table: Value = serde_json::from_str(&std::fs::read_to_string(root.join("cases.json")).unwrap()).unwrap();
    let mut failures: BTreeMap<String, String> = BTreeMap::new();
    let mut ran = 0;
    for case in table["cases"].as_array().unwrap() {
        let call = case["call"].as_str().unwrap();
        if call != "read_nifti" && call != "from_nifti" {
            continue;
        }
        ran += 1;
        let id = case["id"].as_str().unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let out = plain_path(scratch.path().to_path_buf()).join("out");
        std::fs::create_dir(&out).unwrap();
        let mut got = masked(&outcome(case, &root, &out));
        let text = std::fs::read_to_string(root.join("records").join(format!("{id}.json"))).unwrap();
        let mut expected = masked(&serde_json::from_str(&text).unwrap());
        allowed(&mut expected, &mut got);
        if !same(&expected, &got) {
            failures.insert(id.to_string(), first_difference(&expected, &got, ""));
        }
    }
    assert_eq!(ran, 270, "every read_nifti and from_nifti case ran");
    assert!(failures.is_empty(), "{} of {ran} cases differ:\n{failures:#?}", failures.len());
}

/// Where two records first differ, for the failure message.
fn first_difference(a: &Value, b: &Value, at: &str) -> String {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in x {
                match y.get(k) {
                    Some(w) if !same(v, w) => return first_difference(v, w, &format!("{at}/{k}")),
                    None => return format!("{at}/{k}: missing"),
                    _ => {}
                }
            }
            for k in y.keys() {
                if !x.contains_key(k) {
                    return format!("{at}/{k}: unexpected {}", y[k]);
                }
            }
            format!("{at}: ?")
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (v, w)) in x.iter().zip(y).enumerate() {
                if !same(v, w) {
                    return first_difference(v, w, &format!("{at}/{i}"));
                }
            }
            format!("{at}: ?")
        }
        _ => format!("{at}: expected {a}, got {b}"),
    }
}

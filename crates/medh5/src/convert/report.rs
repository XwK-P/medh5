//! What a conversion decided, and where it had to guess.
//!
//! The report is a first-class output, not logging: `medh5 convert` and
//! `medh5 migrate` write it as JSON beside the files, so a curator can review
//! a cohort's conversions without re-running them.

use serde_json::{json, Map, Value};

/// Note severities: `decision` was determined by the data; `guess` was not.
pub const SEVERITIES: [&str; 4] = ["info", "decision", "guess", "warning"];

/// One thing a conversion did that the source did not fully determine.
#[derive(Debug, Clone, PartialEq)]
pub struct Note {
    pub kind: String,
    pub message: String,
    pub severity: String,
    pub detail: Map<String, Value>,
}

impl Note {
    pub fn to_json(&self) -> Value {
        json!({"kind": self.kind, "message": self.message, "severity": self.severity, "detail": self.detail})
    }

    /// `SEVERITY kind: message`, severity padded to eight columns.
    pub fn line(&self) -> String {
        format!("{:8} {}: {}", self.severity.to_uppercase(), self.kind, self.message)
    }
}

/// Every note from one conversion, plus what it produced.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConversionReport {
    pub source: String,
    pub converter: String,
    pub outputs: Vec<String>,
    pub notes: Vec<Note>,
}

fn detail_map(detail: Value) -> Map<String, Value> {
    match detail {
        Value::Object(m) => m,
        Value::Null => Map::new(),
        other => {
            let mut m = Map::new();
            m.insert("value".into(), other);
            m
        }
    }
}

impl ConversionReport {
    pub fn new(converter: &str, source: &str) -> ConversionReport {
        ConversionReport { converter: converter.into(), source: source.into(), ..Default::default() }
    }

    /// Record a note.
    pub fn add(&mut self, kind: &str, message: impl Into<String>, severity: &str, detail: Value) -> &Note {
        self.notes.push(Note { kind: kind.into(), message: message.into(), severity: severity.into(), detail: detail_map(detail) });
        self.notes.last().expect("just pushed")
    }

    /// Something the data determined --- auditable, but not a guess.
    pub fn decision(&mut self, kind: &str, message: impl Into<String>, detail: Value) -> &Note {
        self.add(kind, message, "decision", detail)
    }

    /// Something the source did not say and the converter had to assume.
    pub fn guess(&mut self, kind: &str, message: impl Into<String>, detail: Value) -> &Note {
        self.add(kind, message, "guess", detail)
    }

    pub fn warn(&mut self, kind: &str, message: impl Into<String>, detail: Value) -> &Note {
        self.add(kind, message, "warning", detail)
    }

    pub fn guesses(&self) -> Vec<&Note> {
        self.notes.iter().filter(|n| n.severity == "guess").collect()
    }

    pub fn warnings(&self) -> Vec<&Note> {
        self.notes.iter().filter(|n| n.severity == "warning").collect()
    }

    /// Whether the conversion needed no warning.  Guesses are not failures.
    pub fn ok(&self) -> bool {
        self.warnings().is_empty()
    }

    pub fn of_kind(&self, kind: &str) -> Vec<&Note> {
        self.notes.iter().filter(|n| n.kind == kind).collect()
    }

    pub fn to_json(&self) -> Value {
        let counts: Map<String, Value> = SEVERITIES
            .iter()
            .map(|s| (s.to_string(), json!(self.notes.iter().filter(|n| n.severity == *s).count())))
            .collect();
        json!({
            "source": self.source,
            "converter": self.converter,
            "outputs": self.outputs,
            "ok": self.ok(),
            "counts": counts,
            "notes": self.notes.iter().map(Note::to_json).collect::<Vec<_>>(),
        })
    }

    pub fn format(&self, verbose: bool) -> String {
        let mut lines = vec![format!(
            "{}: {} output(s), {} guess(es), {} warning(s)",
            self.converter,
            self.outputs.len(),
            self.guesses().len(),
            self.warnings().len()
        )];
        for note in &self.notes {
            if verbose || note.severity == "guess" || note.severity == "warning" {
                lines.push(format!("  {}", note.line()));
            }
        }
        lines.join("\n")
    }
}

/// One report over a whole cohort.
pub fn merge_reports(reports: &[ConversionReport], converter: &str) -> ConversionReport {
    let mut out = ConversionReport::new(if converter.is_empty() { "batch" } else { converter }, "<many>");
    for r in reports {
        out.outputs.extend(r.outputs.iter().cloned());
        out.notes.extend(r.notes.iter().cloned());
    }
    out
}

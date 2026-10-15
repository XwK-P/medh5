//! Subject-safe splits of a cohort (§12.2, §12.3).
//!
//! Three rules.  **Groups, never files**: whatever `group_by` names is dealt
//! whole, and a grouping that splits a subject is refused (C204).
//! **Deterministic**: the order is a hash of `(set_id, seed, group)`, so it
//! depends on nothing but the inputs.  **Ratios by deficit**: groups are dealt
//! to whichever partition is furthest below its share, which gives the closest
//! integer allocation the ratios admit, and says so when that is still zero.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::curation::identity::{SplitClaim, PARTITIONS};
use crate::dataset::manifest::{Entry, Manifest};
use crate::h5::{attrs, ops};
use crate::json::{num, repr_list, repr_str};
use crate::sample::open_sample;
use crate::sample::writer::{amend, refuse_failed_digests};
use crate::{Error, Result};

/// Ratios when none are given.
pub fn default_ratios() -> IndexMap<String, f64> {
    [("train", 0.7), ("val", 0.15), ("test", 0.15)].into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Where one group went.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub group: String,
    pub partition: String,
    pub fold: Option<i64>,
    pub stratum: Option<String>,
    pub entries: Vec<String>,
}

impl Assignment {
    pub fn to_json(&self) -> Value {
        json!({
            "group": self.group,
            "partition": self.partition,
            "fold": self.fold,
            "stratum": self.stratum,
            "entries": self.entries,
        })
    }
}

/// A complete assignment of a cohort's groups.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub set_id: String,
    pub manifest_sha256: String,
    pub assignments: Vec<Assignment>,
    pub group_by: String,
    pub stratify_by: Option<String>,
    pub seed: i64,
    pub k_folds: Option<i64>,
    pub ratios: IndexMap<String, f64>,
}

impl Split {
    pub fn partition_of(&self, entry: &Entry) -> Result<Option<&str>> {
        let key = entry.field_str(&self.group_by)?;
        Ok(self.assignments.iter().find(|a| a.group == key).map(|a| a.partition.as_str()))
    }

    pub fn fold_of(&self, entry: &Entry) -> Result<Option<i64>> {
        let key = entry.field_str(&self.group_by)?;
        Ok(self.assignments.iter().find(|a| a.group == key).and_then(|a| a.fold))
    }

    /// The entries of `partition`: a file's path, or `path::key` for a sample
    /// inside a collection --- the locator task sources use.  Two members of
    /// one collection in different partitions were both its path, so a loader
    /// building per-partition file lists put the whole collection in both.
    /// A split written before carries paths alone.
    pub fn paths(&self, partition: &str) -> Vec<String> {
        self.assignments.iter().filter(|a| a.partition == partition).flat_map(|a| a.entries.iter().cloned()).collect()
    }

    /// Samples per partition.
    pub fn counts(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for a in &self.assignments {
            *out.entry(a.partition.clone()).or_insert(0) += a.entries.len();
        }
        out
    }

    /// Achieved stratum counts per partition --- what balance really came out.
    pub fn balance(&self) -> BTreeMap<String, BTreeMap<String, usize>> {
        let mut out: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
        for a in &self.assignments {
            let stratum = a.stratum.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "-".into());
            *out.entry(a.partition.clone()).or_default().entry(stratum).or_insert(0) += a.entries.len();
        }
        out
    }

    /// Folds that were asked for and got no groups.  [`make_splits`] refuses
    /// more folds than groups, so only a split written before 2.0 has any.
    pub fn empty_folds(&self) -> Vec<i64> {
        let Some(k) = self.k_folds else { return Vec::new() };
        let filled: BTreeSet<i64> = self.assignments.iter().filter_map(|a| a.fold).collect();
        (0..k).filter(|f| !filled.contains(f)).collect()
    }

    /// Partitions that were asked for and got nothing.
    pub fn underfilled(&self) -> Vec<String> {
        if self.k_folds.is_some() {
            return Vec::new();
        }
        let counts = self.counts();
        self.ratios
            .iter()
            .filter(|(p, share)| **share > 0.0 && counts.get(*p).copied().unwrap_or(0) == 0)
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// Groups that ended up in more than one partition, and entries placed in
    /// more than one: structurally impossible here, so a self-check --- and
    /// the check a split loaded from a file needs.  A fold is a partition of
    /// its own: every k-fold assignment is `holdout`, so comparing partitions
    /// alone saw no group placed in two folds.
    pub fn leaks(&self) -> Vec<String> {
        let mut seen: IndexMap<&str, (&str, Option<i64>)> = IndexMap::new();
        let mut placed: IndexMap<&str, (&str, Option<i64>)> = IndexMap::new();
        let mut bad = BTreeSet::new();
        for a in &self.assignments {
            let here = (a.partition.as_str(), a.fold);
            if *seen.entry(a.group.as_str()).or_insert(here) != here {
                bad.insert(a.group.clone());
            }
            for entry in &a.entries {
                if *placed.entry(entry.as_str()).or_insert(here) != here {
                    bad.insert(entry.clone());
                }
            }
        }
        bad.into_iter().collect()
    }

    /// Groups this split places elsewhere than `other` does, each with
    /// where `other` put it and where this one does: a partition, or
    /// `fold N`.  Groups only one of the two places are not listed.
    pub fn moved_from(&self, other: &Split) -> IndexMap<String, (String, String)> {
        let place = |a: &Assignment| match a.fold {
            Some(fold) => format!("fold {fold}"),
            None => a.partition.clone(),
        };
        let before: IndexMap<&str, String> = other.assignments.iter().map(|a| (a.group.as_str(), place(a))).collect();
        let mut out = IndexMap::new();
        for a in &self.assignments {
            if let Some(was) = before.get(a.group.as_str()) {
                let now = place(a);
                if *was != now && !out.contains_key(&a.group) {
                    out.insert(a.group.clone(), (was.clone(), now));
                }
            }
        }
        out
    }

    pub fn to_json(&self) -> Value {
        let ratios: Map<String, Value> = self.ratios.iter().map(|(k, v)| (k.clone(), num(*v))).collect();
        json!({
            "set_id": self.set_id,
            "manifest_sha256": self.manifest_sha256,
            "group_by": self.group_by,
            "stratify_by": self.stratify_by,
            "seed": self.seed,
            "k_folds": self.k_folds,
            "ratios": ratios,
            "counts": self.counts(),
            "underfilled": self.underfilled(),
            "empty_folds": self.empty_folds(),
            "balance": self.balance(),
            "assignments": self.assignments.iter().map(Assignment::to_json).collect::<Vec<_>>(),
        })
    }

    pub fn from_json(doc: &Value) -> Result<Split> {
        let text =
            |k: &str| -> Result<String> { doc.get(k).map(crate::json::py_str).ok_or_else(|| Error::Key(repr_str(k))) };
        let assignments = doc
            .get("assignments")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|a| {
                        Ok(Assignment {
                            group: a
                                .get("group")
                                .map(crate::json::py_str)
                                .ok_or_else(|| Error::Key("'group'".into()))?,
                            partition: a
                                .get("partition")
                                .map(crate::json::py_str)
                                .ok_or_else(|| Error::Key("'partition'".into()))?,
                            fold: a.get("fold").and_then(Value::as_i64),
                            stratum: a.get("stratum").and_then(Value::as_str).map(str::to_string),
                            entries: a
                                .get("entries")
                                .and_then(Value::as_array)
                                .map(|e| e.iter().map(crate::json::py_str).collect())
                                .unwrap_or_default(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Split {
            set_id: text("set_id")?,
            manifest_sha256: text("manifest_sha256")?,
            assignments,
            group_by: doc.get("group_by").map(crate::json::py_str).unwrap_or_else(|| "group_id".into()),
            stratify_by: doc.get("stratify_by").and_then(Value::as_str).map(str::to_string),
            seed: doc.get("seed").and_then(Value::as_i64).unwrap_or(0),
            k_folds: doc.get("k_folds").and_then(Value::as_i64),
            ratios: match doc.get("ratios") {
                Some(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0))).collect(),
                _ => default_ratios(),
            },
        })
    }
}

/// A stable pseudo-random ordering key for a group: a hash, so the order
/// depends only on the inputs.
fn rank(set_id: &str, seed: i64, group: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(format!("{set_id}:{seed}:{group}").as_bytes()))
}

/// Refuse a grouping finer than the subject it is supposed to contain (C204).
fn refuse_split_subjects(grouped: &IndexMap<String, Vec<&Entry>>, group_by: &str) -> Result<()> {
    let mut subjects: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (group, entries) in grouped {
        for entry in entries {
            subjects.entry(entry.subject_id.as_str()).or_default().insert(group.as_str());
        }
    }
    let straddling: Vec<(&str, Vec<&str>)> =
        subjects.into_iter().filter(|(_, g)| g.len() > 1).map(|(s, g)| (s, g.into_iter().collect())).collect();
    if straddling.is_empty() {
        return Ok(());
    }
    let detail: Vec<String> =
        straddling.iter().map(|(subject, groups)| format!("{} in {}", repr_str(subject), repr_list(groups))).collect();
    Err(Error::coded(
        "C204",
        format!(
            "grouping by {} puts {} subject(s) in more than one group: {}. A group has to contain whole subjects or \
             the split is not subject-safe --- these would be dealt independently and the same subject could land in \
             two partitions. Give every file of a subject the same {group_by}, or split on 'subject_id'.",
            repr_str(group_by),
            straddling.len(),
            detail.join("; ")
        ),
    ))
}

/// A group's stratum: the majority value among its samples, ties broken on
/// the lexically first value.
fn stratum_of(entries: &[&Entry], by: Option<&str>) -> Result<Option<String>> {
    let Some(by) = by else { return Ok(None) };
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    for entry in entries {
        *tally.entry(entry.field_str(by)?).or_insert(0) += 1;
    }
    let mut best: Option<(&String, usize)> = None;
    for (value, n) in &tally {
        if best.is_none_or(|(_, m)| *n > m) {
            best = Some((value, *n));
        }
    }
    Ok(best.map(|(v, _)| v.clone()))
}

/// Options of [`make_splits`], with the 1.x defaults.
#[derive(Debug, Clone)]
pub struct SplitOptions {
    pub set_id: String,
    pub group_by: String,
    pub stratify_by: Option<String>,
    pub ratios: Option<IndexMap<String, f64>>,
    pub k_folds: Option<i64>,
    pub seed: i64,
}

impl Default for SplitOptions {
    fn default() -> Self {
        SplitOptions {
            set_id: "default".into(),
            group_by: "group_id".into(),
            stratify_by: None,
            ratios: None,
            k_folds: None,
            seed: 0,
        }
    }
}

/// One stratum's groups, keyed by the stratum value.
type Stratum<'a> = (Option<String>, Vec<(String, Vec<&'a Entry>)>);

/// What every algorithm starts from: the checked options, the split without
/// assignments, the normalised shares, and the strata in key order, each
/// stratum's groups in rank order.
struct Prepared<'a> {
    split: Split,
    shares: IndexMap<String, f64>,
    ordered: Vec<Stratum<'a>>,
}

fn prepare<'a>(manifest: &'a Manifest, options: &SplitOptions) -> Result<Prepared<'a>> {
    if manifest.is_empty() {
        return Err(Error::invalid("cannot split an empty manifest"));
    }
    let mut shares = options.ratios.clone().filter(|r| !r.is_empty()).unwrap_or_else(default_ratios);
    match options.k_folds {
        None => {
            let mut unknown: Vec<&String> = shares.keys().filter(|k| !PARTITIONS.contains(&k.as_str())).collect();
            unknown.sort();
            unknown.dedup();
            if !unknown.is_empty() {
                return Err(Error::invalid(format!(
                    "unknown partition(s) {}; expected {}",
                    repr_list(&unknown),
                    repr_list(&PARTITIONS)
                )));
            }
            // A share is a finite number of at least zero.  NaN and infinity
            // left no partition with a positive share, and dealing from none
            // panicked; a negative share was accepted and written down.
            if let Some((name, share)) = shares.iter().find(|(_, v)| !v.is_finite() || **v < 0.0) {
                return Err(Error::invalid(format!(
                    "split ratio {} is {}; a ratio is a finite number of at least 0",
                    repr_str(name),
                    crate::json::py_float(*share)
                )));
            }
            let mut total = shares.values().fold(0.0, |acc, v| acc + v);
            if total.is_infinite() {
                // Finite shares can still overflow their sum: scale them first.
                let largest = shares.values().copied().fold(0.0, f64::max);
                shares = shares.into_iter().map(|(k, v)| (k, v / largest)).collect();
                total = shares.values().fold(0.0, |acc, v| acc + v);
            }
            if total <= 0.0 {
                return Err(Error::invalid("split ratios must sum to more than zero"));
            }
            shares = shares.into_iter().map(|(k, v)| (k, v / total)).collect();
        }
        Some(k) if k < 2 => return Err(Error::invalid("k_folds must be at least 2")),
        Some(_) => {
            // Folds take every group; a ratio has nothing to apply to, and
            // one written into the split, unchecked, said otherwise.
            if options.ratios.as_ref().is_some_and(|r| !r.is_empty()) {
                return Err(Error::invalid(
                    "ratios and k_folds are two ways to split a cohort: a k-fold split deals every group to a \
                     fold, so give one or the other",
                ));
            }
            shares = IndexMap::new();
        }
    }
    let grouped = manifest.groups(&options.group_by)?;
    refuse_split_subjects(&grouped, &options.group_by)?;
    if let Some(k) = options.k_folds.filter(|k| *k > grouped.len() as i64) {
        // An empty fold validates on nothing, and listing a billion of them
        // took the memory the process had.
        return Err(Error::invalid(format!(
            "k_folds {k} is more than the {} group(s) to deal by {}: every fold needs at least one group, so this \
             cohort splits at most {}-fold",
            grouped.len(),
            repr_str(&options.group_by),
            grouped.len()
        )));
    }
    let split = Split {
        set_id: options.set_id.clone(),
        manifest_sha256: manifest.sha256(),
        assignments: Vec::new(),
        group_by: options.group_by.clone(),
        stratify_by: options.stratify_by.clone(),
        seed: options.seed,
        k_folds: options.k_folds,
        ratios: shares.clone(),
    };
    let mut strata: IndexMap<Option<String>, Vec<(String, Vec<&Entry>)>> = IndexMap::new();
    for (group, entries) in grouped {
        let stratum = stratum_of(&entries, options.stratify_by.as_deref())?;
        strata.entry(stratum).or_default().push((group, entries));
    }
    // `str(None)` is "None": the unstratified stratum sorts as that word.
    let mut ordered: Vec<Stratum<'_>> = strata.into_iter().collect();
    ordered.sort_by(|a, b| {
        let key = |s: &Option<String>| s.clone().unwrap_or_else(|| "None".into());
        key(&a.0).cmp(&key(&b.0))
    });
    for (_, members) in ordered.iter_mut() {
        members.sort_by_cached_key(|(group, _)| rank(&options.set_id, options.seed, group));
    }
    Ok(Prepared { split, shares, ordered })
}

fn sort_assignments(split: &mut Split) {
    split.assignments.sort_by(|a, b| {
        (&a.partition, a.fold.unwrap_or(0), &a.group).cmp(&(&b.partition, b.fold.unwrap_or(0), &b.group))
    });
}

/// Assign every group in `manifest` to a partition, or to a fold, by
/// `medh5.split/1`.
///
/// With `k_folds` the folds are recorded as `holdout` assignments with a fold
/// number, and every fold gets at least one group; without it, groups go to
/// `train`/`val`/`test` in the ratios.  [`make_splits_with`] deals by
/// `medh5.split/2`, and continues a split.
pub fn make_splits(manifest: &Manifest, options: &SplitOptions) -> Result<Split> {
    let Prepared { mut split, shares, ordered } = prepare(manifest, options)?;
    // Interleave the strata round-robin, rotating which leads each round, so
    // every stratum spreads over the partitions and the global ratios hold.
    let depth = ordered.iter().map(|(_, m)| m.len()).max().unwrap_or(0);
    let mut interleaved: Vec<(Option<String>, String, Vec<&Entry>)> = Vec::new();
    for position in 0..depth {
        for offset in 0..ordered.len() {
            let (stratum, members) = &ordered[(position + offset) % ordered.len()];
            if let Some((group, entries)) = members.get(position) {
                interleaved.push((stratum.clone(), group.clone(), entries.clone()));
            }
        }
    }

    match options.k_folds {
        Some(k) => {
            for (position, (stratum, group, entries)) in interleaved.into_iter().enumerate() {
                split.assignments.push(Assignment {
                    group,
                    partition: "holdout".into(),
                    fold: Some(position as i64 % k),
                    stratum,
                    entries: entries.iter().map(|e| locator(e)).collect(),
                });
            }
        }
        None => {
            for (stratum, group, entries, partition) in deal(interleaved, &shares) {
                split.assignments.push(Assignment {
                    group,
                    partition,
                    fold: None,
                    stratum,
                    entries: entries.iter().map(|e| locator(e)).collect(),
                });
            }
        }
    }
    sort_assignments(&mut split);
    Ok(split)
}

/// How a split dealt its groups: the algorithm a split JSON names under
/// `algorithm`, `medh5.split/1` when it names none.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitAlgorithm {
    /// `medh5.split/1`: strata interleaved, groups dealt by the largest
    /// deficit against the global shares, so one group's partition depends
    /// on every other group.  The default in 2.x.
    #[default]
    V1,
    /// `medh5.split/2`: totals by largest remainder, each stratum's counts by
    /// controlled rounding, groups filling partitions in hash-rank order
    /// within their stratum.  Adding a group moves few others.
    V2,
}

impl SplitAlgorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            SplitAlgorithm::V1 => "medh5.split/1",
            SplitAlgorithm::V2 => "medh5.split/2",
        }
    }

    /// `"medh5.split/1"` or `"medh5.split/2"`, or `"1"` or `"2"` as the
    /// command line spells them.
    pub fn parse(text: &str) -> Result<SplitAlgorithm> {
        match text {
            "medh5.split/1" | "1" => Ok(SplitAlgorithm::V1),
            "medh5.split/2" | "2" => Ok(SplitAlgorithm::V2),
            other => Err(Error::Value(format!(
                "the split algorithm is 'medh5.split/1' or 'medh5.split/2', not {}",
                repr_str(other)
            ))),
        }
    }
}

/// What a split continued from: the previous split's manifest digest, how
/// many of its groups kept their place, and the groups added and dropped.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Lineage {
    pub manifest_sha256: String,
    pub kept: usize,
    pub added: Vec<String>,
    pub dropped: Vec<String>,
}

impl Lineage {
    pub fn new(manifest_sha256: impl Into<String>, kept: usize, added: Vec<String>, dropped: Vec<String>) -> Self {
        Lineage { manifest_sha256: manifest_sha256.into(), kept, added, dropped }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "manifest_sha256": self.manifest_sha256,
            "kept": self.kept,
            "added": self.added,
            "dropped": self.dropped,
        })
    }

    pub fn from_json(doc: &Value) -> Result<Lineage> {
        let strings = |k: &str| -> Vec<String> {
            doc.get(k)
                .and_then(Value::as_array)
                .map(|a| a.iter().map(crate::json::py_str).collect())
                .unwrap_or_default()
        };
        Ok(Lineage {
            manifest_sha256: doc
                .get("manifest_sha256")
                .map(crate::json::py_str)
                .ok_or_else(|| Error::Key("'manifest_sha256'".into()))?,
            kept: doc.get("kept").and_then(Value::as_u64).unwrap_or(0) as usize,
            added: strings("added"),
            dropped: strings("dropped"),
        })
    }
}

/// A split with the algorithm that dealt it and, when it continued another,
/// its lineage: what a split JSON holds.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SplitRecord {
    pub split: Split,
    pub algorithm: SplitAlgorithm,
    pub lineage: Option<Lineage>,
}

impl SplitRecord {
    pub fn new(split: Split, algorithm: SplitAlgorithm) -> Self {
        SplitRecord { split, algorithm, lineage: None }
    }

    pub fn with_lineage(mut self, lineage: Option<Lineage>) -> Self {
        self.lineage = lineage;
        self
    }

    /// The split's JSON with `algorithm`, and `previous` when it continued
    /// another split.
    pub fn to_json(&self) -> Value {
        let Value::Object(fields) = self.split.to_json() else { unreachable!("a split is an object") };
        let mut out = Map::new();
        for (key, value) in fields {
            let after = key == "ratios";
            out.insert(key, value);
            if after {
                out.insert("algorithm".into(), json!(self.algorithm.as_str()));
                out.insert("previous".into(), self.lineage.as_ref().map(Lineage::to_json).unwrap_or(Value::Null));
            }
        }
        Value::Object(out)
    }

    /// A split JSON; one without `algorithm`, written before 2.1, was dealt
    /// by `medh5.split/1`.
    pub fn from_json(doc: &Value) -> Result<SplitRecord> {
        let algorithm = match doc.get("algorithm") {
            None | Some(Value::Null) => SplitAlgorithm::V1,
            Some(Value::String(text)) => SplitAlgorithm::parse(text)?,
            Some(other) => SplitAlgorithm::parse(&crate::json::repr(other))?,
        };
        let lineage = match doc.get("previous") {
            None | Some(Value::Null) => None,
            Some(previous) => Some(Lineage::from_json(previous)?),
        };
        Ok(SplitRecord { split: Split::from_json(doc)?, algorithm, lineage })
    }

    /// Read a split written by `medh5 dataset split -o`.
    pub fn load(path: &Path) -> Result<SplitRecord> {
        let text = std::fs::read_to_string(path).map_err(|e| crate::error::os_error(&e, path))?;
        SplitRecord::from_json(&crate::json::loads(&text).map_err(|e| Error::Value(e.to_string()))?)
    }
}

/// What [`make_splits_with`] makes: [`SplitOptions`]' fields, the algorithm,
/// and a previous split to continue.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SplitRequest {
    pub set_id: String,
    pub group_by: String,
    pub stratify_by: Option<String>,
    pub ratios: Option<IndexMap<String, f64>>,
    pub k_folds: Option<i64>,
    pub seed: i64,
    pub algorithm: SplitAlgorithm,
    /// Continue this split: every group it placed keeps its partition or
    /// fold, new groups are dealt by `medh5.split/2`, and `algorithm` is
    /// not consulted.
    pub previous: Option<SplitRecord>,
}

impl Default for SplitRequest {
    fn default() -> Self {
        SplitRequest::from(SplitOptions::default())
    }
}

impl From<SplitOptions> for SplitRequest {
    fn from(options: SplitOptions) -> Self {
        SplitRequest {
            set_id: options.set_id,
            group_by: options.group_by,
            stratify_by: options.stratify_by,
            ratios: options.ratios,
            k_folds: options.k_folds,
            seed: options.seed,
            algorithm: SplitAlgorithm::default(),
            previous: None,
        }
    }
}

impl SplitRequest {
    pub fn with_algorithm(mut self, algorithm: SplitAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    pub fn with_previous(mut self, previous: Option<SplitRecord>) -> Self {
        self.previous = previous;
        self
    }

    fn options(&self) -> SplitOptions {
        SplitOptions {
            set_id: self.set_id.clone(),
            group_by: self.group_by.clone(),
            stratify_by: self.stratify_by.clone(),
            ratios: self.ratios.clone(),
            k_folds: self.k_folds,
            seed: self.seed,
        }
    }
}

/// [`make_splits`] by the algorithm `request` names, or continuing
/// `request.previous` (D22).
///
/// `medh5.split/2` deals the partitions' totals by the largest remainder of
/// `N·r`, each stratum's counts by a controlled rounding of `n_s·r_p` (every
/// count its floor or ceiling, every stratum and every total met), and fills a
/// stratum's partitions with its groups in hash-rank order.  A group's place
/// then depends on its stratum's count and its rank, so adding a group moves
/// few others; `medh5.split/1` moved a third of them.
///
/// With `previous`, every group the previous split placed keeps its
/// partition or fold; each new group fills its stratum's largest deficit
/// against `medh5.split/2`'s counts for the whole cohort, in rank order.  The
/// set id, seed, ratios, grouping, stratification and folds must be the
/// previous split's: anything else is another split.
pub fn make_splits_with(manifest: &Manifest, request: &SplitRequest) -> Result<SplitRecord> {
    let options = request.options();
    if let Some(previous) = &request.previous {
        return continue_split(manifest, &options, previous);
    }
    match request.algorithm {
        SplitAlgorithm::V1 => Ok(SplitRecord::new(make_splits(manifest, &options)?, SplitAlgorithm::V1)),
        SplitAlgorithm::V2 => {
            let prepared = prepare(manifest, &options)?;
            let mut split = prepared.split.clone();
            let targets = targets_v2(&prepared)?;
            let places = labels(&prepared);
            let layout = layout(&prepared, places.len());
            for ((stratum, members), counts) in prepared.ordered.iter().zip(&targets) {
                let mut members = members.iter();
                for &i in &layout {
                    for (group, entries) in members.by_ref().take(counts[i] as usize) {
                        split.assignments.push(assignment(&places[i], stratum, group, entries));
                    }
                }
            }
            sort_assignments(&mut split);
            Ok(SplitRecord::new(split, SplitAlgorithm::V2))
        }
    }
}

/// A partition or fold: `(partition, fold)`.
type Place = (String, Option<i64>);

/// The places groups are dealt to, in order: the partitions with a positive
/// share, in [`PARTITIONS`] order, or the folds.
fn labels(prepared: &Prepared<'_>) -> Vec<Place> {
    match prepared.split.k_folds {
        Some(k) => (0..k).map(|f| ("holdout".to_string(), Some(f))).collect(),
        None => PARTITIONS
            .iter()
            .filter(|p| prepared.shares.get(**p).is_some_and(|s| *s > 0.0))
            .map(|p| (p.to_string(), None))
            .collect(),
    }
}

/// The order a stratum's groups, in rank order, fill the places: the place
/// with the largest share in the middle, the next on its left, the next on
/// its right, and so on outwards --- `val`, `train`, `test` at 70/15/15.  A
/// count that moves between the middle place and a neighbour then moves one
/// group across one boundary, and a group added to a stratum moves another
/// only where it lands outside the middle block.
fn layout(prepared: &Prepared<'_>, n: usize) -> Vec<usize> {
    let share = |i: usize| match prepared.split.k_folds {
        Some(_) => 1.0,
        None => prepared.shares[labels(prepared)[i].0.as_str()],
    };
    let mut by_share: Vec<usize> = (0..n).collect();
    by_share.sort_by(|a, b| share(*b).total_cmp(&share(*a)).then(a.cmp(b)));
    let mut out = std::collections::VecDeque::new();
    for (k, i) in by_share.into_iter().enumerate() {
        if k % 2 == 1 {
            out.push_front(i);
        } else {
            out.push_back(i);
        }
    }
    out.into_iter().collect()
}

/// `medh5.split/2`'s count of each stratum's groups in each place, strata in
/// key order, places in [`labels`] order.
fn targets_v2(prepared: &Prepared<'_>) -> Result<Vec<Vec<u64>>> {
    let places = labels(prepared);
    let shares: Vec<f64> = match prepared.split.k_folds {
        Some(k) => vec![1.0 / k as f64; places.len()],
        None => places.iter().map(|(p, _)| prepared.shares[p.as_str()]).collect(),
    };
    let rows: Vec<u64> = prepared.ordered.iter().map(|(_, m)| m.len() as u64).collect();
    let totals = super::rounding::largest_remainder(rows.iter().sum(), &shares);
    super::rounding::controlled(&rows, &shares, &totals)
}

fn assignment(place: &Place, stratum: &Option<String>, group: &str, entries: &[&Entry]) -> Assignment {
    Assignment {
        group: group.to_string(),
        partition: place.0.clone(),
        fold: place.1,
        stratum: stratum.clone(),
        entries: entries.iter().map(|e| locator(e)).collect(),
    }
}

fn continue_split(manifest: &Manifest, options: &SplitOptions, previous: &SplitRecord) -> Result<SplitRecord> {
    let prepared = prepare(manifest, options)?;
    let old = &previous.split;
    let mut differs: Vec<&str> = Vec::new();
    if old.set_id != options.set_id {
        differs.push("set_id");
    }
    if old.seed != options.seed {
        differs.push("seed");
    }
    if old.group_by != options.group_by {
        differs.push("group_by");
    }
    if old.stratify_by != options.stratify_by {
        differs.push("stratify_by");
    }
    if old.k_folds != options.k_folds {
        differs.push("k_folds");
    }
    let positive = |m: &IndexMap<String, f64>| -> BTreeMap<String, f64> {
        m.iter().filter(|(_, v)| **v > 0.0).map(|(k, v)| (k.clone(), *v)).collect()
    };
    let (was, now) = (positive(&old.ratios), positive(&prepared.shares));
    if old.k_folds.is_none()
        && (was.len() != now.len() || was.iter().zip(&now).any(|(a, b)| a.0 != b.0 || (a.1 - b.1).abs() > 1e-9))
    {
        differs.push("ratios");
    }
    if !differs.is_empty() {
        return Err(Error::invalid(format!(
            "the previous split was made with another {}: continuing it with these would be another split; use a \
             new set_id",
            differs.join(", ")
        )));
    }
    let places = labels(&prepared);
    let targets = targets_v2(&prepared)?;
    let placed: IndexMap<&str, Place> =
        old.assignments.iter().map(|a| (a.group.as_str(), (a.partition.clone(), a.fold))).collect();
    let mut split = prepared.split.clone();
    let (mut kept, mut added) = (0usize, Vec::new());
    let mut present: BTreeSet<&str> = BTreeSet::new();
    for ((stratum, members), counts) in prepared.ordered.iter().zip(&targets) {
        let mut have: Vec<i64> = vec![0; places.len()];
        let mut fresh = Vec::new();
        for (group, entries) in members {
            present.insert(group.as_str());
            match placed.get(group.as_str()) {
                Some(place) => {
                    if let Some(i) = places.iter().position(|p| p == place) {
                        have[i] += 1;
                    }
                    kept += 1;
                    split.assignments.push(assignment(place, stratum, group, entries));
                }
                None => fresh.push((group, entries)),
            }
        }
        // In rank order, each new group to the place furthest below its
        // count, the earlier place on a tie.
        for (group, entries) in fresh {
            let mut best = 0;
            for i in 1..places.len() {
                if counts[i] as i64 - have[i] > counts[best] as i64 - have[best] {
                    best = i;
                }
            }
            have[best] += 1;
            added.push(group.clone());
            split.assignments.push(assignment(&places[best], stratum, group, entries));
        }
    }
    added.sort();
    let mut dropped: Vec<String> = placed.keys().filter(|g| !present.contains(**g)).map(|g| g.to_string()).collect();
    dropped.sort();
    dropped.dedup();
    sort_assignments(&mut split);
    let lineage = Lineage::new(old.manifest_sha256.clone(), kept, added, dropped);
    Ok(SplitRecord::new(split, SplitAlgorithm::V2).with_lineage(Some(lineage)))
}

/// The frontends' split: `algorithm` as the caller gave it, or `None`.
///
/// Left to its 2.x default without a previous split, the cohort is dealt
/// both ways and the `medh5.split/1` split comes back with how many of its
/// groups `medh5.split/2` places elsewhere, which the frontends warn of when
/// it is not 0.  A previous split with `medh5.split/1` named is refused: /1
/// has no continuation.
#[doc(hidden)]
pub fn make_splits_asked(
    manifest: &Manifest,
    request: &SplitRequest,
    algorithm: Option<SplitAlgorithm>,
) -> Result<(SplitRecord, Option<usize>)> {
    match (algorithm, &request.previous) {
        (Some(SplitAlgorithm::V1), Some(_)) => Err(Error::invalid(
            "medh5.split/1 cannot continue a split: a previous split's new groups are dealt by medh5.split/2, so \
             leave the algorithm out or name medh5.split/2",
        )),
        (Some(algorithm), _) => Ok((make_splits_with(manifest, &request.clone().with_algorithm(algorithm))?, None)),
        (None, Some(_)) => Ok((make_splits_with(manifest, request)?, None)),
        (None, None) => {
            let v1 = make_splits_with(manifest, &request.clone().with_algorithm(SplitAlgorithm::V1))?;
            let v2 = make_splits_with(manifest, &request.clone().with_algorithm(SplitAlgorithm::V2))?;
            let moved = v2.split.moved_from(&v1.split).len();
            Ok((v1, Some(moved)))
        }
    }
}

/// What the frontends warn of when the algorithm was left out and
/// `medh5.split/2` would place `moved` groups elsewhere; `keep` says how to
/// keep this split in the frontend's spelling.
#[doc(hidden)]
pub fn default_algorithm_warning(moved: usize, keep: &str) -> String {
    format!(
        "medh5 3.0 splits with medh5.split/2 by default, which places {moved} of these groups differently: {keep} \
         to keep this split"
    )
}

type Dealt<'a> = (Option<String>, String, Vec<&'a Entry>, String);

/// Hand out groups by largest remaining deficit against the target shares.
fn deal<'a>(ordered: Vec<(Option<String>, String, Vec<&'a Entry>)>, shares: &IndexMap<String, f64>) -> Vec<Dealt<'a>> {
    let names: Vec<&str> = PARTITIONS.iter().copied().filter(|p| shares.get(*p).is_some_and(|s| *s > 0.0)).collect();
    let mut assigned: Vec<i64> = vec![0; names.len()];
    let mut out = Vec::with_capacity(ordered.len());
    for (position, (stratum, group, entries)) in ordered.into_iter().enumerate() {
        let done = (position + 1) as f64;
        let mut best = 0;
        let mut best_deficit = f64::NEG_INFINITY;
        for (i, name) in names.iter().enumerate() {
            let deficit = shares[*name] * done - assigned[i] as f64;
            // `max` keeps the first maximum; on equal deficits the earlier
            // partition wins (the `-names.index(p)` tiebreak).
            if deficit > best_deficit {
                best = i;
                best_deficit = deficit;
            }
        }
        assigned[best] += 1;
        out.push((stratum, group, entries, names[best].to_string()));
    }
    out
}

/// Stamp each sample with its partition and the manifest digest (§12.3).
///
/// With `k_folds`, `fold` selects which fold is validation for this claim;
/// every other fold becomes `train`.  Amending rewrites each file, so this is
/// optional: a split is fully usable as a JSON file.
///
/// Every refusal is met before the first file is amended: the fold, a split
/// that places something twice, a manifest holding a sample the split does
/// not place (one it was not made from), a sample inside a collection, and
/// each file
/// amending would refuse --- another major or a later minor (1.1 §2.3), a
/// profile this engine does not implement, HDF5 references, a file the
/// validator rejects, which a commit refuses to write (§15), and a dataset
/// whose bytes fail its digest, which a commit refuses to restamp (E701).
/// Only a failure while writing, such as a full disk, can stop it part way.
pub fn write_claims(
    split: &Split,
    manifest: &Manifest,
    assigned_by: Option<&str>,
    fold: Option<i64>,
) -> Result<Vec<String>> {
    // A refusal halfway through left some files claiming the split and the
    // rest not, and a fold outside the split's range was not a refusal at
    // all: no assignment matched it, so every file was written as `train`
    // and the split had no validation partition.
    match (split.k_folds, fold) {
        (Some(_), None) => {
            return Err(Error::invalid("a k-fold split needs --fold N to say which fold is validation"));
        }
        (Some(k), Some(chosen)) if !(0..k).contains(&chosen) => {
            return Err(Error::invalid(format!(
                "fold {chosen} is not a fold of this {k}-fold split, which numbers them 0 to {}",
                k - 1
            )));
        }
        (Some(k), Some(chosen)) if !split.assignments.iter().any(|a| a.fold == Some(chosen)) => {
            return Err(Error::invalid(format!(
                "fold {chosen} of this {k}-fold split holds no group, so claims naming it validation would have \
                 no validation partition"
            )));
        }
        (None, Some(chosen)) => {
            // Ignored, it left the caller believing fold `chosen` was validation.
            return Err(Error::invalid(format!(
                "fold {chosen} names a fold, but this split has none: it was made with ratios, not k_folds"
            )));
        }
        _ => {}
    }
    let leaks = split.leaks();
    if !leaks.is_empty() {
        return Err(Error::invalid(format!(
            "the split places {} in more than one partition, so a claim could record only one of them; it is not \
             a split to write",
            repr_list(&leaks)
        )));
    }
    let mut by_path: IndexMap<&str, &Assignment> = IndexMap::new();
    for assignment in &split.assignments {
        for path in &assignment.entries {
            by_path.insert(path.as_str(), assignment);
        }
    }
    let place = |entry: &Entry| -> Option<&Assignment> {
        by_path.get(locator(entry).as_str()).or_else(|| by_path.get(entry.path.as_str())).copied()
    };
    // A manifest with a sample the split does not place is not the one it
    // was made from, nor part of it: the cohort grew, or the paths moved.
    // Its claims were stamped with the split's digest --- stale against this
    // manifest the moment they were written (C201) --- and the samples the
    // split did not name were passed over in silence.
    let unplaced: Vec<String> = manifest.entries.iter().filter(|e| place(e).is_none()).map(locator).collect();
    if !unplaced.is_empty() {
        let shown: Vec<&str> = unplaced.iter().take(3).map(String::as_str).collect();
        let more = if unplaced.len() > 3 { format!(" and {} more", unplaced.len() - 3) } else { String::new() };
        let why = if split.manifest_sha256 == manifest.sha256() {
            "the split names its samples by other paths --- re-split this manifest".to_string()
        } else {
            format!(
                "it was made from another manifest (digest {}, this one's is {}) --- re-split this manifest, or pass \
                 the one the split was made from",
                &split.manifest_sha256[..12.min(split.manifest_sha256.len())],
                &manifest.sha256()[..12]
            )
        };
        return Err(Error::invalid(format!(
            "the split places no partition for {} of this manifest's samples ({}{more}): {why}",
            unplaced.len(),
            shown.join(", ")
        )));
    }
    let mut planned = Vec::new();
    for entry in &manifest.entries {
        let Some(placed) = place(entry) else { continue };
        if let Some(key) = &entry.key {
            return Err(Error::invalid(format!(
                "{}#{key} is inside a collection --- unpack it before writing split claims",
                entry.path
            )));
        }
        let mut partition = placed.partition.clone();
        if split.k_folds.is_some() {
            partition = if placed.fold == fold { "val".into() } else { "train".into() };
        }
        let claim = SplitClaim {
            set_id: split.set_id.clone(),
            partition,
            fold: placed.fold.map(serde_json::Number::from),
            assigned_by: assigned_by.map(str::to_string),
            assigned_at: None,
            manifest_sha256: Some(split.manifest_sha256.clone()),
        };
        claim.check()?;
        planned.push((entry.path.clone(), claim));
    }
    for (path, _) in &planned {
        claimable(Path::new(path))?;
    }
    let mut written = Vec::new();
    for (path, claim) in planned {
        let mut writer = amend(Path::new(&path), None)?;
        let fields = match claim.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        writer.split(fields)?;
        writer.commit(true)?;
        written.push(path);
    }
    Ok(written)
}

/// What amending `path` and committing it would refuse, found by reading:
/// a collection or another major (on opening), a later minor or a profile
/// this engine does not implement (1.1 §2.3), HDF5 references, an error the
/// validator reports at the level a commit checks (§15), and a digest the
/// bytes fail, which the commit would restamp (E701) --- met at commit, it
/// stopped the run after the files before it had claimed the split.
fn claimable(path: &Path) -> Result<()> {
    let shown = repr_str(&path.to_string_lossy());
    let sample = open_sample(path)?;
    let declared = attrs::get_strs(&sample.root, "medh5_profiles")?.unwrap_or_default();
    crate::version::require_amendable(&sample.version()?, &declared, &shown)?;
    ops::refuse_references(&sample.root, "an amendment")?;
    let report = crate::validate::validate_root(&sample.root, Some(path), "semantic", true)?;
    if let Some(error) = report.diagnostics.iter().find(|d| d.severity == "error") {
        return Err(Error::coded(
            &error.code,
            format!(
                "{shown} fails validation, so amending it to claim the split would be refused: {} {}: {}",
                error.code, error.location, error.message
            ),
        ));
    }
    refuse_failed_digests(&sample.root, &shown)
}

/// How a split names one manifest entry: its path, or `path::key` for a
/// sample inside a collection (the locator task sources use).
pub fn locator(entry: &Entry) -> String {
    match entry.key.as_deref() {
        Some(key) if !key.is_empty() => format!("{}::{key}", entry.path),
        _ => entry.path.clone(),
    }
}

/// Read a split written by `medh5 dataset split -o`.
pub fn load(path: &Path) -> Result<Split> {
    let text = std::fs::read_to_string(path)?;
    Split::from_json(&crate::json::loads(&text).map_err(|e| Error::Value(e.to_string()))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split cohort of the plan: three strata of 60 %, 30 % and 10 % of
    /// `n`, and rare strata of 3, 5 and 7 groups, one entry per group.
    pub(crate) fn cohort(n: usize, extra: usize) -> Manifest {
        let sizes = [("a", n * 6 / 10), ("b", n * 3 / 10), ("c", n / 10), ("r3", 3), ("r5", 5), ("r7", 7)];
        let entry = |id: String, site: &str| Entry {
            path: format!("{id}.medh5"),
            sample_id: id.clone(),
            subject_id: id.clone(),
            group_id: id,
            site_id: Some(site.into()),
            ..Default::default()
        };
        let mut entries = Vec::new();
        for (site, count) in sizes {
            for i in 0..count {
                entries.push(entry(format!("{site}-{i:03}"), site));
            }
        }
        // Added groups join the largest stratum, every other id unchanged.
        for i in 0..extra {
            entries.push(entry(format!("new-{i:03}"), "a"));
        }
        Manifest { entries, ..Default::default() }
    }

    fn request(seed: i64, algorithm: SplitAlgorithm) -> SplitRequest {
        let mut r = SplitRequest::default().with_algorithm(algorithm);
        r.seed = seed;
        r.stratify_by = Some("site_id".into());
        r
    }

    /// One group added to the largest stratum of 115, over seeds 0--19:
    /// `medh5.split/1` moves a median of 15 existing groups (up to 29),
    /// `medh5.split/2` a median of 1 (at most 2).
    #[test]
    fn s12_3_split2_one_new_group_moves_few() {
        let mut moved = [Vec::new(), Vec::new()];
        for seed in 0..20 {
            for (i, algorithm) in [SplitAlgorithm::V1, SplitAlgorithm::V2].into_iter().enumerate() {
                let before = make_splits_with(&cohort(100, 0), &request(seed, algorithm)).unwrap();
                let after = make_splits_with(&cohort(100, 1), &request(seed, algorithm)).unwrap();
                moved[i].push(after.split.moved_from(&before.split).len());
            }
        }
        for m in &mut moved {
            m.sort();
        }
        assert!(moved[0][10] >= 10, "{:?}", moved[0]);
        assert!(moved[1][10] <= 2, "{:?}", moved[1]);
    }

    #[test]
    fn s12_3_previous_moves_no_group_and_meets_the_shares() {
        let before = make_splits_with(&cohort(100, 0), &request(3, SplitAlgorithm::V1)).unwrap();
        let grown = cohort(100, 200);
        let after =
            make_splits_with(&grown, &request(3, SplitAlgorithm::V1).with_previous(Some(before.clone()))).unwrap();
        assert!(after.split.moved_from(&before.split).is_empty());
        assert_eq!(after.algorithm, SplitAlgorithm::V2);
        let lineage = after.lineage.clone().unwrap();
        assert_eq!((lineage.kept, lineage.added.len(), lineage.dropped.len()), (115, 200, 0));
        assert_eq!(lineage.manifest_sha256, before.split.manifest_sha256);
        let counts = after.split.counts();
        assert!((counts["train"] as f64 / 315.0 - 0.7).abs() < 0.02, "{counts:?}");
        // The JSON carries both, and reads back.
        assert_eq!(SplitRecord::from_json(&after.to_json()).unwrap(), after);
        // Another seed is another split.
        let err = make_splits_with(&grown, &request(4, SplitAlgorithm::V2).with_previous(Some(before))).unwrap_err();
        assert!(err.to_string().contains("seed"), "{err}");
    }
}

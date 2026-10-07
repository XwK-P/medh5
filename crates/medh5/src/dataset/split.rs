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
use crate::json::{num, repr_list, repr_str};
use crate::sample::writer::amend;
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

    /// Folds that were asked for and got no groups.
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

    /// Groups that ended up in more than one partition: structurally
    /// impossible here, so a self-check.
    pub fn leaks(&self) -> Vec<String> {
        let mut seen: IndexMap<&str, &str> = IndexMap::new();
        let mut bad = BTreeSet::new();
        for a in &self.assignments {
            let previous = *seen.entry(a.group.as_str()).or_insert(a.partition.as_str());
            if previous != a.partition {
                bad.insert(a.group.clone());
            }
        }
        bad.into_iter().collect()
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
/// Assign every group in `manifest` to a partition, or to a fold.
///
/// With `k_folds` the folds are recorded as `holdout` assignments with a fold
/// number; without it, groups go to `train`/`val`/`test` in the ratios.
pub fn make_splits(manifest: &Manifest, options: &SplitOptions) -> Result<Split> {
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
            let total = shares.values().fold(0.0, |acc, v| acc + v);
            if total <= 0.0 {
                return Err(Error::invalid("split ratios must sum to more than zero"));
            }
            shares = shares.into_iter().map(|(k, v)| (k, v / total)).collect();
        }
        Some(k) if k < 2 => return Err(Error::invalid("k_folds must be at least 2")),
        Some(_) => {}
    }
    let grouped = manifest.groups(&options.group_by)?;
    refuse_split_subjects(&grouped, &options.group_by)?;
    let mut split = Split {
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
    for (group, entries) in &grouped {
        let stratum = stratum_of(entries, options.stratify_by.as_deref())?;
        strata.entry(stratum).or_default().push((group.clone(), entries.clone()));
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
                    entries: entries.iter().map(|e| e.path.clone()).collect(),
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
                    entries: entries.iter().map(|e| e.path.clone()).collect(),
                });
            }
        }
    }
    split.assignments.sort_by(|a, b| {
        (&a.partition, a.fold.unwrap_or(0), &a.group).cmp(&(&b.partition, b.fold.unwrap_or(0), &b.group))
    });
    Ok(split)
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
pub fn write_claims(
    split: &Split,
    manifest: &Manifest,
    assigned_by: Option<&str>,
    fold: Option<i64>,
) -> Result<Vec<String>> {
    let mut by_path: IndexMap<&str, &Assignment> = IndexMap::new();
    for assignment in &split.assignments {
        for path in &assignment.entries {
            by_path.insert(path.as_str(), assignment);
        }
    }
    let mut written = Vec::new();
    for entry in &manifest.entries {
        let Some(placed) = by_path.get(entry.path.as_str()) else { continue };
        if let Some(key) = &entry.key {
            return Err(Error::invalid(format!(
                "{}#{key} is inside a collection --- unpack it before writing split claims",
                entry.path
            )));
        }
        let mut partition = placed.partition.clone();
        if split.k_folds.is_some() {
            let Some(chosen) = fold else {
                return Err(Error::invalid("a k-fold split needs --fold N to say which fold is validation"));
            };
            partition = if placed.fold == Some(chosen) { "val".into() } else { "train".into() };
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
        let mut writer = amend(Path::new(&entry.path), None)?;
        let fields = match claim.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        writer.split(fields)?;
        writer.commit(true)?;
        written.push(entry.path.clone());
    }
    Ok(written)
}

/// Read a split written by `medh5 dataset split -o`.
pub fn load(path: &Path) -> Result<Split> {
    let text = std::fs::read_to_string(path)?;
    Split::from_json(&crate::json::loads(&text).map_err(|e| Error::Value(e.to_string()))?)
}

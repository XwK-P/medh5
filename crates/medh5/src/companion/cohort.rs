//! A task drawn from a cohort: its subjects, sources and partitions taken
//! from a cohort manifest and its split (task-cache-1 §3.3).
//!
//! The cohort says which samples there are and whose; the split, or the
//! claims its samples carry, says where each subject goes.  Nothing is
//! guessed: an entry the split does not place, or places `unassigned`, a
//! subject placed in two partitions, a claim made against another manifest
//! and an entry whose recorded `content_id` is not its file's are refused,
//! naming what was found.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use crate::curation::identity::{SplitClaim, PARTITIONS};
use crate::dataset::manifest::{Entry, Manifest};
use crate::dataset::split::{locator, Split};
use crate::json::{repr_list, repr_str};
use crate::{Error, Result};

use super::source::{SourceAccess, SourceRef};
use super::task::{Subject, TaskManifest};

/// Where a cohort's partitions come from.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Partitions<'a> {
    /// A split of the cohort (`medh5 dataset split -o`).  A k-fold split
    /// needs `fold`: that fold is `val`, every other `train`.
    Split { split: &'a Split, fold: Option<u32> },
    /// The claims the samples carry for `set_id` (`medh5 dataset split
    /// --write-claims`), each made against this cohort manifest.
    Claims { set_id: &'a str },
}

/// What a task drawn from a cohort is called, and where its URIs are
/// relative to.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CohortTask<'a> {
    pub task_id: &'a str,
    pub version: &'a str,
    pub identity_namespace: &'a str,
    /// The directory the manifest will be saved in: source URIs are written
    /// relative to it where they lie beneath it.  `None` writes them
    /// absolute.
    pub base: Option<&'a Path>,
    /// The partition learned preprocessing is fitted on, listed first.
    pub training_partition: &'a str,
}

impl<'a> CohortTask<'a> {
    /// A task named so, its URIs absolute, fitted on `train`.
    pub fn new(task_id: &'a str, version: &'a str, identity_namespace: &'a str) -> CohortTask<'a> {
        CohortTask { task_id, version, identity_namespace, base: None, training_partition: "train" }
    }

    /// Source URIs relative to `base`, where they lie beneath it.
    pub fn with_base(mut self, base: Option<&'a Path>) -> CohortTask<'a> {
        self.base = base;
        self
    }

    /// Fitted on `partition` rather than `train`.
    pub fn with_training_partition(mut self, partition: &'a str) -> CohortTask<'a> {
        self.training_partition = partition;
        self
    }
}

impl TaskManifest {
    /// A task over a cohort: one subject per cohort `subject_id`, its entries
    /// as its sources, pinned to what each file holds now, each subject in
    /// the partition `partitions` gives it.  The split is `(set_id,
    /// partitions)` with the training partition first; there are no rows.
    /// Fragments of one subject are reconciled.
    ///
    /// Entry paths are read as the cohort manifest holds them in memory
    /// (relative to the working directory, as [`Manifest::load`] leaves
    /// them).  Source ids are the defaults ([`SourceRef::default_id`]).
    pub fn from_cohort(cohort: &Manifest, partitions: Partitions<'_>, task: &CohortTask<'_>) -> Result<TaskManifest> {
        if task.training_partition.is_empty() || task.training_partition == "unassigned" {
            return Err(Error::invalid(format!(
                "the training partition cannot be {}",
                repr_str(task.training_partition)
            )));
        }
        let place = Placer::new(cohort, partitions)?;
        let base = task.base.map(crate::dataset::manifest::absolute).transpose()?;
        // Subjects in the cohort's order, each with its partition and entries.
        let mut subjects: IndexMap<&str, (String, Vec<&Entry>)> = IndexMap::new();
        for entry in &cohort.entries {
            let partition = place.partition(entry)?;
            match subjects.get_mut(entry.subject_id.as_str()) {
                Some((held, entries)) => {
                    if *held != partition {
                        return Err(Error::invalid(format!(
                            "subject {} is in two partitions, {} and {} ({}): a task places a subject whole",
                            repr_str(&entry.subject_id),
                            repr_str(held),
                            repr_str(&partition),
                            locator(entry)
                        )));
                    }
                    entries.push(entry);
                }
                None => {
                    subjects.insert(&entry.subject_id, (partition, vec![entry]));
                }
            }
        }
        let mut out = TaskManifest::new(task.task_id, task.version, task.identity_namespace);
        let mut present: BTreeSet<String> = BTreeSet::new();
        for (subject_id, (partition, entries)) in &subjects {
            let mut sources = Vec::with_capacity(entries.len());
            for entry in entries {
                sources.push(pin_entry(entry, base.as_deref())?);
            }
            present.insert(partition.clone());
            out.add_subject(Subject {
                subject_id: subject_id.to_string(),
                clock_id: None,
                partition: Some(partition.clone()),
                sources,
                reconciled: Vec::new(),
            });
        }
        // The training partition first, then the standard ones in their
        // order, then any other a split named, alphabetically.
        present.remove(task.training_partition);
        let rank = |p: &String| (PARTITIONS.iter().position(|q| q == p).unwrap_or(PARTITIONS.len()), p.clone());
        let mut others: Vec<String> = present.into_iter().collect();
        others.sort_by_key(rank);
        let mut listed = vec![task.training_partition.to_string()];
        listed.extend(others);
        out.split = Some((place.set_id.to_string(), listed));
        // Fragments of one subject holding the same events are recorded, as
        // `task reconcile` would, so the new task preflights clean.
        out.reconcile_all(&SourceAccess::new(base.as_deref()))?;
        Ok(out)
    }
}

/// The source an entry names, pinned to what its file holds now.
fn pin_entry(entry: &Entry, base: Option<&Path>) -> Result<SourceRef> {
    let path = PathBuf::from(&entry.path);
    let key = entry.key.clone().filter(|k| !k.is_empty());
    let sample = SourceRef::new(entry.path.clone(), key.clone(), "").open(None)?;
    let uri = match base {
        Some(base) => {
            crate::dataset::manifest::rebase(&entry.path, base, crate::dataset::manifest::PathStyle::Relative)?
        }
        None => crate::dataset::manifest::absolute(&path)?.to_string_lossy().into_owned(),
    };
    let pinned = SourceRef::pin(uri, key, &sample)?;
    if let Some(recorded) = entry.content_id.as_deref().filter(|c| !c.is_empty()) {
        if recorded != pinned.content_id {
            return Err(Error::invalid(format!(
                "{} records content_id {} but the file holds {}: the cohort manifest is stale --- re-index it \
                 (`medh5 dataset index`)",
                locator(entry),
                repr_str(recorded),
                repr_str(&pinned.content_id)
            )));
        }
    }
    Ok(pinned)
}

/// Where each entry goes.
struct Placer<'a> {
    set_id: &'a str,
    how: How<'a>,
}

enum How<'a> {
    Split { by_locator: BTreeMap<&'a str, (&'a str, Option<i64>)>, fold: Option<i64> },
    Claims { sha256: String },
}

impl<'a> Placer<'a> {
    fn new(cohort: &'a Manifest, partitions: Partitions<'a>) -> Result<Placer<'a>> {
        Ok(match partitions {
            Partitions::Split { split, fold } => {
                let fold = fold.map(i64::from);
                match (split.k_folds, fold) {
                    (Some(_), None) => {
                        return Err(Error::invalid("a k-fold split needs a fold to say which fold is validation"));
                    }
                    (Some(k), Some(chosen)) if !(0..k).contains(&chosen) => {
                        return Err(Error::invalid(format!(
                            "fold {chosen} is not a fold of this {k}-fold split, which numbers them 0 to {}",
                            k - 1
                        )));
                    }
                    (None, Some(chosen)) => {
                        return Err(Error::invalid(format!(
                            "fold {chosen} names a fold, but this split has none: it was made with ratios, not \
                             k_folds"
                        )));
                    }
                    _ => {}
                }
                let leaks = split.leaks();
                if !leaks.is_empty() {
                    return Err(Error::invalid(format!(
                        "the split places {} in more than one partition",
                        repr_list(&leaks)
                    )));
                }
                let mut by_locator = BTreeMap::new();
                for a in &split.assignments {
                    for e in &a.entries {
                        by_locator.insert(e.as_str(), (a.partition.as_str(), a.fold));
                    }
                }
                Placer { set_id: &split.set_id, how: How::Split { by_locator, fold } }
            }
            Partitions::Claims { set_id } => Placer { set_id, how: How::Claims { sha256: cohort.sha256() } },
        })
    }

    fn partition(&self, entry: &Entry) -> Result<String> {
        let found = match &self.how {
            How::Split { by_locator, fold } => {
                let placed = by_locator.get(locator(entry).as_str()).or_else(|| by_locator.get(entry.path.as_str()));
                placed.map(|(partition, f)| match fold {
                    Some(chosen) => if *f == Some(*chosen) { "val" } else { "train" }.to_string(),
                    None => partition.to_string(),
                })
            }
            How::Claims { sha256 } => {
                let mut found = None;
                for doc in &entry.splits {
                    let claim = SplitClaim::from_json(&serde_json::Value::Object(doc.clone()))?;
                    if claim.set_id != self.set_id {
                        continue;
                    }
                    if claim.manifest_sha256.as_deref() != Some(sha256.as_str()) {
                        return Err(Error::invalid(format!(
                            "{} claims split {} against another manifest (digest {}, this one's is {}): re-claim \
                             it from this manifest, or pass the one it was made from",
                            locator(entry),
                            repr_str(self.set_id),
                            claim.manifest_sha256.as_deref().map(|d| &d[..12.min(d.len())]).unwrap_or("none"),
                            &sha256[..12]
                        )));
                    }
                    found = Some(claim.partition);
                    break;
                }
                found
            }
        };
        match found {
            Some(p) if p == "unassigned" => Err(Error::invalid(format!(
                "{} is unassigned in split {}: a task places every source",
                locator(entry),
                repr_str(self.set_id)
            ))),
            Some(p) => Ok(p),
            None => Err(Error::invalid(format!(
                "{} has no partition in split {}: a task places every source",
                locator(entry),
                repr_str(self.set_id)
            ))),
        }
    }
}

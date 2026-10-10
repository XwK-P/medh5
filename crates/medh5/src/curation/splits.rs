//! Auditing split claims across a cohort (§12.3).
//!
//! A split claim in a file is a **membership claim, not an authority**: the
//! dataset manifest decides.  Two findings matter and they are different:
//! *conflicting claims* (W906: one `set_id` against two manifest hashes, so a
//! file predates a re-split) and *subject leakage* (one unit of anatomy in two
//! partitions of one split).  Both are invisible in any one file, so this is a
//! cross-file operation.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::collection::{open_any, AnyFile};
use crate::curation::identity::SplitClaim;
use crate::json::repr_str;
use crate::sample::Sample;
use crate::Result;

/// One file's claim to one partition of one split.
#[derive(Debug, Clone, PartialEq)]
pub struct Membership {
    pub path: String,
    pub sample_id: String,
    pub subject_id: String,
    pub group_id: String,
    pub claim: SplitClaim,
}

impl Membership {
    pub fn set_id(&self) -> &str {
        &self.claim.set_id
    }

    pub fn partition(&self) -> &str {
        &self.claim.partition
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("path".into(), json!(self.path));
        out.insert("sample_id".into(), json!(self.sample_id));
        out.insert("subject_id".into(), json!(self.subject_id));
        out.insert("group_id".into(), json!(self.group_id));
        if let Value::Object(claim) = self.claim.to_json() {
            out.extend(claim);
        }
        Value::Object(out)
    }
}

/// One unit of anatomy appearing in more than one partition of one split.
///
/// The unit is a connected component of `subject_id` and grouping key: two
/// files are one unit if they share either, transitively.
#[derive(Debug, Clone, PartialEq)]
pub struct Leak {
    pub set_id: String,
    pub group_id: String,
    pub partitions: Vec<String>,
    pub paths: Vec<String>,
    pub groups: Vec<String>,
    pub subjects: Vec<String>,
}

impl Leak {
    pub fn to_json(&self) -> Value {
        let groups = if self.groups.is_empty() { vec![self.group_id.clone()] } else { self.groups.clone() };
        json!({
            "set_id": self.set_id,
            "group_id": self.group_id,
            "partitions": self.partitions,
            "paths": self.paths,
            "groups": groups,
            "subjects": self.subjects,
        })
    }
}

impl std::fmt::Display for Leak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let joined = if self.groups.len() > 1 {
            format!(
                " (joined with {} by subject {})",
                self.groups[1..].iter().map(|g| repr_str(g)).collect::<Vec<_>>().join(", "),
                self.subjects.iter().map(|s| repr_str(s)).collect::<Vec<_>>().join(", ")
            )
        } else {
            String::new()
        };
        write!(
            f,
            "{}: group {}{joined} is in {} ({} files)",
            self.set_id,
            repr_str(&self.group_id),
            self.partitions.join(", "),
            self.paths.len()
        )
    }
}

/// One `set_id` claimed against more than one manifest (W906).
#[derive(Debug, Clone, PartialEq)]
pub struct Conflict {
    pub set_id: String,
    pub manifests: Vec<String>,
    pub paths_by_manifest: BTreeMap<String, Vec<String>>,
}

impl Conflict {
    pub fn to_json(&self) -> Value {
        json!({"set_id": self.set_id, "manifests": self.manifests, "paths_by_manifest": self.paths_by_manifest})
    }
}

impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let short: Vec<String> = self.manifests.iter().map(|m| m.chars().take(12).collect()).collect();
        write!(f, "{}: {} different manifest hashes ({})", self.set_id, self.manifests.len(), short.join(", "))
    }
}

/// What a cohort's split claims say, and where they disagree.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SplitAudit {
    pub memberships: Vec<Membership>,
    pub conflicts: Vec<Conflict>,
    pub leaks: Vec<Leak>,
    /// Files carrying no split claim at all --- not an error, but easy to lose.
    pub unclaimed: Vec<String>,
    pub unreadable: Vec<(String, String)>,
}

impl SplitAudit {
    pub fn ok(&self) -> bool {
        self.conflicts.is_empty() && self.leaks.is_empty() && self.unreadable.is_empty()
    }

    pub fn set_ids(&self) -> Vec<String> {
        self.memberships.iter().map(|m| m.claim.set_id.clone()).collect::<BTreeSet<_>>().into_iter().collect()
    }

    /// `partition -> sample ids` for one split.
    pub fn partitions(&self, set_id: &str) -> BTreeMap<String, Vec<String>> {
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for member in self.memberships.iter().filter(|m| m.set_id() == set_id) {
            out.entry(member.partition().to_string()).or_default().push(member.sample_id.clone());
        }
        for ids in out.values_mut() {
            ids.sort();
        }
        out
    }

    pub fn counts(&self) -> BTreeMap<String, BTreeMap<String, usize>> {
        self.set_ids()
            .into_iter()
            .map(|set_id| {
                let counts = self.partitions(&set_id).into_iter().map(|(k, v)| (k, v.len())).collect();
                (set_id, counts)
            })
            .collect()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "ok": self.ok(),
            "sets": self.set_ids(),
            "counts": self.counts(),
            "conflicts": self.conflicts.iter().map(Conflict::to_json).collect::<Vec<_>>(),
            "leaks": self.leaks.iter().map(Leak::to_json).collect::<Vec<_>>(),
            "unclaimed": self.unclaimed,
            "unreadable": self.unreadable.iter().map(|(p, e)| json!({"path": p, "error": e})).collect::<Vec<_>>(),
        })
    }
}

/// Read every file's claims and cross-check them (§12.3).
pub fn audit_splits<P: AsRef<Path>>(paths: &[P]) -> SplitAudit {
    let mut memberships = Vec::new();
    let mut unclaimed = Vec::new();
    let mut unreadable = Vec::new();
    for path in paths {
        let path = path.as_ref();
        let text = path.to_string_lossy().into_owned();
        match memberships_of(path) {
            Err(e) => unreadable.push((text, e.python_line())),
            Ok(found) => {
                if found.is_empty() {
                    unclaimed.push(text);
                }
                memberships.extend(found);
            }
        }
    }
    unclaimed.sort();
    unreadable.sort();
    let conflicts = conflicts(&memberships);
    let leaks = leaks(&memberships);
    SplitAudit { memberships, conflicts, leaks, unclaimed, unreadable }
}

/// Every claim in a file, whether it holds one sample or many.
pub fn memberships_of(path: &Path) -> Result<Vec<Membership>> {
    let text = path.to_string_lossy().into_owned();
    let samples: Vec<(String, Sample)> = match open_any(path, None)? {
        AnyFile::Sample(s) => vec![(String::new(), s)],
        AnyFile::Collection(c) => c.keys()?.into_iter().map(|k| Ok((k.clone(), c.get(&k)?))).collect::<Result<_>>()?,
    };
    let mut out = Vec::new();
    for (key, sample) in samples {
        let document = sample.document()?;
        let identity = &document.identity;
        let location = if key.is_empty() { text.clone() } else { format!("{text}::{key}") };
        for claim in &document.splits {
            out.push(Membership {
                path: location.clone(),
                sample_id: identity.sample_id.clone(),
                subject_id: identity.subject_id.clone(),
                group_id: document.cohort.grouping_key(&identity.subject_id).to_string(),
                claim: claim.clone(),
            });
        }
    }
    Ok(out)
}

fn conflicts(memberships: &[Membership]) -> Vec<Conflict> {
    let mut by_set: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
    for member in memberships {
        let Some(digest) = member.claim.manifest_sha256.as_deref().filter(|d| !d.is_empty()) else { continue };
        by_set
            .entry(member.claim.set_id.clone())
            .or_default()
            .entry(digest.to_string())
            .or_default()
            .push(member.path.clone());
    }
    by_set
        .into_iter()
        .filter(|(_, by_manifest)| by_manifest.len() > 1)
        .map(|(set_id, by_manifest)| Conflict {
            set_id,
            manifests: by_manifest.keys().cloned().collect(),
            paths_by_manifest: by_manifest
                .into_iter()
                .map(|(k, mut v)| {
                    v.sort();
                    (k, v)
                })
                .collect(),
        })
        .collect()
}

/// Grouping key -> every grouping key sharing anatomy with it, sorted.
///
/// `pairs` are `(subject_id, group_id)`.  The unit is the connected component
/// of subjects and keys (union--find): two visits of one subject curated under
/// two `group_id` values are one unit, which grouping by the key alone misses.
pub fn anatomy_units<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> BTreeMap<String, Vec<String>> {
    type Node = (bool, String); // (is_subject, name); subjects sort after groups like ("group" < "subject")
    let mut parent: HashMap<Node, Node> = HashMap::new();
    let mut order: Vec<Node> = Vec::new();
    fn find(parent: &mut HashMap<Node, Node>, order: &mut Vec<Node>, node: Node) -> Node {
        if !parent.contains_key(&node) {
            parent.insert(node.clone(), node.clone());
            order.push(node.clone());
        }
        let mut node = node;
        loop {
            let up = parent[&node].clone();
            if up == node {
                return node;
            }
            let grand = parent[&up].clone();
            parent.insert(node.clone(), grand.clone());
            node = grand;
        }
    }
    for (subject, group) in pairs {
        let a = find(&mut parent, &mut order, (true, subject.to_string()));
        let b = find(&mut parent, &mut order, (false, group.to_string()));
        if a != b {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            parent.insert(hi, lo);
        }
    }
    let mut members: Vec<(Node, Vec<String>)> = Vec::new();
    for node in order.clone() {
        if !node.0 {
            let root = find(&mut parent, &mut order, node.clone());
            match members.iter_mut().find(|(r, _)| *r == root) {
                Some((_, unit)) => unit.push(node.1.clone()),
                None => members.push((root, vec![node.1.clone()])),
            }
        }
    }
    let mut out = BTreeMap::new();
    for (_, mut unit) in members {
        unit.sort();
        for group in &unit {
            out.insert(group.clone(), unit.clone());
        }
    }
    out
}

fn leaks(memberships: &[Membership]) -> Vec<Leak> {
    let units = anatomy_units(memberships.iter().map(|m| (m.subject_id.as_str(), m.group_id.as_str())));
    let mut by_unit: BTreeMap<(String, Vec<String>), Vec<&Membership>> = BTreeMap::new();
    for member in memberships {
        let unit = units.get(&member.group_id).cloned().unwrap_or_else(|| vec![member.group_id.clone()]);
        by_unit.entry((member.claim.set_id.clone(), unit)).or_default().push(member);
    }
    let mut out = Vec::new();
    for ((set_id, groups), members) in by_unit {
        let partitions: Vec<String> =
            members.iter().map(|m| m.claim.partition.clone()).collect::<BTreeSet<_>>().into_iter().collect();
        if partitions.len() > 1 {
            let mut paths: Vec<String> = members.iter().map(|m| m.path.clone()).collect();
            paths.sort();
            out.push(Leak {
                set_id,
                group_id: groups[0].clone(),
                partitions,
                paths,
                subjects: members.iter().map(|m| m.subject_id.clone()).collect::<BTreeSet<_>>().into_iter().collect(),
                groups,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_join_keys_through_a_shared_subject() {
        let units = anatomy_units([("s1", "g1"), ("s1", "g2"), ("s2", "g3")]);
        assert_eq!(units["g1"], vec!["g1", "g2"]);
        assert_eq!(units["g2"], vec!["g1", "g2"]);
        assert_eq!(units["g3"], vec!["g3"]);
    }
}

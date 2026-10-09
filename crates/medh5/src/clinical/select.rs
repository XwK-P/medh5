//! Revision chains and prospective selection at a cutoff (1.1 §9;
//! task-and-cache contract §4).
//!
//! What a model may see at cutoff `c` is decided here, once, for every
//! frontend.  Strictly:
//!
//! 1. only event versions whose availability is *known* and at or before `c`;
//! 2. per record, the newest such version along its `supersedes` chain ---
//!    unless it was entered in error; a newer version that is *definitely*
//!    after `c` leaves it usable, one whose availability is unknown or
//!    straddles `c` makes the whole selection **uncertifiable**, because the
//!    record's absence would then depend on history after the cutoff;
//! 3. the task's kinds, context window, plan and static policy;
//! 4. only the links and payloads those versions attest.
//!
//! Stable ids break ties for *storage*; they are not evidence of order.  Two
//! events whose ordering times overlap share a tie group, and an event limit
//! says what it does with the group it cuts.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde_json::{json, Map, Value};

use super::model::{Bounds, Event, Link, EVENT_KINDS};
use crate::json::{repr_list, repr_str};
use crate::{Error, Result};

/// Selection policies: the strict one, and its one named alternative.
pub const POLICIES: [&str; 2] = ["strict_prospective", "latest_provable"];
/// What events are ordered by.
pub const ORDER_BY: [&str; 2] = ["effective", "available"];
/// How the context window's lower edge is treated.
pub const BOUNDARIES: [&str; 2] = ["closed", "open"];
/// What an uncertain ordering time must do to count as inside the window.
pub const UNCERTAINTY: [&str; 2] = ["contained", "overlaps"];
/// Which end of the sequence an event limit keeps.
pub const KEEP: [&str; 2] = ["latest", "earliest"];
/// What an event limit does with the tie group it cuts.
pub const TIES: [&str; 2] = ["keep_group", "drop_group"];

/// The versions of each record, oldest first, ordered by `supersedes`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Chains {
    pub records: BTreeMap<String, Vec<String>>,
}

/// Every record's versions, oldest first, as indices into `events`, in
/// record-id order.  The chains must be valid (no E816).
fn chain_indices<'a>(
    events: &'a [Event],
    by_id: &HashMap<&'a str, usize>,
    links: impl IntoIterator<Item = &'a Link>,
) -> Result<Vec<Vec<usize>>> {
    let mut versions: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, e) in events.iter().enumerate() {
        versions.entry(e.record_id.as_str()).or_default().push(i);
    }
    let mut next: HashMap<usize, usize> = HashMap::new();
    let mut previous: HashMap<usize, usize> = HashMap::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for l in links {
        if l.relation != "supersedes" || l.source_type != "event" || l.target_type != "event" {
            continue;
        }
        let (Some(&new), Some(&old)) = (by_id.get(l.source_id.as_str()), by_id.get(l.target_id.as_str())) else {
            continue;
        };
        if !seen.insert((new, old)) {
            continue; // the same link from another fragment
        }
        if next.insert(old, new).is_some() || previous.insert(new, old).is_some() {
            return Err(Error::coded(
                "E816",
                format!("the revision chain through {} branches", repr_str(&l.target_id)),
            ));
        }
    }
    let mut out = Vec::with_capacity(versions.len());
    for (record, members) in versions {
        let roots: Vec<usize> = members.iter().copied().filter(|v| !previous.contains_key(v)).collect();
        let disjoint = || {
            Error::coded(
                "E816",
                format!("record {} has {} versions that do not form one chain", repr_str(record), members.len()),
            )
        };
        if roots.len() != 1 {
            return Err(disjoint());
        }
        let mut chain = vec![roots[0]];
        while let Some(&n) = next.get(chain.last().expect("a root")) {
            if chain.len() > members.len() {
                return Err(Error::coded("E816", format!("the revision chain of {} is cyclic", repr_str(record))));
            }
            chain.push(n);
        }
        if chain.len() != members.len() {
            return Err(disjoint());
        }
        out.push(chain);
    }
    Ok(out)
}

fn index_of(events: &[Event]) -> HashMap<&str, usize> {
    events.iter().enumerate().map(|(i, e)| (e.event_id.as_str(), i)).collect()
}

impl Chains {
    /// Order every record's versions.  The chains must be valid (no E816).
    pub fn build<'a>(events: &'a [Event], links: impl IntoIterator<Item = &'a Link>) -> Result<Chains> {
        let by_id = index_of(events);
        let records = chain_indices(events, &by_id, links)?
            .into_iter()
            .map(|chain| {
                let record = events[chain[0]].record_id.clone();
                (record, chain.into_iter().map(|i| events[i].event_id.clone()).collect())
            })
            .collect();
        Ok(Chains { records })
    }

    /// A record's versions, oldest first.
    pub fn versions(&self, record_id: &str) -> &[String] {
        self.records.get(record_id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// The newest version of every record --- the full history's view.
    pub fn latest(&self) -> impl Iterator<Item = &str> {
        self.records.values().filter_map(|c| c.last().map(String::as_str))
    }
}

/// How inputs are chosen at a cutoff (task-and-cache contract §3.4).
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionPolicy {
    /// `strict_prospective` or `latest_provable`.
    pub selection: String,
    /// `effective` or `available`.
    pub order_by: String,
    /// The context window's length before the cutoff; `None` for all history.
    pub context_us: Option<i64>,
    /// `closed` includes an event exactly at the window's lower edge.
    pub context_boundary: String,
    /// `contained` or `overlaps`: how an uncertain time meets the window.
    pub uncertainty: String,
    /// Event kinds allowed as input; `None` for all.
    pub kinds: Option<Vec<String>>,
    /// Admit planned and future-effective events, flagged as plans.
    pub plans: bool,
    /// Admit `static` events (they are never windowed).
    pub include_static: bool,
    /// At most this many timed events; `None` for no limit.
    pub max_events: Option<usize>,
    /// `latest` or `earliest`.
    pub keep: String,
    /// `keep_group` or `drop_group`.
    pub ties: String,
}

impl Default for SelectionPolicy {
    fn default() -> Self {
        SelectionPolicy {
            selection: "strict_prospective".into(),
            order_by: "effective".into(),
            context_us: None,
            context_boundary: "closed".into(),
            uncertainty: "contained".into(),
            kinds: None,
            plans: false,
            include_static: true,
            max_events: None,
            keep: "latest".into(),
            ties: "keep_group".into(),
        }
    }
}

fn choose(value: &str, field: &str, allowed: &[&str]) -> Result<()> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(Error::invalid(format!("policy `{field}` is {}, not one of {}", repr_str(value), repr_list(allowed))))
    }
}

impl SelectionPolicy {
    /// The strict policy with everything else at its default.
    pub fn strict() -> SelectionPolicy {
        SelectionPolicy::default()
    }

    pub fn check(&self) -> Result<()> {
        choose(&self.selection, "selection", &POLICIES)?;
        choose(&self.order_by, "order_by", &ORDER_BY)?;
        choose(&self.context_boundary, "context_boundary", &BOUNDARIES)?;
        choose(&self.uncertainty, "uncertainty", &UNCERTAINTY)?;
        choose(&self.keep, "keep", &KEEP)?;
        choose(&self.ties, "ties", &TIES)?;
        if let Some(w) = self.context_us {
            if w < 0 {
                return Err(Error::invalid("policy `context_us` is a length and cannot be negative"));
            }
        }
        if let Some(kinds) = &self.kinds {
            for k in kinds {
                choose(k, "kinds", &EVENT_KINDS)?;
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        json!({
            "selection": self.selection,
            "order_by": self.order_by,
            "context_us": self.context_us,
            "context_boundary": self.context_boundary,
            "uncertainty": self.uncertainty,
            "kinds": self.kinds,
            "plans": self.plans,
            "static": self.include_static,
            "max_events": self.max_events,
            "keep": self.keep,
            "ties": self.ties,
        })
    }

    /// Parse a policy object; every member is optional and defaults as above.
    pub fn from_json(value: &Value) -> Result<SelectionPolicy> {
        let map = value.as_object().ok_or_else(|| Error::invalid("a selection policy is a JSON object"))?;
        const FIELDS: [&str; 11] = [
            "selection",
            "order_by",
            "context_us",
            "context_boundary",
            "uncertainty",
            "kinds",
            "plans",
            "static",
            "max_events",
            "keep",
            "ties",
        ];
        if let Some(k) = map.keys().find(|k| !FIELDS.contains(&k.as_str())) {
            return Err(Error::invalid(format!("unknown policy field {}", repr_str(k))));
        }
        let d = SelectionPolicy::default();
        let text = |k: &str, default: &str| -> Result<String> {
            match map.get(k) {
                None | Some(Value::Null) => Ok(default.to_string()),
                Some(Value::String(s)) => Ok(s.clone()),
                Some(_) => Err(Error::invalid(format!("policy `{k}` must be a string"))),
            }
        };
        let flag = |k: &str, default: bool| -> Result<bool> {
            match map.get(k) {
                None | Some(Value::Null) => Ok(default),
                Some(Value::Bool(b)) => Ok(*b),
                Some(_) => Err(Error::invalid(format!("policy `{k}` must be a boolean"))),
            }
        };
        let policy = SelectionPolicy {
            selection: text("selection", &d.selection)?,
            order_by: text("order_by", &d.order_by)?,
            context_us: match map.get("context_us") {
                None | Some(Value::Null) => None,
                Some(v) => Some(v.as_i64().ok_or_else(|| Error::invalid("policy `context_us` must be an integer"))?),
            },
            context_boundary: text("context_boundary", &d.context_boundary)?,
            uncertainty: text("uncertainty", &d.uncertainty)?,
            kinds: match map.get("kinds") {
                None | Some(Value::Null) => None,
                Some(Value::Array(items)) => Some(
                    items
                        .iter()
                        .map(|v| {
                            v.as_str().map(str::to_string).ok_or_else(|| Error::invalid("policy `kinds` lists strings"))
                        })
                        .collect::<Result<_>>()?,
                ),
                Some(_) => return Err(Error::invalid("policy `kinds` must be a list")),
            },
            plans: flag("plans", d.plans)?,
            include_static: flag("static", d.include_static)?,
            max_events: match map.get("max_events") {
                None | Some(Value::Null) => None,
                Some(v) => Some(
                    v.as_u64().ok_or_else(|| Error::invalid("policy `max_events` must be a non-negative integer"))?
                        as usize,
                ),
            },
            keep: text("keep", &d.keep)?,
            ties: text("ties", &d.ties)?,
        };
        policy.check()?;
        Ok(policy)
    }
}

/// One event version a selection admits, in input order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selected {
    /// The version: an index into the events selected from.
    pub index: usize,
    /// The time the sequence is ordered by: effective start or availability;
    /// `None` for a static event.
    pub order: Option<Bounds>,
    /// Events whose ordering times overlap share a group: their relative order
    /// is unknown, and nothing here invents one.
    pub tie_group: usize,
    /// Admitted as a plan: planned, or not yet started at the cutoff.
    pub plan: bool,
}

impl Selected {
    pub fn to_json(&self, events: &[Event]) -> Value {
        self.to_json_as(names(&events[self.index]))
    }

    /// [`Selected::to_json`] for a version named `[event_id, record_id, kind]`.
    pub fn to_json_as(&self, [event_id, record_id, kind]: [&str; 3]) -> Value {
        json!({
            "event_id": event_id,
            "record_id": record_id,
            "kind": kind,
            "order_us": self.order.map(|b| b.to_json()),
            "tie_group": self.tie_group,
            "plan": self.plan,
        })
    }
}

/// What a selection's JSON names a version by: `[event_id, record_id, kind]`.
pub fn names(event: &Event) -> [&str; 3] {
    [&event.event_id, &event.record_id, &event.kind]
}

/// What a cutoff admits.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub cutoff_us: i64,
    pub policy: String,
    /// `certified` (the strict guarantee holds), `uncertifiable` (a later
    /// revision's availability is unknown or straddles the cutoff: exclude
    /// the row), or `provable` (`latest_provable`: the newest *provably*
    /// available versions, with no claim that they were the source's newest).
    pub status: String,
    pub events: Vec<Selected>,
    /// Indices into the links given, of those the selection attests.
    pub links: Vec<usize>,
    /// `(fragment, type, id)` of every payload the inputs may read.
    pub payloads: BTreeSet<(usize, String, String)>,
    /// Records whose later revision made the row uncertifiable (or, under
    /// `latest_provable`, was ignored).
    pub uncertain_records: Vec<String>,
    /// Why event versions stayed out, and how many.
    pub excluded: BTreeMap<String, usize>,
}

impl Selection {
    pub fn certified(&self) -> bool {
        self.status == "certified"
    }

    /// The admitted versions' ids, in input order; `events` is what was
    /// selected from.
    pub fn event_ids<'a>(&self, events: &'a [Event]) -> Vec<&'a str> {
        self.events.iter().map(|s| events[s.index].event_id.as_str()).collect()
    }

    /// Whether `(fragment, kind, id)` may be read as input.
    pub fn admits(&self, fragment: usize, kind: &str, id: &str) -> bool {
        self.payloads.contains(&(fragment, kind.to_string(), id.to_string()))
    }

    pub fn to_json(&self, events: &[Event]) -> Value {
        self.to_json_named(&|i| names(&events[i]))
    }

    /// [`Selection::to_json`], naming the version at each index through
    /// `name` --- for a caller that kept the names and not the events.
    pub fn to_json_named<'a>(&self, name: &dyn Fn(usize) -> [&'a str; 3]) -> Value {
        let payloads: Vec<Value> = self.payloads.iter().map(|(f, k, i)| json!([f, k, i])).collect();
        let mut excluded = Map::new();
        for (k, v) in &self.excluded {
            excluded.insert(k.clone(), json!(v));
        }
        json!({
            "cutoff_us": self.cutoff_us,
            "policy": self.policy,
            "status": self.status,
            "events": self.events.iter().map(|s| s.to_json_as(name(s.index))).collect::<Vec<_>>(),
            "links": self.links,
            "payloads": payloads,
            "uncertain_records": self.uncertain_records,
            "excluded": excluded,
        })
    }
}

fn count(excluded: &mut BTreeMap<String, usize>, reason: &str) {
    *excluded.entry(reason.to_string()).or_default() += 1;
}

/// How a link can attest a payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Attests {
    /// The structural link from a `document` or `imaging` event to its own
    /// payload, attested when its source is selected.
    Structure { source: usize },
    /// A link carrying `asserted_by_event_id`, attested when that version is
    /// selected and its event and document endpoints are admitted too.
    Assertion { by: usize, source: Option<usize>, target: Option<usize> },
}

/// One subject's event versions and links, prepared once for selection at
/// any number of cutoffs: revision chains ordered, versions indexed, links
/// classified, and a stable rank standing in for each event id when ties
/// are broken for storage.
#[derive(Debug, Clone)]
pub struct Prepared<'a> {
    events: &'a [Event],
    links: Vec<(usize, &'a Link)>,
    chains: Vec<Vec<usize>>,
    /// The position of each event's id in id order.
    rank: Vec<usize>,
    attests: Vec<(usize, Attests)>,
}

impl<'a> Prepared<'a> {
    /// Prepare merged event versions (unique by id, reconciled across
    /// fragments first) and their links, each with the fragment it came from.
    pub fn new(events: &'a [Event], links: &[(usize, &'a Link)]) -> Result<Prepared<'a>> {
        let by_id = index_of(events);
        let chains = chain_indices(events, &by_id, links.iter().map(|(_, l)| *l))?;
        let mut order: Vec<usize> = (0..events.len()).collect();
        order.sort_unstable_by(|a, b| events[*a].event_id.cmp(&events[*b].event_id));
        let mut rank = vec![0; events.len()];
        for (r, i) in order.into_iter().enumerate() {
            rank[i] = r;
        }
        let event = |kind: &str, id: &str| if kind == "event" { by_id.get(id).copied() } else { None };
        let mut attests = Vec::new();
        for (i, (_, l)) in links.iter().enumerate() {
            let source = event(&l.source_type, &l.source_id);
            let structural = l.relation == "describes"
                && source.is_some_and(|s| {
                    matches!(
                        (events[s].kind.as_str(), l.target_type.as_str()),
                        ("document", "document") | ("imaging", "image")
                    )
                });
            if structural {
                attests.push((i, Attests::Structure { source: source.expect("structural") }));
            } else if l.relation != "supersedes" {
                if let Some(by) = l.asserted_by_event_id.as_deref().and_then(|b| by_id.get(b).copied()) {
                    let target = event(&l.target_type, &l.target_id);
                    attests.push((i, Attests::Assertion { by, source, target }));
                }
            }
        }
        Ok(Prepared { events, links: links.to_vec(), chains, rank, attests })
    }

    pub fn events(&self) -> &'a [Event] {
        self.events
    }

    /// The newest version of every record, oldest record id first.
    pub fn latest(&self) -> impl Iterator<Item = usize> + '_ {
        self.chains.iter().filter_map(|c| c.last().copied())
    }

    /// Select at `cutoff` (1.1 §9).
    pub fn select(&self, cutoff: i64, policy: &SelectionPolicy) -> Result<Selection> {
        policy.check()?;
        let events = self.events;
        let mut excluded = BTreeMap::new();
        let mut uncertain_records = Vec::new();
        let mut current: Vec<usize> = Vec::new();

        // 1-2: per record, the newest version available at the cutoff.
        for chain in &self.chains {
            let available_at = |i: &usize| events[*i].available.is_some_and(|a| a.at_or_before(cutoff));
            let Some(k) = chain.iter().rposition(available_at) else {
                let reason = if chain.iter().any(|i| events[*i].available.is_none()) {
                    "unknown_availability"
                } else if chain.iter().any(|i| events[*i].available.is_some_and(|a| a.straddles(cutoff))) {
                    "straddles_cutoff"
                } else {
                    "after_cutoff"
                };
                count(&mut excluded, reason);
                continue;
            };
            if chain[k + 1..].iter().any(|i| !events[*i].available.is_some_and(|a| a.after(cutoff))) {
                uncertain_records.push(events[chain[k]].record_id.clone());
            }
            if events[chain[k]].status == "entered_in_error" {
                count(&mut excluded, "entered_in_error");
                continue;
            }
            current.push(chain[k]);
        }
        let status = match (policy.selection.as_str(), uncertain_records.is_empty()) {
            ("latest_provable", _) => "provable",
            (_, true) => "certified",
            (_, false) => "uncertifiable",
        };

        // 3: kinds, plans, static, context.
        let mut admitted: Vec<(Option<Bounds>, bool, usize)> = Vec::new();
        let window_lo = policy.context_us.map(|w| cutoff.saturating_sub(w));
        for i in current {
            let e = &events[i];
            if let Some(kinds) = &policy.kinds {
                if !kinds.contains(&e.kind) {
                    count(&mut excluded, "kind");
                    continue;
                }
            }
            if e.temporal_type == "static" {
                if !policy.include_static {
                    count(&mut excluded, "static");
                    continue;
                }
                let order = if policy.order_by == "available" { e.available } else { None };
                admitted.push((order, false, i));
                continue;
            }
            let order = match policy.order_by.as_str() {
                "available" => e.available,
                _ => e.effective_start,
            };
            let Some(order) = order else {
                count(&mut excluded, "unknown_time");
                continue;
            };
            let started = e.effective_start.is_some_and(|s| s.at_or_before(cutoff));
            let plan = e.status == "planned" || (e.temporal_type != "unknown" && !started);
            if plan {
                if !policy.plans {
                    count(&mut excluded, "plan");
                    continue;
                }
                admitted.push((Some(order), true, i));
                continue;
            }
            if let Some(lo_edge) = window_lo {
                let inside = |t: i64| if policy.context_boundary == "closed" { t >= lo_edge } else { t > lo_edge };
                let (fully, partly) = (inside(order.lo), inside(order.hi));
                let ok = if policy.uncertainty == "contained" { fully } else { partly };
                if !ok {
                    count(&mut excluded, if partly { "uncertain_context" } else { "outside_context" });
                    continue;
                }
            }
            admitted.push((Some(order), false, i));
        }

        // 4: order, tie groups, and the limit.  Ids break ties for storage
        // only; their rank stands in for them.
        admitted.sort_unstable_by_key(|(order, _, i)| (order.is_some(), order.map(|o| (o.lo, o.hi)), self.rank[*i]));
        // Static events (sorted first) share group 0: they have no order at all.
        // Timed events open a new group unless their time overlaps the group's
        // reach so far.
        let mut selected: Vec<Selected> = Vec::with_capacity(admitted.len());
        let has_static = admitted.first().is_some_and(|a| a.0.is_none());
        let mut group = 0usize;
        let mut reach: Option<i64> = None;
        for (order, plan, i) in &admitted {
            if let Some(o) = order {
                match reach {
                    Some(r) if o.lo <= r => reach = Some(r.max(o.hi)),
                    Some(_) => {
                        group += 1;
                        reach = Some(o.hi);
                    }
                    None => {
                        group = usize::from(has_static);
                        reach = Some(o.hi);
                    }
                }
            }
            selected.push(Selected { index: *i, order: *order, tie_group: group, plan: *plan });
        }
        if let Some(max) = policy.max_events {
            let timed: Vec<usize> = (0..selected.len()).filter(|i| selected[*i].order.is_some()).collect();
            if timed.len() > max {
                // A limit of zero keeps no timed event, so no tie group
                // straddles it (the schema admits 0; static events stay).
                let mut keep: BTreeSet<usize> = BTreeSet::new();
                if max > 0 {
                    let (keep_range, cut_at) = if policy.keep == "latest" {
                        let first_kept = timed.len() - max;
                        (first_kept..timed.len(), first_kept)
                    } else {
                        (0..max, max - 1)
                    };
                    let boundary = selected[timed[cut_at]].tie_group;
                    keep = keep_range.map(|j| timed[j]).collect();
                    let in_boundary: Vec<usize> =
                        timed.iter().copied().filter(|i| selected[*i].tie_group == boundary).collect();
                    let splits =
                        in_boundary.iter().any(|i| keep.contains(i)) && in_boundary.iter().any(|i| !keep.contains(i));
                    if splits {
                        if policy.ties == "keep_group" {
                            keep.extend(in_boundary);
                        } else {
                            for i in in_boundary {
                                keep.remove(&i);
                            }
                        }
                    }
                }
                let before = selected.len();
                selected = selected
                    .into_iter()
                    .enumerate()
                    .filter(|(i, s)| s.order.is_none() || keep.contains(i))
                    .map(|(_, s)| s)
                    .collect();
                *excluded.entry("event_limit".into()).or_default() += before - selected.len();
            }
        }

        // 5: the links and payloads the selected versions attest.
        let mut chosen = vec![false; events.len()];
        for s in &selected {
            chosen[s.index] = true;
        }
        let mut payloads: BTreeSet<(usize, String, String)> = BTreeSet::new();
        let mut attested = Vec::new();
        // Structural links first: a document or imaging event owns its payload.
        for (i, how) in &self.attests {
            if let Attests::Structure { source } = how {
                if chosen[*source] {
                    let (fragment, l) = self.links[*i];
                    payloads.insert((fragment, l.target_type.clone(), l.target_id.clone()));
                    attested.push(*i);
                }
            }
        }
        let documents: BTreeSet<(usize, &str)> =
            payloads.iter().filter(|(_, k, _)| k == "document").map(|(f, _, id)| (*f, id.as_str())).collect();
        let mut asserted: Vec<(usize, String, String)> = Vec::new();
        for (i, how) in &self.attests {
            let Attests::Assertion { by, source, target } = how else { continue };
            if !chosen[*by] {
                continue;
            }
            let (fragment, l) = self.links[*i];
            let endpoint_ok = |kind: &str, id: &str, event: &Option<usize>| match kind {
                "event" => event.is_some_and(|e| chosen[e]),
                "document" => documents.contains(&(fragment, id)),
                _ => true,
            };
            if !(endpoint_ok(&l.source_type, &l.source_id, source) && endpoint_ok(&l.target_type, &l.target_id, target))
            {
                continue;
            }
            for (kind, id) in [(&l.source_type, &l.source_id), (&l.target_type, &l.target_id)] {
                if matches!(kind.as_str(), "image" | "annotation" | "transform" | "grid" | "instance" | "document") {
                    asserted.push((fragment, kind.clone(), id.clone()));
                }
            }
            if let Some(ann) = &l.target_annotation_id {
                asserted.push((fragment, "annotation".into(), ann.clone()));
            }
            attested.push(*i);
        }
        payloads.extend(asserted);
        attested.sort_unstable();
        Ok(Selection {
            cutoff_us: cutoff,
            policy: policy.selection.clone(),
            status: status.into(),
            events: selected,
            links: attested,
            payloads,
            uncertain_records,
            excluded,
        })
    }
}

/// Select at `cutoff` from merged event versions and their links.
///
/// `links` pairs each link with the fragment (source) it came from, so the
/// payloads it attests are named in that fragment.  The events must be unique
/// by id (reconciled across fragments first) and their chains valid.  To
/// select one history at many cutoffs, prepare it once ([`Prepared`]).
pub fn select(events: &[Event], links: &[(usize, &Link)], cutoff: i64, policy: &SelectionPolicy) -> Result<Selection> {
    Prepared::new(events, links)?.select(cutoff, policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clinical::model::HOUR;

    fn version(id: &str, record: &str, effective: i64, available: Option<(i64, i64)>) -> Event {
        Event {
            event_id: id.into(),
            record_id: record.into(),
            kind: "observation".into(),
            temporal_type: "point".into(),
            effective_start: Some(Bounds::exact(effective)),
            available: available.map(|(a, b)| Bounds::new(a, b)),
            status: "final".into(),
            ..Default::default()
        }
    }

    fn sup(new: &str, old: &str) -> Link {
        Link::new(("event", new), "supersedes", ("event", old))
    }

    fn ids(s: &Selection, events: &[Event]) -> Vec<String> {
        s.event_ids(events).into_iter().map(str::to_string).collect()
    }

    #[test]
    fn s9_1_a_definitely_later_revision_leaves_the_earlier_usable() {
        let events = vec![
            version("r_v1", "r", 0, Some((4 * HOUR, 4 * HOUR))),
            version("r_v2", "r", 0, Some((48 * HOUR, 48 * HOUR))),
        ];
        let link = sup("r_v2", "r_v1");
        let s = select(&events, &[(0, &link)], 24 * HOUR, &SelectionPolicy::strict()).unwrap();
        assert_eq!(ids(&s, &events), ["r_v1"]);
        assert!(s.certified());
        let s = select(&events, &[(0, &link)], 48 * HOUR, &SelectionPolicy::strict()).unwrap();
        assert_eq!(ids(&s, &events), ["r_v2"]);
    }

    #[test]
    fn s9_1_an_uncertain_later_revision_makes_the_row_uncertifiable() {
        for later in [None, Some((20 * HOUR, 30 * HOUR))] {
            let events = vec![version("v1", "r", 0, Some((HOUR, HOUR))), version("v2", "r", 0, later)];
            let link = sup("v2", "v1");
            let s = select(&events, &[(0, &link)], 24 * HOUR, &SelectionPolicy::strict()).unwrap();
            assert_eq!(s.status, "uncertifiable");
            assert_eq!(s.uncertain_records, ["r"]);
            let mut provable = SelectionPolicy::strict();
            provable.selection = "latest_provable".into();
            let s = select(&events, &[(0, &link)], 24 * HOUR, &provable).unwrap();
            assert_eq!(s.status, "provable");
            assert_eq!(ids(&s, &events), ["v1"]);
        }
    }

    #[test]
    fn s9_1_unknown_availability_is_never_assumed() {
        let events = vec![version("lab", "lab", -48 * HOUR, None)];
        let s = select(&events, &[], 24 * HOUR, &SelectionPolicy::strict()).unwrap();
        assert!(s.events.is_empty());
        assert_eq!(s.excluded.get("unknown_availability"), Some(&1));
        assert!(s.certified());
    }

    #[test]
    fn s9_1_entered_in_error_withdraws_the_record() {
        let mut v2 = version("v2", "r", 0, Some((2, 2)));
        v2.status = "entered_in_error".into();
        let events = vec![version("v1", "r", 0, Some((1, 1))), v2];
        let link = sup("v2", "v1");
        let s = select(&events, &[(0, &link)], 10, &SelectionPolicy::strict()).unwrap();
        assert!(s.events.is_empty());
    }

    #[test]
    fn s9_1_ties_are_groups_and_limits_say_what_they_cut() {
        // a and b overlap in time; c is later.
        let mut a = version("a", "a", 0, Some((0, 0)));
        a.effective_start = Some(Bounds::new(0, 10));
        let b = version("b", "b", 5, Some((5, 5)));
        let c = version("c", "c", 20, Some((20, 20)));
        let events = vec![c, b, a];
        let s = select(&events, &[], 100, &SelectionPolicy::strict()).unwrap();
        assert_eq!(ids(&s, &events), ["a", "b", "c"]);
        assert_eq!(s.events[0].tie_group, s.events[1].tie_group);
        assert_ne!(s.events[1].tie_group, s.events[2].tie_group);
        let mut limited = SelectionPolicy::strict();
        limited.max_events = Some(2);
        let kept = select(&events, &[], 100, &limited).unwrap();
        assert_eq!(ids(&kept, &events), ["a", "b", "c"], "keep_group keeps the whole boundary group");
        limited.ties = "drop_group".into();
        let dropped = select(&events, &[], 100, &limited).unwrap();
        assert_eq!(ids(&dropped, &events), ["c"]);
        assert_eq!(dropped.excluded.get("event_limit"), Some(&2));
    }

    /// C01: the schema admits `max_events = 0`, which indexed one past the
    /// kept range (`keep = latest`) or underflowed `max - 1` (`earliest`).
    #[test]
    fn s9_1_a_limit_of_zero_keeps_no_timed_event() {
        let timed = version("t", "t", 5, Some((5, 5)));
        let mut fact = version("s", "s", 0, Some((1, 1)));
        fact.temporal_type = "static".into();
        fact.effective_start = None;
        for keep in ["latest", "earliest"] {
            for ties in ["keep_group", "drop_group"] {
                let mut policy = SelectionPolicy::strict();
                policy.max_events = Some(0);
                policy.keep = keep.into();
                policy.ties = ties.into();
                let events = vec![timed.clone(), fact.clone()];
                let s = select(&events, &[], 100, &policy).unwrap();
                assert_eq!(ids(&s, &events), ["s"], "{keep} {ties}");
                assert_eq!(s.excluded.get("event_limit"), Some(&1));
                let only = select(&events[1..], &[], 100, &policy).unwrap();
                assert_eq!(ids(&only, &events[1..]), ["s"]);
            }
        }
    }

    #[test]
    fn s9_1_plans_are_never_completed_outcomes() {
        let mut order = version("order", "order", 50, Some((10, 10)));
        order.kind = "medication_order".into();
        order.status = "planned".into();
        let s = select(&[order.clone()], &[], 20, &SelectionPolicy::strict()).unwrap();
        assert!(s.events.is_empty());
        assert_eq!(s.excluded.get("plan"), Some(&1));
        let mut with_plans = SelectionPolicy::strict();
        with_plans.plans = true;
        let s = select(&[order], &[], 20, &with_plans).unwrap();
        assert!(s.events[0].plan);
    }

    #[test]
    fn s9_1_the_context_window_respects_uncertainty() {
        let mut coarse = version("coarse", "coarse", 0, Some((0, 0)));
        coarse.effective_start = Some(Bounds::new(-20, 0));
        let mut policy = SelectionPolicy::strict();
        policy.context_us = Some(10);
        let s = select(&[coarse.clone()], &[], 5, &policy).unwrap();
        assert!(s.events.is_empty());
        assert_eq!(s.excluded.get("uncertain_context"), Some(&1));
        policy.uncertainty = "overlaps".into();
        let s = select(&[coarse], &[], 5, &policy).unwrap();
        assert_eq!(s.events.len(), 1);
    }

    #[test]
    fn s9_1_payloads_come_only_through_selected_versions() {
        let mut ct0 = version("ct0", "ct0", 0, Some((HOUR, HOUR)));
        ct0.kind = "imaging".into();
        let mut ct1 = version("ct1", "ct1", 2160 * HOUR, Some((2161 * HOUR, 2161 * HOUR)));
        ct1.kind = "imaging".into();
        let mut grounding = version("ground", "ground", 0, Some((2200 * HOUR, 2200 * HOUR)));
        grounding.kind = "other".into();
        let l0 = Link::new(("event", "ct0"), "describes", ("image", "CT_tp0"));
        let l1 = Link::new(("event", "ct1"), "describes", ("image", "CT_tp1"));
        let mut late = Link::new(("event", "ct0"), "describes", ("annotation", "lesions_tp0"));
        late.asserted_by_event_id = Some("ground".into());
        let links = [(0, &l0), (0, &l1), (0, &late)];
        let s = select(&[ct0, ct1, grounding], &links, 24 * HOUR, &SelectionPolicy::strict()).unwrap();
        assert!(s.admits(0, "image", "CT_tp0"));
        assert!(!s.admits(0, "image", "CT_tp1"));
        assert!(!s.admits(0, "annotation", "lesions_tp0"), "a later grounding is not available at baseline");
    }

    #[test]
    fn s9_1_one_preparation_answers_every_cutoff_as_select_does() {
        let mut events = Vec::new();
        let mut links = Vec::new();
        for i in 0..40i64 {
            events.push(version(
                &format!("e{i}"),
                &format!("r{}", i / 2),
                i * HOUR,
                Some(((i + 3) * HOUR, (i + 5) * HOUR)),
            ));
        }
        for i in (1..40).step_by(2) {
            links.push(sup(&format!("e{i}"), &format!("e{}", i - 1)));
        }
        let refs: Vec<(usize, &Link)> = links.iter().map(|l| (0, l)).collect();
        let prepared = Prepared::new(&events, &refs).unwrap();
        let mut policy = SelectionPolicy::strict();
        policy.context_us = Some(10 * HOUR);
        for cutoff in (-2..50).map(|h| h * HOUR) {
            let once = select(&events, &refs, cutoff, &policy).unwrap();
            assert_eq!(prepared.select(cutoff, &policy).unwrap(), once, "cutoff {cutoff}");
        }
    }
}

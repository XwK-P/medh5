//! The clinical half of [`SampleWriter`] (1.1 §3--§8, §10).
//!
//! Rows are checked as they arrive by the rules that need only the row; the
//! rules that need every row and the sample --- references, ownership,
//! revision chains --- are the validator's, which `commit` runs over the
//! finished file before it replaces anything.
//!
//! An amended file's clinical tables are copied through untouched until
//! something changes them; then they are read, changed and rewritten whole.
//! A `clinical` group that is not the profile's is never reinterpreted: adding
//! clinical records to such a file is refused (§10).
//!
//! What an event version the amended file held attests is fixed with it
//! (§7.3): the commit compares it with the file being amended.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::writer::{ClinicalSource, SampleWriter};
use crate::clinical::check::{check_document, check_event, check_link, parse_instance, Finding, SampleContext};
use crate::clinical::model::{ClinicalRecords, Clock, Descriptor, Document, Event, Link, GROUP};
use crate::clinical::table::{self, Clinical};
use crate::h5::file::open_read;
use crate::h5::{attrs, ops};
use crate::json::repr_str;
use crate::storage::codecs::resolve_profile;
use crate::{Error, Result};

fn refuse(findings: Vec<Finding>, what: &str) -> Result<()> {
    match findings.into_iter().find(|f| !f.code.starts_with('W')) {
        None => Ok(()),
        Some(f) => Err(Error::coded(f.code, format!("{what}: {}", f.message))),
    }
}

impl SampleWriter {
    /// The records this writer will write, read from an amended file the
    /// first time anything changes them.
    fn clinical_records(&mut self) -> Result<&mut ClinicalRecords> {
        if self.clinical.is_none() {
            match self.clinical_source {
                ClinicalSource::Foreign => {
                    return Err(Error::coded(
                        "E803",
                        "this sample's `clinical` group is not the clinical profile's (it holds no declared \
                         `medh5.clinical/1` descriptor); it was allowed as an extension, and is neither overwritten \
                         nor reinterpreted --- rename it, or start the profile in a sample without it (1.1 §10)",
                    ))
                }
                ClinicalSource::Inherited => {
                    let root = self.root()?;
                    let found = Clinical::open(&root, false)?
                        .ok_or_else(|| Error::coded("E801", "the inherited `clinical` group vanished"))?;
                    let records = found.records()?;
                    self.clinical_ids = (
                        records.events.iter().map(|e| e.event_id.clone()).collect(),
                        records.documents.iter().map(|d| d.document_id.clone()).collect(),
                    );
                    self.inherited_events =
                        records.events.iter().map(|e| (e.event_id.clone(), e.kind.clone())).collect();
                    self.clinical = Some(records);
                }
                ClinicalSource::Absent => {
                    return Err(Error::invalid(
                        "declare the subject clock first (`set_clock`): every clinical time is measured on it (1.1 §3)",
                    ))
                }
            }
        }
        Ok(self.clinical.as_mut().expect("loaded above"))
    }

    /// Declare the subject clock, starting the `clinical` profile.
    ///
    /// A sample's clock is fixed once it has one: changing it would
    /// reinterpret every time already written, so a different clock is
    /// refused, and the same one is a no-op.
    pub fn set_clock(&mut self, clock: Clock) -> Result<Descriptor> {
        let descriptor = Descriptor::new(clock);
        let errors = crate::clinical::schema::validate_descriptor(&descriptor.to_json());
        if let Some(first) = errors.first() {
            return Err(Error::coded("E802", format!("clinical clock: {first}")));
        }
        if self.clinical.is_none() && self.clinical_source == ClinicalSource::Absent {
            self.clinical = Some(ClinicalRecords::new(descriptor.clone()));
            return Ok(descriptor);
        }
        let records = self.clinical_records()?;
        if records.descriptor.clock != descriptor.clock {
            return Err(Error::coded(
                "E802",
                format!(
                    "this sample's clinical clock is {}; times already written are measured on it, so it cannot \
                     become {} (a manifest records verified conversions between clocks)",
                    repr_str(&records.descriptor.clock.id),
                    repr_str(&descriptor.clock.id)
                ),
            ));
        }
        Ok(records.descriptor.clone())
    }

    /// Add one event version (§5).  Its own rules are checked now.
    pub fn add_event(&mut self, event: Event) -> Result<Event> {
        refuse(check_event(&event, ""), &format!("event {}", repr_str(&event.event_id)))?;
        self.clinical_records()?;
        if self.clinical_ids.0.contains(&event.event_id) {
            return Err(Error::coded(
                "E809",
                format!(
                    "event {} already exists; an event version is immutable --- a revision is a new event \
                     superseding it",
                    repr_str(&event.event_id)
                ),
            ));
        }
        self.clinical_ids.0.insert(event.event_id.clone());
        self.clinical_records()?.events.push(event.clone());
        Ok(event)
    }

    /// Add one source document (§6).  Link it from its `document` event.
    pub fn add_document(&mut self, document: Document) -> Result<Document> {
        refuse(check_document(&document, ""), &format!("document {}", repr_str(&document.document_id)))?;
        self.clinical_records()?;
        if self.clinical_ids.1.contains(&document.document_id) {
            return Err(Error::coded(
                "E809",
                format!(
                    "document {} already exists; changed text is a new document and a new event version",
                    repr_str(&document.document_id)
                ),
            ));
        }
        self.clinical_ids.1.insert(document.document_id.clone());
        self.clinical_records()?.documents.push(document.clone());
        Ok(document)
    }

    /// Add one typed link (§7).
    ///
    /// An event version the amended file already held is immutable, and so
    /// is what it attests (§7.3): a structural `describes` link from it --- a
    /// document event's text, an imaging event's image --- is refused, and so
    /// is a link it would assert.  The version was available when it was, so
    /// a payload or a relationship attached later became an input at every
    /// cutoff after that time, before it existed.  A new version superseding
    /// it owns a payload instead, and a new version, available when it was
    /// drawn, asserts a relationship.
    pub fn add_link(&mut self, link: Link) -> Result<Link> {
        refuse(check_link(&link, ""), &format!("link {}", link.describe()))?;
        self.clinical_records()?;
        // After `drop_clinical` the held versions are gone, and one added
        // again is held to what it was by the commit (`check_held`).
        if self.clinical_source == ClinicalSource::Inherited {
            let owned = link.source_type == "event"
                && self.inherited_events.get(&link.source_id).is_some_and(|kind| link.is_structural_from(kind));
            if owned {
                return Err(Error::coded(
                    "E809",
                    format!(
                        "link {}: event {} is a version this sample already held, and an event version is immutable \
                         --- adding or replacing a payload it owns needs a new event version that supersedes it \
                         (1.1 §7.3)",
                        link.describe(),
                        repr_str(&link.source_id)
                    ),
                ));
            }
            let held = link.relation != "supersedes"
                && link.asserted_by_event_id.as_ref().is_some_and(|by| self.inherited_events.contains_key(by));
            // A structural link is its source's, whatever it names as asserter.
            let structural = || {
                let records = self.clinical.as_ref().expect("loaded above");
                let source = records.events.iter().rev().find(|e| e.event_id == link.source_id);
                link.source_type == "event" && source.is_some_and(|e| link.is_structural_from(&e.kind))
            };
            if held && !structural() {
                return Err(Error::coded(
                    "E809",
                    format!(
                        "link {}: it is asserted by event {}, a version this sample already held, and an event \
                         version is immutable --- a relationship drawn now was not known with it: assert it by a new \
                         event version, available when it was drawn (1.1 §7.3)",
                        link.describe(),
                        repr_str(link.asserted_by_event_id.as_deref().unwrap_or_default())
                    ),
                ));
            }
        }
        self.clinical_records()?.links.push(link.clone());
        Ok(link)
    }

    /// Add a bundle of records: its clock must be this sample's (or start it),
    /// and no id may collide with one already here.
    pub fn add_records(&mut self, records: ClinicalRecords) -> Result<()> {
        self.set_clock(records.descriptor.clock.clone())?;
        for event in records.events {
            self.add_event(event)?;
        }
        for document in records.documents {
            self.add_document(document)?;
        }
        for link in records.links {
            self.add_link(link)?;
        }
        Ok(())
    }

    /// The records written so far, or an amended file's once loaded.
    pub fn clinical(&mut self) -> Result<Option<&ClinicalRecords>> {
        if self.clinical.is_none() && self.clinical_source == ClinicalSource::Inherited {
            self.clinical_records()?;
        }
        Ok(self.clinical.as_ref())
    }

    /// Whether this writer holds the clinical profile (new or inherited).
    pub fn has_clinical(&self) -> bool {
        self.clinical.is_some() || self.clinical_source == ClinicalSource::Inherited
    }

    /// Write changed records over the copied group, at commit --- once what
    /// the amended file held is found unchanged ([`Self::check_held`]).
    pub(crate) fn write_clinical(&mut self) -> Result<()> {
        self.check_held()?;
        let Some(records) = &self.clinical else { return Ok(()) };
        let root = self.root()?;
        ops::unlink(&root, GROUP)?;
        let profile = resolve_profile(Some(&self.codec))?;
        table::write(&root, records, &profile)
    }

    /// Refuse an amendment that changes what an event version the amended
    /// file held attests (§7.3, E809): the version itself, the links it
    /// attests --- its structural link, and the links it asserts --- and every
    /// payload those links admit with it (§9.1 step 5).  The version was
    /// available when it was, so anything attached to it later became an
    /// input at every cutoff after that time, before it existed: a later
    /// grounding asserted by the original report, a mask replaced under the
    /// id the report's procedure attests, a link `add_link` refuses but a
    /// dropped and re-added history took.
    ///
    /// The amended file is still in place --- the commit replaces it only
    /// after this --- so what it held is read from it, payload digests
    /// included: a payload removed and written again under its own id is
    /// compared with the one it replaced, by its datasets' digests.  A
    /// version the amendment dropped (`drop_clinical`) and did not add again
    /// is gone, not changed; one added again is held to what it was.
    fn check_held(&self) -> Result<()> {
        let inherited = self.clinical_source == ClinicalSource::Inherited;
        if !inherited && (self.inherited_events.is_empty() || self.clinical.is_none()) {
            return Ok(());
        }
        // Read nothing that cannot be read: a source whose tables do not
        // read fails the commit's validation anyway.
        let Ok(file) = open_read(&self.path) else { return Ok(()) };
        let source = file.as_group()?;
        let Ok(Some(held)) = Clinical::open(&source, false) else { return Ok(()) };
        let (events, links) = match &self.clinical {
            Some(records) => (records.events.as_slice(), records.links.as_slice()),
            None => (held.events.as_slice(), held.links.as_slice()),
        };
        let present: HashMap<&str, &Event> = events.iter().map(|e| (e.event_id.as_str(), e)).collect();
        if !held.events.iter().any(|e| present.contains_key(e.event_id.as_str())) {
            return Ok(());
        }
        if let Some(records) = &self.clinical {
            if records.descriptor.clock != held.descriptor.clock {
                return Err(Error::coded(
                    "E802",
                    format!(
                        "this amendment keeps event versions the sample held on clock {}, now on clock {}: times \
                         already written are measured on the clock they were written on (1.1 §3)",
                        repr_str(&held.descriptor.clock.id),
                        repr_str(&records.descriptor.clock.id)
                    ),
                ));
            }
        }
        let before = attested(&held.events, &held.links);
        let after = attested(events, links);
        let root = self.root()?;
        let algo = crate::integrity::digest::root_algo(&source)?;
        let changed = |what: String, by: &str| {
            Error::coded(
                "E809",
                format!(
                    "{what}, which event {} attests, is not what the amended sample held: an event version is \
                     immutable, and so is what it attests --- a changed payload takes a new id, attested by a new \
                     event version (1.1 §7.3)",
                    repr_str(by)
                ),
            )
        };
        // The annotations carrying each instance id, in the amended file.
        let mut carriers: Option<SampleContext> = None;
        let mut compared: BTreeSet<(&str, &str)> = BTreeSet::new();
        for version in &held.events {
            let id = version.event_id.as_str();
            let Some(now) = present.get(id) else { continue };
            if *now != version {
                return Err(Error::coded(
                    "E809",
                    format!(
                        "event {} is a version the amended sample held, and this amendment changes it: an event \
                         version is immutable --- a revision is a new event superseding it (1.1 §7.3)",
                        repr_str(id)
                    ),
                ));
            }
            let was = before.get(id).map(Vec::as_slice).unwrap_or_default();
            let is = after.get(id).map(Vec::as_slice).unwrap_or_default();
            if was != is {
                let link = is.iter().find(|l| !was.contains(l)).or_else(|| was.iter().find(|l| !is.contains(l)));
                return Err(changed(format!("link {}", link.map_or_else(String::new, |(l, _)| l.describe())), id));
            }
            for (link, structural) in was {
                for (kind, target) in link.payloads(*structural) {
                    if !compared.insert((kind, target)) {
                        continue;
                    }
                    let objects: Vec<(&str, &str)> = match kind {
                        // Without `drop_clinical` no document the file held
                        // can change: an id is never added twice.
                        "document" => {
                            if !inherited && held.document_info(target).is_some() {
                                let now = self.clinical.as_ref().and_then(|r| r.document(target));
                                if now != Some(&held.document(target)?) {
                                    return Err(changed(format!("document {}", repr_str(target)), id));
                                }
                            }
                            Vec::new()
                        }
                        "instance" => {
                            if carriers.is_none() {
                                carriers = Some(SampleContext::from_root(&source, None)?);
                            }
                            let n = parse_instance(target);
                            let found = &carriers.as_ref().expect("read above").annotations;
                            found
                                .iter()
                                .filter(|(_, ids)| n.is_some_and(|n| ids.as_ref().is_some_and(|s| s.contains(&n))))
                                .map(|(annotation, _)| ("annotations", annotation.as_str()))
                                .collect()
                        }
                        other => OBJECT_GROUPS.iter().filter(|(k, _)| *k == other).map(|(_, g)| (*g, target)).collect(),
                    };
                    for (group, name) in objects {
                        let was = object_digests(&source, group, name, &algo)?;
                        if was.is_some() && was != object_digests(&root, group, name, &algo)? {
                            return Err(changed(format!("{kind} {}", repr_str(target)), id));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// The group each kind of payload object lives in.
const OBJECT_GROUPS: [(&str, &str); 4] =
    [("image", "images"), ("annotation", "annotations"), ("transform", "transforms"), ("grid", "grids")];

/// Each event version's attested links, sorted, each with whether it is
/// the version's structural link (§7.3).
fn attested<'a>(events: &'a [Event], links: &'a [Link]) -> HashMap<&'a str, Vec<(&'a Link, bool)>> {
    let kinds: HashMap<&str, &str> = events.iter().map(|e| (e.event_id.as_str(), e.kind.as_str())).collect();
    let kind_of = |id: &str| kinds.get(id).copied();
    let mut out: HashMap<&str, Vec<(&Link, bool)>> = HashMap::new();
    for link in links {
        let structural =
            link.source_type == "event" && kind_of(&link.source_id).is_some_and(|k| link.is_structural_from(k));
        if let Some(by) = link.attester(&kind_of) {
            out.entry(by).or_default().push((link, structural));
        }
    }
    for found in out.values_mut() {
        found.sort();
        found.dedup();
    }
    out
}

/// The digest of every dataset of object `name` in `group` (the object
/// itself, when it is a dataset), by path: the one it carries, or its
/// content's when it carries none --- a dataset written in this amendment,
/// stamped only at commit.  `None` when there is no such object.
fn object_digests(root: &hdf5::Group, group: &str, name: &str, algo: &str) -> Result<Option<BTreeMap<String, String>>> {
    let Some(parent) = ops::child_group(root, group) else { return Ok(None) };
    let digest = |ds: &hdf5::Dataset, path: &str| -> Result<String> {
        match attrs::get_str(ds, "digest")? {
            Some(stored) => Ok(stored),
            None => crate::integrity::dataset_digest(ds, path, algo),
        }
    };
    let path = format!("{group}/{name}");
    let mut out = BTreeMap::new();
    match ops::node_kind(&parent, name) {
        None => return Ok(None),
        Some(ops::NodeKind::Dataset) => {
            out.insert(path.clone(), digest(&parent.dataset(name)?, &path)?);
        }
        Some(ops::NodeKind::Group) => {
            for (relative, ds) in ops::datasets(&parent.group(name)?)? {
                let at = format!("{path}/{relative}");
                out.insert(at.clone(), digest(&ds, &at)?);
            }
        }
        Some(_) => {}
    }
    Ok(Some(out))
}

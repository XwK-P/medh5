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

use super::writer::{ClinicalSource, SampleWriter};
use crate::clinical::check::{check_document, check_event, check_link, Finding};
use crate::clinical::model::{ClinicalRecords, Clock, Descriptor, Document, Event, Link, GROUP};
use crate::clinical::table::{self, Clinical};
use crate::h5::ops;
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
    /// is what it owns: a structural `describes` link from it --- a document
    /// event's text, an imaging event's image --- is refused.  The version was
    /// available when it was, so a payload attached later became an input at
    /// every cutoff after that time, before it existed (F02 of the round-4
    /// audit).  A new version superseding it carries the payload instead.
    pub fn add_link(&mut self, link: Link) -> Result<Link> {
        refuse(check_link(&link, ""), &format!("link {}", link.describe()))?;
        self.clinical_records()?;
        if link.relation == "describes" && link.source_type == "event" {
            let owned = self.inherited_events.get(&link.source_id).is_some_and(|kind| {
                matches!((kind.as_str(), link.target_type.as_str()), ("document", "document") | ("imaging", "image"))
            });
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

    /// Write changed records over the copied group, at commit.
    pub(crate) fn write_clinical(&mut self) -> Result<()> {
        let Some(records) = &self.clinical else { return Ok(()) };
        let root = self.root()?;
        ops::unlink(&root, GROUP)?;
        let profile = resolve_profile(Some(&self.codec))?;
        table::write(&root, records, &profile)
    }
}

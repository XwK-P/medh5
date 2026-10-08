//! [`SampleWriter`]: where the pieces become a file.
//!
//! Writing is a builder that validates as it goes and commits atomically
//! (§14.4): create writes a sibling temporary file and renames it over the
//! target, so a reader never sees a half-written sample and a crash leaves the
//! previous file intact.  Amend is copy-on-write, because HDF5 does not
//! reclaim space on delete.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use ndarray::Array2;
use serde_json::{Map, Value};

use super::reader::{annotation_id, attr_name_map_of, frame_references, require_major, FRAME_ATTRS, PROFILES};
use crate::array::NdArray;
use crate::clinical::ClinicalRecords;
use crate::curation::identity::{Cohort, Deidentification, Identity, SplitClaim, ID_SOURCE};
use crate::curation::provenance::{Activity, Agent};
use crate::curation::quality::QualityRecord;
use crate::curation::timeline::{Timeline, Timepoint};
use crate::document::{SampleDocument, META_DATASET};
use crate::geometry::grid::{
    default_axis_kinds, default_axis_names, read_grids, write_grid, Grid, KNOWN_UNITS, TIME_UNITS,
};
use crate::geometry::multiscale::{check_pyramid, pyramid_factors, Pyramid};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data;
use crate::h5::file::{open_read, AtomicFile};
use crate::h5::ops;
use crate::ids::validate_id;
use crate::integrity::{compute_content_id, stamp_digests};
use crate::json::{repr_int_tuple, repr_list, repr_str};
use crate::labels::{ClassKey, LabelSet};
use crate::storage::chunking::grid_chunks;
use crate::storage::codecs::{dataset_layout, profile_family, resolve_profile, Role};
use crate::{Error, Result, VERSION};

/// Root attributes `commit` writes itself, so an amend must not copy them.
pub const MANAGED_ROOT_ATTRS: [&str; 7] =
    ["medh5_version", "medh5_kind", "medh5_profiles", "created", "generator", "digest_algo", "content_id"];

/// The groups the writer manages; anything else is copied through on amend.
pub const STANDARD_GROUPS: [&str; 6] = ["meta", "grids", "images", "annotations", "transforms", "index"];

/// `YYYY-mm-ddTHH:MM:SSZ`, now, in UTC.
pub fn utcnow() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}

/// Days since 1970-01-01 to a proleptic Gregorian date.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The `medh5_version` a commit writes: the lowest version the declared
/// profiles need (1.0, or 1.1 with `clinical`), never below what an amended
/// file already declared --- a no-op amend does not downgrade (1.1 §2.3).
pub fn written_version(source: Option<&str>, profiles: &[String]) -> String {
    let needed = crate::version::required_for(profiles);
    match source {
        Some(v) if crate::version::support(v) == crate::version::Support::Full => {
            crate::version::later(v, needed).to_string()
        }
        _ => needed.to_string(),
    }
}

/// Where a writer's `clinical/` group came from (1.1 §3, §10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClinicalSource {
    /// None: the sample has no `clinical` group.
    #[default]
    Absent,
    /// The profile's own content, declared and recognised, copied through
    /// untouched until something changes it.
    Inherited,
    /// A `clinical` group that is not the profile's --- a 1.0 file's own
    /// extension.  Copied through untouched, and never reinterpreted.
    Foreign,
}

/// Which classes an annotation claims were looked for (§11.3).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Annotated {
    /// Only the classes supplied.
    #[default]
    AllGiven,
    /// The whole label set.
    All,
    /// Exactly these.
    Classes(Vec<ClassKey>),
}

/// A quality reference: an existing record's key, or a record to create.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityArg {
    Key(String),
    Record(Map<String, Value>),
}

/// Options for [`SampleWriter::add_image`].
#[derive(Debug, Clone, Default)]
pub struct ImageOptions {
    pub value_type: Option<String>,
    pub value_units: Option<String>,
    pub channel_names: Option<Vec<String>>,
    pub rescale_slope: Option<f64>,
    pub rescale_intercept: Option<f64>,
    pub window_center: Option<Vec<f64>>,
    pub window_width: Option<Vec<f64>>,
    pub valid_mask: Option<String>,
    pub prov: Option<String>,
    pub codec: Option<String>,
}

/// Options for [`SampleWriter::add_grid`].
#[derive(Debug, Clone, Default)]
pub struct GridOptions {
    pub origin: Option<Vec<f64>>,
    pub direction: Option<Array2<f64>>,
    pub axis_names: Option<Vec<String>>,
    pub axis_kinds: Option<Vec<String>>,
    pub coord_system: Option<String>,
    pub units: Option<String>,
    pub timepoint: Option<String>,
    pub frame_uid: Option<String>,
    pub patch_hint: Option<Vec<i64>>,
    pub chunk_hint: Option<Vec<i64>>,
    pub time_values: Option<Vec<f64>>,
    pub time_units: Option<String>,
}

/// Builder for one sample.  Every `add_*` validates immediately.
pub struct SampleWriter {
    pub path: PathBuf,
    /// The codec profile name datasets default to.
    pub codec: String,
    file: Option<AtomicFile>,
    committed: bool,
    pub(crate) grids: IndexMap<String, Grid>,
    pub(crate) images: IndexMap<String, (String, String)>,
    pub(crate) image_multiscale: HashMap<String, bool>,
    pub(crate) annotation_kinds: IndexMap<String, String>,
    pub(crate) transform_frames: IndexMap<String, (String, String)>,
    declared_profiles: BTreeSet<String>,
    default_timeline: bool,
    source_version: Option<String>,
    pub(crate) document: SampleDocument,
    /// The clinical records, once written to or changed in this writer.
    pub(crate) clinical: Option<ClinicalRecords>,
    pub(crate) clinical_source: ClinicalSource,
    /// Event and document ids already in `clinical`, so a duplicate is
    /// refused without scanning every row.
    pub(crate) clinical_ids: (std::collections::HashSet<String>, std::collections::HashSet<String>),
}

impl std::fmt::Debug for SampleWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SampleWriter").field("path", &self.path).field("codec", &self.codec).finish()
    }
}

/// Create a new sample.  Call [`SampleWriter::commit`] to write it.
pub fn create(
    path: &Path,
    sample_id: Option<&str>,
    subject_id: Option<&str>,
    codec: &str,
    profiles: &[String],
) -> Result<SampleWriter> {
    SampleWriter::new(path, sample_id, subject_id, codec, profiles, None)
}

/// Copy-on-write amend: build a new file from the old and replace it (§14.4).
///
/// Unknown objects are copied through untouched; known profiles are re-derived
/// from what the amended file holds, and a recognised `clinical` profile is
/// carried with its group.  A file this engine cannot preserve --- another
/// major, a higher minor, a profile it does not implement --- is refused before
/// anything is written (1.1 §2.3).  `codec` defaults to the family the file was
/// written in.
pub fn amend(path: &Path, codec: Option<&str>) -> Result<SampleWriter> {
    let source = open_read(path)?;
    let version = require_major(&source, path)?;
    let declared = attrs::get_strs(&source, "medh5_profiles")?.unwrap_or_default();
    crate::version::require_amendable(&version, &declared, &repr_str(&path.to_string_lossy()))?;
    let kind = attrs::get_str(&source, "medh5_kind")?.unwrap_or_else(|| "sample".into());
    if kind != "sample" {
        return Err(Error::File(format!(
            "{} is a {}; amend works on one sample --- extract the member with `medh5 unpack`, amend it, and pack again",
            repr_str(&path.to_string_lossy()),
            repr_str(&kind)
        )));
    }
    let root = source.as_group()?;
    let chosen = match codec {
        Some(c) => c.to_string(),
        None => profile_family(&root)?.to_string(),
    };
    let writer = SampleWriter::new(path, None, None, &chosen, &[], Some(&root));
    drop(root);
    drop(source);
    writer
}

impl SampleWriter {
    /// A writer for `path`; `source` copies an existing sample in (amend).
    pub fn new(
        path: &Path,
        sample_id: Option<&str>,
        subject_id: Option<&str>,
        codec: &str,
        profiles: &[String],
        source: Option<&hdf5::Group>,
    ) -> Result<SampleWriter> {
        let codec = resolve_profile(Some(codec))?.name.to_string();
        let file = AtomicFile::create(path)?;
        let stem = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let default_id =
            sample_id.map(str::to_string).unwrap_or_else(|| stem.split('.').next().unwrap_or("").to_string());
        let identity =
            Identity::new(default_id.clone(), subject_id.map(str::to_string).unwrap_or_else(|| default_id.clone()));
        let mut writer = SampleWriter {
            path: path.to_path_buf(),
            codec,
            file: Some(file),
            committed: false,
            grids: IndexMap::new(),
            images: IndexMap::new(),
            image_multiscale: HashMap::new(),
            annotation_kinds: IndexMap::new(),
            transform_frames: IndexMap::new(),
            declared_profiles: profiles.iter().cloned().collect(),
            default_timeline: true,
            source_version: None,
            document: SampleDocument::new(identity?, Timeline::single("tp0")?),
            clinical: None,
            clinical_source: ClinicalSource::Absent,
            clinical_ids: Default::default(),
        };
        let setup = (|| -> Result<()> {
            for name in ["grids", "images", "annotations"] {
                writer.root()?.create_group(name)?;
            }
            if let Some(src) = source {
                writer.inherit(src)?;
            }
            Ok(())
        })();
        if let Err(e) = setup {
            writer.abort();
            return Err(e);
        }
        Ok(writer)
    }

    /// The file being built.
    pub fn handle(&self) -> Result<&hdf5::File> {
        match &self.file {
            Some(f) => Ok(f.handle()),
            None => Err(Error::invalid("this writer has already committed or aborted")),
        }
    }

    /// The root group of the file being built.
    pub fn root(&self) -> Result<hdf5::Group> {
        Ok(self.handle()?.as_group()?)
    }

    fn group(&self, name: &str) -> Result<hdf5::Group> {
        Ok(self.root()?.group(name)?)
    }

    /// Discard the in-progress file, leaving any existing one untouched.
    pub fn abort(&mut self) {
        if let Some(f) = self.file.take() {
            f.abort();
        }
        self.committed = true;
    }

    /// Whether `commit` or `abort` has run.
    pub fn is_closed(&self) -> bool {
        self.committed
    }

    pub fn document(&self) -> &SampleDocument {
        &self.document
    }

    pub fn document_mut(&mut self) -> &mut SampleDocument {
        &mut self.document
    }

    /// Replace the whole sample document; `commit` validates it as usual.
    pub fn set_document(&mut self, document: SampleDocument) {
        self.document = document;
    }

    /// Remove the `clinical` profile --- its group, its records and its
    /// declaration --- for the imaging projection (1.1 §10).  The sample is
    /// then written in the lowest version its remaining profiles need, and is
    /// a different sample with a different `content_id`.
    pub fn drop_clinical(&mut self) -> Result<()> {
        let root = self.root()?;
        ops::unlink(&root, crate::clinical::GROUP)?;
        self.clinical = None;
        self.clinical_ids = Default::default();
        self.clinical_source = ClinicalSource::Absent;
        self.declared_profiles.remove(crate::clinical::PROFILE);
        self.source_version = None;
        Ok(())
    }

    fn inherit(&mut self, source: &hdf5::Group) -> Result<()> {
        self.source_version = attrs::get_str(source, "medh5_version")?;
        self.document = super::reader::read_document(source)?;
        self.default_timeline = false;
        // Known profiles are re-derived from the amended content, so a claim
        // an edit no longer justifies is dropped rather than kept false; an
        // unknown one was refused before the amend began (1.1 §2.3), and a
        // recognised `clinical` group carries its profile with it.
        let declared = attrs::get_strs(source, "medh5_profiles")?.unwrap_or_default();
        if ops::exists(source, crate::clinical::GROUP) {
            self.clinical_source =
                if declared.iter().any(|p| p == crate::clinical::PROFILE) && crate::clinical::recognised(source) {
                    ClinicalSource::Inherited
                } else {
                    ClinicalSource::Foreign
                };
        }
        let root = self.root()?;
        for name in ["grids", "images", "annotations", "transforms", "index"] {
            if !ops::exists(source, name) {
                continue;
            }
            ops::unlink(&root, name)?;
            ops::copy_object(source, name, &root, name)?;
        }
        self.grids = read_grids(&root)?.into_iter().collect();
        if let Some(images) = ops::child_group(&root, "images") {
            for name in ops::members(&images)? {
                let multiscale = ops::is_group(&images, &name);
                let (grid, modality) = if multiscale {
                    let g = images.group(&name)?;
                    (attrs::get_str(&g, "grid")?, attrs::get_str(&g, "modality")?)
                } else {
                    let d = images.dataset(&name)?;
                    (attrs::get_str(&d, "grid")?, attrs::get_str(&d, "modality")?)
                };
                self.images.insert(name.clone(), (grid.unwrap_or_default(), modality.unwrap_or_else(|| "OT".into())));
                self.image_multiscale.insert(name, multiscale);
            }
        }
        if let Some(anns) = ops::child_group(&root, "annotations") {
            for name in ops::members(&anns)? {
                if let Some(g) = ops::child_group(&anns, &name) {
                    self.annotation_kinds.insert(name, attrs::get_str(&g, "kind")?.unwrap_or_else(|| "mask".into()));
                }
            }
        }
        if let Some(ts) = ops::child_group(&root, "transforms") {
            for name in ops::members(&ts)? {
                if let Some(g) = ops::child_group(&ts, &name) {
                    let frames = (
                        attrs::get_str(&g, "from_frame")?.unwrap_or_default(),
                        attrs::get_str(&g, "to_frame")?.unwrap_or_default(),
                    );
                    self.transform_frames.insert(name, frames);
                }
            }
        }
        ops::copy_unknown(source, &root, &STANDARD_GROUPS)?;
        for name in attrs::names(source)? {
            if MANAGED_ROOT_ATTRS.contains(&name.as_str()) {
                continue;
            }
            attrs::copy_raw(source, &root, &name)?;
        }
        Ok(())
    }

    // -- document ---------------------------------------------------------------

    /// Merge fields into the identity.  Changing an id drops what `id_source`
    /// recorded about it, unless the fields record a new source.
    pub fn identity(&mut self, fields: Map<String, Value>) -> Result<Identity> {
        let current = match self.document.identity.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        let mut merged = current.clone();
        merged.extend(fields.clone());
        if !fields.contains_key(ID_SOURCE) {
            if let Some(Value::Object(source)) = merged.get(ID_SOURCE).cloned() {
                let kept: Map<String, Value> = source
                    .into_iter()
                    .filter(|(name, _)| match fields.get(name) {
                        None => true,
                        Some(v) => Some(v) == current.get(name),
                    })
                    .collect();
                if kept.is_empty() {
                    merged.remove(ID_SOURCE);
                } else {
                    merged.insert(ID_SOURCE.into(), Value::Object(kept));
                }
            }
        }
        self.document.identity = Identity::from_json(&Value::Object(merged))?;
        Ok(self.document.identity.clone())
    }

    /// Merge fields into the cohort record.
    pub fn cohort(&mut self, fields: Map<String, Value>) -> Result<Cohort> {
        let mut merged = match self.document.cohort.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        merged.extend(fields);
        self.document.cohort = Cohort::from_json(Some(&Value::Object(merged)))?;
        Ok(self.document.cohort.clone())
    }

    /// Declare a timepoint; the first explicit one replaces the implicit `tp0`.
    pub fn add_timepoint(&mut self, timepoint_id: &str, mut fields: Map<String, Value>) -> Result<Timepoint> {
        let existing: Vec<Timepoint> =
            if self.default_timeline { Vec::new() } else { self.document.timepoints.points().to_vec() };
        let index = fields.remove("index").unwrap_or_else(|| Value::from(existing.len()));
        let mut doc = Map::new();
        doc.insert("id".into(), Value::String(timepoint_id.into()));
        doc.insert("index".into(), index);
        doc.extend(fields);
        let tp = Timepoint::from_json(&Value::Object(doc))?;
        let mut all = existing;
        all.push(tp.clone());
        self.document.timepoints = Timeline::new(all)?;
        self.default_timeline = false;
        Ok(tp)
    }

    pub fn label_set(&mut self, label_set: LabelSet) -> LabelSet {
        self.document.label_set = Some(label_set.clone());
        label_set
    }

    /// Declare an agent; an explicit id must be free, an automatic one is
    /// `<type initial><n>`, skipping ids already taken.
    pub fn agent(
        &mut self,
        agent_type: &str,
        name: &str,
        agent_id: Option<&str>,
        fields: Map<String, Value>,
    ) -> Result<Agent> {
        let prov = &self.document.provenance;
        let id = match agent_id {
            Some(id) => id.to_string(),
            None => {
                let initial = agent_type.chars().next().unwrap_or('a');
                let mut n = prov.n_agents() + 1;
                while prov.has_agent(&format!("{initial}{n}")) {
                    n += 1;
                }
                format!("{initial}{n}")
            }
        };
        let mut doc = Map::new();
        doc.insert("id".into(), Value::String(id));
        doc.insert("type".into(), Value::String(agent_type.into()));
        doc.insert("name".into(), Value::String(name.into()));
        for (k, v) in fields {
            if !v.is_null() {
                doc.insert(k, v);
            }
        }
        let agent = Agent::from_json(&Value::Object(doc))?;
        self.document.provenance.add_agent(agent, false)
    }

    pub fn person(&mut self, name: &str, agent_id: Option<&str>, fields: Map<String, Value>) -> Result<Agent> {
        self.agent("person", name, agent_id, fields)
    }

    pub fn software(&mut self, name: &str, version: Option<&str>, mut fields: Map<String, Value>) -> Result<Agent> {
        if let Some(v) = version {
            fields.insert("version".into(), Value::String(v.into()));
        }
        self.agent("software", name, None, fields)
    }

    pub fn organization(&mut self, name: &str, fields: Map<String, Value>) -> Result<Agent> {
        self.agent("organization", name, None, fields)
    }

    /// Record an activity; ids are `act_<type>_<n>`, skipping taken ones.
    pub fn activity(
        &mut self,
        activity_type: &str,
        agent: Option<&str>,
        activity_id: Option<&str>,
        fields: Map<String, Value>,
    ) -> Result<Activity> {
        let prov = &self.document.provenance;
        let id = match activity_id {
            Some(id) => id.to_string(),
            None => {
                let mut n = prov.n_activities() + 1;
                while prov.has_activity(&format!("act_{activity_type}_{n}")) {
                    n += 1;
                }
                format!("act_{activity_type}_{n}")
            }
        };
        let mut doc = Map::new();
        doc.insert("id".into(), Value::String(id));
        doc.insert("type".into(), Value::String(activity_type.into()));
        if let Some(a) = agent {
            doc.insert("agent".into(), Value::String(a.into()));
        }
        for (k, v) in fields {
            if !v.is_null() {
                doc.insert(k, v);
            }
        }
        let activity = Activity::from_json(&Value::Object(doc))?;
        self.document.provenance.add_activity(activity, false)
    }

    /// Create or replace a quality record (status defaults to `draft`).
    pub fn set_quality(&mut self, key: &str, fields: Map<String, Value>) -> Result<QualityRecord> {
        let mut doc = Map::new();
        doc.insert("status".into(), Value::String("draft".into()));
        doc.extend(fields);
        let record = QualityRecord::from_json(&Value::Object(doc))?;
        self.document.quality.insert(key.to_string(), record.clone());
        Ok(record)
    }

    /// Record a split claim, replacing any earlier claim for the same set.
    pub fn split(&mut self, fields: Map<String, Value>) -> Result<SplitClaim> {
        let claim = SplitClaim::from_json(&Value::Object(fields))?;
        self.document.splits.retain(|s| s.set_id != claim.set_id);
        self.document.splits.push(claim.clone());
        Ok(claim)
    }

    /// Set the de-identification record (it needs at least `method`).
    pub fn deidentification(&mut self, fields: Map<String, Value>) -> Result<Deidentification> {
        let record = Deidentification::from_json(Some(&Value::Object(fields)))?
            .ok_or_else(|| Error::invalid("a de-identification record needs at least its `method` (§11.4)"))?;
        self.document.deidentification = Some(record.clone());
        Ok(record)
    }

    /// Merge acquisition parameters for one image.
    pub fn acquisition(&mut self, image_id: &str, params: Map<String, Value>) -> Value {
        let entry = self.document.acquisition.entry(image_id.to_string()).or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(m) = entry {
            m.extend(params);
        }
        entry.clone()
    }

    /// Set a namespaced extension member of the document.
    pub fn extra(&mut self, namespace: &str, value: Value) {
        self.document.extra.insert(namespace.to_string(), value);
    }

    // -- grids ------------------------------------------------------------------

    /// Declare a grid.  Geometry lives here and nowhere else.
    pub fn add_grid(&mut self, grid_id: &str, shape: &[i64], spacing: &[f64], options: GridOptions) -> Result<Grid> {
        validate_id(grid_id, "grid id")?;
        if self.grids.contains_key(grid_id) {
            return Err(Error::invalid(format!("grid {} is already declared", repr_str(grid_id))));
        }
        let units = options.units.clone().unwrap_or_else(|| "mm".into());
        if !KNOWN_UNITS.contains(&units.as_str()) {
            return Err(Error::coded(
                "E109",
                format!(
                    "grid {}: units {} is not one of {} (§3.2); convert spacing and origin rather than declaring another unit",
                    repr_str(grid_id),
                    repr_str(&units),
                    repr_list(&KNOWN_UNITS)
                ),
            ));
        }
        if let Some(tu) = &options.time_units {
            if !TIME_UNITS.contains(&tu.as_str()) {
                return Err(Error::coded(
                    "E109",
                    format!(
                        "grid {}: time_units {} is not one of {} (§3.2)",
                        repr_str(grid_id),
                        repr_str(tu),
                        repr_list(&TIME_UNITS)
                    ),
                ));
            }
        }
        let axis_kinds = match options.axis_kinds {
            Some(k) => k,
            None => default_axis_kinds(shape.len())?,
        };
        let axis_names = options.axis_names.unwrap_or_else(|| default_axis_names(&axis_kinds));
        if axis_kinds.iter().any(|k| k == "time") && options.time_values.is_none() {
            return Err(Error::coded(
                "E109",
                format!(
                    "grid {} declares a time axis but no time_values; §3.2 requires one acquisition time per frame",
                    repr_str(grid_id)
                ),
            ));
        }
        let n_spatial = axis_kinds.iter().filter(|k| *k == "spatial").count();
        let mut grid = Grid::new(
            grid_id,
            shape.to_vec(),
            axis_names,
            axis_kinds,
            spacing.to_vec(),
            options.origin.unwrap_or_else(|| vec![0.0; n_spatial]),
            options.direction.unwrap_or_else(|| Array2::eye(n_spatial)),
            options.coord_system.unwrap_or_else(|| "LPS".into()),
            units,
        );
        if let Ok(g) = &mut grid {
            g.timepoint = options.timepoint;
            g.frame_uid = options.frame_uid;
            g.time_values = options.time_values;
            g.time_units = options.time_units;
            g.patch_hint = options.patch_hint;
            g.chunk_hint = options.chunk_hint;
            g.check()?;
        }
        let grid = grid?;
        write_grid(&self.group("grids")?, &grid)?;
        self.grids.insert(grid_id.to_string(), grid.clone());
        Ok(grid)
    }

    /// Grids declared so far.
    pub fn grids(&self) -> &IndexMap<String, Grid> {
        &self.grids
    }

    /// A declared grid, or E101.
    pub fn grid_ref(&self, grid_id: &str) -> Result<&Grid> {
        self.grids.get(grid_id).ok_or_else(|| {
            Error::coded(
                "E101",
                format!("grid {} is not declared; declare it before referencing it", repr_str(grid_id)),
            )
        })
    }

    /// Rename frames of reference everywhere they are named (§3.4): on
    /// grids, world-space annotations and transforms at once.  Returns the
    /// attributes rewritten.
    pub fn remap_frame_uids(&mut self, mapping: &HashMap<String, String>) -> Result<Vec<String>> {
        let root = self.root()?;
        let mut changed = Vec::new();
        for (group, names) in FRAME_ATTRS {
            let Some(node) = ops::child_group(&root, group) else { continue };
            for name in ops::members(&node)? {
                let Some(obj) = ops::child_group(&node, &name) else { continue };
                for attr in names {
                    let current = attrs::get_str(&obj, attr)?.unwrap_or_default();
                    match mapping.get(&current) {
                        Some(replacement) if replacement != &current => {
                            attrs::write(&obj, attr, &AttrValue::Str(replacement.clone()))?;
                            changed.push(format!("{group}.{name}.{attr}"));
                        }
                        _ => {}
                    }
                }
            }
        }
        if !changed.is_empty() {
            self.grids = read_grids(&root)?.into_iter().collect();
            let transforms = ops::child_group(&root, "transforms");
            for (name, frames) in self.transform_frames.iter_mut() {
                if let Some(g) = transforms.as_ref().and_then(|t| ops::child_group(t, name)) {
                    *frames = (
                        attrs::get_str(&g, "from_frame")?.unwrap_or_default(),
                        attrs::get_str(&g, "to_frame")?.unwrap_or_default(),
                    );
                }
            }
        }
        Ok(changed)
    }

    /// Every frame UID in the file so far, and the attributes naming it.
    pub fn frame_uids(&self) -> Result<IndexMap<String, Vec<String>>> {
        frame_references(&self.root()?)
    }

    // -- images -------------------------------------------------------------------

    pub(crate) fn chunks_for(&self, grid: &Grid, itemsize: usize, leading: usize) -> Result<Vec<usize>> {
        grid_chunks(grid, itemsize, leading)
    }

    fn profile(&self, codec: Option<&str>) -> Result<crate::storage::codecs::CodecProfile> {
        resolve_profile(Some(codec.unwrap_or(&self.codec)))
    }

    /// Write one image, chunked for its grid's patch hint.
    pub fn add_image(
        &mut self,
        image_id: &str,
        data: &NdArray,
        grid: &str,
        modality: &str,
        options: ImageOptions,
    ) -> Result<hdf5::Dataset> {
        validate_id(image_id, "image id")?;
        if self.images.contains_key(image_id) {
            return Err(Error::invalid(format!("image {} already exists", repr_str(image_id))));
        }
        let value_type = options.value_type.clone().unwrap_or_else(|| "intensity".into());
        super::image::check_value_type(&value_type)?;
        let target = self.grid_ref(grid)?.clone();
        let shape: Vec<i64> = data.shape().iter().map(|v| *v as i64).collect();
        if shape != target.shape {
            return Err(Error::coded(
                "E202",
                format!(
                    "image {} has shape {}; grid {} declares {}",
                    repr_str(image_id),
                    repr_int_tuple(&data.shape()),
                    repr_str(grid),
                    repr_int_tuple(&target.shape)
                ),
            ));
        }
        if let Some(names) = &options.channel_names {
            let ok = match target.channel_axis() {
                Some(axis) => names.len() as i64 == target.shape[axis],
                None => false,
            };
            if !ok {
                return Err(Error::coded(
                    "E204",
                    format!("image {}: channel_names does not match the channel axis", repr_str(image_id)),
                ));
            }
        }
        let chunks = self.chunks_for(&target, data.dtype().itemsize(), 0)?;
        let layout = dataset_layout(
            &data.shape(),
            data.dtype().itemsize(),
            &self.profile(options.codec.as_deref())?,
            Role::Image,
            Some(chunks),
        );
        let node = data::create(&self.group("images")?, image_id, data, &layout)?;
        let mut values: Vec<(&str, Option<AttrValue>)> = vec![
            ("grid", Some(AttrValue::Str(grid.into()))),
            ("modality", Some(AttrValue::Str(modality.into()))),
            ("value_type", Some(AttrValue::Str(value_type))),
            ("value_units", options.value_units.map(AttrValue::Str)),
            ("channel_names", options.channel_names.filter(|v| !v.is_empty()).map(|v| AttrValue::strs(&v))),
            ("rescale_slope", options.rescale_slope.map(AttrValue::Float)),
            ("rescale_intercept", options.rescale_intercept.map(AttrValue::Float)),
            ("window_center", options.window_center.filter(|v| !v.is_empty()).map(|v| AttrValue::floats(&v))),
            ("window_width", options.window_width.filter(|v| !v.is_empty()).map(|v| AttrValue::floats(&v))),
            ("valid_mask", options.valid_mask.map(AttrValue::Str)),
            ("prov", options.prov.map(AttrValue::Str)),
        ];
        values.retain(|(_, v)| v.is_some());
        attrs::write_all(&node, &values)?;
        self.images.insert(image_id.to_string(), (grid.to_string(), modality.to_string()));
        self.image_multiscale.insert(image_id.to_string(), false);
        Ok(node)
    }

    /// Write a multiscale image (§4.3); the level geometry is checked here.
    #[allow(clippy::too_many_arguments)]
    pub fn add_pyramid(
        &mut self,
        image_id: &str,
        levels: &[NdArray],
        grid_levels: &[String],
        modality: &str,
        downsample_method: &str,
        options: ImageOptions,
    ) -> Result<hdf5::Group> {
        validate_id(image_id, "image id")?;
        if self.images.contains_key(image_id) {
            return Err(Error::invalid(format!("image {} already exists", repr_str(image_id))));
        }
        let value_type = options.value_type.clone().unwrap_or_else(|| "intensity".into());
        super::image::check_value_type(&value_type)?;
        if levels.len() != grid_levels.len() {
            return Err(Error::coded(
                "E105",
                format!("image {}: {} arrays for {} grid levels", repr_str(image_id), levels.len(), grid_levels.len()),
            ));
        }
        let grids: Vec<Grid> = grid_levels.iter().map(|g| self.grid_ref(g).cloned()).collect::<Result<_>>()?;
        let refs: Vec<&Grid> = grids.iter().collect();
        let factors = pyramid_factors(&grids[0], &refs);
        let problems = check_pyramid(&grids[0], &refs, &factors, crate::geometry::multiscale::GEOMETRY_RTOL)?;
        if !problems.is_empty() {
            return Err(Error::coded(
                "E105",
                format!("image {}: inconsistent pyramid geometry: {}", repr_str(image_id), problems.join("; ")),
            ));
        }
        let group = self.group("images")?.create_group(image_id)?;
        let profile = self.profile(options.codec.as_deref())?;
        for (level, (array, target)) in levels.iter().zip(&grids).enumerate() {
            let shape: Vec<i64> = array.shape().iter().map(|v| *v as i64).collect();
            if shape != target.shape {
                return Err(Error::coded(
                    "E202",
                    format!(
                        "image {} level {level} has shape {}; grid {} declares {}",
                        repr_str(image_id),
                        repr_int_tuple(&array.shape()),
                        repr_str(&target.grid_id),
                        repr_int_tuple(&target.shape)
                    ),
                ));
            }
            let chunks = self.chunks_for(target, array.dtype().itemsize(), 0)?;
            let layout = dataset_layout(&array.shape(), array.dtype().itemsize(), &profile, Role::Image, Some(chunks));
            data::create(&group, &level.to_string(), array, &layout)?;
        }
        let pyramid = Pyramid::new(levels.len(), factors, downsample_method, grid_levels.to_vec())?;
        let mut values: Vec<(String, AttrValue)> = vec![
            ("grid".into(), AttrValue::Str(grid_levels[0].clone())),
            ("modality".into(), AttrValue::Str(modality.into())),
            ("value_type".into(), AttrValue::Str(value_type)),
        ];
        if let Some(v) = options.value_units {
            values.push(("value_units".into(), AttrValue::Str(v)));
        }
        if let Some(v) = options.rescale_slope {
            values.push(("rescale_slope".into(), AttrValue::Float(v)));
        }
        if let Some(v) = options.rescale_intercept {
            values.push(("rescale_intercept".into(), AttrValue::Float(v)));
        }
        if let Some(v) = options.prov {
            values.push(("prov".into(), AttrValue::Str(v)));
        }
        values.extend(pyramid.attrs());
        for (k, v) in &values {
            attrs::write(&group, k, v)?;
        }
        self.images.insert(image_id.to_string(), (grid_levels[0].clone(), modality.to_string()));
        self.image_multiscale.insert(image_id.to_string(), true);
        Ok(group)
    }

    // -- commit -------------------------------------------------------------------

    /// Profiles this sample actually satisfies, unioned with declared ones.
    pub fn infer_profiles(&self) -> Result<BTreeSet<String>> {
        let kinds: BTreeSet<&str> = self.annotation_kinds.values().map(String::as_str).collect();
        let mut found: BTreeSet<String> = BTreeSet::from(["core".to_string()]);
        let doc = &self.document;
        let has_labels = doc.label_set.is_some();
        if ["labelmap", "layers", "bitmask", "instances", "probmap"].iter().any(|k| kinds.contains(k)) && has_labels {
            found.insert("seg".into());
        }
        if has_labels && self.has_task("detection")? {
            found.insert("det".into());
        }
        if kinds.contains("classification") && has_labels {
            found.insert("cls".into());
        }
        if !self.transform_frames.is_empty() {
            found.insert("reg".into());
        }
        if !doc.provenance.is_empty() && !self.annotation_kinds.is_empty() {
            let anns = self.group("annotations")?;
            let mut all = true;
            for name in self.annotation_kinds.keys() {
                let g = anns.group(name)?;
                if !attrs::has(&g, "quality") {
                    all = false;
                    break;
                }
            }
            if all {
                found.insert("curation".into());
            }
        }
        if !self.images.is_empty() && self.images.keys().all(|i| self.image_multiscale.get(i).copied().unwrap_or(false))
        {
            found.insert("multiscale".into());
        }
        if let Some(index) = ops::child_group(&self.root()?, "index") {
            if !ops::members(&index)?.is_empty() {
                found.insert("training".into());
            }
        }
        if doc.timepoints.is_longitudinal()
            && self.grids.values().all(|g| g.timepoint.as_deref().is_some_and(|t| !t.is_empty()))
        {
            found.insert("longitudinal".into());
        }
        if self.clinical.is_some() || self.clinical_source == ClinicalSource::Inherited {
            found.insert(crate::clinical::PROFILE.into());
        }
        let mut unknown: Vec<String> =
            self.declared_profiles.union(&found).filter(|p| !PROFILES.contains(&p.as_str())).cloned().collect();
        unknown.sort();
        if !unknown.is_empty() {
            return Err(Error::coded("E007", format!("unknown profile(s) {}", repr_list(&unknown))));
        }
        Ok(found.union(&self.declared_profiles).cloned().collect())
    }

    fn has_task(&self, task: &str) -> Result<bool> {
        let anns = self.group("annotations")?;
        for name in ops::members(&anns)? {
            if let Some(g) = ops::child_group(&anns, &name) {
                if attrs::get_str(&g, "task")?.as_deref() == Some(task) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn check_references(&self) -> Result<()> {
        let doc = &self.document;
        let declared: BTreeSet<String> = doc.timepoints.ids().into_iter().collect();
        let multi = declared.len() > 1;
        for grid in self.grids.values() {
            match &grid.timepoint {
                None => {
                    if multi {
                        return Err(Error::coded(
                            "E106",
                            format!(
                                "grid {} has no `timepoint`, but the sample declares {} timepoints",
                                repr_str(&grid.grid_id),
                                declared.len()
                            ),
                        ));
                    }
                }
                Some(tp) if !declared.contains(tp) => {
                    return Err(Error::coded(
                        "E107",
                        format!("grid {} names undeclared timepoint {}", repr_str(&grid.grid_id), repr_str(tp)),
                    ))
                }
                _ => {}
            }
        }
        let anns = self.group("annotations")?;
        for name in self.annotation_kinds.keys() {
            let node = anns.group(name)?;
            if let Some(ls) = &doc.label_set {
                if ls.form == "inline" {
                    if let Some(ids) = attrs::get_i64s(&node, "class_ids")? {
                        let missing = ls.missing(ids);
                        if !missing.is_empty() {
                            return Err(Error::coded(
                                "E402",
                                format!(
                                    "annotation {} uses class ids {} that are not in label set {}",
                                    repr_str(name),
                                    crate::json::repr_int_list(&missing),
                                    repr_str(&ls.id)
                                ),
                            ));
                        }
                    }
                }
            }
            if let Some(prov) = attrs::get_str(&node, "prov")? {
                if !doc.provenance.has_activity(&prov) {
                    return Err(Error::coded(
                        "E601",
                        format!(
                            "annotation {} names activity {}, which is not in the provenance graph",
                            repr_str(name),
                            repr_str(&prov)
                        ),
                    ));
                }
            }
            if let Some(q) = attrs::get_str(&node, "quality")? {
                if !doc.quality.contains_key(&q) {
                    return Err(Error::coded(
                        "E602",
                        format!(
                            "annotation {} names quality record {}, which does not exist",
                            repr_str(name),
                            repr_str(&q)
                        ),
                    ));
                }
            }
            if let Some(tps) = attrs::get_strs(&node, "timepoints")? {
                for tp in tps {
                    if !declared.contains(&tp) {
                        return Err(Error::coded(
                            "E409",
                            format!("annotation {} names undeclared timepoint {}", repr_str(name), repr_str(&tp)),
                        ));
                    }
                }
            }
            if let Some(mask) = attrs::get_str(&node, "ignore_mask")? {
                self.check_mask_reference(&format!("annotation {}", repr_str(name)), "ignore_mask", &mask, &node)?;
            }
            if let Some(derived) = attrs::get_strs(&node, "derived_from")? {
                for reference in derived {
                    if !self.annotation_kinds.contains_key(annotation_id(&reference)) {
                        return Err(Error::coded(
                            "E413",
                            format!(
                                "annotation {}: derived_from names {}, which does not exist",
                                repr_str(name),
                                repr_str(&reference)
                            ),
                        ));
                    }
                }
            }
        }
        let images = self.group("images")?;
        for image_id in self.images.keys() {
            let loc: hdf5::Location = if ops::is_group(&images, image_id) {
                (*images.group(image_id)?).clone()
            } else {
                let d = images.dataset(image_id)?;
                (**d).clone()
            };
            if let Some(prov) = attrs::get_str(&loc, "prov")? {
                if !doc.provenance.has_activity(&prov) {
                    return Err(Error::coded(
                        "E601",
                        format!(
                            "image {} names activity {}, which is not in the provenance graph",
                            repr_str(image_id),
                            repr_str(&prov)
                        ),
                    ));
                }
            }
            if let Some(mask) = attrs::get_str(&loc, "valid_mask")? {
                self.check_mask_reference(&format!("image {}", repr_str(image_id)), "valid_mask", &mask, &loc)?;
            }
        }
        if !self.transform_frames.is_empty() {
            let ts = self.group("transforms")?;
            for name in self.transform_frames.keys() {
                let g = ts.group(name)?;
                if let Some(prov) = attrs::get_str(&g, "prov")? {
                    if !doc.provenance.has_activity(&prov) {
                        return Err(Error::coded(
                            "E601",
                            format!(
                                "transform {} names activity {}, which is not in the provenance graph",
                                repr_str(name),
                                repr_str(&prov)
                            ),
                        ));
                    }
                }
                if let Some(m) = attrs::get_str(&g, "metrics")? {
                    if !doc.quality.contains_key(&m) {
                        return Err(Error::coded(
                            "E602",
                            format!(
                                "transform {} names quality record {}, which does not exist",
                                repr_str(name),
                                repr_str(&m)
                            ),
                        ));
                    }
                }
            }
        }
        let dangling = doc.provenance.dangling_agent_refs();
        if !dangling.is_empty() {
            let pairs: Vec<String> = dangling.iter().map(|(a, g)| format!("{a} -> {g}")).collect();
            return Err(Error::coded("E605", format!("activities name undeclared agents: {}", pairs.join(", "))));
        }
        Ok(())
    }

    fn check_mask_reference(&self, owner: &str, attr: &str, target: &str, node: &hdf5::Location) -> Result<()> {
        let Some(kind) = self.annotation_kinds.get(target) else {
            return Err(Error::coded(
                "E413",
                format!("{owner}: {attr} names annotation {}, which does not exist", repr_str(target)),
            ));
        };
        if kind != "mask" {
            return Err(Error::coded(
                "E413",
                format!(
                    "{owner}: {attr} names {}, whose kind is {}; §4.4 and §7.7 require a `mask` annotation",
                    repr_str(target),
                    repr_str(kind)
                ),
            ));
        }
        let mine = attrs::get_str(node, "grid")?;
        let target_group = self.group("annotations")?.group(target)?;
        let theirs = attrs::get_str(&target_group, "grid")?;
        if let (Some(m), Some(t)) = (mine, theirs) {
            if m != t {
                return Err(Error::coded(
                    "E413",
                    format!(
                        "{owner}: {attr} names {} on grid {}, but this object is on grid {}",
                        repr_str(target),
                        repr_str(&t),
                        repr_str(&m)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Validate, write `/meta`, stamp digests and atomically replace.
    ///
    /// `digests = false` stamps only the datasets that carry no digest yet;
    /// `content_id` is always computed.  Returns the `content_id`, or `None`
    /// when the writer had already committed.
    pub fn commit(&mut self, digests: bool) -> Result<Option<String>> {
        if self.committed {
            return Ok(None);
        }
        let result = self.commit_inner(digests);
        if result.is_err() {
            self.abort();
        }
        result.map(Some)
    }

    fn commit_inner(&mut self, digests: bool) -> Result<String> {
        if self.images.is_empty() {
            return Err(Error::coded("E201", "a sample must contain at least one image"));
        }
        self.check_references()?;
        let errors = self.document.check_schema();
        if !errors.is_empty() {
            return Err(Error::coded(
                "E005",
                format!(
                    "sample document fails its JSON Schema: {}",
                    errors.iter().take(5).cloned().collect::<Vec<_>>().join("; ")
                ),
            ));
        }
        let root = self.root()?;
        ops::unlink(&root, META_DATASET)?;
        data::create_scalar_string(&root, META_DATASET, &self.document.dumps())?;
        self.write_clinical()?;
        let profiles: Vec<String> = self.infer_profiles()?.into_iter().collect();
        let version = written_version(self.source_version.as_deref(), &profiles);
        attrs::write(&root, "medh5_version", &AttrValue::Str(version))?;
        attrs::write(&root, "medh5_kind", &AttrValue::Str("sample".into()))?;
        attrs::write(&root, "medh5_profiles", &crate::annotations::header::list_attr(&profiles))?;
        attrs::write(&root, "created", &AttrValue::Str(utcnow()))?;
        attrs::write(&root, "generator", &AttrValue::Str(format!("medh5 {VERSION}")))?;
        attrs::write(&root, "digest_algo", &AttrValue::Str("sha256".into()))?;
        stamp_digests(&root, "sha256", &["index"], !digests)?;
        let content_id = compute_content_id(&root, &attr_name_map_of(&root)?, "sha256", None)?;
        attrs::write(&root, "content_id", &AttrValue::Str(content_id.clone()))?;
        self.check_valid()?;
        drop(root);
        self.committed = true;
        if let Some(file) = self.file.take() {
            file.commit()?;
        }
        Ok(content_id)
    }

    /// Refuse to write a file the validator would reject (§15): the
    /// validator's structural and semantic *error* rules over the finished
    /// temporary file are the one definition of a valid file.
    fn check_valid(&mut self) -> Result<()> {
        let root = self.root()?;
        let report = crate::validate::validate_root(&root, Some(&self.path), "semantic", true)?;
        let errors: Vec<_> = report.diagnostics.iter().filter(|d| d.severity == "error").collect();
        if errors.is_empty() {
            return Ok(());
        }
        let listed: Vec<String> =
            errors.iter().take(5).map(|d| format!("{} {}: {}", d.code, d.location, d.message)).collect();
        let more = if errors.len() > 5 { format!(" (and {} more)", errors.len() - 5) } else { String::new() };
        Err(Error::coded(
            &errors[0].code,
            format!(
                "refusing to write {}: the file would fail validation: {}{more}",
                repr_str(&self.path.to_string_lossy()),
                listed.join("; ")
            ),
        ))
    }
}

impl Drop for SampleWriter {
    fn drop(&mut self) {
        // A writer dropped without `commit` is an abort: the temporary
        // sibling goes, the target is untouched.
        if let Some(f) = self.file.take() {
            f.abort();
        }
    }
}

//! Label sets: id -> meaning (spec §5).
//!
//! Annotations reference classes by `uint16` id and never by name, so a label
//! set is the only thing standing between an integer and a diagnosis.  Two
//! properties matter more than the data model:
//!
//! * The hierarchy is a **DAG, not a tree**.  `left_kidney` is a `kidney` and is
//!   part of the urinary system; forcing that into a tree loses one of the two.
//! * `closure` is declared per annotation, never inferred.  A reader that adds
//!   `liver` because `liver_segment_iv` is present has invented ground truth,
//!   so the spec forbids it unless `closure = "implicit"` says otherwise.

pub mod registry;

use std::collections::{BTreeSet, HashMap, HashSet};

use serde_json::{json, Map, Value};

use crate::digest::hash_hex;
use crate::json::{canonical, repr_list, repr_str};
use crate::pyval::{self, get, get_list, get_str, require, to_int, to_str};
use crate::{Error, Result};

/// Explicitly *not* any class.  Never appears in `classes`.
pub const BACKGROUND_ID: i64 = 0;
/// Outside what was annotated: neither foreground nor background.
pub const IGNORE_ID: i64 = 65535;
/// The largest assignable class id.
pub const MAX_CLASS_ID: i64 = 65534;
/// The two closures an annotation may declare (§5.4).
pub const CLOSURES: [&str; 2] = ["explicit", "implicit"];
/// The two ways a label set may be carried (§5.1).
pub const FORMS: [&str; 2] = ["inline", "ref"];
/// At or below this class count, `form: inline` is REQUIRED (§5.1).
pub const INLINE_REQUIRED_BELOW: usize = 4096;

/// A class id in the writable range `1..=65534`, or a coded refusal (E303).
///
/// Every encoder that stores class ids goes through this before its `uint16`
/// cast; unchecked, `-1` became 65535 (the ignore value) and `70000` became
/// `4464`, each decoding as a different class than was asked for.
pub fn check_class_id(class_id: i64) -> Result<u16> {
    if !(BACKGROUND_ID < class_id && class_id <= MAX_CLASS_ID) {
        return Err(Error::coded(
            "E303",
            format!(
                "class id {class_id} is outside the writable range [{}, {MAX_CLASS_ID}]: \
                 {BACKGROUND_ID} is background and {IGNORE_ID} is ignore (spec §5.3)",
                BACKGROUND_ID + 1
            ),
        ));
    }
    Ok(class_id as u16)
}

/// A binding to an external vocabulary (SNOMED-CT, RadLex, FMA, UBERON, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct OntologyCode {
    pub system: String,
    pub code: String,
    pub name: Option<String>,
}

impl OntologyCode {
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("system".into(), json!(self.system));
        out.insert("code".into(), json!(self.code));
        if let Some(name) = &self.name {
            out.insert("name".into(), json!(name));
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "an ontology code")?;
        Ok(OntologyCode {
            system: to_str(require(doc, "system")?),
            code: to_str(require(doc, "code")?),
            name: get_str(doc, "name"),
        })
    }
}

/// A non-`is_a` edge between classes, e.g. `part_of` or `adjacent_to`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Relation {
    pub subject: i64,
    pub predicate: String,
    pub object: i64,
}

impl Relation {
    pub fn to_json(&self) -> Value {
        json!({"subject": self.subject, "predicate": self.predicate, "object": self.object})
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "a relation")?;
        Ok(Relation {
            subject: to_int(require(doc, "subject")?)?,
            predicate: to_str(require(doc, "predicate")?),
            object: to_int(require(doc, "object")?)?,
        })
    }
}

/// A keypoint topology declared by the vocabulary (spec §5.5).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Skeleton {
    pub id: String,
    pub keypoints: Vec<i64>,
    pub edges: Vec<(i64, i64)>,
}

impl Skeleton {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "keypoints": self.keypoints,
            "edges": self.edges.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
        })
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "a skeleton")?;
        let keypoints = match require(doc, "keypoints")? {
            Value::Array(items) => items.iter().map(to_int).collect::<Result<Vec<_>>>()?,
            other => return Err(Error::Type(format!("keypoints must be a list, not {}", pyval::type_name(other)))),
        };
        let mut edges = Vec::new();
        for edge in get_list(doc, "edges") {
            let pair = edge
                .as_array()
                .filter(|p| p.len() == 2)
                .ok_or_else(|| Error::Value("a skeleton edge must be a pair of keypoint ids".into()))?;
            edges.push((to_int(&pair[0])?, to_int(&pair[1])?));
        }
        Ok(Skeleton { id: to_str(require(doc, "id")?), keypoints, edges })
    }
}

/// One semantic class (spec §5.2).
#[derive(Debug, Clone, PartialEq)]
pub struct LabelClass {
    pub id: i64,
    pub key: String,
    pub name: String,
    pub parents: Vec<i64>,
    pub category: Option<String>,
    pub color: Option<[i64; 4]>,
    pub codes: Vec<OntologyCode>,
    pub laterality: Option<String>,
    pub properties: Map<String, Value>,
}

impl LabelClass {
    /// A class with no optional fields, validated.
    pub fn new(id: i64, key: impl Into<String>, name: impl Into<String>) -> Result<Self> {
        Self::build(id, key.into(), name.into(), Vec::new(), None, None, Vec::new(), None, Map::new())
    }

    /// Construct and validate every field.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        id: i64,
        key: String,
        name: String,
        parents: Vec<i64>,
        category: Option<String>,
        color: Option<Vec<i64>>,
        codes: Vec<OntologyCode>,
        laterality: Option<String>,
        properties: Map<String, Value>,
    ) -> Result<Self> {
        if !(BACKGROUND_ID < id && id <= MAX_CLASS_ID) {
            return Err(Error::coded(
                "E303",
                format!(
                    "class {}: id {id} must be in 1..{MAX_CLASS_ID} ({BACKGROUND_ID} is background, \
                     {IGNORE_ID} is ignore)",
                    repr_str(&key)
                ),
            ));
        }
        if key.is_empty() || name.is_empty() {
            return Err(Error::coded("E306", format!("class id {id}: both key and name are required")));
        }
        let color = match color {
            None => None,
            Some(values) => {
                if values.len() != 4 || values.iter().any(|c| !(0..=255).contains(c)) {
                    return Err(Error::coded(
                        "E306",
                        format!("class {}: color must be four 0-255 RGBA values", repr_str(&key)),
                    ));
                }
                Some([values[0], values[1], values[2], values[3]])
            }
        };
        Ok(LabelClass { id, key, name, parents, category, color, codes, laterality, properties })
    }

    /// Whether the class marks a lesion: `properties.is_lesion`, else category.
    pub fn is_lesion(&self) -> bool {
        match self.properties.get("is_lesion") {
            Some(v) => pyval::truthy(v),
            None => self.category.as_deref() == Some("lesion"),
        }
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.id));
        out.insert("key".into(), json!(self.key));
        out.insert("name".into(), json!(self.name));
        if !self.parents.is_empty() {
            out.insert("parents".into(), json!(self.parents));
        }
        if let Some(category) = &self.category {
            out.insert("category".into(), json!(category));
        }
        if let Some(color) = &self.color {
            out.insert("color".into(), json!(color));
        }
        if !self.codes.is_empty() {
            out.insert("codes".into(), Value::Array(self.codes.iter().map(|c| c.to_json()).collect()));
        }
        if let Some(laterality) = &self.laterality {
            out.insert("laterality".into(), json!(laterality));
        }
        if !self.properties.is_empty() {
            out.insert("properties".into(), Value::Object(self.properties.clone()));
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "a label class")?;
        let parents = get_list(doc, "parents").iter().map(to_int).collect::<Result<Vec<_>>>()?;
        let color = match get(doc, "color") {
            Some(Value::Array(items)) if !items.is_empty() => {
                Some(items.iter().map(to_int).collect::<Result<Vec<_>>>()?)
            }
            _ => None,
        };
        let codes = get_list(doc, "codes").iter().map(OntologyCode::from_json).collect::<Result<Vec<_>>>()?;
        let properties = match get(doc, "properties") {
            Some(Value::Object(map)) => map.clone(),
            _ => Map::new(),
        };
        Self::build(
            to_int(require(doc, "id")?)?,
            to_str(require(doc, "key")?),
            to_str(require(doc, "name")?),
            parents,
            get_str(doc, "category"),
            color,
            codes,
            get_str(doc, "laterality"),
            properties,
        )
    }
}

/// Deterministic UTF-8 serialization used for label-set digests (§5.1):
/// sorted keys, no insignificant whitespace, non-ASCII kept as UTF-8.
pub fn canonical_json(doc: &Value) -> Vec<u8> {
    canonical(doc).into_bytes()
}

/// A controlled vocabulary, inline in the file or referenced by URI.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelSet {
    pub id: String,
    pub version: String,
    pub form: String,
    pub uri: Option<String>,
    declared_sha256: Option<String>,
    pub relations: Vec<Relation>,
    pub skeletons: Vec<Skeleton>,
    classes: Vec<LabelClass>,
    by_id: HashMap<i64, usize>,
    by_key: HashMap<String, usize>,
}

/// A class lookup key: an id or a machine key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClassKey {
    Id(i64),
    Key(String),
}

impl ClassKey {
    /// Python's `repr()` of the key.
    pub fn repr(&self) -> String {
        match self {
            ClassKey::Id(i) => i.to_string(),
            ClassKey::Key(k) => repr_str(k),
        }
    }
}

impl From<i64> for ClassKey {
    fn from(v: i64) -> Self {
        ClassKey::Id(v)
    }
}

impl From<u16> for ClassKey {
    fn from(v: u16) -> Self {
        ClassKey::Id(i64::from(v))
    }
}

impl From<&str> for ClassKey {
    fn from(v: &str) -> Self {
        ClassKey::Key(v.to_string())
    }
}

impl From<String> for ClassKey {
    fn from(v: String) -> Self {
        ClassKey::Key(v)
    }
}

impl LabelSet {
    /// Build and validate a label set (§5.1-§5.3).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        classes: Vec<LabelClass>,
        version: impl Into<String>,
        relations: Vec<Relation>,
        skeletons: Vec<Skeleton>,
        form: impl Into<String>,
        uri: Option<String>,
        sha256: Option<String>,
    ) -> Result<Self> {
        let id = id.into();
        let form = form.into();
        if !FORMS.contains(&form.as_str()) {
            return Err(Error::coded(
                "E305",
                format!("label set form {} must be one of {}", repr_str(&form), repr_list(&FORMS)),
            ));
        }
        let mut classes = classes;
        classes.sort_by_key(|c| c.id);
        let mut by_id = HashMap::new();
        let mut by_key = HashMap::new();
        for (i, c) in classes.iter().enumerate() {
            by_id.entry(c.id).or_insert(i);
            by_key.entry(c.key.clone()).or_insert(i);
        }
        let set = LabelSet {
            id,
            version: version.into(),
            form,
            uri,
            declared_sha256: sha256,
            relations,
            skeletons,
            classes,
            by_id,
            by_key,
        };
        set.check()?;
        Ok(set)
    }

    /// An inline label set of `classes`, version `1.0.0`.
    pub fn inline(id: impl Into<String>, classes: Vec<LabelClass>) -> Result<Self> {
        Self::new(id, classes, "1.0.0", Vec::new(), Vec::new(), "inline", None, None)
    }

    /// Validate spec §5.1-§5.3 (E302, E303, E304, E305, E306).
    pub fn check(&self) -> Result<()> {
        if self.by_id.len() != self.classes.len() {
            let dupes = duplicates(self.classes.iter().map(|c| c.id));
            let mut dupes: Vec<i64> = dupes.into_iter().collect();
            dupes.sort();
            return Err(Error::coded(
                "E302",
                format!("label set {}: duplicate class ids {}", repr_str(&self.id), crate::json::repr_int_list(&dupes)),
            ));
        }
        if self.by_key.len() != self.classes.len() {
            let mut dupes: Vec<String> = duplicates(self.classes.iter().map(|c| c.key.clone())).into_iter().collect();
            dupes.sort();
            return Err(Error::coded(
                "E302",
                format!("label set {}: duplicate class keys {}", repr_str(&self.id), repr_list(&dupes)),
            ));
        }
        for class in &self.classes {
            for parent in &class.parents {
                if !self.by_id.contains_key(parent) {
                    return Err(Error::coded(
                        "E306",
                        format!(
                            "label set {}: class {} names unknown parent id {parent}",
                            repr_str(&self.id),
                            repr_str(&class.key)
                        ),
                    ));
                }
            }
        }
        self.check_acyclic()?;
        if self.form == "ref" && self.uri.as_deref().map(str::is_empty).unwrap_or(true) {
            return Err(Error::coded("E305", format!("label set {}: form 'ref' requires a uri", repr_str(&self.id))));
        }
        if self.form == "ref" && self.declared_sha256.as_deref().map(str::is_empty).unwrap_or(true) {
            return Err(Error::coded(
                "E305",
                format!(
                    "label set {}: form 'ref' requires a sha256, so a reader that cannot resolve \
                     the uri still knows which vocabulary it needs",
                    repr_str(&self.id)
                ),
            ));
        }
        if self.form == "ref" && !self.classes.is_empty() {
            return Err(Error::coded(
                "E305",
                format!("label set {}: form 'ref' must not carry inline classes", repr_str(&self.id)),
            ));
        }
        for relation in &self.relations {
            for endpoint in [relation.subject, relation.object] {
                if !self.by_id.contains_key(&endpoint) && self.form == "inline" {
                    return Err(Error::coded(
                        "E306",
                        format!("label set {}: relation names unknown class {endpoint}", repr_str(&self.id)),
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_acyclic(&self) -> Result<()> {
        // 0 = unvisited, 1 = on the current path, 2 = done.
        let mut colour: HashMap<i64, u8> = HashMap::new();
        fn visit(set: &LabelSet, node: i64, path: &mut Vec<i64>, colour: &mut HashMap<i64, u8>) -> Result<()> {
            match colour.get(&node).copied().unwrap_or(0) {
                1 => {
                    let mut names: Vec<&str> = path.iter().map(|n| set.class_by_id(*n).key.as_str()).collect();
                    names.push(&set.class_by_id(node).key);
                    return Err(Error::coded(
                        "E304",
                        format!("label set {}: hierarchy cycle {}", repr_str(&set.id), names.join(" -> ")),
                    ));
                }
                2 => return Ok(()),
                _ => {}
            }
            colour.insert(node, 1);
            path.push(node);
            for parent in set.class_by_id(node).parents.clone() {
                visit(set, parent, path, colour)?;
            }
            path.pop();
            colour.insert(node, 2);
            Ok(())
        }
        for class in &self.classes {
            visit(self, class.id, &mut Vec::new(), &mut colour)?;
        }
        Ok(())
    }

    fn class_by_id(&self, id: i64) -> &LabelClass {
        &self.classes[self.by_id[&id]]
    }

    // -- lookup -----------------------------------------------------------

    /// The number of classes.
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// Whether the set has no inline classes.
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// Classes, sorted by id.
    pub fn classes(&self) -> &[LabelClass] {
        &self.classes
    }

    /// Class ids, ascending.
    pub fn ids(&self) -> Vec<i64> {
        self.classes.iter().map(|c| c.id).collect()
    }

    /// Class keys, in id order.
    pub fn keys(&self) -> Vec<String> {
        self.classes.iter().map(|c| c.key.clone()).collect()
    }

    /// Whether an id or key names a class.
    pub fn contains(&self, key: &ClassKey) -> bool {
        match key {
            ClassKey::Id(i) => self.by_id.contains_key(i),
            ClassKey::Key(k) => self.by_key.contains_key(k),
        }
    }

    /// Whether `id` names a class.
    pub fn contains_id(&self, id: i64) -> bool {
        self.by_id.contains_key(&id)
    }

    /// Look up a class by id or key.
    pub fn get(&self, key: &ClassKey) -> Option<&LabelClass> {
        let index = match key {
            ClassKey::Id(i) => self.by_id.get(i),
            ClassKey::Key(k) => self.by_key.get(k),
        }?;
        Some(&self.classes[*index])
    }

    /// Look up a class by id or key, or a `KeyError`.
    pub fn lookup(&self, key: &ClassKey) -> Result<&LabelClass> {
        self.get(key).ok_or_else(|| Error::Key(format!("label set {} has no class {}", repr_str(&self.id), key.repr())))
    }

    /// Look up a class by id.
    pub fn by_id(&self, id: i64) -> Option<&LabelClass> {
        self.by_id.get(&id).map(|i| &self.classes[*i])
    }

    /// Look up a class by key.
    pub fn by_key(&self, key: &str) -> Option<&LabelClass> {
        self.by_key.get(key).map(|i| &self.classes[*i])
    }

    /// Resolve a mixed sequence of ids and keys to classes, in order.
    pub fn resolve(&self, keys: &[ClassKey]) -> Result<Vec<&LabelClass>> {
        keys.iter().map(|k| self.lookup(k)).collect()
    }

    /// Resolve a mixed sequence of ids and keys to class ids, in order.
    pub fn ids_for(&self, keys: &[ClassKey]) -> Result<Vec<i64>> {
        keys.iter().map(|k| self.lookup(k).map(|c| c.id)).collect()
    }

    /// Ids not present in this label set --- the E402 check.
    pub fn missing(&self, ids: impl IntoIterator<Item = i64>) -> Vec<i64> {
        let set: BTreeSet<i64> = ids.into_iter().filter(|i| !self.by_id.contains_key(i)).collect();
        set.into_iter().collect()
    }

    // -- hierarchy --------------------------------------------------------

    /// Transitive `is_a` ancestors of a class, nearest first, deduplicated.
    pub fn ancestors(&self, key: &ClassKey) -> Result<Vec<i64>> {
        let start = self.lookup(key)?;
        let mut seen: Vec<i64> = Vec::new();
        let mut seen_set: HashSet<i64> = HashSet::new();
        let mut frontier: std::collections::VecDeque<i64> = start.parents.iter().copied().collect();
        while let Some(node) = frontier.pop_front() {
            if !seen_set.insert(node) {
                continue;
            }
            seen.push(node);
            if let Some(class) = self.by_id(node) {
                frontier.extend(class.parents.iter().copied());
            }
        }
        Ok(seen)
    }

    /// Every class having this one among its transitive ancestors.
    pub fn descendants(&self, key: &ClassKey) -> Result<Vec<i64>> {
        let target = self.lookup(key)?.id;
        let mut out = Vec::new();
        for class in &self.classes {
            if self.ancestors(&ClassKey::Id(class.id))?.contains(&target) {
                out.push(class.id);
            }
        }
        Ok(out)
    }

    /// Apply a `closure` (spec §5.4) to a set of class ids.
    pub fn close(&self, ids: &[i64], closure: &str) -> Result<Vec<i64>> {
        if !CLOSURES.contains(&closure) {
            return Err(Error::coded(
                "E412",
                format!("closure {} must be one of {}", repr_str(closure), repr_list(&CLOSURES)),
            ));
        }
        let mut ordered: Vec<i64> = Vec::new();
        for id in ids {
            if !ordered.contains(id) {
                ordered.push(*id);
            }
        }
        if closure == "explicit" {
            return Ok(ordered);
        }
        let mut out = ordered.clone();
        for id in &ordered {
            for ancestor in self.ancestors(&ClassKey::Id(*id))? {
                if !out.contains(&ancestor) {
                    out.push(ancestor);
                }
            }
        }
        Ok(out)
    }

    /// Relations whose subject is the class, optionally of one predicate.
    pub fn relations_of(&self, key: &ClassKey, predicate: Option<&str>) -> Result<Vec<&Relation>> {
        let subject = self.lookup(key)?.id;
        Ok(self
            .relations
            .iter()
            .filter(|r| r.subject == subject && predicate.map(|p| r.predicate == p).unwrap_or(true))
            .collect())
    }

    /// A declared skeleton, or a `KeyError`.
    pub fn skeleton(&self, skeleton_id: &str) -> Result<&Skeleton> {
        self.skeletons.iter().find(|s| s.id == skeleton_id).ok_or_else(|| {
            Error::Key(format!("label set {} has no skeleton {}", repr_str(&self.id), repr_str(skeleton_id)))
        })
    }

    /// Class id -> RGBA, for viewers.  Classes without a colour are omitted.
    pub fn colors(&self) -> Vec<(i64, [i64; 4])> {
        self.classes.iter().filter_map(|c| c.color.map(|col| (c.id, col))).collect()
    }

    // -- serialization ----------------------------------------------------

    /// The digested part of the label set: identity plus content, no carriage.
    pub fn content_doc(&self) -> Value {
        let mut doc = Map::new();
        doc.insert("id".into(), json!(self.id));
        doc.insert("version".into(), json!(self.version));
        doc.insert("classes".into(), Value::Array(self.classes.iter().map(|c| c.to_json()).collect()));
        if !self.relations.is_empty() {
            let mut rels: Vec<&Relation> = self.relations.iter().collect();
            rels.sort_by(|a, b| (a.subject, &a.predicate, a.object).cmp(&(b.subject, &b.predicate, b.object)));
            doc.insert("relations".into(), Value::Array(rels.iter().map(|r| r.to_json()).collect()));
        }
        if !self.skeletons.is_empty() {
            let mut sks: Vec<&Skeleton> = self.skeletons.iter().collect();
            sks.sort_by(|a, b| a.id.cmp(&b.id));
            doc.insert("skeletons".into(), Value::Array(sks.iter().map(|s| s.to_json()).collect()));
        }
        Value::Object(doc)
    }

    /// Hex digest of [`content_doc`](Self::content_doc) under canonical JSON.
    ///
    /// A `form: ref` set carries the digest rather than computing one: it has
    /// no inline classes to hash.
    pub fn digest(&self, algo: &str) -> Result<String> {
        if let Some(declared) = &self.declared_sha256 {
            if self.classes.is_empty() {
                return Ok(declared.clone());
            }
        }
        hash_hex(algo, &canonical_json(&self.content_doc()))
    }

    /// The `sha256` the set declares or computes.
    pub fn sha256(&self) -> String {
        self.digest("sha256").expect("sha256 is always available")
    }

    /// The declared `sha256`, as read.
    pub fn declared_sha256(&self) -> Option<&str> {
        self.declared_sha256.as_deref()
    }

    /// The `/meta -> label_set` document.
    pub fn to_json(&self, form: Option<&str>) -> Value {
        let resolved = form.unwrap_or(&self.form);
        let mut doc = Map::new();
        doc.insert("id".into(), json!(self.id));
        doc.insert("version".into(), json!(self.version));
        doc.insert("sha256".into(), json!(self.sha256()));
        doc.insert("form".into(), json!(resolved));
        if let Some(uri) = &self.uri {
            doc.insert("uri".into(), json!(uri));
        }
        if resolved == "inline" {
            doc.insert("classes".into(), Value::Array(self.classes.iter().map(|c| c.to_json()).collect()));
            if !self.relations.is_empty() {
                doc.insert("relations".into(), Value::Array(self.relations.iter().map(|r| r.to_json()).collect()));
            }
            if !self.skeletons.is_empty() {
                doc.insert("skeletons".into(), Value::Array(self.skeletons.iter().map(|s| s.to_json()).collect()));
            }
        }
        Value::Object(doc)
    }

    /// Parse `/meta -> label_set`; `None` for an absent or empty value.
    pub fn from_json(doc: Option<&Value>) -> Result<Option<Self>> {
        let Some(doc) = doc else { return Ok(None) };
        if !pyval::truthy(doc) {
            return Ok(None);
        }
        let doc = pyval::as_object(doc, "a label set")?;
        let classes = get_list(doc, "classes").iter().map(LabelClass::from_json).collect::<Result<Vec<_>>>()?;
        let relations = get_list(doc, "relations").iter().map(Relation::from_json).collect::<Result<Vec<_>>>()?;
        let skeletons = get_list(doc, "skeletons").iter().map(Skeleton::from_json).collect::<Result<Vec<_>>>()?;
        Ok(Some(Self::new(
            to_str(require(doc, "id")?),
            classes,
            get(doc, "version").map(to_str).unwrap_or_else(|| "1.0.0".into()),
            relations,
            skeletons,
            get(doc, "form").map(to_str).unwrap_or_else(|| "inline".into()),
            get_str(doc, "uri"),
            get_str(doc, "sha256"),
        )?))
    }

    /// A `form: ref` view of this vocabulary, for collection-level sharing.
    pub fn as_ref(&self, uri: &str) -> Result<LabelSet> {
        LabelSet::new(
            self.id.clone(),
            Vec::new(),
            self.version.clone(),
            Vec::new(),
            Vec::new(),
            "ref",
            Some(uri.to_string()),
            Some(self.sha256()),
        )
    }

    /// A vocabulary restricted to `keys` plus their ancestors.
    pub fn subset(&self, keys: &[ClassKey], id: Option<&str>) -> Result<LabelSet> {
        let mut wanted: HashSet<i64> = HashSet::new();
        for key in keys {
            wanted.insert(self.lookup(key)?.id);
        }
        for cid in wanted.clone() {
            wanted.extend(self.ancestors(&ClassKey::Id(cid))?);
        }
        LabelSet::new(
            id.map(str::to_string).unwrap_or_else(|| format!("{}-subset", self.id)),
            self.classes.iter().filter(|c| wanted.contains(&c.id)).cloned().collect(),
            self.version.clone(),
            self.relations
                .iter()
                .filter(|r| wanted.contains(&r.subject) && wanted.contains(&r.object))
                .cloned()
                .collect(),
            Vec::new(),
            "inline",
            None,
            None,
        )
    }

    /// Python's `repr()` of the set.
    pub fn repr(&self) -> String {
        format!(
            "LabelSet({}, version={}, {} classes, form={})",
            repr_str(&self.id),
            repr_str(&self.version),
            self.classes.len(),
            repr_str(&self.form)
        )
    }
}

fn duplicates<T: Eq + std::hash::Hash + Clone>(values: impl IntoIterator<Item = T>) -> HashSet<T> {
    let mut seen = HashSet::new();
    let mut dupes = HashSet::new();
    for v in values {
        if !seen.insert(v.clone()) {
            dupes.insert(v);
        }
    }
    dupes
}

/// Mint a vocabulary from bare names --- what converters do on ingest.
pub fn from_keys(keys: &[String], id: &str, version: &str, start: i64) -> Result<LabelSet> {
    let classes = keys
        .iter()
        .enumerate()
        .map(|(i, key)| LabelClass::new(start + i as i64, key.clone(), pyval::title_case(&key.replace('_', " "))))
        .collect::<Result<Vec<_>>>()?;
    LabelSet::new(id, classes, version, Vec::new(), Vec::new(), "inline", None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LabelSet {
        let mut lesion = LabelClass::new(3, "lesion", "Lesion").unwrap();
        lesion.parents = vec![1];
        lesion.category = Some("lesion".into());
        LabelSet::new(
            "test-v1",
            vec![
                LabelClass::new(1, "liver", "Liver").unwrap(),
                LabelClass::new(2, "spleen", "Spleen").unwrap(),
                lesion,
            ],
            "1.0.0",
            vec![],
            vec![],
            "inline",
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn hierarchy_and_closure() {
        let ls = sample();
        assert_eq!(ls.ancestors(&ClassKey::Key("lesion".into())).unwrap(), vec![1]);
        assert_eq!(ls.close(&[3], "implicit").unwrap(), vec![3, 1]);
        assert_eq!(ls.close(&[3], "explicit").unwrap(), vec![3]);
        assert_eq!(ls.descendants(&ClassKey::Id(1)).unwrap(), vec![3]);
    }

    #[test]
    fn duplicate_and_cycle_refused() {
        let err =
            LabelSet::inline("x", vec![LabelClass::new(1, "a", "A").unwrap(), LabelClass::new(1, "b", "B").unwrap()])
                .unwrap_err();
        assert_eq!(err.code(), Some("E302"));
        let mut a = LabelClass::new(1, "a", "A").unwrap();
        a.parents = vec![2];
        let mut b = LabelClass::new(2, "b", "B").unwrap();
        b.parents = vec![1];
        let err = LabelSet::inline("x", vec![a, b]).unwrap_err();
        assert_eq!(err.code(), Some("E304"));
        assert!(err.message().ends_with("hierarchy cycle a -> b -> a"));
    }

    #[test]
    fn reserved_ids_refused() {
        assert_eq!(LabelClass::new(0, "bg", "Bg").unwrap_err().code(), Some("E303"));
        assert_eq!(check_class_id(65535).unwrap_err().code(), Some("E303"));
        assert_eq!(check_class_id(5).unwrap(), 5);
    }
}

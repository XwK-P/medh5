//! `medh5.labels`: the vocabulary (spec §5).

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyString, PyTuple};
use serde_json::{Map, Value};

use medh5::labels::{self as engine, registry, ClassKey};

use crate::convert::{class_key, class_keys, json_to_py, map_to_py, py_to_json};
use crate::errors::R;
use crate::values::{dataclass_repr, json_hash, opt, reduce_via_json};

const MODULE: &str = "medh5.labels.labelset";

// -- OntologyCode ----------------------------------------------------------------------

#[pyclass(module = "medh5.labels.labelset", name = "OntologyCode", skip_from_py_object, frozen, eq)]
#[derive(Clone, PartialEq)]
pub struct OntologyCode(pub engine::OntologyCode);

#[pymethods]
impl OntologyCode {
    #[new]
    #[pyo3(signature = (system, code, name=None))]
    fn new(system: String, code: String, name: Option<String>) -> Self {
        OntologyCode(engine::OntologyCode { system, code, name })
    }
    #[getter]
    fn system(&self) -> &str {
        &self.0.system
    }
    #[getter]
    fn code(&self) -> &str {
        &self.0.code
    }
    #[getter]
    fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(OntologyCode(engine::OntologyCode::from_json(&py_to_json(doc)?)?))
    }
    fn __hash__(&self) -> isize {
        json_hash(&self.0.to_json())
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        dataclass_repr(
            "OntologyCode",
            &[
                ("system", PyString::new(py, &self.0.system).into_any()),
                ("code", PyString::new(py, &self.0.code).into_any()),
                ("name", opt(py, self.0.name.as_deref())?),
            ],
        )
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        reduce_via_json(slf.as_any())
    }
}

// -- Relation ----------------------------------------------------------------------------

#[pyclass(module = "medh5.labels.labelset", name = "Relation", skip_from_py_object, frozen, eq, hash)]
#[derive(Clone, PartialEq, Hash)]
pub struct Relation(pub engine::Relation);

#[pymethods]
impl Relation {
    #[new]
    fn new(subject: i64, predicate: String, object: i64) -> Self {
        Relation(engine::Relation { subject, predicate, object })
    }
    #[getter]
    fn subject(&self) -> i64 {
        self.0.subject
    }
    #[getter]
    fn predicate(&self) -> &str {
        &self.0.predicate
    }
    #[getter]
    fn object(&self) -> i64 {
        self.0.object
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(Relation(engine::Relation::from_json(&py_to_json(doc)?)?))
    }
    fn __repr__(&self) -> String {
        format!(
            "Relation(subject={}, predicate={}, object={})",
            self.0.subject,
            medh5::json::repr_str(&self.0.predicate),
            self.0.object
        )
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        reduce_via_json(slf.as_any())
    }
}

// -- Skeleton ----------------------------------------------------------------------------

#[pyclass(module = "medh5.labels.labelset", name = "Skeleton", skip_from_py_object, frozen, eq, hash)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Skeleton(pub engine::Skeleton);

#[pymethods]
impl Skeleton {
    #[new]
    #[pyo3(signature = (id, keypoints, edges=Vec::new()))]
    fn new(id: String, keypoints: Vec<i64>, edges: Vec<(i64, i64)>) -> Self {
        Skeleton(engine::Skeleton { id, keypoints, edges })
    }
    #[getter]
    fn id(&self) -> &str {
        &self.0.id
    }
    #[getter]
    fn keypoints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.keypoints)
    }
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.edges)
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(Skeleton(engine::Skeleton::from_json(&py_to_json(doc)?)?))
    }
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        dataclass_repr(
            "Skeleton",
            &[
                ("id", PyString::new(py, &slf.get().0.id).into_any()),
                ("keypoints", slf.getattr("keypoints")?),
                ("edges", slf.getattr("edges")?),
            ],
        )
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        reduce_via_json(slf.as_any())
    }
}

// -- LabelClass --------------------------------------------------------------------------

#[pyclass(module = "medh5.labels.labelset", name = "LabelClass", skip_from_py_object, frozen, eq)]
#[derive(Clone, PartialEq)]
pub struct LabelClass(pub engine::LabelClass);

fn ontology_codes(codes: Option<&Bound<'_, PyAny>>) -> R<Vec<engine::OntologyCode>> {
    let mut out = Vec::new();
    let Some(codes) = codes else { return Ok(out) };
    for item in codes.try_iter()? {
        let item = item?;
        match item.cast::<OntologyCode>() {
            Ok(c) => out.push(c.get().0.clone()),
            Err(_) => out.push(engine::OntologyCode::from_json(&py_to_json(&item)?)?),
        }
    }
    Ok(out)
}

impl LabelClass {
    pub fn tuple_color<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.0.color {
            Some(c) => Ok(PyTuple::new(py, c)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }
}

#[pymethods]
impl LabelClass {
    #[new]
    #[pyo3(signature = (id, key, name, parents=None, category=None, color=None, codes=None, laterality=None, properties=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: i64,
        key: String,
        name: String,
        parents: Option<Vec<i64>>,
        category: Option<String>,
        color: Option<Vec<i64>>,
        codes: Option<&Bound<'_, PyAny>>,
        laterality: Option<String>,
        properties: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let properties = match properties {
            Some(p) if !p.is_none() => match py_to_json(p)? {
                Value::Object(m) => m,
                _ => Map::new(),
            },
            _ => Map::new(),
        };
        Ok(LabelClass(engine::LabelClass::build(
            id,
            key,
            name,
            parents.unwrap_or_default(),
            category,
            color,
            ontology_codes(codes)?,
            laterality,
            properties,
        )?))
    }
    #[getter]
    fn id(&self) -> i64 {
        self.0.id
    }
    #[getter]
    fn key(&self) -> &str {
        &self.0.key
    }
    #[getter]
    fn name(&self) -> &str {
        &self.0.name
    }
    #[getter]
    fn parents<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.parents)
    }
    #[getter]
    fn category(&self) -> Option<&str> {
        self.0.category.as_deref()
    }
    #[getter]
    fn color<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.tuple_color(py)
    }
    #[getter]
    fn codes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.codes.iter().map(|c| OntologyCode(c.clone())))
    }
    #[getter]
    fn laterality(&self) -> Option<&str> {
        self.0.laterality.as_deref()
    }
    #[getter]
    fn properties<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_py(py, &self.0.properties)
    }
    #[getter]
    fn is_lesion(&self) -> bool {
        self.0.is_lesion()
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(LabelClass(engine::LabelClass::from_json(&py_to_json(doc)?)?))
    }
    fn __hash__(&self) -> isize {
        json_hash(&self.0.to_json())
    }
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let names = ["id", "key", "name", "parents", "category", "color", "codes", "laterality", "properties"];
        let fields = names.iter().map(|n| Ok((*n, slf.getattr(*n)?))).collect::<PyResult<Vec<_>>>()?;
        dataclass_repr("LabelClass", &fields)
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        reduce_via_json(slf.as_any())
    }
}

// -- LabelSet ------------------------------------------------------------------------------

#[pyclass(module = "medh5.labels.labelset", name = "LabelSet", skip_from_py_object, frozen)]
pub struct LabelSet(pub engine::LabelSet, PyOnceLock<Py<PyTuple>>);

impl LabelSet {
    pub fn wrap(inner: engine::LabelSet) -> Self {
        LabelSet(inner, PyOnceLock::new())
    }

    /// The classes as Python objects, made once: `ls[2] is ls["liver"]`.
    fn class_objects<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyTuple>> {
        Ok(self
            .1
            .get_or_try_init(py, || {
                PyTuple::new(py, self.0.classes().iter().map(|c| LabelClass(c.clone()))).map(Bound::unbind)
            })?
            .bind(py))
    }

    fn class_object<'py>(&self, py: Python<'py>, class: &engine::LabelClass) -> PyResult<Bound<'py, PyAny>> {
        let position = self.0.classes().iter().position(|c| c.id == class.id).unwrap_or(0);
        self.class_objects(py)?.get_item(position)
    }
}

fn label_classes(classes: Option<&Bound<'_, PyAny>>) -> R<Vec<engine::LabelClass>> {
    let mut out = Vec::new();
    let Some(classes) = classes else { return Ok(out) };
    for item in classes.try_iter()? {
        let item = item?;
        match item.cast::<LabelClass>() {
            Ok(c) => out.push(c.get().0.clone()),
            Err(_) => out.push(engine::LabelClass::from_json(&py_to_json(&item)?)?),
        }
    }
    Ok(out)
}

fn relations(items: Option<&Bound<'_, PyAny>>) -> R<Vec<engine::Relation>> {
    let mut out = Vec::new();
    let Some(items) = items else { return Ok(out) };
    for item in items.try_iter()? {
        let item = item?;
        match item.cast::<Relation>() {
            Ok(r) => out.push(r.get().0.clone()),
            Err(_) => out.push(engine::Relation::from_json(&py_to_json(&item)?)?),
        }
    }
    Ok(out)
}

fn skeletons(items: Option<&Bound<'_, PyAny>>) -> R<Vec<engine::Skeleton>> {
    let mut out = Vec::new();
    let Some(items) = items else { return Ok(out) };
    for item in items.try_iter()? {
        let item = item?;
        match item.cast::<Skeleton>() {
            Ok(s) => out.push(s.get().0.clone()),
            Err(_) => out.push(engine::Skeleton::from_json(&py_to_json(&item)?)?),
        }
    }
    Ok(out)
}

/// A label set argument: the binding class, or its JSON form.
pub fn label_set_arg(obj: &Bound<'_, PyAny>) -> R<engine::LabelSet> {
    if let Ok(ls) = obj.cast::<LabelSet>() {
        return Ok(ls.get().0.clone());
    }
    engine::LabelSet::from_json(Some(&py_to_json(obj)?))?
        .ok_or_else(|| medh5::Error::invalid("an empty label set document").into())
}

#[pymethods]
impl LabelSet {
    #[new]
    #[pyo3(signature = (id, classes=None, *, version="1.0.0".to_string(), relations=None, skeletons=None, form="inline".to_string(), uri=None, sha256=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: String,
        classes: Option<&Bound<'_, PyAny>>,
        version: String,
        relations: Option<&Bound<'_, PyAny>>,
        skeletons: Option<&Bound<'_, PyAny>>,
        form: String,
        uri: Option<String>,
        sha256: Option<String>,
    ) -> R<Self> {
        Ok(LabelSet::wrap(engine::LabelSet::new(
            id,
            label_classes(classes)?,
            version,
            self::relations(relations)?,
            self::skeletons(skeletons)?,
            form,
            uri,
            sha256,
        )?))
    }
    #[getter]
    fn id(&self) -> &str {
        &self.0.id
    }
    #[getter]
    fn version(&self) -> &str {
        &self.0.version
    }
    #[getter]
    fn form(&self) -> &str {
        &self.0.form
    }
    #[getter]
    fn uri(&self) -> Option<&str> {
        self.0.uri.as_deref()
    }
    #[getter]
    fn relations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.relations.iter().map(|r| Relation(r.clone())))
    }
    #[getter]
    fn skeletons<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.skeletons.iter().map(|s| Skeleton(s.clone())))
    }
    fn check(&self) -> R<()> {
        Ok(self.0.check()?)
    }
    fn __len__(&self) -> usize {
        self.0.len()
    }
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(slf.getattr("classes")?.try_iter()?.into_any())
    }
    fn __contains__(&self, item: &Bound<'_, PyAny>) -> bool {
        if let Ok(s) = item.cast::<PyString>() {
            return self.0.by_key(&s.to_string()).is_some();
        }
        match item.extract::<i64>() {
            Ok(i) => self.0.contains_id(i),
            Err(_) => false,
        }
    }
    fn __repr__(&self) -> String {
        self.0.repr()
    }
    #[getter]
    fn classes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(self.class_objects(py)?.clone())
    }
    #[getter]
    fn ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.ids())
    }
    #[getter]
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.keys())
    }
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        let key = lookup_key(key)?;
        Ok(self.class_object(py, self.0.lookup(&key)?)?)
    }
    fn get<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Ok(key) = lookup_key(key) else { return Ok(None) };
        match self.0.get(&key) {
            Some(c) => Ok(Some(self.class_object(py, c)?)),
            None => Ok(None),
        }
    }
    fn resolve<'py>(&self, py: Python<'py>, keys: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
        let keys = keys_arg(keys)?;
        let found = self.0.resolve(&keys)?;
        let objects = found.iter().map(|c| self.class_object(py, c)).collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, objects)?)
    }
    fn ids_for<'py>(&self, py: Python<'py>, keys: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
        let keys = keys_arg(keys)?;
        Ok(PyTuple::new(py, self.0.ids_for(&keys)?)?)
    }
    fn missing<'py>(&self, py: Python<'py>, ids: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.missing(crate::convert::class_ids(ids)?))
    }
    fn ancestors<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.0.ancestors(&lookup_key(key)?)?)?)
    }
    fn descendants<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.0.descendants(&lookup_key(key)?)?)?)
    }
    fn close<'py>(&self, py: Python<'py>, ids: &Bound<'py, PyAny>, closure: &str) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.0.close(&crate::convert::class_ids(ids)?, closure)?)?)
    }
    #[pyo3(signature = (key, predicate=None))]
    fn relations_of<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
        predicate: Option<&str>,
    ) -> R<Bound<'py, PyTuple>> {
        let found = self.0.relations_of(&lookup_key(key)?, predicate)?;
        Ok(PyTuple::new(py, found.into_iter().map(|r| Relation(r.clone())))?)
    }
    fn skeleton(&self, skeleton_id: &str) -> R<Skeleton> {
        Ok(Skeleton(self.0.skeleton(skeleton_id)?.clone()))
    }
    fn colors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (id, rgba) in self.0.colors() {
            out.set_item(id, PyTuple::new(py, rgba)?)?;
        }
        Ok(out)
    }
    fn content_doc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.content_doc())
    }
    #[pyo3(signature = (algo="sha256"))]
    fn digest(&self, algo: &str) -> R<String> {
        Ok(self.0.digest(algo)?)
    }
    #[getter]
    fn sha256(&self) -> String {
        self.0.sha256()
    }
    #[pyo3(signature = (*, form=None))]
    fn to_json<'py>(&self, py: Python<'py>, form: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json(form))
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Option<Self>> {
        if doc.is_none() {
            return Ok(None);
        }
        Ok(engine::LabelSet::from_json(Some(&py_to_json(doc)?))?.map(LabelSet::wrap))
    }
    fn as_ref(&self, uri: &str) -> R<LabelSet> {
        Ok(LabelSet::wrap(self.0.as_ref(uri)?))
    }
    #[pyo3(signature = (keys, *, id=None))]
    fn subset(&self, keys: &Bound<'_, PyAny>, id: Option<&str>) -> R<LabelSet> {
        Ok(LabelSet::wrap(self.0.subset(&keys_arg(keys)?, id)?))
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        reduce_via_json(slf.as_any())
    }
}

/// `ls[key]`: an `int` looks up an id, anything else a key.
fn lookup_key(key: &Bound<'_, PyAny>) -> PyResult<ClassKey> {
    if let Ok(s) = key.cast::<PyString>() {
        return Ok(ClassKey::Key(s.to_string()));
    }
    match key.extract::<i64>() {
        Ok(i) => Ok(ClassKey::Id(i)),
        Err(_) => Ok(ClassKey::Key(key.str()?.to_string())),
    }
}

fn keys_arg(keys: &Bound<'_, PyAny>) -> PyResult<Vec<ClassKey>> {
    let mut out = Vec::new();
    for item in keys.try_iter()? {
        out.push(lookup_key(&item?)?);
    }
    Ok(out)
}

// -- module functions ------------------------------------------------------------------------

#[pyfunction]
fn check_class_id(class_id: i64) -> R<i64> {
    Ok(i64::from(engine::check_class_id(class_id)?))
}

#[pyfunction]
fn canonical_json<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    Ok(pyo3::types::PyBytes::new(py, &engine::canonical_json(&py_to_json(doc)?)))
}

#[pyfunction]
#[pyo3(signature = (keys, *, id, version="1.0.0", start=1))]
fn from_keys(keys: Vec<String>, id: &str, version: &str, start: i64) -> R<LabelSet> {
    Ok(LabelSet::wrap(engine::from_keys(&keys, id, version, start)?))
}

#[pyfunction]
fn registry_available<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, registry::available())
}

#[pyfunction]
fn registry_load(name: &str) -> R<LabelSet> {
    Ok(LabelSet::wrap(registry::load(name)?))
}

#[pyfunction]
fn registry_load_file(path: std::path::PathBuf) -> R<LabelSet> {
    Ok(LabelSet::wrap(registry::load_file(&path)?))
}

#[pyfunction]
fn registry_register(name: &str, label_set: &Bound<'_, PyAny>) -> R<LabelSet> {
    Ok(LabelSet::wrap(registry::register(name, label_set_arg(label_set)?)))
}

#[pyfunction]
fn registry_unregister(name: &str) {
    registry::unregister(name)
}

#[pyfunction]
fn registry_describe<'py>(py: Python<'py>) -> R<Bound<'py, PyDict>> {
    Ok(map_to_py(py, &registry::describe()?)?)
}

#[pyfunction]
fn registry_from_doc(doc: &Bound<'_, PyAny>) -> R<LabelSet> {
    Ok(LabelSet::wrap(registry::labelset_from_doc(&py_to_json(doc)?)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = MODULE;
    m.add_class::<OntologyCode>()?;
    m.add_class::<Relation>()?;
    m.add_class::<Skeleton>()?;
    m.add_class::<LabelClass>()?;
    m.add_class::<LabelSet>()?;
    m.add_function(wrap_pyfunction!(check_class_id, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_json, m)?)?;
    m.add_function(wrap_pyfunction!(from_keys, m)?)?;
    m.add_function(wrap_pyfunction!(registry_available, m)?)?;
    m.add_function(wrap_pyfunction!(registry_load, m)?)?;
    m.add_function(wrap_pyfunction!(registry_load_file, m)?)?;
    m.add_function(wrap_pyfunction!(registry_register, m)?)?;
    m.add_function(wrap_pyfunction!(registry_unregister, m)?)?;
    m.add_function(wrap_pyfunction!(registry_describe, m)?)?;
    m.add_function(wrap_pyfunction!(registry_from_doc, m)?)?;
    m.add("BACKGROUND_ID", engine::BACKGROUND_ID)?;
    m.add("IGNORE_ID", engine::IGNORE_ID)?;
    m.add("MAX_CLASS_ID", engine::MAX_CLASS_ID)?;
    m.add("CLOSURES", PyTuple::new(m.py(), engine::CLOSURES)?)?;
    m.add("FORMS", PyTuple::new(m.py(), engine::FORMS)?)?;
    m.add("INLINE_REQUIRED_BELOW", engine::INLINE_REQUIRED_BELOW)?;
    let _ = (class_key, class_keys);
    Ok(())
}

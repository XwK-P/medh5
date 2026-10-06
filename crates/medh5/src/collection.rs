//! Collections --- many sample roots in one file (spec §2.2).
//!
//! A sample root inside a collection is *exactly* a sample root, so packing
//! and extraction are pure copies: chunks move as raw bytes and every
//! `content_id` is unchanged.  A shard is an I/O decision, never a different
//! encoding of its samples.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::h5::attrs::{self, AttrValue};
use crate::h5::file::{open_read, AtomicFile};
use crate::h5::ops;
use crate::ids::validate_sample_key;
use crate::json::{repr_list, repr_str};
use crate::sample::{require_major, Sample};
use crate::{Error, Result, FORMAT_VERSION};

/// Conventional extension for a collection file (§2.1).
pub const SUFFIX: &str = ".medh5c";
/// Where sample roots live inside a collection (§2.2).
pub const SAMPLES_GROUP: &str = crate::validate::SAMPLES_GROUP;

/// Whether an open root declares itself a collection.
pub fn is_collection(root: &hdf5::Location) -> Result<bool> {
    Ok(attrs::get_str(root, "medh5_kind")?.as_deref() == Some("collection"))
}

/// A read-only mapping of `sample_key -> Sample` over one shard.
#[derive(Debug)]
pub struct Collection {
    pub path: Option<PathBuf>,
    pub root: hdf5::Group,
    handle: Option<hdf5::File>,
}

impl Collection {
    pub fn from_root(root: hdf5::Group, handle: Option<hdf5::File>, path: Option<PathBuf>) -> Result<Collection> {
        if !ops::exists(&root, SAMPLES_GROUP) {
            return Err(Error::coded(
                "E008",
                format!(
                    "collection {} has no {} group",
                    path.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "<memory>".into()),
                    repr_str(SAMPLES_GROUP)
                ),
            ));
        }
        Ok(Collection { path, root, handle })
    }

    pub fn close(&mut self) {
        self.handle = None;
    }

    /// The `samples` group.
    pub fn group(&self) -> Result<hdf5::Group> {
        Ok(self.root.group(SAMPLES_GROUP)?)
    }

    /// Member keys, sorted.
    pub fn keys(&self) -> Result<Vec<String>> {
        ops::members(&self.group()?)
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.keys()?.len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn contains(&self, key: &str) -> Result<bool> {
        Ok(ops::exists(&self.group()?, key))
    }

    /// One member as a sample (`KeyError` naming the known keys).
    pub fn get(&self, key: &str) -> Result<Sample> {
        let group = self.group()?;
        let Some(member) = ops::child_group(&group, key) else {
            return Err(Error::Key(format!(
                "collection has no sample {}; known keys: {}",
                repr_str(key),
                repr_list(&ops::members(&group)?)
            )));
        };
        let shown = format!(
            "{}::{key}",
            self.path.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "None".into())
        );
        Ok(Sample::from_root(member, self.handle.clone(), Some(PathBuf::from(shown))))
    }

    pub fn version(&self) -> Result<String> {
        let v = attrs::require(&self.root, "medh5_version", "E001")?;
        Ok(v.as_str().unwrap_or_else(|| attrs::stringify_value(&v)))
    }

    pub fn kind(&self) -> Result<String> {
        Ok(attrs::get_str(&self.root, "medh5_kind")?.unwrap_or_else(|| "sample".into()))
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> Result<String> {
        Ok(format!(
            "Collection({}, {} samples)",
            self.path.as_ref().map(|p| repr_str(&p.to_string_lossy())).unwrap_or_else(|| "None".into()),
            self.len()?
        ))
    }

    pub fn summary(&self) -> Result<Value> {
        let mut samples = Vec::new();
        for key in self.keys()? {
            let s = self.get(&key)?;
            let identity = s.identity()?;
            let mut images: Vec<String> = s.images()?.keys().cloned().collect();
            images.sort();
            let mut anns: Vec<String> = s.annotations()?.keys().cloned().collect();
            anns.sort();
            samples.push(json!({
                "key": key,
                "sample_id": identity.sample_id,
                "subject_id": identity.subject_id,
                "profiles": s.profiles()?.into_iter().collect::<Vec<_>>(),
                "content_id": s.content_id()?,
                "timepoints": s.timepoints()?.ids(),
                "images": images,
                "annotations": anns,
            }));
        }
        Ok(json!({
            "path": self.path.as_ref().map(|p| p.to_string_lossy().into_owned()),
            "version": self.version()?,
            "kind": self.kind()?,
            "samples": samples,
        }))
    }

    /// `sample_key -> subject_id`: what a split has to group by (§12.2).
    pub fn subject_ids(&self) -> Result<Vec<(String, String)>> {
        self.keys()?.into_iter().map(|k| Ok((k.clone(), self.get(&k)?.identity()?.subject_id.clone()))).collect()
    }
}

/// Open a `.medh5c` shard, read-only.
pub fn open_collection(path: &Path) -> Result<Collection> {
    let handle = open_read(path)?;
    require_major(&handle, path)?;
    if !is_collection(&handle)? {
        let kind = attrs::get_str(&handle, "medh5_kind")?.unwrap_or_else(|| "sample".into());
        return Err(Error::coded(
            "E006",
            format!(
                "{} declares medh5_kind={}, not 'collection'; open it with `medh5.open`",
                repr_str(&path.to_string_lossy()),
                repr_str(&kind)
            ),
        ));
    }
    let root = handle.as_group()?;
    Collection::from_root(root, Some(handle), Some(path.to_path_buf()))
}

/// A file opened whatever its kind.
#[derive(Debug)]
pub enum AnyFile {
    Sample(Sample),
    Collection(Collection),
}

/// Open a file whatever its kind, resolving `key` inside a collection.
pub fn open_any(path: &Path, key: Option<&str>) -> Result<AnyFile> {
    let handle = open_read(path)?;
    require_major(&handle, path)?;
    let root = handle.as_group()?;
    if is_collection(&handle)? {
        let collection = Collection::from_root(root, Some(handle), Some(path.to_path_buf()))?;
        return match key {
            None => Ok(AnyFile::Collection(collection)),
            Some(k) => Ok(AnyFile::Sample(collection.get(k)?)),
        };
    }
    if key.is_some() {
        return Err(Error::File(format!(
            "{} is a single sample; `key` applies to collections",
            repr_str(&path.to_string_lossy())
        )));
    }
    Ok(AnyFile::Sample(Sample::from_root(root, Some(handle), Some(path.to_path_buf()))))
}

/// The sample key a file gets when none is supplied: its stem.
pub fn default_key(path: &Path) -> Result<String> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(validate_sample_key(name.split('.').next().unwrap_or(""))?.to_string())
}

fn copy_root(src: &hdf5::Group, dst: &hdf5::Group) -> Result<()> {
    for name in ops::members(src)? {
        ops::copy_object(src, &name, dst, &name)?;
    }
    for key in attrs::names(src)? {
        attrs::copy_raw(src, dst, &key)?;
    }
    Ok(())
}

/// Copy sample files into one collection shard (§2.2); chunks move as raw
/// bytes, so `unpack(pack(x)) == x` down to the compressed bytes.
pub fn pack(sources: &[&Path], out: &Path, keys: Option<&[String]>) -> Result<PathBuf> {
    if sources.is_empty() {
        return Err(Error::coded("E008", "pack needs at least one sample file"));
    }
    if let Some(k) = keys {
        if k.len() != sources.len() {
            return Err(Error::coded("E003", format!("{} keys for {} sources", k.len(), sources.len())));
        }
    }
    let chosen: Vec<String> = sources
        .iter()
        .enumerate()
        .map(|(i, p)| match keys {
            Some(k) => Ok(validate_sample_key(&k[i])?.to_string()),
            None => default_key(p),
        })
        .collect::<Result<_>>()?;
    let mut duplicates: Vec<&String> = chosen.iter().filter(|k| chosen.iter().filter(|x| x == k).count() > 1).collect();
    duplicates.sort();
    duplicates.dedup();
    if !duplicates.is_empty() {
        return Err(Error::coded(
            "E003",
            format!(
                "sample key(s) {} are not unique in the collection; pass explicit --key values",
                repr_list(&duplicates)
            ),
        ));
    }
    let file = AtomicFile::create(out)?;
    let result = (|| -> Result<()> {
        let handle = file.handle();
        attrs::write(handle, "medh5_version", &AttrValue::Str(FORMAT_VERSION.into()))?;
        attrs::write(handle, "medh5_kind", &AttrValue::Str("collection".into()))?;
        let group = handle.create_group(SAMPLES_GROUP)?;
        for (key, source) in chosen.iter().zip(sources) {
            let src = open_read(source)?;
            require_major(&src, source)?;
            if is_collection(&src)? {
                return Err(Error::coded(
                    "E006",
                    format!("{} is already a collection; pack takes sample files", source.to_string_lossy()),
                ));
            }
            let member = group.create_group(key)?;
            copy_root(&src.as_group()?, &member)?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => {
            file.commit()?;
            Ok(out.to_path_buf())
        }
        Err(e) => {
            file.abort();
            Err(e)
        }
    }
}

fn write_sample_root(src: &hdf5::Group, destination: &Path) -> Result<()> {
    let file = AtomicFile::create(destination)?;
    let result = (|| -> Result<()> {
        let handle = file.handle();
        let root = handle.as_group()?;
        copy_root(src, &root)?;
        attrs::write(handle, "medh5_kind", &AttrValue::Str("sample".into()))?;
        if !attrs::has(handle, "medh5_version") {
            attrs::write(handle, "medh5_version", &AttrValue::Str(FORMAT_VERSION.into()))?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => file.commit(),
        Err(e) => {
            file.abort();
            Err(e)
        }
    }
}

/// Extract sample roots back into standalone files (§2.2).
pub fn unpack(path: &Path, outdir: &Path, keys: Option<&[String]>, suffix: &str) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(outdir)?;
    let collection = open_collection(path)?;
    let known = collection.keys()?;
    let wanted: Vec<String> = match keys {
        None => known
            .iter()
            .map(|k| match validate_sample_key(k) {
                Ok(v) => Ok(v.to_string()),
                Err(_) => Err(Error::coded(
                    "E003",
                    format!(
                        "{} has a member named {}, which is not a valid sample key (§2.2) and cannot be used as a file name; refusing to unpack it",
                        repr_str(&path.to_string_lossy()),
                        repr_str(k)
                    ),
                )),
            })
            .collect::<Result<_>>()?,
        Some(k) => k.iter().map(|x| Ok(validate_sample_key(x)?.to_string())).collect::<Result<_>>()?,
    };
    let missing: Vec<&String> = wanted.iter().filter(|k| !known.contains(k)).collect();
    if !missing.is_empty() {
        return Err(Error::coded(
            "E003",
            format!("collection has no sample(s) {}; known keys: {}", repr_list(&missing), repr_list(&known)),
        ));
    }
    let group = collection.group()?;
    let mut written = Vec::new();
    for key in wanted {
        let destination = outdir.join(format!("{key}{suffix}"));
        write_sample_root(&group.group(&key)?, &destination)?;
        written.push(destination);
    }
    Ok(written)
}

/// Extract one member into a standalone sample file at `out` (and nothing
/// else is written).
pub fn extract(path: &Path, key: &str, out: &Path) -> Result<PathBuf> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let wanted = validate_sample_key(key)?.to_string();
    let collection = open_collection(path)?;
    let known = collection.keys()?;
    if !known.contains(&wanted) {
        return Err(Error::coded(
            "E003",
            format!("collection has no sample {}; known keys: {}", repr_str(&wanted), repr_list(&known)),
        ));
    }
    write_sample_root(&collection.group()?.group(&wanted)?, out)?;
    Ok(out.to_path_buf())
}

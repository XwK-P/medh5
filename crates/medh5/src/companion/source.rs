//! Source references: a locator, an optional member key, and a pin.
//!
//! A URI says where bytes are; it is not what they are.  The pinned
//! `content_id` is: a reference resolves to a sample, and the sample counts
//! only while its address is the pinned one.  Since `content_id` is a Merkle
//! root over *stored* digests, the check also recomputes the root and
//! re-verifies the clinical datasets' actual bytes --- a stored root alone
//! would miss an edit under unchanged digests (1.1 §8).

use std::path::{Component, Path, PathBuf};

use serde_json::{json, Value};

use super::Finding;
use crate::collection::{open_any, AnyFile};
use crate::json::repr_str;
use crate::sample::Sample;
use crate::{Error, Result};

/// The local path a `file:` URI (RFC 8089) names, or `None` when `uri` is
/// not one that names a path on this machine: its host is empty or
/// `localhost`, its path is percent-decoded, and on Windows (`windows`) a
/// drive path loses its leading slash (`file:///C:/x` is `C:/x`) and another
/// host is a UNC share.  What Python's `Path.as_uri()` writes reads back as
/// the path it was made from; stripping the prefix made `file://localhost/x`
/// the relative `localhost/x`, left `%20` in a name, and joined `/C:/x` onto
/// the base.
fn file_uri_path(uri: &str, windows: bool) -> Option<String> {
    let scheme = uri.get(..5).filter(|s| s.eq_ignore_ascii_case("file:"))?;
    let rest = &uri[scheme.len()..];
    let (host, path) = match rest.strip_prefix("//") {
        Some(authority) => {
            let at = authority.find('/').unwrap_or(authority.len());
            (&authority[..at], &authority[at..])
        }
        None if rest.starts_with('/') => ("", rest),
        None => return None,
    };
    let path = percent_decode(path)?;
    let local = host.is_empty() || host.eq_ignore_ascii_case("localhost");
    if !windows {
        return local.then_some(path);
    }
    if !local {
        return Some(format!(r"\\{host}{}", path.replace('/', r"\")));
    }
    let drive = path.as_bytes();
    let drive_path = drive.len() >= 3
        && drive[0] == b'/'
        && drive[1].is_ascii_alphabetic()
        && matches!(drive[2], b':' | b'|')
        && drive.get(3).is_none_or(|c| *c == b'/');
    Some(if drive_path { format!("{}:{}", &path[1..2], &path[3..]) } else { path })
}

/// `%XX` escapes decoded, or `None` for a malformed escape or a path that
/// is not UTF-8 once decoded.
fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = text.get(i + 1..i + 3).filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()))?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// The characters an id may hold: `medh5-task-1.schema.json`'s `$defs/id`,
/// as a regular-expression class.
pub const ID_CHARS: &str = "A-Za-z0-9_.@:-";
/// The most characters an id may have.
pub const ID_MAX: usize = 128;

fn is_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "_.@:-".contains(c)
}

/// `text` as id characters: each other character is `_`, and it is cut at
/// `max` characters.
fn id_of(text: &str, max: usize) -> String {
    text.chars().map(|c| if is_id_char(c) { c } else { '_' }).take(max).collect()
}

/// The `/`-separated segments of `uri` --- a `file:` URI's path once decoded
/// --- that name something: empty and `.` segments are dropped, as a path
/// drops them.  A backslash is an ordinary character, on every platform, so
/// one manifest derives one id everywhere.
fn uri_segments(uri: &str) -> Vec<String> {
    let path = file_uri_path(uri, false).unwrap_or_else(|| uri.to_string());
    path.split('/').filter(|s| !s.is_empty() && *s != ".").map(str::to_string).collect()
}

/// A file name without its last suffix, as Python's `PurePath.stem`: a
/// leading or trailing dot is no suffix.
fn stem(name: &str) -> &str {
    match name.rfind('.') {
        Some(i) if i > 0 && i < name.len() - 1 => &name[..i],
        _ => name,
    }
}

/// The id `depth` directories deep: the file's stem after the `depth`
/// directories holding it, and `:key` for a member, as id characters.
fn id_at_depth(segments: &[String], sample_key: Option<&str>, depth: usize) -> String {
    let Some((name, dirs)) = segments.split_last() else {
        return id_of(&sample_key.filter(|k| !k.is_empty()).map(|k| format!(":{k}")).unwrap_or_default(), ID_MAX);
    };
    let mut parts: Vec<&str> = dirs[dirs.len().saturating_sub(depth)..].iter().map(String::as_str).collect();
    parts.push(stem(name));
    let key = sample_key.filter(|k| !k.is_empty()).map(|k| format!(":{k}")).unwrap_or_default();
    id_of(&format!("{}{key}", parts.join("/")), ID_MAX)
}

/// `path` with its `.` and `..` segments resolved lexically, as text: no
/// file is examined, so a symbolic link does not move where `..` leads.
fn lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                let climbed = matches!(out.components().next_back(), Some(Component::Normal(_))) && out.pop();
                if !climbed && !out.has_root() {
                    out.push("..");
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// A directory the runner names, made absolute and canonical once.  On
/// Windows a drive path keeps its plain form (`C:\x`, not `\\?\C:\x`), as
/// a source URI would spell it.
fn runner_dir(path: &Path) -> Result<PathBuf> {
    let found = std::fs::canonicalize(path).map_err(|e| crate::error::os_error(&e, path))?;
    if cfg!(windows) {
        if let Some(plain) = found.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
            if plain.as_bytes().get(1) == Some(&b':') {
                return Ok(PathBuf::from(plain));
            }
        }
    }
    Ok(found)
}

/// Where a runner lets a task's sources be opened from (task-cache-1 §2):
/// the base relative URIs resolve against and, when confined, the one
/// directory a source must lie in.
///
/// A manifest's author chooses its URIs; whoever runs it chooses what may be
/// opened.  Confined, a URI is joined to the base and its `.` and `..`
/// segments resolved lexically --- no file is examined --- and a path outside
/// the root is refused (T301) naming the URI and nothing about the path, the
/// same whether or not it exists; the path so resolved is what is opened.
/// Lexically, so the symbolic links data directories are full of stay
/// usable: `link/../x` is `root/x`, wherever `link` points.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SourceAccess {
    /// What relative URIs resolve against; `None` is the current directory.
    pub base: Option<PathBuf>,
    /// The directory every source must lie in; `None` confines nothing.
    pub root: Option<PathBuf>,
}

impl SourceAccess {
    /// Unconfined: relative URIs resolve against `base`, and any path opens.
    pub fn new(base: Option<&Path>) -> SourceAccess {
        SourceAccess { base: base.map(Path::to_path_buf), root: None }
    }

    /// Confined to `root`, by default the base (itself by default the
    /// current directory).  Both are canonicalized here, once: what the
    /// runner names may be a link; what a URI names is judged as text.
    pub fn confined(base: Option<&Path>, root: Option<&Path>) -> Result<SourceAccess> {
        let base = runner_dir(base.unwrap_or(Path::new(".")))?;
        let root = match root {
            Some(r) => runner_dir(r)?,
            None => base.clone(),
        };
        Ok(SourceAccess { base: Some(base), root: Some(root) })
    }

    /// Whether sources are confined.
    pub fn is_confined(&self) -> bool {
        self.root.is_some()
    }

    /// The file `source` names, or T301 when it lies outside the root.
    pub fn locate(&self, source: &SourceRef) -> Result<PathBuf> {
        let joined = source.resolve(self.base.as_deref());
        let Some(root) = &self.root else { return Ok(joined) };
        let path = lexical(&joined);
        if path.starts_with(root) {
            return Ok(path);
        }
        Err(Error::coded(
            "T301",
            format!(
                "{} is outside the directory this task's sources are confined to; a runner that trusts the manifest \
                 can widen it (--source-root, or confine=) or lift it (--any-source-path, or confine=False)",
                repr_str(&source.uri)
            ),
        ))
    }
}

/// One sample, standalone or a collection member, at a pinned version.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceRef {
    /// The id rows and reconciliation records use for this source.
    pub source_id: String,
    /// A path, relative to the manifest's directory, or absolute; `file://`
    /// URIs are accepted.
    pub uri: String,
    /// The member of a `.medh5c` collection, or `None`.
    pub sample_key: Option<String>,
    /// The sample version the reference pins.
    pub content_id: String,
    /// The sample's own `identity.subject_id`, as the manifest maps it.
    pub local_subject_id: Option<String>,
}

impl SourceRef {
    pub fn new(uri: impl Into<String>, sample_key: Option<String>, content_id: impl Into<String>) -> SourceRef {
        let uri = uri.into();
        SourceRef { source_id: String::new(), uri, sample_key, content_id: content_id.into(), local_subject_id: None }
    }

    /// The id a reference gets when none is given (task-cache-1 §2): the
    /// file's stem after the directory holding it when the URI names one
    /// (`site-a_P-01`), and `:<key>` for a member, as id characters within
    /// 128.  A `file:` URI is read as the path it names; any other URI is
    /// split on `/` only.
    pub fn default_id(uri: &str, sample_key: Option<&str>) -> String {
        id_at_depth(&uri_segments(uri), sample_key, 1)
    }

    /// [`default_id`](Self::default_id), widened until `taken` refuses it no
    /// longer: first by the directories above it, then numbered `@2`, `@3`,
    /// within 128 characters.  What a manifest gives a derived id that
    /// another source already holds; the holder keeps its id.
    pub fn default_id_among(uri: &str, sample_key: Option<&str>, taken: &dyn Fn(&str) -> bool) -> String {
        let segments = uri_segments(uri);
        let first = id_at_depth(&segments, sample_key, 1);
        // No id is no id however it is widened: the schema refuses it (T101).
        if first.is_empty() || !taken(&first) {
            return first;
        }
        for depth in 2..segments.len() {
            let wider = id_at_depth(&segments, sample_key, depth);
            if !taken(&wider) {
                return wider;
            }
        }
        (2usize..)
            .map(|n| {
                let suffix = format!("@{n}");
                let head: String = first.chars().take(ID_MAX - suffix.len()).collect();
                format!("{head}{suffix}")
            })
            .find(|id| !taken(id))
            .expect("some number is free")
    }

    /// A reference pinned to what `sample` is now.
    ///
    /// Refused when the root does not bind every attested path to its bytes
    /// (1.0 §13.2): such a pin could never check clean, and a file crafted
    /// with an alias holding an attested name --- relinkable afterwards to
    /// other covered bytes under the same root --- would pin without
    /// complaint.
    pub fn pin(uri: impl Into<String>, sample_key: Option<String>, sample: &Sample) -> Result<SourceRef> {
        let content_id = sample
            .content_id()?
            .ok_or_else(|| Error::invalid("a sample without a content_id cannot be pinned; commit it first"))?;
        let mut out = SourceRef::new(uri, sample_key, content_id);
        let uncovered = crate::integrity::unattested(&sample.root)?;
        if !uncovered.is_empty() {
            return Err(Error::Integrity(format!(
                "{} cannot be pinned: no line of its content_id binds {} to {} bytes (1.0 §13.2); `medh5 validate \
                 --level integrity` says why",
                out.locator(),
                uncovered.join(", "),
                if uncovered.len() == 1 { "its" } else { "their" },
            )));
        }
        out.local_subject_id = Some(sample.identity()?.subject_id.clone());
        Ok(out)
    }

    /// The file the locator names, relative to `base` when it is relative.
    /// A `file:` URI is decoded as [`file_uri_path`] decodes it.
    pub fn resolve(&self, base: Option<&Path>) -> PathBuf {
        let path = file_uri_path(&self.uri, cfg!(windows)).map_or_else(|| PathBuf::from(&self.uri), PathBuf::from);
        match base {
            Some(b) if path.is_relative() => b.join(path),
            _ => path,
        }
    }

    /// `uri` or `uri::key`, for messages.
    pub fn locator(&self) -> String {
        match &self.sample_key {
            Some(k) => format!("{}::{k}", self.uri),
            None => self.uri.clone(),
        }
    }

    /// Open the sample the reference names (not yet checking the pin).
    pub fn open(&self, base: Option<&Path>) -> Result<Sample> {
        self.open_at(&self.resolve(base))
    }

    /// Open the sample the reference names where `access` lets it be
    /// opened: T301, without examining the path, when it lies outside a
    /// confined runner's root.
    pub fn open_in(&self, access: &SourceAccess) -> Result<Sample> {
        self.open_at(&access.locate(self)?)
    }

    fn open_at(&self, path: &Path) -> Result<Sample> {
        #[cfg(test)]
        OPENS.with(|n| n.set(n.get() + 1));
        match open_any(path, self.sample_key.as_deref())? {
            AnyFile::Sample(s) => Ok(s),
            AnyFile::Collection(_) => Err(Error::invalid(format!(
                "{} is a collection; a source names one of its members with `sample_key`",
                repr_str(&self.uri)
            ))),
        }
    }

    /// Whether `sample` is the pinned version: its stored `content_id` is the
    /// pin, every dataset the root speaks for carries a digest, the root
    /// recomputes to it, and every clinical dataset's bytes match their
    /// digests.  `deep` verifies every dataset.  Empty when it is.
    ///
    /// The sample is walked once: the stored digests the root is recomputed
    /// from are the ones the bytes are checked against.  A dataset whose
    /// bytes cannot be read --- a damaged chunk --- does not match.
    pub fn check(&self, sample: &Sample, deep: bool) -> Result<Vec<Finding>> {
        let mut out = Vec::new();
        let at = if self.source_id.is_empty() { self.locator() } else { self.source_id.clone() };
        let stored = sample.content_id()?;
        if stored.as_deref() != Some(self.content_id.as_str()) {
            out.push(Finding::new(
                "T302",
                &at,
                // Only the pin: what a file off its pin stores is nothing the
                // manifest's author named (task-cache-1 Appendix A).
                format!(
                    "{} is not the version the reference pins ({}) --- repin it explicitly if the change is meant",
                    self.locator(),
                    self.content_id
                ),
            ));
            return Ok(out);
        }
        let digests = crate::integrity::collect_digests(&sample.root, &["index"])?;
        // The root covers stored digests at the paths their lines name, so a
        // dataset that carries none --- a column added later, say --- is bytes
        // no pin speaks for, and so is one reached at a path that is not its
        // own: refused before anything is recomputed (1.0 §13.2, 1.1 §8,
        // E818).  Every path through the attested groups is judged, soft links
        // followed as readers follow them: an object listed under no line ---
        // reached first through `index/`, or only through a soft link --- is
        // no more covered than one without a digest, and a name held by
        // an alias could be relinked to other covered bytes under the same
        // root.
        let uncovered = crate::integrity::unattested(&sample.root)?;
        if !uncovered.is_empty() {
            out.push(Finding::new(
                "T302",
                &at,
                format!(
                    "{}: no line of the root binds {} to {} bytes, so the pinned content_id does not cover {}",
                    self.locator(),
                    uncovered.join(", "),
                    if uncovered.len() == 1 { "its" } else { "their" },
                    if uncovered.len() == 1 { "it" } else { "them" }
                ),
            ));
            return Ok(out);
        }
        let algo = crate::integrity::digest::root_algo(&sample.root)?;
        let recomputed =
            crate::integrity::compute_content_id(&sample.root, &sample.attr_name_map()?, &algo, Some(&digests))?;
        if recomputed != self.content_id {
            out.push(Finding::new(
                "T302",
                &at,
                format!("{} declares the pinned content_id but its digests recompute to {recomputed}", self.locator()),
            ));
            return Ok(out);
        }
        // Every path through `clinical/` is its object's own, checked above,
        // so the clinical datasets are the lines under it.
        let mut failed = Vec::new();
        for (path, digest) in &digests {
            if !deep && !path.starts_with("clinical/") {
                continue;
            }
            let matches = crate::digest::parse_digest(digest).and_then(|(algo, _)| {
                let ds = sample.root.dataset(path)?;
                Ok(&crate::integrity::dataset_digest(&ds, path, &algo)? == digest)
            });
            if !matches.unwrap_or(false) {
                failed.push(path.clone());
            }
        }
        if !failed.is_empty() {
            failed.sort();
            out.push(Finding::new(
                "T302",
                &at,
                format!(
                    "{}: dataset bytes no longer match their digests ({}), whatever the stored root says",
                    self.locator(),
                    failed.join(", ")
                ),
            ));
        }
        Ok(out)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "source_id": self.source_id,
            "uri": self.uri,
            "sample_key": self.sample_key,
            "content_id": self.content_id,
            "local_subject_id": self.local_subject_id,
        })
    }

    /// The `{uri, sample_key, content_id}` a cache entry records.
    pub fn to_pin_json(&self) -> Value {
        json!({"uri": self.uri, "sample_key": self.sample_key, "content_id": self.content_id})
    }

    pub fn from_json(value: &Value) -> Result<SourceRef> {
        let text = |k: &str| value.get(k).and_then(Value::as_str).map(str::to_string);
        Ok(SourceRef {
            source_id: text("source_id").unwrap_or_default(),
            uri: text("uri").ok_or_else(|| Error::invalid("a source names its `uri`"))?,
            sample_key: text("sample_key"),
            content_id: text("content_id").ok_or_else(|| Error::invalid("a source pins its `content_id`"))?,
            local_subject_id: text("local_subject_id"),
        })
    }
}

#[cfg(test)]
thread_local! {
    /// How many sources this thread has opened: what a test of "one open per
    /// source" counts.
    pub(crate) static OPENS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s2_a_file_uri_names_the_path_it_was_made_from() {
        for (uri, posix, windows) in [
            ("file:///data/a%20b.medh5", Some("/data/a b.medh5"), Some("/data/a b.medh5")),
            ("file://localhost/data/x.medh5", Some("/data/x.medh5"), Some("/data/x.medh5")),
            ("FILE:/data/%C3%A9.medh5", Some("/data/é.medh5"), Some("/data/é.medh5")),
            ("file:///C:/data/x.medh5", Some("/C:/data/x.medh5"), Some("C:/data/x.medh5")),
            ("file:///c|/x.medh5", Some("/c|/x.medh5"), Some("c:/x.medh5")),
            ("file://server/share/x.medh5", None, Some(r"\\server\share\x.medh5")),
            ("file:///bad%2", None, None),
            ("file:///bad%+1", None, None),
            ("file:relative.medh5", None, None),
            ("data/x.medh5", None, None),
        ] {
            assert_eq!(file_uri_path(uri, false).as_deref(), posix, "{uri}");
            assert_eq!(file_uri_path(uri, true).as_deref(), windows, "{uri}");
        }
        let at = |uri: &str| SourceRef::new(uri, None, "").resolve(Some(Path::new("/base")));
        if cfg!(unix) {
            assert_eq!(at("file://localhost/data/x.medh5"), PathBuf::from("/data/x.medh5"));
        }
        assert_eq!(at("x.medh5"), PathBuf::from("/base/x.medh5"));
    }

    /// The ids 2.0.2's Python derived on POSIX, now derived by the engine and
    /// identical on every platform: a backslash is no separator anywhere.
    #[test]
    fn s2_default_ids_are_pythons() {
        let long = "n".repeat(200);
        for (uri, key, id) in [
            ("P-01.medh5", None, "P-01"),
            ("site-a/P-01.medh5", None, "site-a_P-01"),
            ("data/site-a/P-01.medh5", None, "site-a_P-01"),
            ("../data/P-01.medh5", None, "data_P-01"),
            ("../P-01.medh5", None, ".._P-01"),
            ("a/./b//P-01.medh5", None, "b_P-01"),
            ("/abs/P 01.medh5", None, "abs_P_01"),
            ("file:///data/site%20b/P-01.medh5", None, "site_b_P-01"),
            ("file:///C:/data/P-01.medh5", None, "data_P-01"),
            ("s3://b/x.medh5", None, "b_x"),
            (r"..\x\y.medh5c", None, ".._x_y"),
            (r"site\P-01.medh5", Some("k"), "site_P-01:k"),
            ("shard.medh5c", Some("P-01"), "shard:P-01"),
            ("x.tar.gz", None, "x.tar"),
            (".hidden", None, ".hidden"),
            ("x.", None, "x."),
            ("", None, ""),
            ("", Some("k"), ":k"),
        ] {
            assert_eq!(SourceRef::default_id(uri, key), id, "{uri} {key:?}");
        }
        let cut = SourceRef::default_id(&format!("{long}.medh5"), None);
        assert_eq!(cut, "n".repeat(128));
        let schema: Value = serde_json::from_str(super::super::task::schema_text()).unwrap();
        assert_eq!(schema["$defs"]["id"]["pattern"], format!("^[{ID_CHARS}]{{1,{ID_MAX}}}$"));
    }

    /// Confined, a URI is judged as text: `.` and `..` resolved, nothing
    /// examined.  A path that climbs out, an absolute one and a `file:` URI
    /// elsewhere are refused alike, whether or not they exist; a link inside
    /// the root is followed as text, so `link/../x` is `root/x`.
    #[test]
    fn s2_confinement_is_lexical() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        std::fs::create_dir_all(root.join("b")).unwrap();
        std::fs::create_dir_all(dir.path().join("elsewhere/deep")).unwrap();
        let access = SourceAccess::confined(Some(&root), None).unwrap();
        let canonical = runner_dir(&root).unwrap();
        let at = |uri: &str| access.locate(&SourceRef::new(uri, None, ""));
        assert_eq!(at("a/../b/x.medh5").unwrap(), canonical.join("b").join("x.medh5"));
        assert_eq!(at("./x.medh5").unwrap(), canonical.join("x.medh5"));
        let outside = format!("{}", dir.path().join("elsewhere").join("x.medh5").display());
        let file_uri = format!("file://{}", if cfg!(windows) { format!("/{outside}") } else { outside.clone() });
        let mut messages = Vec::new();
        for uri in ["../x.medh5", "b/../../x.medh5", "../elsewhere/x.medh5", &outside, &file_uri, "/abs/x.medh5"] {
            let err = at(uri).unwrap_err();
            assert_eq!(err.code(), Some("T301"), "{uri}: {err}");
            assert!(err.message().starts_with(&repr_str(uri)), "{err}");
            messages.push(err.message().replacen(&repr_str(uri), "URI", 1));
        }
        // One message, naming the URI only: nothing says whether it exists.
        messages.dedup();
        assert_eq!(messages.len(), 1, "{messages:?}");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(dir.path().join("elsewhere/deep"), root.join("link")).unwrap();
            assert_eq!(at("link/../x.medh5").unwrap(), canonical.join("x.medh5"));
            assert!(at("link/../../x.medh5").is_err());
        }
        // Unconfined, the path is the URI joined to the base, as in 2.0.
        let open = SourceAccess::new(Some(&root));
        assert_eq!(open.locate(&SourceRef::new("../x.medh5", None, "")).unwrap(), root.join("../x.medh5"));
        assert!(!open.is_confined() && access.is_confined());
    }

    #[test]
    fn s2_a_derived_id_that_collides_is_widened() {
        let taken = |ids: &[&str]| {
            let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
            move |id: &str| ids.iter().any(|t| t == id)
        };
        let among = |uri: &str, ids: &[&str]| SourceRef::default_id_among(uri, None, &taken(ids));
        assert_eq!(among("b/site-a/P-01.medh5", &[]), "site-a_P-01");
        assert_eq!(among("b/site-a/P-01.medh5", &["site-a_P-01"]), "b_site-a_P-01");
        assert_eq!(among("x/b/site-a/P-01.medh5", &["site-a_P-01", "b_site-a_P-01"]), "x_b_site-a_P-01");
        assert_eq!(among("site-a/P-01.h5", &["site-a_P-01"]), "site-a_P-01@2");
        assert_eq!(among("site-a/P-01.h5", &["site-a_P-01", "site-a_P-01@2"]), "site-a_P-01@3");
        let long = format!("{}.medh5", "n".repeat(200));
        let numbered = among(&long, &[&"n".repeat(128)]);
        assert_eq!(numbered, format!("{}@2", "n".repeat(126)));
    }
}

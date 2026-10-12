# writer report (w202/writer c7b091e) — condensed from the agent hand-back
Items 1-13 DONE. Item 9 docs-only (physical=True returns stored dtype when not rescaled; documented). Item 7: refuse float masks with values other than 0/1 (E411), keep float 0/1 and ints/bools.
Localized edit: integrity/repair.rs:177 stamp in writer.digest_algo() instead of "sha256" (needed: fix --rewrite-digests on edited sha512 file was refused E701).
Shared stub medh5/_core.pyi docstrings edited.
## CHANGELOG drafts
Fixed:
- An amendment no longer leaves the old /meta behind under a hard link to it. /meta was copied by path, so an alias such as x_acme/meta kept the old document. After scrub --pseudonymise-ids the original identifiers survived under the alias and in the bytes. The commit now re-links every alias to the new document and rewrites the file.
- remap_frame_uids leaves no replaced UID in the file's bytes. HDF5 keeps an overwritten attribute's old value in free space, so the commit now rewrites the file from its root after a remap.
- W907 no longer recommends int16 for a float image holding NaN or infinity. Only finite values were tested.
- repr() of a closed Sample or Image says it is closed instead of raising.
Behaviour changes:
- add_segmentation(encoding="mask") is refused (E404). It used to merge every class into one volume with empty class_ids and annotated_class_ids, and the file still validated. encoding takes "auto" or one of TRANSCODABLE.
- A float mask holding values other than 0 and 1 is refused (E411). add_mask, add_segmentation(masks=…, ignore=…) and InstanceInput used to read 0.2 as set. Threshold the mask first, or pass a bool array.
- A commit refused before it changes anything leaves the writer open, and committing an aborted writer raises. A second commit() used to return None as if the file had been written. In a with block, a refused commit still aborts.
- An amendment keeps the file's digest_algo. sha512 and blake2b files used to be restamped in sha256 with a new content_id, or, with digests=False, left holding two algorithms. fix --rewrite-digests restamps in the file's algorithm too.
- drop_clinical() on a sample without the clinical profile keeps its version (1.1 §2.1).
Performance:
- Writing layers measures the overlap graph once instead of twice.
- Image reads copy less: read(physical=True) converts and rescales in one pass, and asking for the stored dtype no longer copies. Values identical.
Documentation:
- Image.read(physical=True) documented as it behaves: float32 for a rescaled image; any other image is read as stored.
- amend documents Windows: close every reader before the commit.
- written_version names its drop_clinical exception.
## Handoffs
- scrub: writer temp file holds old doc at each alias until commit; scan the source or skip writer.meta_aliases (pub(crate)). After a remap the commit compacts, so scrub's separate repack is redundant.
- annot: writer validates encoding before select_encoding (writer_annotations.rs:~405); every encode passes colouring: Some(stats.colouring).
- valid: repair.rs one-liner above. Index source_digest is always sha256 (writer_annotations.rs:1230, verify.rs:286): say whether §13.3 means the file's algorithm.
## Doc wording
- python-api.md:419-420: "On POSIX, anything holding the file open across an amend keeps reading the old version; Windows cannot replace a file that is open, so close every reader of it before the commit, whose rename otherwise fails and leaves the file as it was."
- Writer reference: a refused commit leaves the writer open to mend; add_segmentation(encoding=…) takes "auto" or a transcodable kind; a float mask may hold only 0 and 1.
- Spec optional §2.1/§13.1: an amendment keeps the file's digest_algo, and every digest a writer stamps uses it.
## Decisions / open
- a_first/meta alias sorting before meta: the document object's first path is the alias; it carries a digest; content_id differs from a file without the alias. Validation and verify pass.
- Unknown digest_algo (E703) still falls back to sha256.
- Latent bug not fixed: check_held opens self.path rather than the amend source (unreachable today).
## Tests: 179 lib tests, 480+569+565+346 Python, corpus 157/157 with identical content_ids across all 158 files vs main. 16 new cases fail on main.

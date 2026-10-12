# Workstream `annot`: annotations, labels, F14 closure, and label-path performance (W2 annotations, W6 label/index/re-reads)

Worktree: `SCR/wt/annot` (branch `w202/annot`).

## Files you own
- `crates/medh5/src/annotations/**`: `encode.rs`, `encode_geometric.rs`, `header.rs`, `layout.rs`, `payload.rs`, `read.rs`, `read_geometric.rs`, `select.rs`, `mod.rs`.
- `crates/medh5/src/labels/**`, including the bundled vocabularies under `crates/medh5/data/`. Find where the label sets are stored.
- `crates/medh5/src/storage/index.rs` and `crates/medh5/src/sample/reader.rs`.
- `crates/medh5-python/src/{annotations,labels}.rs`.
- `medh5/annotations/**`, `medh5/labels.py`.
- `docs/reference/annotations.md`.
- Tests: `tests/format/{test_annotations,test_encodings,test_geometric,test_classification,test_labels,test_readers}.py`.

## Not yours
- `add_segmentation(encoding=…)` accepting only `auto` plus transcodable kinds, and calling `check_target` in `sample/writer_annotations.rs`: the writer agent. You make `select_encoding` refuse an unknown `prefer`; they make the writer refuse.
- The validator rules: the valid agent. You do the open-time and encoder checks.
- `box_to_slices` in `geometry/affine.rs`: the geom agent.
- Tracking and agreement: the curation agent.

## Items (quoted from the audit; verify each against the code first)
1. **Top issue 3, select part:** `add_segmentation(encoding="mask")` ORs every class into one mask and writes `class_ids=[]`, `annotated_class_ids=[]`; `annotations/select.rs` returns `prefer` unchecked. **Fix:** make `select_encoding` refuse an unknown `prefer`.
2. `transcode_payload`/`check_roundtrip` (public as `medh5.annotations.voxel.transcode_payload`) (`annotations/encode.rs:990-1026`):
   - apply the writer's refusals (dense→`instances`, in-band ignore);
   - compare the ignore regions;
   - the `check_roundtrip` docstring says A→B→A; make it describe what it actually checks (W7, code docs).
3. Range-check a stored `ignore_id` at open: 1..65535, and not a class id (`annotations/header.rs`). The validator side is the valid agent's.
4. Check at open (`annotations/layout.rs`; `read.rs:~1046`):
   - `bit_class_ids`: unique, equal as a set to `class_ids`, shape `(C,)`;
   - `layer_class_ids`: rows equal planes.
5. `payload_to_masks` panics (Python `PanicException`) on inconsistent hand-built payloads. Run the shape table first (`encode.rs:~799, 814, 823`).
6. **F14** (medium-high): `closure = "implicit"` is stored but never applied by any reader.
   - A classification asserting only `left_kidney`, with `kidney` examined, answers `state(kidney) = negative`: a false negative for exactly the pattern §9 recommends.
   - Spec 1.0 §5.4: under `implicit`, "All `is_a` ancestors of a listed class also hold". "Readers MUST NOT infer ancestors when `closure = "explicit"`".
   - Spec §9: hierarchical classification "asserts only the most specific class; ancestors follow from the label-set DAG".
   - **Fix:**
     - `Classification.state` (`read_geometric.rs:745-751`): a class is positive when it, or any descendant, is asserted > 0 in the same scope unit. Keep `value()`, `labels()` and `asserted_class_ids` literal.
     - `dense_class`/`dense`/`contains` (`read.rs:589-688`): OR in the descendants' planes. Use `LabelSet::descendants` (`labels/mod.rs:602`), inside `dense_class`, so every encoding inherits it.
     - `explicit` stays as is.
   - `LabelSet.close()` already implements the closure, but no reader calls it.
   - Document it in `docs/reference/annotations.md` and the `Classification.state` docstring: what `closure` means to `state()`, `dense()` and `contains()`.
   - Reproductions: `SCR/reports/indep-labels-tracking.md` and `SCR/agents/indep/labels-tracking/`.
   - Report the spec erratum wording (§9, §5.4) for the spec agent.
7. Refuse non-finite box coordinates in the encoders (`encode_geometric.rs:~175`): use `!(lo <= hi)` plus `is_finite`.
8. `dense([0])` answers differently per encoding. Refuse class 0, as 65535 is refused (`read.rs:388`).
9. Smaller items:
   - `instances_from_masks` emits objects that `encode_instances` refuses;
   - `encode_mesh` offsets are unchecked until commit;
   - `boxes` without `instance_ids` get 0-based ids; check the spec for the id rule;
   - NaN scores are not filtered;
   - `has_ignore_region` disagrees with `ignore_region`.
10. **W6, label path** (top issue 7). Measured: 60 classes take 86 ms vs 8 ms, and every training item gains 55–90% label cost.
    - `dense()` on a `labelmap` re-reads the window once per class. Read once and fill every class plane in one pass.
    - `labelmap()` on layers/bitmask: read each plane once.
    - Cache `encodes_ignore`, and skip the in-band ignore read when the annotation has none.
    - Return the ignore region from the block already decoded (`annotations/read.rs:672-676, 731-741, 360-364, 885-887`; `sample/reader.rs:555-558`).
11. **W6, index, bbox, analyse:**
    - Work on `Ix3` or flat offsets instead of `IxDyn` `indexed_iter`. Fuse count, bbox, coords and occupancy into one pass. Read each plane once (`storage/index.rs:94-145`; `select.rs:108-173`).
    - `analyse` copies every mask and is quadratic in classes: 11 s at 200 classes.
    - The 8-class index build takes 2.9–5.3 s.
    - **The outputs must be identical:** index contents, bboxes, occupancy, and `content_id` of written files. Add equivalence tests against the old behaviour where practical.
12. **W6, re-reads:**
    - `BoxesAnnotation.boxes` and the other geometric columns re-read the dataset on every access. Use the handle's cached columns (`medh5/annotations/geometric.py:74-124`).
    - `polygons()` re-reads all vertices per polygon (`read_geometric.rs:533-560`).
    - `instance(id)` decodes every mask (`read.rs:1302`).
13. **W7:** the bundled `brats-subregions` vocabulary's id 4 is a synthetic "tumor" parent, but in BraTS ≤2021 label 4 is enhancing tumour. Document the 2021→2023 relabelling in the vocabulary's description, or wherever the vocabularies are documented. Don't change ids.

## Notes
- **Performance changes must be byte-for-byte neutral** for what gets written and value-for-value neutral for what gets read. Keep the window-read invariant (CLAUDE.md).
- Measure before and after for items 10–12, and put the numbers in your report.
- Tests to run:
  - `wcargo annot test -p medh5 --lib annotations:: labels:: storage::index sample::reader`;
  - `wext annot`, then `wpy annot -m pytest tests/format/test_annotations.py tests/format/test_encodings.py tests/format/test_geometric.py tests/format/test_classification.py tests/format/test_labels.py tests/format/test_readers.py tests/tools/test_sampling.py tests/integrations/test_loaders.py -q`.
  - The torch loaders use your readers, and torch is installed.

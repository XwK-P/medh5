# annot report (w202/annot 7 commits to 009169b) — all 13 items DONE
Localized edits outside ownership: sample/writer_annotations.rs (select_encoding(...)?; annotation_to_masks uses stored_class), medh5-cli/src/seg.rs:76 (?), tests/format/test_properties.py (dense->instances accepts E404).
Tests: Rust 34; owned pytest 124+245, new 96+10+2; sampling+loaders+properties 158; writer/sample/validate/conformance/typing/docs_examples 629 (1 skip); docs_python annotations 23; CLI seg/label 23; fmt/clippy/ruff/mypy clean; equivalence script 530 arrays (incl. content_id under every encoding): 0 differ.
Timings before (main release) -> after (dev opt1): 60-class labelmap dense roi 98.8->9.1 ms; dense whole 1176->44 ms; class_bboxes 8877->44 ms; ignore_region roi 1.3->0.01 ms; layers labelmap roi 12.6->4.3 ms; index class_bboxes 2406->244 ms; build_index 2802->169 ms; occupancy=8 5115->176 ms; occupancy 308->29 ms; analyse 60 cls 8547->561 ms; 200 cls 11839->424 ms; boxes x1000 1132->1.2 ms; polygons() 5125->8.3 ms; instance(id) x50 910->2.5 ms.
## CHANGELOG drafts
Fixed:
- closure = "implicit" is applied by readers: dense, contains, counts, bboxes, the index and Classification.state treat a class as present where a descendant is; before, an examined ancestor read as a false negative (F14).
- transcode_payload applies the writer's refusals (dense->instances, dropping in-band ignore); check_roundtrip compares ignore regions.
- payload_to_masks reports inconsistent payloads (E405/E404) instead of PanicException.
- has_ignore_region agrees with ignore_region (an empty sibling mask ignores nothing).
- instances_from_masks skips empty masks, which encode_instances refused.
Behaviour changes:
- Refusals instead of silent wrong results: select_encoding refuses an unknown or mask preference; dense([0]) and contains(0, …) refused (E404); non-finite box/OBB coordinates (E406) and bad mesh offsets (E408) refused at encode; stored ignore_id out of range or a class id, and inconsistent bit_class_ids/layer_class_ids, refused at open.
- Geometric columns are read once and read-only, like the instance columns; copy one to edit it. A NaN score reads as None.
Performance:
- Label reads and the index read each volume, layer or bitplane once: a 60-class dense is about 10x faster; build_index, class_bboxes and analyse 15-200x faster; outputs identical.
Documentation:
- brats-subregions documents the BraTS <=2021 -> 2023 relabelling (4 -> 3).
## Handoffs
- valid: validator mirror of open-time checks (ignore_id range/not a class -> E404; bit_class_ids unique, set-equal, (C,); layer rows == planes).
## Spec wording
- §5.4: "Under implicit, readers MUST answer a class as present wherever a class it is an is_a ancestor of is present (contains, dense planes, counts, bounds, classification state, per scope unit); stored values stay literal."
- §8.2/§8.3: coordinates MUST be finite. §8.7: offsets checked when encoded. §8.2: without instance_ids, readers may number boxes by row; such ids join nothing.
## Decisions: Rust select_encoding returns Result (Python unchanged); header's ignore_id and class 0 refused as classes; positive descendant beats asserted-0 ancestor under implicit; scope_id=None state() raises E412 if a descendant is asserted in several scope units; labelmap() and annotation_to_masks literal (stored_class).

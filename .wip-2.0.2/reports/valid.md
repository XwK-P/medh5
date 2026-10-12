# valid report (w202/valid, 3 commits) — all 18 items DONE
Tests: Rust validate/integrity/conformance 20 + --test conformance 5; pytest validate/integrity/repair 243; conformance 46; docs_examples+typing 266; clinical+cli 257; both corpus runs 163/163; clippy/ruff/mypy clean. ~40 new tests incl. verify/validate/diagnose agreement over 11 digest cases.
Corpus: 6 new cases (163 total), every stated count updated; includes E703-every-digest-malformed.
## CHANGELOG drafts
Fixed:
- Integrity and strict validation fail a file in which no digest parses. E703, E702 and W905 were dropped, so validate passed files verify failed (F01).
- The validator reports objects of the wrong kind or that lead nowhere (E004/E008/E001/E109), and images that are not numbers (E001), at every level (F09).
- One unreadable chunk no longer hides the rest of the integrity report; it is E001 at its own path.
- The validator checks missing §6.2 attributes (E412), ignore_id, bit_class_ids, layer_class_ids, non-finite boxes and affines (E406/E504), and that /meta is UTF-8.
- diagnose/fix see malformed digests (Diagnosis.malformed, additive); fix_paths attempts every file (raises one error naming each failed file); --rebuild-index keeps each entry's max_coords, seed and occupancy.
Behaviour changes: the conformance scorer scores severity (an E-code reported as a warning no longer passes); rules_for refuses unknown levels (ValueError); fix --rewrite-digests refuses structurally invalid files (writer's commit gate); validation stops after E002; a non-string version is E001.
Performance: digest slabs follow chunk boundaries, so each chunk is decoded once (digests identical).
Documentation: conformance.md says severity is scored, describes the 1.1 E009/E803 reclassification, cites §2.2; codes.json summaries for E004/E008/E109/E406/E504/E702.
## Handoffs
- annot (sample/reader.rs image_ids()): dangling image raises KeyError; writer (sample/image.rs): image of strings raises TypeError — both should be MEDH5Error; test_conformance smoke test names these two until fixed.
- cli (inspect.rs fix output): print a malformed count (today "needs attention").
- version: require_major refuses " 1.0" but version::support accepts it.
## Spec wording (§15.2): new codes.json summaries E004/E008/E109/E406/E504/E702; §4.2 "images MUST be numeric"; §6.2 E412.
## Decisions: E002 decided by readers' require_major, not support. Open: an undigested dataset under a content_id fails verify but the validator reports nothing (§13.1 SHOULD; no code fits) — excluded from the agreement test.

# W22-07 De-identification: collections, clinical text

| | |
|---|---|
| Branch | `w22/deidentification` |
| Size | L overall: PR 1 M, PR 2 M–L |
| Lands after | v2.1.0 released |
| Builds on (2.1.0) | `ScrubOptions`, `scrub_each`, per-subject shifts, `--out` (W21-10 PRs 1–2); the clinical derivation and text attestation (W21-10 PR 3) |
| Owns | `crates/medh5/src/curation/scrub.rs`, `crates/medh5/src/curation/scrub/clinical.rs` (where 07.2's work lives), `medh5/curation/scrub.py`, `crates/medh5-cli/src/scrub.rs`, `docs/guides/deidentify.md` |
| Decisions | E7, E8 |

## Session prompt

> Implement workstream W22-07 (de-identification: collections, clinical text) of
> medh5 2.2.0 in this repository. First read `plans/README.md`,
> `plans/2.2.0/README.md` and `plans/2.2.0/decisions.md` (E7, E8), then
> `plans/2.2.0/w22-07-deidentification.md` in full. Check that 2.1.0's scrub
> work (W21-10) is on `main`. Work from the latest `main` on
> `w22/deidentification`; land the two items in order, as one pull request.
> Never attest work that was not done; the source is only read; a span is
> remapped or the rewrite refused, never moved by guess. Write tests first, run
> the local gate from `CLAUDE.md`, add fragments, open a `W22-07: …` pull
> request, and arm auto-merge only as plans/README.md's "Review before the
> queue" says.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 07.1 Scrubbing a collection, member by member | M |
| 2 | 07.2 Rewriting clinical text | M–L |

## Items

### 07.1 Scrubbing a collection, member by member (PR 1, E8)

- **State at 2.1.0** (checked at v2.1.0). A collection is refused before
  either step, by `open_sample`'s refusal of anything that is not a sample
  (`reader.rs`), so `scrub` refuses a `.medh5c` when it scans as well as when
  it applies, and `amend_into` refuses collections. `scrub_paths` is
  deprecated for 3.0 (use `scrub_each`).
- **Change.** `scrub --apply COLLECTION --out NEW.medh5c` (required; the source
  read-only): each member scrubbed by the sample rules with one salt, per-subject
  shifts by each member's subject, ids pseudonymised consistently across members,
  written into a new collection whose members list and per-member `content_id`s
  are rebuilt; a member that refuses fails the whole collection with nothing
  written (or, with `--skip-refused`, is left out and listed). The scan reads a
  collection member by member. Python `apply(collection_path, *, out=…, …)`;
  `scrub_each` accepts collections (the deprecated `scrub_paths` gains
  nothing).
- **Tests.** `test_S2_2_a_scrubbed_collection_is_a_new_collection`;
  `test_S11_4_members_of_one_subject_share_a_shift`;
  `test_S2_2_a_refused_member_leaves_no_output`;
  `test_S2_2_the_source_collection_is_byte_identical`.
- **Changelog.** Added, area `deidentification`.

### 07.2 Rewriting clinical text (PR 2, E7)

- **State at 2.1.0** (checked at v2.1.0). Clinical text is scanned one document at a time
  and attested by digest; apply refuses while a `clinical_text` hit remains; text
  bytes are never rewritten.
- **Change.** `--redact-text` (Python `redact_text=True`): for each document with
  hits, a new document whose text replaces each hit with a fixed placeholder of the
  rule (`[NAME]`, `[DATE]`, `[ID]`), owned by a new version of its event; every
  span in the sample pointing into the old text is remapped through the edit (a
  span overlapping a redacted range is refused, naming it); short text columns
  are rewritten the same way, as new versions. The attestation records the
  redaction (`tool: "medh5 scrub"`, its rules, the digests written). Hits the
  rules do not find still need `--attest-text`.
- **Tests.** `test_S6_redaction_writes_a_new_document_and_version`;
  `test_S6_spans_are_remapped_through_the_redaction`;
  `test_S6_a_span_inside_a_redacted_range_is_refused`;
  `test_S6_a_redacted_sample_validates_clean`;
  `test_S6_redaction_is_read_one_document_at_a_time`.
- **Docs.** `docs/guides/deidentify.md`, 1.1 §10's bullet (a sentence on
  rewritten text).
- **Changelog.** Added, area `deidentification`.

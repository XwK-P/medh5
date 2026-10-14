# W21-06 Python API and layering

| | |
|---|---|
| Branch | `w21/python` |
| Size | L overall; each pull request S–M |
| Lands after | W21-00 PR 4, PR 5; PR 6 after W21-00 PR 8 |
| Blocks | nothing |
| Owns | `medh5/{errors.py,validate.py,document.py,conformance.py,storage.py,integrity.py,image.py,sampling.py,bench.py,annotations/base.py}`, `crates/medh5-python/src/{errors.rs,convert.rs,reader.rs,validate.rs}` and the result-type bindings, `crates/medh5/src/lib.rs`, `docs/reference/rust.md`, `crates/medh5/README.md` |
| Decisions | D27, D31, D35, D38 |
| Backport | none |

## Session prompt

> Implement workstream W21-06 (Python API and layering) of medh5 2.1.0 in this
> repository. First read `plans/README.md`, `plans/2.1.0/README.md` and
> `plans/2.1.0/decisions.md` (D27, D31, D35, D38), then
> `plans/2.1.0/w21-06-python.md` in full. Work from the latest `main` on
> `w21/python`; land the six pull requests in order. The layering pull requests
> (3–5) replace Python re-implementations of engine answers with calls into
> `_core`: where the two disagree today, the engine's answer wins and the
> difference is a Fixed entry, never silently absorbed. PR 6 is conditional on
> the semver gate (D35): check it locally with `cargo semver-checks` against
> 2.0.2 before deciding. Write tests first, run the local gate from `CLAUDE.md`,
> add fragments, open `W21-06: …` pull requests, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 06.1 File errors carry `errno` and the standard subclasses | M |
| 2 | 06.2 `physical=True` documented; 06.3 Ellipsis in `read`, `Instance` by value; 06.4 `dataset_layout`; 06.5 `draws(grid=…)` and a lazy `iter_patches` | S–M |
| 3 | 06.6 Result types delegate to the engine; 06.7 three facades that drifted | M |
| 4 | 06.8 Vocabularies come from the engine; the guard test | S–M |
| 5 | 06.9 Engine functions with no caller; 06.10 six named cases | S–M |
| 6 | 06.11 The crate's public surface | S |

## Items

### 06.1 File errors carry `errno` and the standard subclasses (PR 1, D31)

- **State at 2.0.2.** `Error::File(String)` and `Error::Io(String)` carry only a
  message (`crates/medh5/src/error.rs`); `From<hdf5::Error>` and
  `From<std::io::Error>` both produce `Io`; `os_error` produces `Io("[Errno N]
  reason: 'path'")`. `open_read` stats first and maps a failure to fixed words
  with no errno (`os_reason`, the deterministic E001 words). The binding
  (`crates/medh5-python/src/errors.rs`) maps `File` to `MEDH5FileError(m)` and
  `Io` to a plain `OSError(m)`. One missing file gives three errors: a missing
  file or a directory is `MEDH5FileError` with `errno=None`, `filename=None`, not
  a `FileNotFoundError` and no `__cause__`; a damaged chunk a plain `OSError`;
  `conformance.check_checksums` on a missing suite an `OSError` whose text says
  "[Errno 2]" while `errno` is `None`; `TaskManifest.load` a real
  `FileNotFoundError`. `docs/reference/python-api.md` documents this.
- **Change.**
  - `MEDH5FileError.__init__(self, message, errno=None, filename=None)`;
    `__str__` returns the message (E001 words and the CLI unchanged);
    `__reduce__` keeps all three; the defaults keep `exc_type(msg)` working.
  - New, each also a `MEDH5FileError`: `MEDH5FileNotFoundError(MEDH5FileError,
    FileNotFoundError)`, `MEDH5PermissionError(MEDH5FileError, PermissionError)`,
    `MEDH5IsADirectoryError(MEDH5FileError, IsADirectoryError)`; exported from
    `medh5` and `medh5.errors`.
  - On the error path only, the binding's path doors (a `path_error(err, path)`
    used by open, validate, verify, amend, fix, diagnose, pack and the
    converters) re-ask the system (`fs::metadata`, then `fs::File::open`) to pick
    the class, taking `errno` from `raw_os_error()`; the message stays free of
    errno, as `os_reason` intends.
  - The engine's "[Errno N] reason: 'path'" `Io` messages become `OSError(N,
    reason, path)`, which Python maps to the right subclass.
  - HDF5 read failures raise a new `MEDH5IOError(OSError)`, not yet a
    `MEDH5Error`, so catch order is unchanged. 3.0 (ledger): `MEDH5IOError`
    derives from `MEDH5FileError`; Rust `Error` becomes `#[non_exhaustive]` with
    `Os { kind, path, message }`.
- **Tests.** `test_a_missing_file_is_a_file_not_found_error` (`errno == ENOENT`,
  `filename` set, `str` unchanged); `test_a_directory_is_an_is_a_directory_error`;
  `test_an_unreadable_file_is_a_permission_error` (skipped as root);
  `test_a_file_error_pickles_with_its_errno`;
  `test_a_damaged_chunk_is_an_io_error_not_a_medh5_error`;
  `test_S15_a_missing_file_is_reported_the_same_on_every_run` stays green. A
  `STALE_CLAIMS` entry for "so `except FileNotFoundError` does not catch it".
- **Docs.** `docs/reference/python-api.md` (Exceptions), `medh5/errors.py`.
- **Changelog.** Added, area `python-api`; Deprecated (the 3.0 hierarchy change).

### 06.2 `Image.read(physical=True)` is documented, not changed (PR 2, D27)

- **State at 2.0.2.** `Image::read_index` (`sample/image.rs`) applies the rescale
  only when `slope != 1 || intercept != 0` (`is_rescaled`) and then returns
  float32 unless `dtype` says otherwise; other images come back in their stored
  dtype. The 2.0.2 docstring says so (`medh5/image.py`).
- **Change.** The reference states the rule: "a physical read is float32 only
  where a rescale applies; pass `dtype` for one type across a cohort"
  (`docs/reference/python-api.md`, `docs/reference/torch.md`,
  `docs/guides/cohorts.md`, `docs/guides/import-dicom.md`). `STALE_CLAIMS` gains
  "physical=True returns float32".
- **Tests.** `test_S4_2_a_physical_read_without_a_rescale_is_the_stored_values_in_their_type`;
  `test_S4_2_an_identity_rescale_declared_explicitly_reads_as_stored`;
  `test_S4_2_dtype_gives_one_type_across_a_cohort`.
- **Changelog.** Documentation.

### 06.3 Ellipsis in `Image.read`; `Instance` compares by value (PR 2)

- **State at 2.0.2.** `Image.read`'s roi is parsed by `convert::region`
  (`crates/medh5-python/src/convert.rs`, called from `reader.rs`), which has no
  Ellipsis: `read((..., slice(0, 2)))` on a `(2,4,5,6)` image raises
  `TypeError`, while `img.dataset[..., 0:2]` works (`nodes.rs` `dataset_key`).
  `Instance` is `@dataclass(frozen=True, slots=True)` with ndarray `box` and
  `mask` (`medh5/annotations/base.py`): `a == b` raises "truth value …
  ambiguous", `a in [b]` raises, `hash(a)` raises `TypeError`.
- **Change.** `convert::region_for(obj, ndim)` expands a single Ellipsis
  NumPy-style; two raise `IndexError`; every other form behaves as today.
  `Instance` becomes `eq=False` with a value `__eq__` (`np.array_equal` on `box`
  and `mask`; two `None` masks equal) and `__hash__ = None` (hashing still raises
  `TypeError`).
- **Tests.** `test_S14_5_an_ellipsis_fills_the_axes_it_stands_for` (3-D and 4-D);
  `test_S14_5_two_ellipses_are_refused`; `test_S7_4_instances_compare_by_value`
  (with `in` and `!=`); `test_S7_4_an_instance_is_unhashable`.
- **Changelog.** Fixed, area `python-api`.

### 06.4 `storage.dataset_layout` (PR 2)

- **State at 2.0.2.** `medh5.storage.dataset_kwargs(shape, dtype, *,
  profile=None, role="image", chunks=None)` returns the writer's layout, not the
  h5py keywords its name promises (`{'chunks': (8,32,64), 'codec':
  'blosc2:zstd:3:shuffle'}`; `**`-passing it to h5py raises on `codec`).
  `_core.dataset_layout` exists. (2.0.2 already made it honour a custom
  `CodecProfile`.)
- **Change.** `medh5.storage.dataset_layout(shape, dtype, *, profile=None,
  role="image", chunks=None) -> dict[str, Any]`, returning exactly what
  `dataset_kwargs` returns. `dataset_kwargs` moves into `storage._DEPRECATED`
  (plans/README.md, Deprecations), its value the function itself, so reaching
  the name raises `DeprecationWarning`: "medh5 3.0 removes dataset_kwargs: use
  dataset_layout; for h5py keywords, resolve_profile(p).codec(role).kwargs()";
  it leaves `__all__`, and the API snapshot lists it as deprecated. Removed in
  3.0, the name never reused. Move the tests' uses to the new name.
- **Tests.** `test_S14_2_dataset_layout_is_dataset_kwargs_renamed` over the
  existing cases; `test_S14_2_dataset_kwargs_warns`.
- **Docs.** `docs/reference/storage.md`.
- **Changelog.** Deprecated, area `python-api`.

### 06.5 `draws(grid=…)` and a lazy `iter_patches` (PR 2)

- **State at 2.0.2.** The engine's `draw` takes `grid` (`sampling.rs`) but
  `draws` does not (it calls `draw(…, None)` n times from one `Rng`). The binding
  builds a new engine `Rng` each call (`crates/medh5-python/src/rng.rs`).
  `iter_patches` is `yield from sampler.draws(...)` (`medh5/sampling.py`): eager,
  so `iter_patches(n=10_000)` holds every patch before yielding the first; a
  Python loop over `draw` with `rng=42` repeats one patch.
- **Change.** Rust `PatchSampler::draws_in(&self, sample: &Sample, annotation:
  Option<&str>, n: usize, rng: &mut Rng, grid: Option<&str>) ->
  Result<Vec<Patch>>`; `draws` delegates. Binding `draws(sample,
  annotation=None, n=1, rng=None, *, grid=None)` and `iter_draws(...) ->
  PatchDraws`, a `#[pyclass(module = "medh5._core")]` holding a clone of the
  engine sampler, the sample's `Arc`, one `Rng` and the remaining count; each
  `__next__` draws one patch with the GIL released. Facade
  `PatchSampler.draws(..., *, grid=None)`; `iter_patches(sample, sampler, *,
  annotation=None, n=1, rng=None, grid=None) -> Iterator[Patch]`, validating the
  annotation and grid when called. It yields exactly what `draws` returns.
- **Tests.** `test_S14_3_iter_patches_yields_what_draws_returns` (same seed);
  `test_S14_3_iter_patches_draws_one_patch_per_next` (`_core._read_stats()`);
  `test_S14_3_draws_pins_the_window_grid`;
  `test_S14_3_a_misspelt_annotation_is_refused_before_the_first_patch`.
- **Docs.** `docs/reference/torch.md`, the sampling section of
  `docs/reference/python-api.md`.
- **Changelog.** Added and Performance, area `python-api`.

### 06.6 Result types delegate to the engine (PR 3)

- **State at 2.0.2.** These Python result types compute their verdicts, counts,
  text and JSON in Python, while the engine types they are rebuilt from compute
  the same, and the CLI prints the engine's: validate `Report` (strict promotion
  `_effective_severity`, `errors`, `warnings`, `promoted`, `codes`, `ok`,
  `to_json`, `format`) and `merge` (`medh5/validate.py`); `conformance.summarize`,
  `Case`, `CaseResult`; `RecompressResult` (`medh5/storage.py`); `Diagnosis`,
  `Repair`, `VerifyResult` (`medh5/integrity.py`); bench `Measurement`
  (`medh5/bench.py`), whose `to_json` already disagrees (NaN: the engine writes
  `null`, Python `NaN`). The binding already has the cure, used for five types:
  a function takes the dataclass, rebuilds the engine value from its attributes
  (`report_arg` in `crates/medh5-python/src/curation_tools.rs`) and calls the
  engine method (`scrub_report_*`, `audit_*`, `io_report_*`, `dataset_check_*`,
  `dataset_moments_*`). `ClassStats` (W21-08) and `CacheReport` (W21-11) are
  other workstreams'.
- **Change.** Keep every dataclass (public; users construct them) and move its
  logic behind `_core`, one function per method. Engine: make the private
  `Report::effective` public as `pub fn effective_severity(&self, d: &Diagnostic)
  -> &str` (this item's only engine change; 06.7 makes two more private items
  visible). New bindings and stubs:

  ```python
  def validate_report_positions(report: Any) -> tuple[list[int], list[int], list[int]]: ...
  def validate_report_json(report: Any) -> dict[str, Any]: ...
  def validate_report_format(report: Any, *, verbose: bool = False) -> str: ...
  def validate_merge(reports: Sequence[Any], path: str = "<multiple>") -> dict[str, Any]: ...
  def conformance_summary(results: Sequence[Any]) -> dict[str, Any]: ...
  def storage_recompress_json(result: Any) -> dict[str, Any]: ...
  def storage_recompress_line(result: Any) -> str: ...
  def integrity_diagnosis_json(diagnosis: Any) -> dict[str, Any]: ...
  def integrity_repair_json(repair: Any) -> dict[str, Any]: ...
  def integrity_verify_json(result: Any) -> dict[str, Any]: ...
  def bench_measurement_json(measurement: Any) -> dict[str, Any]: ...
  ```

  `Report.errors` becomes `tuple(self.diagnostics[i] for i in positions[0])`, so
  callers keep their own `Diagnostic` objects; `promoted` and `ok` follow from the
  same positions.
- **Tests.** One parametrised test per type: an engine-produced value
  round-tripped through `from_json` gives `to_json()`, `str()` and `format()`
  equal to the engine's output, and to the CLI's `--json` where it has one; a
  hand-built `Report` at `strict` with warnings gives the same counts from
  Python as from `medh5 validate --level strict --json`.
- **Changelog.** Fixed, area `python-api`: `Measurement.to_json()` writes `None`
  for a NaN, as the engine and CLI do.

### 06.7 Three facades that drifted (PR 3)

- **State at 2.0.2.** `medh5/validate.py` `LEVELS` is a literal (the engine's is
  bound as `_core.LEVELS`; the binding also re-spells `check_level` as
  `known_level` and `attr_names_for` inside `validate_root`).
  `medh5/document.py` `schema()` is `@lru_cache`d over `_core.schema()`, which
  returns a fresh dict: every caller shares one mutable dict. `medh5/cli.py`
  `_sample_file` says "works on sample files" where the native
  `sample_file_error` (`crates/medh5-cli/src/common.rs`) says "works on one sample
  file": the wording already drifted.
- **Change.** `LEVELS = _core.LEVELS`, and the binding uses the engine's
  `check_level` and attribute names: both are private in `validate/mod.rs`
  (`fn check_level`, `fn attr_names_for`), and the binding is another crate, so
  they become `#[doc(hidden)] pub` (helpers only a frontend needs).
  `document.schema()` returns `_core.schema()`
  uncached. `#[doc(hidden)] pub fn sample_file_refusal(path: &str) ->
  Option<String>` factored out of `sample_file_error`, re-exported from the CLI
  crate's root beside `top_level_message`, bound as `cli_sample_file_refusal`,
  used by `_sample_file`.
- **Tests.** A collection passed to a Python converter and to a native command
  prints one identical message; `schema() is not schema()` and `schema() ==
  _core.schema()`; `validate.LEVELS is _core.LEVELS`.
- **Changelog.** Fixed, area `python-api` (the refusal's wording; `schema()` no
  longer shares one mutable dict).

### 06.8 Vocabularies come from the engine; the guard test (PR 4)

- **State at 2.0.2.** `medh5/errors.py` `Domain` is never tested against
  `codes::DOMAINS`, and `codes_for` re-implements `codes::for_domain`. Four
  `("LPS", "RAS")` tuples (`medh5/monai.py`, `io/nifti.py`, `io/dicom_seg.py`,
  `io/rtstruct.py`) spell the schema's enum, with no engine constant. (The task,
  cache and clinical constants are W21-11's.)
- **Change.** `pub const COORD_SYSTEMS: &[&str]` in `geometry`; bindings
  `_core.COORD_SYSTEMS` and `_core.CODE_DOMAINS`; `errors.codes_for` calls
  `codes::for_domain`; module-level copies bind the `_core` objects. Guard test
  `tests/project/test_layers.py::test_vocabularies_are_the_engines`: every
  module-level tuple of strings in `medh5/` whose value equals a `_core` constant
  must be that object (`is`); its allow-list holds W21-11's constants until W21-11
  PR 5 and must be empty at release.
- **Tests.** The guard; `typing.get_args(Domain) == _core.CODE_DOMAINS`; the
  schema's `coord_system` enum equals `COORD_SYSTEMS`.
- **Changelog.** `<!-- no entry: no value changes -->`.

### 06.9 Engine functions with no caller (PR 5)

- **State at 2.0.2.** 26 `pub fn`s have no caller outside tests. An unused
  function is an untested promise: `columns_of("evnts")` answers with the link
  columns; `Deidentification::shift_days` reads 0 as a shift, the bug
  `recorded_shift` exists to avoid.
- **Change.** Apply this table; each deprecation names its replacement and 3.0
  (and goes in the ledger as "the engine functions with no caller").

  | Function | Where | Do |
  |---|---|---|
  | `Assertion::is_positive` / `is_negative` | `annotations/read_geometric.rs` | keep (mirror `medh5/annotations/classification.py`); add a test |
  | `Patch::needs_padding` | `sampling.rs` | keep; mirrors `sampling.py` |
  | `Event::is_lesion_assessment` | `clinical/model.rs` | use it in `clinical/check.rs` |
  | `Selection::event_ids` | `clinical/select.rs` | keep (integration tests use it) |
  | `ConversionReport::of_kind` | `convert/report.rs` | keep; mirrors `io/report.py` |
  | `Grouping::timepoint_ids` | `convert/grouping.rs` | use it in `convert/legacy.rs` (the `tp{i}` ids) |
  | `Deidentification::shift_days` | `curation/identity.rs` | left to W21-10 PR 1 (reads 0 as no shift, used in `recorded_shift`) |
  | `Clinical::links_from` / `links_to` | `clinical/table.rs` | keep; add a test |
  | `TaskManifest::rows_in` | `companion/task.rs` | left to W21-11 PR 5 (one pass) |
  | `PatchSampler::with_size` | `sampling.rs` | keep; add a test |
  | `Grid::from_spec`, `strings_digest`, `codes::for_domain` | | keep |
  | `columns_of` | `clinical/columns.rs` | deprecate (unknown names return the link columns) |
  | `array_data` | `annotations/encode.rs` | deprecate (`PayloadData::Array(a)` is the API) |
  | `bbox_volume` | `annotations/select.rs` | deprecate |
  | `targeted_metrics` | `bench.rs` | deprecate |
  | `profiles_repr` | `storage/codecs.rs` | deprecate |
  | `matmul` | `geometry/linalg.rs` | deprecate (it is `a.dot(b)`) |
  | `sampling::PairReport` and `add_skip` | `sampling.rs` | deprecate the type (Python's `PairReport` is the live one) |
  | `attrs::object_name`, `data::slice_array`, `ops::datasets_resolving` | `h5/…` | leave; hidden with `h5` (06.11), deleted in 3.0 |

- **Tests.** One unit test per kept function.
- **Changelog.** Deprecated, area `release` (Rust only).

### 06.10 Six named cases (PR 5)

- **`codes::code`** (`codes.rs`) has no caller and quotes with `{name:?}` (`"X"`)
  where `medh5.errors.code` quotes `'X'`: keep it, quote with
  `json::repr_str` as Python does, and have `errors.code` call it.
- **`Chains`** (`clinical/select.rs`, re-exported) has no caller (the engine uses
  the private `chain_indices`): keep it as a reader of `supersedes` chains, with a
  test.
- **`Deidentification.extra`** (`curation/identity.rs`): the schema's
  `deidentification` object is `additionalProperties: false`, so anything it
  holds makes the document invalid. `#[deprecated]` on the field (its uses get
  `#[allow(deprecated)]`); the Python getter warns through `deprecation.rs`;
  removed in 3.0.
- **`data::filters`** (`h5/data.rs`) passes a 64-byte `name` buffer nothing reads
  and pushes an entry even when `H5Pget_filter2` fails: pass a zero length and a
  null name, and return `Error::Io` when the id is negative, closing the
  property list on that path too. It also truncates client values to its
  32-value buffer: `H5Pget_filter2` fills at most the capacity it is given but
  returns the filter's full count in `nelmts`, so when that count exceeds 32,
  call it again for that filter with a buffer of `nelmts` values and return
  them all.
- **`validate_file`** (`validate/mod.rs`) runs `check_level` first, so its later
  "unknown validation level" arm is unreachable: delete it.
- **`scrub_paths`** is W21-10's (`scrub_each`).
- **Tests.** Rust `h5::data::tests::filters_returns_every_client_value`: a
  dataset created with an optional filter (`H5Z_FLAG_OPTIONAL`, an id no library
  registers) carrying 40 client values returns all 40, in order (32 at 2.0.2);
  `codes::code` quotes as Python does.
- **Changelog.** Deprecated (`extra`), Fixed (`filters`), area `release`.

### 06.11 The crate's public surface (PR 6, D35)

- **State at 2.0.2.** `crates/medh5/src/lib.rs` has `pub mod` `bench`, `h5`,
  `json`, `numeric`, `pyval`, `rng`, plus `pub use hdf5;` and `pub use ndarray;`.
  `docs/reference/rust.md` and `crates/medh5/README.md` call `h5`, `json`,
  `numeric`, `pyval` and `rng` "public too, … the engine's plumbing". The CLI uses
  `h5::{attrs, ops, data, file::same_file}`, `hdf5`, `json`, `pyval`, `bench`; the
  binding uses `h5::{…}`, `AttrValue`, `hdf5::{Dataset, Group, Location, Error}`,
  `json` (64 uses), `rng::Rng`, `bench`; nothing uses `numeric`. 193 public
  signatures name `hdf5::` types, about 119 outside `h5` (`Sample.root`,
  `Image::dataset`, the writer's `add_*`, `validate_root`). `AttrValue` lives in
  `h5::attrs` but appears in documented fields (`Grid.extra`,
  `TransformHeader.extra`, `AnnotationHeader.extra`, `AnnotationPayload.attrs`,
  `Image::attr`).
- **Change.**
  - Add `pub use h5::attrs::AttrValue;` at the crate root, documented.
  - Check locally whether `cargo semver-checks` (against 2.0.2) accepts
    `#[doc(hidden)]` on `pub mod h5`, `json`, `pyval`, `numeric`. If it does, add
    it; if it reports a break, do not (D35): 2.1.0 only documents them as
    plumbing, and hiding moves to 3.0 (ledger).
  - `pub use hdf5` and `pub use ndarray` stay visible. The README and
    `docs/reference/rust.md` declare hdf5-metno 0.15, ndarray 0.17 and IndexMap 2
    public dependencies (a bump of any is a major medh5 release), and say `h5`,
    `json`, `numeric` and `pyval` are plumbing outside semver from 2.1. `bench`
    and `rng` join the documented table; `labels::canonical_json` stays the
    documented way to §5.1.
  - The README's Rust example is a doctest and `rust.md` shows the same block (a
    test compares them): change both together.
- **Tests.** A project test that every `pub mod` in `lib.rs` is either in
  `rust.md`'s table or `#[doc(hidden)]`; the README-versus-`rust.md` comparison
  extended to the module tables.
- **Changelog.** Documentation (and Behaviour changes, area `release`, if the
  modules are hidden).

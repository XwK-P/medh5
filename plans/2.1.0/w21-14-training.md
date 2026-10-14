# W21-14 Training: MONAI, DDP, stitching, class masks

| | |
|---|---|
| Branch | `w21/training` |
| Size | M overall; each pull request S–M |
| Lands after | W21-00 PR 4, PR 5; PR 4's write site after W21-05 PR 1 |
| Blocks | nothing |
| Owns | `medh5/torch/**`, `medh5/monai.py`, `medh5/_monai_readers.py` (new), `docs/reference/torch.md`; touches `medh5/sampling.py` (W21-06's) to add the aggregator, a new class |
| Decisions | D30, D34 |
| Backport | none: PR 1's collate fix changes what code reading the metadata sees (`"none"` for `None`, D30), which a patch release does not |

## Session prompt

> Implement workstream W21-14 (training: MONAI, DDP, stitching, class masks) of
> medh5 2.1.0 in this repository. First read `plans/README.md`,
> `plans/2.1.0/README.md` and `plans/2.1.0/decisions.md` (D30, D34), then
> `plans/2.1.0/w21-14-training.md` in full. Work from the latest `main` on
> `w21/training`; land the four items in order as one pull request. `import
> medh5.monai` must still import no MONAI; MONAI and torch tests `importorskip`,
> and run in the `test-monai` and `test-min-deps` jobs (MONAI 1.3, torch 2.0):
> check the floors. HDF5 handles must not cross `fork`, and rows index, never
> copy. Write tests first, run the local gate from `CLAUDE.md` (and `pytest
> tests/ -k "Monai or monai"` with the `monai` extra), add a fragment per item,
> open the `W21-14: …` pull request, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 14.1 MONAI batches medh5 items; 14.2 a MONAI reader and `LoadMedH5d` | M |
| 2 | 14.3 DDP sharding for `FileGroupedSampler`; clinical file groups | S |
| 3 | 14.4 Stitching patch predictions back into volumes | M |
| 4 | 14.5 Per-class ignore masks; 14.6 the vocabulary's file and digest | S |

## Items

### 14.1 MONAI batches medh5 items (PR 1, D30)

- **State at 2.0.2.** `meta["medh5"]` carries `None` for an absent `frame_uid`,
  `value_units`, `modality`, `timepoint` or `subject_id` (`medh5/monai.py`
  `meta_dict`), and MONAI's `list_data_collate` and `DataLoader` raise
  `TypeError: default_collate: … found <class 'NoneType'>`: every sample without a
  `frame_uid` (every NIfTI and nnU-Net import) cannot be batched. The tests never
  collate (`tests/integrations/test_monai.py` checks JSON safety only). MONAI's
  own readers spell `None` as `"none"` (`TraceKeys.NONE`).
- **Change.** In the MONAI adapter's metadata only, `None` is `"none"`;
  `from_metatensor` maps it back. `meta["medh5"]` gains `sample_key`.
- **Tests.** `test_a_sample_without_frame_uid_batches` (the regression, through
  `list_data_collate`); `test_from_metatensor_maps_none_back`.
- **Changelog.** Behaviour changes, area `torch`: "**MONAI can batch medh5 items.**
  … code testing `meta["medh5"]["frame_uid"] is None` now sees `"none"`."

### 14.2 A MONAI reader and `LoadMedH5d` (PR 1)

- **State at 2.0.2.** `medh5/monai.py` converts (`meta_dict`, `to_metatensor`,
  `from_metatensor`, `to_dict`) but has no `ImageReader`, so `LoadImaged` cannot
  open a `.medh5`; MONAI warns on readers that do not subclass `ImageReader`.
- **Change.** A private `medh5/_monai_readers.py` (imports MONAI), loaded by
  `medh5.monai.__getattr__` (with `__dir__`), so `import medh5.monai` imports no
  MONAI; the new names in `__all__`:

  ```python
  class MedH5Reader(ImageReader):
      def __init__(self, image: str | None = None, *, annotation: str | None = None,
                   sample_key: str | None = None, physical: bool = True, space: str | None = None,
                   level: int = 0, dtype: npt.DTypeLike = np.float32) -> None: ...
      def verify_suffix(self, filename) -> bool: ...          # .medh5, .medh5c, path::key
      def read(self, data, **kwargs) -> list[tuple[str, str | None]]: ...  # references only
      def get_data(self, img) -> tuple[np.ndarray, dict]: ...  # leases through medh5.torch.handles.CACHE

  class LoadMedH5d(MapTransform):
      def __init__(self, keys: KeysCollection = "path", *, images: Sequence[str] | None = None,
                   annotations: Sequence[str] = (), space: str | None = None,
                   physical: bool = True, allow_missing_keys: bool = False) -> None: ...
  ```

  With `image=None` the reader reads the sample's only image or refuses, naming
  the choices; `annotation=` returns `to_dict`'s label (int64, 65535 marks
  ignore); several filenames stack as channels only when they share one grid. A
  collection member is written `path::key` (as splits name it), split only after
  a `.medh5c` suffix. `LoadMedH5d` is `to_dict` over one leased sample (same
  parameters, same output keys); a second input key is refused (the outputs would
  collide).
- **Tests.** `TestMonaiReader`: `test_reader_affine_matches_to_metatensor`,
  `test_label_is_int64_with_65535_ignore`,
  `test_an_unnamed_image_is_refused_when_there_are_several`,
  `test_loadimaged_batches_through_list_data_collate`,
  `test_a_collection_member_by_path_key`, `test_workers_reuse_handles`.
  `TestMonaiLoadMedH5d`: `test_one_lease_per_item`,
  `test_spacingd_and_orientationd_accept_the_output`. `TestWithoutMonai`:
  `test_importing_medh5_monai_imports_no_monai`, `test_the_reader_names_the_extra`.
- **Docs.** `docs/reference/torch.md` (MONAI): blocks start with `from medh5.monai
  import`, so the `test-monai` job selects them.
- **Changelog.** Added, area `torch`.

### 14.3 DDP sharding for `FileGroupedSampler`; clinical file groups (PR 2)

- **State at 2.0.2.** `FileGroupedSampler(dataset, *, shuffle=True, seed=0)`
  (`medh5/torch/samplers.py`) permutes files with `default_rng((seed, epoch))`,
  then shuffles items within each file from the same stream; `__len__` counts
  every item; it refuses a dataset without `file_groups()`, which is
  `ClinicalTaskDataset` (`medh5/torch/clinical.py`). Nothing supports DDP, and
  `DistributedSampler` breaks the grouping. (Whole files per rank would give
  unequal lengths, which hang DDP collectives.)
- **Change.** `FileGroupedSampler(dataset, *, shuffle=True, seed=0,
  num_replicas: int | None = None, rank: int | None = None, drop_last: bool =
  False)`: defaults from `torch.distributed.get_world_size()`/`get_rank()` when
  available and initialised, else 1 and 0; `ValueError` unless `num_replicas >= 1`
  and `0 <= rank < num_replicas`. `__iter__` builds the 2.0.2 order unchanged
  (same `rng` calls), extends it cyclically to `R·⌈N/R⌉` (or cuts to `R·⌊N/R⌋`
  with `drop_last`), and yields `order[rank·q:(rank+1)·q]`: at most R−1 files per
  epoch are read by two neighbouring ranks. `__len__` is `q`, the same every
  epoch; read-only `num_replicas` and `rank`. The docstring: under DDP call
  `set_epoch` every epoch. `ClinicalTaskDataset.file_groups() -> list[list[int]]`:
  row indices grouped by `row.subject_index` in order of first appearance (every
  row of a subject leases the same sources; indices only).
- **Tests.** `TestFileGroupedSamplerDDP` (beside `test_one_open_per_file_per_epoch`
  in `tests/integrations/test_loaders.py`): `test_ddp_ranks_partition_the_epoch`,
  `test_a_file_is_split_only_at_a_slice_boundary` (at most R−1 split groups),
  `test_one_replica_is_the_2_0_2_order` (a golden list), `test_drop_last_truncates`,
  `test_the_length_is_the_same_every_epoch`, `test_a_rank_out_of_range_is_refused`,
  `test_defaults_come_from_torch_distributed` (monkeypatched). In
  `test_clinical_training.py`: `test_rows_are_grouped_by_subject`,
  `test_the_sampler_takes_a_clinical_dataset`.
- **Docs.** `docs/reference/torch.md` (a DDP block passing `num_replicas=2,
  rank=1`, so it runs without a process group), `docs/guides/clinical-training.md`.
- **Changelog.** Added, area `torch`.

### 14.4 Stitching patch predictions back into volumes (PR 3)

- **State at 2.0.2.** `GridPatchDataset` (`medh5/torch/datasets.py`) plans `(path,
  Patch)` per file with `grid_patches` (`medh5/sampling.py`); a `Patch` carries
  `slices`, `pad`, `grid_id`; `_meta` stores the normalised path and
  `patch.to_json()`. "Stitch back" is documented, but nothing reassembles
  predictions.
- **Change.** A NumPy core in `medh5.sampling`, plus a thin torch adapter, so MONAI
  and non-torch users share the arithmetic:

  ```python
  BLEND_MODES = ("constant", "gaussian")
  def blend_weights(patch_size: Sequence[int], *, mode: str = "gaussian",
                    sigma_scale: float = 0.125) -> npt.NDArray[np.float32]: ...
  class PatchAggregator:
      def __init__(self, spatial_shape: Sequence[int], *, channels: int | None = None,
                   mode: str = "gaussian", sigma_scale: float = 0.125,
                   dtype: npt.DTypeLike = np.float32, grid_id: str | None = None) -> None: ...
      def add(self, prediction: npt.ArrayLike, patch: Patch) -> None: ...  # cuts patch.pad
      count: int
      coverage: npt.NDArray[np.bool_]
      def result(self, *, uncovered: float = float("nan")) -> npt.NDArray[Any]: ...
  ```

  Weights floored at 1e-6·max (no covered voxel divides by zero); `add` refuses
  another `grid_id` or a prediction not of the patch's shape. `GridPatchDataset`
  items gain `meta["cover"] = {"file": i, "index": k, "of": n}` (completion counted
  from it, not by matching normalised paths). In `medh5.torch`:

  ```python
  class GridStitcher:
      def __init__(self, dataset: GridPatchDataset, *, channels: int | None = None,
                   mode: str = "gaussian", sigma_scale: float = 0.125,
                   dtype: npt.DTypeLike = np.float32) -> None: ...
      def add(self, predictions: Any, meta: Sequence[Mapping[str, Any]]) -> list[tuple[str, np.ndarray]]: ...
      pending: dict[str, int]   # path -> patches still missing
      def finish(self, *, partial: bool = False) -> list[tuple[str, np.ndarray]]: ...
  ```

  `add` returns the files a batch completes, reading `meta` as
  `medh5.torch.collate` batches it; `finish` raises `MEDH5ValidationError` naming
  each incomplete file (one dense accumulator per open volume: refuse, never a
  silent partial volume).
- **Tests.** `tests/tools/test_sampling.py::TestPatchAggregator`:
  `test_constant_blend_reassembles_the_volume_exactly`,
  `test_gaussian_blend_of_a_constant_is_that_constant`, `test_padding_is_cut_off`,
  `test_weights_are_never_zero`, `test_another_grid_is_refused`;
  `tests/integrations/test_loaders.py`: `test_a_cover_stitches_back_into_each_volume`
  (an identity model, two workers, shuffled: equals the image),
  `test_finish_names_a_file_missing_patches`.
- **Docs.** `docs/reference/torch.md` (inference), `docs/guides/segmentation.md`
  (sliding-window inference), `docs/guides/performance.md` (accumulator memory).
- **Changelog.** Added, area `torch`.

### 14.5 Per-class ignore masks (PR 4)

- **State at 2.0.2.** `_slot` (`medh5/torch/clinical.py`) builds one `ignore` per
  slot: the padding joined with `sample.ignore_region(ann_id, roi)` of every
  label annotation that labels or examined any of the slot's classes;
  `_SLOTTED = ("images", "valid", "ignore", "label")`; `TestSlotSupervision`
  asserts the union. When one slot draws classes from different annotations
  (rater A labels liver with an ignore band, rater B tumour without), the union
  masks tumour voxels inside A's band, and that supervision is lost.
- **Change.** `ClinicalTaskDataset(..., class_ignore: bool = False)`: an opt-in
  key `class_ignore[slot]` of shape `(K, *patch)`, bool; channel k is the padding
  joined with the ignore regions of the annotations that label or examined class
  k (1.0 §7.7: a region belongs to its annotation); two annotations covering one
  class are joined (conservative). Collected from the same `ignore_region` reads
  (no extra I/O); a placeholder slot is all True; channels follow the slot's class
  order; `"class_ignore"` joins `_SLOTTED`. `ignore` is unchanged and equals the OR
  over channels.
- **Tests.** In `TestSlotSupervision`:
  `test_S7_7_a_class_is_masked_only_by_the_annotations_of_that_class`,
  `test_class_ignore_includes_the_padding`, `test_an_empty_slot_ignores_every_class`,
  `test_the_union_of_class_ignore_is_ignore`, `test_collate_stacks_class_ignore`.
- **Docs.** `docs/reference/torch.md`, `docs/guides/clinical-training.md` (masking
  a loss per class), `docs/guides/partial-labels.md`.
- **Changelog.** Added, area `torch`.

### 14.6 The vocabulary's file and digest (PR 4, D34)

- **State at 2.0.2.** `ConceptVocabulary.save` (`medh5/torch/clinical.py`) writes
  in place with `Path.write_text`; `ConceptVocabulary.digest` is ASCII-escaped
  JSON, not §5.1's canonical form.
- **Change.** `save` uses `_core.write_atomic` (off W21-05's allow-list). The
  digest stays; its docstring says it is not §5.1 and why it cannot change under
  `medh5.cache/1` (D34).
- **Changelog.** Fixed (the in-place write), area `torch`.

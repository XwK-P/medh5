# Workstream `geom`: geometry, transforms and storage (F19; W2 geometry/transforms; W6 chunking/cubic; W9 L3 test)

Worktree: `SCR/wt/geom` (branch `w202/geom`). It is created from the integration branch, so it already contains the merged wave-1 work.

## Files you own
- `crates/medh5/src/geometry/**`: `grid.rs`, `affine.rs`, `linalg.rs`, `multiscale.rs`, `mod.rs`.
- `crates/medh5/src/transforms/**`: `model.rs`, `apply.rs`, `encode.rs`, `resolve.rs`, `mod.rs`.
- `crates/medh5/src/storage/{chunking,codecs,recompress,mod}.rs`.
- `crates/medh5-python/src/{geometry,transforms,storage}.rs`.
- `medh5/{geometry,transforms,storage}.py`.
- `docs/reference/storage.md`.
- Tests: `tests/format/{test_geometry,test_transforms,test_storage}.py`.

## Items (quoted from the audit; verify each against the code first)
1. `Transform::chain`/`ChainTransform` must check that frames connect (`transforms/model.rs:295-338`). Today a wrong-order chain silently maps points.
2. **Pyramid origin check:** measure against voxel size, not |origin| (`geometry/multiscale.rs:179-187`). Today it misses a missing half-voxel shift whenever |origin| is more than about 500× the spacing (whole-body CT, µm stages).
3. **F19:** `jacobian_determinant(roi)`/`folding_fraction(roi)` must resolve slices like `read_field` (`model.rs:771-778, 1108-1131`).
   - Today strided ROIs are silently wrong: `np.s_[:-1]` raises E109 for shape (-1, 8, 8), and `::2` reports 100% folding on a valid field.
   - **Revision 1's fix ("multiply the spacing by the step") was wrong:** det(I + S·M) = det(I + M·S), so index-space vectors stay exactly as wrong.
   - **For 2.0.2:**
     - refuse an ROI with more than S axes, and pad a shorter one with full slices;
     - resolve each slice with `Slice::resolve(n)`;
     - refuse step ≠ 1, with the wording of `annotations/read.rs:527`;
     - give a clear error for an axis with fewer than 2 voxels, or an empty ROI;
     - pass the resolved (start, stop) pairs to `cropped_grid`.
   - Fix the `Transform.jacobian_determinant` docstring: what `roi` accepts.
   - Reproductions and the algebra: `SCR/reports/indep-transforms-nodes.md`, `SCR/agents/indep/transforms-nodes/`.
   - Add a relationship test: on a linear field, the analytic Jacobian equals the computed one for any valid non-strided ROI.
4. Python `box_corners`/`apply_affine_to_box` must check shape `(S, 2)`: a `(2, 3)` box is silently misread. `slices_to_box` should refuse open slices (`crates/medh5-python/src/geometry.rs`).
5. **Storage:**
   - honour `chunk_hint` for stacked encodings (`storage/chunking.rs:172-178`);
   - `recompress --profile portable` should rewrite small plugin-filtered datasets;
   - `describe_filters` should name zstd (32015) (`storage/codecs.rs:192`).
6. **Non-finite values:**
   - refuse non-finite box coordinates in `box_to_slices` (`geometry/affine.rs:196`), with `!(lo <= hi)` plus `is_finite`;
   - refuse non-finite affine matrix entries (`transforms/encode.rs:22-45`);
   - refuse non-finite pyramid factors (`multiscale.rs:65`).
7. `recompress_paths` stops at the first failure and loses the results of files already rewritten (`storage/recompress.rs:411-413`). Return per-path outcomes, as the CLI does, or continue and report every failure, with the smallest API impact. If the valid agent already changed `fix_paths` (see `crates/medh5/src/integrity/repair.rs` in your worktree), use the same pattern.
8. `storage.dataset_kwargs` changed meaning from 1.x (h5py keywords → `{"chunks","codec"}`) without a CHANGELOG note, and it ignores a custom `CodecProfile` (`medh5/storage.py:156-174`). Honour the profile, and give a CHANGELOG draft noting the 2.0 change.
9. **W6:**
   - Cubic displacement fields re-prefilter on every evaluation (`transforms/apply.rs:354-390`). Cache the prefiltered coefficients per field, with identical results.
   - The chunk optimiser: clamp the starting chunk to the 4 MiB ceiling (today patch 96 gives 8 MiB chunks and 160 gives 32 MiB), and read L3 per sharing CPU (`storage/chunking.rs:28-31, 104-105`). Or correct the docs, which promise 512 KiB–4 MiB and per-core L3. Prefer the code fix if it matches the documented promise. New chunk shapes change the bytes of *newly written* files, but no `content_id` changes, since digests cover decompressed content. Confirm that, and report it as a behaviour change.
10. **W9:** half of the L3-probe test patches Python functions the Rust probe never calls (`tests/format/test_storage.py`; find it). Make the test exercise the real probe, or delete the dead half.

## Notes
- Tests to run:
  - `wcargo geom test -p medh5 --lib geometry:: transforms:: storage::`;
  - `wext geom`, then `wpy geom -m pytest tests/format/test_geometry.py tests/format/test_transforms.py tests/format/test_storage.py -q`.

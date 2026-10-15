<!-- area: converters -->
### Added

- **`from_nnunetv2(include_test=True)` imports an nnU-Net dataset's test
  cases.** `imagesTs` cases are imported with labels from `labelsTs` where
  present, and each case records its folder as a split claim in set
  `nnunetv2:<dataset>` (`--include-test`).
- **`to_nnunetv2(split=SET_ID)` writes a split's test cases to `imagesTs`.**
  A case claiming `test` in that set goes to `imagesTs` and `labelsTs`, one
  claiming `train` or `val` to `imagesTr`; an unplaced or held-out case is
  refused (`--split`). The stray-file check now covers `labelsTs`.
- **`to_nnunetv2(replace=True)` replaces an existing dataset.** The new
  dataset is written over it in place, file by file, each file whole, with
  `dataset.json` last; the NIfTI files of the case folders that the export
  no longer writes are removed after it, and every other file is left where
  it is (`--replace`). The report lists the files removed and left. A crash
  part way leaves each file whole but the dataset a mix of the old and the
  new until the export is run again.

### Behaviour changes

- **An nnU-Net import that leaves test cases out warns.** `from_nnunetv2`
  imports `imagesTr` only, as before, and the `imagesTs` cases it leaves out
  are now a warning that counts them, so `medh5 convert from-nnunet` exits 1
  on a dataset with an `imagesTs`. Pass `include_test=True`
  (`--include-test`) to import them.

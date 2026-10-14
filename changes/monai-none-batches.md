<!-- area: torch -->
### Behaviour changes

- **MONAI can batch medh5 items.** `meta["medh5"]` carried `None` for a
  sample's absent `frame_uid`, `value_units`, `modality`, `timepoint` or
  `subject_id`, and MONAI's `list_data_collate` refuses a `None`, so no sample
  without a frame of reference — every NIfTI and nnU-Net import — could be
  batched. The MONAI metadata now spells an absent value `"none"`, MONAI's own
  `TraceKeys.NONE`, and `from_metatensor` reads it back as `None`: code testing
  `meta["medh5"]["frame_uid"] is None` now sees `"none"`. `meta["medh5"]` also
  names the collection member it came from, as `sample_key`.

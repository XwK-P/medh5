<!-- area: writer -->
### Added

- **An image or a grid can be removed, and an image's attributes edited,
  without rewriting the sample.** `remove_image` and `remove_grid` join
  `remove_annotation` and `remove_transform`; removing an image drops its
  `acquisition` entry and its `series_uids` mentions too. `set_image_attrs`
  changes `modality`, `value_type`, `value_units`, `channel_names`, the
  rescale, the window, `valid_mask` or `prov` in place, `None` removing one,
  where correcting a modality used to mean writing the pixels again. Each is
  checked as `add_image` checks it, and the commit refuses what still names
  a removed object (E101, E503) or leaves no image (E201) or grid (E111).

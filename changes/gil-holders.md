<!-- area: python-api -->
### Performance

- **More calls let other Python threads run while they work.** A large array
  argument (1 MiB or more) is copied into the engine's layout with the GIL
  released, and so are `curation.agreement.dice` and `iou`, which now count on
  the masks themselves, broadcasting a plane against a volume without copying
  it; `encode_contours` packs with the GIL released; `encode_boxes`,
  `encode_obb`, `encode_keypoints`, `encode_points`, the writer's `add_boxes`,
  `add_obb`, `add_keypoints` and `add_points`, an annotation's `to_world` and
  `to_index`, and `transforms.refuse_outside` release it from 4096 rows, and
  read class ids given as a NumPy integer array as an array rather than one
  Python int at a time. Letting go of the last handle to a dataset just
  written flushes its chunk cache with the GIL released too: 58 ms that held
  it for a 192 × 256 × 256 image. A `dice` of two 300 × 512 × 512 masks took
  0.88 s, every millisecond of it holding the GIL; it now takes 0.06 s and
  holds it for 1 ms. `normalize_masks` no longer copies a C-contiguous boolean
  mask: it returns the array it was given. Do not change an array while a call
  reads it.

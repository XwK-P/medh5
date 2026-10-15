<!-- area: python-api -->
### Fixed

- **`Image.read` takes an Ellipsis.** `read((..., slice(0, 2)))` raised
  `TypeError` where the dataset's own `[..., 0:2]` worked; one `...` now
  stands for the whole axes the other items leave, and two raise
  `IndexError`, as in NumPy.
- **Two `Instance`s compare by value.** `a == b` and `a in [b]` raised
  "truth value … ambiguous" on the box array; equality now compares every
  field, the box and mask by value. An instance is still unhashable.

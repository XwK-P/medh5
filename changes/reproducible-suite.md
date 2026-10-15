<!-- area: validation -->
### Added

- **The conformance suite builds to the same bytes every time.** Two
  publishes a few minutes apart differed in every case file, with the same
  `content_id`s: each sample recorded when it was built, and every group
  HDF5's creation and modification times. The corpus now pins `created`, and
  the engine creates every group, the root included, without timestamps, as
  it already wrote datasets. A mirror can check a published suite by
  publishing it again and comparing `SHA256SUMS`. Any file the engine writes
  differs from another of the same content only in `created`; no digest or
  `content_id` changes.

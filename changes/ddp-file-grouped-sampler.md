<!-- area: torch -->
### Added

- **`FileGroupedSampler` shards an epoch across DDP ranks.** New
  `num_replicas`, `rank` and `drop_last` (defaulting to `torch.distributed`'s
  world size and rank) give each rank one contiguous share of the epoch's file
  order, the same length on every rank and every epoch, so a file is split
  between ranks only where a share ends; with one replica the order is
  2.0.2's. `ClinicalTaskDataset` gains `file_groups()`, its rows grouped by
  subject, so the sampler takes a clinical task too.

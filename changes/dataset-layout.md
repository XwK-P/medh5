<!-- area: python-api -->
### Deprecated

- **`medh5.storage.dataset_kwargs` is now `dataset_layout`.** Since 2.0 it has
  returned the writer's layout (`{"chunks": ..., "codec": ...}`), not `h5py`
  keywords, so the old name misled. The old name is the same function and
  warns; 3.0 removes it. For `h5py` keywords, pass
  `resolve_profile(p).codec(role).kwargs()` to `create_dataset`.

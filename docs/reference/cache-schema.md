# Cache manifest schema

A feature cache records what it was derived from in a dependency manifest
(`medh5.cache/1`), stored as canonical JSON in the cache file's `manifest`
dataset. This schema validates it. It is part of the separately versioned
[task and cache contract](../spec/task-cache-1.md), not of the format.

```python
import json
import medh5.cache

schema = json.loads(medh5.cache.schema_text())
schema["properties"]["level"]
```

A manifest that fails this schema is `T401`; the checksums and source pins it
records are checked by:

```bash
medh5 cache validate reports.medh5cache
```

The tables below are generated from the schema itself at build time.

<!--@schema medh5-cache-1-->

## Related

- [Task and cache contract §7](../spec/task-cache-1.md#7-feature-caches) — caches, normatively.
- [Task manifest schema](task-schema.md) — the other schema of the contract.
- [Train on clinical tasks](../guides/clinical-training.md) — building and validating a cache from Python.

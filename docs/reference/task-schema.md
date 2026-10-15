# Task manifest schema

A task manifest (`medh5.task/1`) says what a prediction task may read at a
cutoff, from which pinned sample versions, for which subjects in which
partition, and what the target is. This schema validates it. It is part of the
separately versioned [task and cache contract](../spec/task-cache-1.md), not of
the format.

```python
import json
import medh5.task

schema = json.loads(medh5.task.schema_text())
schema["required"]
```

JSON Schema checks shapes and vocabularies; the engine checks identity, splits,
pins and fingerprints. A manifest that is not JSON, or fails this schema, is
`T101`:

```bash
medh5 task validate manifest.json
```

The tables below are generated from the schema itself at build time.

<!--@schema medh5-task-1-->

## Related

- [Task and cache contract §3](../spec/task-cache-1.md#3-the-task-manifest) — the manifest, normatively.
- [Cache manifest schema](cache-schema.md) — the other schema of the contract.
- [Train on clinical tasks](../guides/clinical-training.md) — building and using a manifest from Python.

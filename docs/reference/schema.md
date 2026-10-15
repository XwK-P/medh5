# Sample document schema

Every `.medh5` file carries a JSON document at `/meta` describing the subject,
the timepoints, the label set, the provenance and the curation records. This is
its schema.

The file's *arrays* — images, masks, boxes, transforms — are HDF5 datasets and
are not described here; see [Storage](storage.md) for the layout and
[the specification](../spec/medh5-1.0.md) for what each object means.

```python
import h5py, json

with h5py.File("case_0001.medh5") as f:
    doc = json.loads(f["meta"][()])
    doc["identity"]["subject_id"]
```

Validation against this schema is `E005`, and is what `--level structural`
checks. The format engine embeds the schema and its validator, so every install
checks it — and every `commit()` refuses a document that fails it:

```bash
medh5 validate case_0001.medh5 --level structural
```

The schema itself, as the engine embeds it:

```python
import json
import medh5.document

schema = json.loads(medh5.document.schema_text())
schema["$id"]
```

The tables below are generated from the schema itself at build time.

<!--@schema medh5-sample-1.0-->

## Related

- [Diagnostic codes](diagnostic-codes.md) — `E004` and `E005` are the failures this document can cause.
- [Curation records](curation.md) — the Python API for provenance, quality and identity.
- [Specification §2.4](../spec/medh5-1.0.md) — the normative description.

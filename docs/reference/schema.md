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
checks. It needs the `schema` extra:

```bash
pip install "medh5[schema]"
medh5 validate case_0001.medh5 --level structural
```

The tables below are generated from the schema itself at build time.

<!--@schema-->

## Related

- [Diagnostic codes](diagnostic-codes.md) — `E004` and `E005` are the failures this document can cause.
- [Curation records](curation.md) — the Python API for provenance, quality and identity.
- [Specification §2.4](../spec/medh5-1.0.md) — the normative description.

# Clinical schema

A sample that declares the `clinical` profile (format 1.1) keeps its descriptor
— the subject's clock and the profile's declaration — as canonical JSON in
`clinical/meta`. This schema validates that descriptor, and, as
`#/$defs/records`, the logical-record form of the profile's tables (events,
documents, links) that `medh5 clinical export` writes and `medh5 clinical
augment` reads.

```python
import json
import medh5.clinical

schema = json.loads(medh5.clinical.schema_text())
sorted(schema["$defs"])
```

JSON Schema checks shapes and vocabularies. It cannot check column shapes,
references, revision chains or the order of time bounds; the validator does,
with the `E8xx` codes, and a descriptor that fails this schema is `E802`.

The tables below are generated from the schema itself at build time.

<!--@schema medh5-clinical-1-->

## Related

- [Specification 1.1 §3](../spec/medh5-1.1.md#3-layout-and-the-clinical-descriptor) — the descriptor, normatively.
- [Specification 1.1 Appendix B](../spec/medh5-1.1.md#appendix-b--schemas) — where each schema is used.
- [`medh5 clinical export`](cli.md#medh5-clinical-export) and [`medh5 clinical augment`](cli.md#medh5-clinical-augment) — the commands that write and read the records.
- [Diagnostic codes](diagnostic-codes.md) — the `E8xx` checks this schema cannot express.

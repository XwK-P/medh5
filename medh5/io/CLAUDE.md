## 0.x

Deleted at 1.0. `io/_legacy_reader.py` is a read-only reader of the old layout
so `medh5 migrate` works; there is no 0.x writer, deliberately. 0.x files are
converted once, and the migration reports every non-mechanical decision.

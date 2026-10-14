<!-- area: task-cache -->
### Behaviour changes

- **`TaskManifest.save` and `load` are the engine's.** Python wrote a
  manifest with its non-ASCII text as is, the engine with it escaped, so one
  document saved as two files; Python now writes the engine's bytes, the
  same as `medh5 task reconcile`. `TaskManifest.load` refuses a manifest
  holding `NaN` or an infinity (T101), as the engine did: JSON has neither.
  `TaskManifest.save` refuses a path that is one of the manifest's sources,
  by any name or link, as `medh5 task reconcile --out` already did, where it
  replaced the sample with JSON. `TaskManifest.reconcile` drops the declared
  fingerprint, as the command line does: the manifest changed.

<!-- area: files -->
### Behaviour changes

- **An amendment refuses to replace a file that changed after it read it.**
  Two amendments, or an amendment and a `repack`, both renamed over the file,
  and the first one's work was lost without an error. An amendment, and a
  `repack` or `recompress` in place, now refuses to commit when its source was
  replaced or edited in place since it was read (`MEDH5FileError`, "changed
  after this amendment read it; nothing was written"). This detects a lost
  update; it does not lock one out (§14.4).

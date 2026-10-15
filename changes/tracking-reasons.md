<!-- area: curation -->
### Added

- **Tracking says why each absence is, or is not, a finding.**
  `Tracking.reason_at(instance_id, timepoint)` and `reasons(instance_id)`
  give one of `medh5.curation.REASONS`: `located` or `class_examined` where
  the object is `resolved`, `outside_fov`, `ignored`, `unrelated`,
  `no_identity` or `not_examined` where it is `unexamined`. `is_new` and
  `is_resolved` take `located=True` to count only located absences; the
  JSON of `Sample.tracks()` and `medh5 track --json` gives each track its
  `reasons`, and `medh5 track` prints a `reasons` column. In Rust,
  `curation::tracking::locate_tracks` returns a `LocatedTracking`.

### Deprecated

- **The class-level `Tracking::{state_at, states, is_new, is_resolved}`
  are deprecated in the Rust crate.** Under them an object outside a
  visit's field of view reads as resolved; use `locate_tracks`. 3.0 removes
  them.

### Behaviour changes

- **An object outside a follow-up's field of view is no longer `resolved`
  there.** Tracking maps the object's boxes from the visits that saw it into
  the follow-up's grid, through the shared frame or affine transforms (§7.4,
  as corrected in Appendix C.1): outside the grid or its images' acquired
  voxels it is `unexamined` (`outside_fov`), and so is it inside an ignore
  region (`ignored`). An ignore region away from the object no longer makes
  the absence `unexamined`, and a baseline that never imaged the place finds
  no new lesion there. Where nothing relates the visits, the class-level rule
  stands (`class_examined`).

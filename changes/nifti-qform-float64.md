<!-- area: converters -->
### Behaviour changes

- **A qform-only NIfTI imports to the same grid on every platform.** nibabel
  builds the qform in `np.longdouble`, 80-bit on x86-64 and 64-bit elsewhere,
  so direction and spacing — and the sample's `content_id` — differed in the
  last bit between platforms. `read_nifti` now computes the qform in float64
  from the quaternion parameters, as NIfTI-1 defines it: imports on x86-64
  can change in the last bit, and are the same everywhere.

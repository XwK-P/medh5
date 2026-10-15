<!-- area: converters -->
### Added

- **`add_segmentation(labelmap=...)` writes a segmentation from one integer
  volume.** Each nonzero value is the class of that id, or the classes
  `label_values=` gives it (a value may be in several, as an nnU-Net region's
  component is), and `ignore_value=` marks the ignore region. It writes what
  the masks the volume implies write, byte for byte, without building them;
  the volume is borrowed from NumPy, not copied. In Rust,
  `SampleWriter::add_segmentation_labelmap` and `LabelmapSource`.

### Performance

- **The nnU-Net import no longer builds a mask per class.** A case's label
  volume is written through `labelmap=`: at 104 classes of 384³, the masks
  cost 11.2 GB of peak memory above the image write, and the label volume
  nothing.

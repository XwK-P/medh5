<!-- area: files -->
### Fixed

- **An amendment keeps the file's group.** A replaced file belonged to the
  amending user's primary group, so a `0o640` file of a shared project group
  became readable by the wrong group and unreadable by the right one. The
  replacement now takes the file's group (and, run as root, its owner) before
  its mode. Where you are not in the file's group, the replacement grants its
  own group only what it grants everyone, which never widens access. Every
  file the engine writes — samples, manifests, reports — does the same. Access
  control lists and extended attributes are not carried.

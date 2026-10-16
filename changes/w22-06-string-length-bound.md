<!-- area: validation -->
### Security

- **A string stated longer than its file is refused before it is read.** A
  variable-length string's stored length is the file's to state, up to
  4 GiB, and a small crafted file could make a read set aside that much
  memory before failing. A contiguous string dataset — `/meta` among them —
  is now checked first, and a length longer than the file is reported as
  unreadable (`E001`) at once.

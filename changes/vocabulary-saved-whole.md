<!-- area: torch -->
### Fixed

- **`ConceptVocabulary.save` writes the file whole.** It wrote through the
  file in place, so a reader could see half a vocabulary and a failed write
  lost the previous one; it now writes a temporary beside the file and renames
  it over. `ConceptVocabulary.digest` is unchanged, and its documentation now
  says why it is not §5.1's canonical JSON: feature caches record it.

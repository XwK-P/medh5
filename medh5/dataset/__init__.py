"""Dataset-level utilities: directory indexing, filtering, splitting.

The :class:`Dataset` collection scans a directory of ``.medh5`` files,
caches lightweight metadata (via :func:`MEDH5File.read_meta`, no array
reads), and supports filtering and JSON manifest persistence.

:func:`make_splits` produces reproducible stratified / grouped /
k-fold splits backed only by stdlib + numpy.
"""

from medh5.dataset.index import Dataset, DatasetRecord
from medh5.dataset.split import make_splits

__all__ = ["Dataset", "DatasetRecord", "make_splits"]

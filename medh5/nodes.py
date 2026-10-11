"""Views of stored HDF5 objects: what 1.x handed out as ``h5py`` objects.

``Sample.root``, ``Image.dataset``, ``Annotation.group``, ``Transform.group``,
``SampleWriter.handle`` and the writer's ``add_*`` methods return these.  They
read and write through the format engine, so the package needs no ``h5py``:

* :class:`Group` --- members by name or ``a/b`` path, ``keys()``, ``in``,
  ``del``, and ``attrs``;
* :class:`Dataset` --- ``shape``, ``dtype``, ``chunks``, ``filters``,
  ``storage_size``, ``attrs`` and ``ds[...]`` / ``numpy.asarray(ds)``;
* :class:`Attrs` --- a mutable mapping over an object's attributes, decoded
  and encoded as spec §2.5 fixes.

A view is valid while the sample or writer it came from is open.
"""

from __future__ import annotations

from medh5._core import Attrs, Dataset, Group

__all__ = ["Attrs", "Dataset", "Group"]

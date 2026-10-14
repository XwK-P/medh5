"""Views of stored HDF5 objects, ``medh5.nodes`` (spec §2.5).

``Sample.root``, ``Image.dataset``, ``Annotation.group`` and
``SampleWriter.handle`` hand these out where 1.x handed out ``h5py`` objects:
a group's members by name or path, a dataset's metadata and values, and
attributes decoded and encoded as §2.5 fixes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.nodes import Attrs, Dataset, Group
from tests.kits import Flat


@pytest.fixture
def written(tmp_path: Path) -> Path:
    path = tmp_path / "nodes.medh5"
    with Flat.open_writer(path) as w:
        w.add_segmentation("organs", grid="g", masks={1: Flat.mask()})
        w.handle.attrs["note"] = "kept"
        w.handle.attrs["scale"] = np.array([1.5, 2.0])
    return path


class TestGroup:
    def test_S2_5_members_by_name_or_path(self, written: Path):
        with medh5.open(written) as sample:
            root = sample.root
            assert isinstance(root, Group)
            assert "images" in root and "images/CT" in root
            assert "absent" not in root and "images/absent" not in root
            assert {"images", "annotations"} <= set(root.keys())
            assert isinstance(root["images"], Group)
            assert isinstance(root["images/CT"], Dataset)
            assert root["images"]["CT"].name == root["images/CT"].name
            assert root.get("absent") is None
            assert len(root["images"]) == len(list(root["images"])) == 1
            assert sample.annotations["organs"].group.name.endswith("organs")
            with pytest.raises(KeyError):
                root["absent"]

    def test_the_classes_are_found_where_their_module_says(self):
        for cls in (Attrs, Dataset, Group):
            assert cls.__module__ == "medh5.nodes"


class TestDataset:
    def test_S2_5_shape_dtype_and_values(self, written: Path):
        image = Flat.image()
        with medh5.open(written) as sample:
            ds = sample.root["images/CT"]
            assert sample.images["CT"].dataset.name == ds.name
            assert ds.shape == image.shape and ds.ndim == 3
            assert ds.dtype == image.dtype and len(ds) == image.shape[0]
            assert ds.size == image.size and ds.nbytes == image.nbytes
            assert np.array_equal(ds[0:2, :, 1], image[0:2, :, 1])
            assert np.array_equal(ds.read(), image)
            assert np.array_equal(np.asarray(ds), image)

    def test_an_array_of_a_dataset_is_always_a_copy(self, written: Path):
        """A read makes a new array: ``copy=False`` cannot be honoured, so it
        is refused, as ``h5py`` refuses it; a ``dtype`` converts once."""
        image = Flat.image()
        with medh5.open(written) as sample:
            ds = sample.root["images/CT"]
            with pytest.raises(ValueError, match="copy"):
                ds.__array__(copy=False)
            assert np.array_equal(ds.__array__(copy=True), image)
            converted = np.asarray(ds, dtype=np.float32)
            assert converted.dtype == np.float32
            assert np.array_equal(converted, image.astype(np.float32))
            first, second = np.asarray(ds), np.asarray(ds)
            first[0, 0, 0] += 1
            assert second[0, 0, 0] == image[0, 0, 0], "each read its own array"

    def test_S2_5_string_datasets_read_as_object_arrays(self, tmp_path: Path):
        """A string dataset reads as an object array of ``str`` in its stored
        shape, as a numeric one reads as its dtype, and a key means what it
        means to NumPy.  ``read()`` and ``numpy.asarray`` refused one,
        ``ds[...]`` raised, ``ds[0]`` of a 2-D dataset was one element, and a
        scalar ignored its key."""
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "strings.medh5"
        with Flat.open_writer(path) as w:
            w.add_classification(
                "staging", {"lesion": 1.0}, schemes=["Lung-RADS"], scheme_values=["4A"]
            )
        table = np.array([["a", "b", "c"], ["d", "e", "ü"]], dtype=object)
        with h5py.File(path, "a") as f:
            f.create_dataset("x_ext/table", data=table, dtype=h5py.string_dtype())
        with medh5.open(path) as sample:
            staging = sample.annotations["staging"]
            schemes = staging.group["schemes"]
            assert schemes.dtype == object and schemes.shape == (1,)
            for read in (
                schemes.read(),
                np.asarray(schemes),
                schemes[...],
                schemes[:,],
            ):
                assert isinstance(read, np.ndarray) and read.dtype == object
                assert read.tolist() == ["Lung-RADS"] == list(staging.schemes)
            assert schemes[0] == schemes[-1] == "Lung-RADS"
            with pytest.raises(IndexError):
                schemes[1]

            meta = sample.root["meta"]
            assert meta.shape == () and isinstance(meta[()], str)
            whole = meta.read()
            assert whole.shape == () == meta[...].shape and whole.dtype == object
            assert json.loads(whole.item()) == json.loads(meta[()])
            assert json.loads(np.asarray(meta).item())["identity"]["sample_id"]
            with pytest.raises(IndexError):
                meta[0]

            stored = sample.root["x_ext/table"]
            assert stored.shape == (2, 3)
            assert np.array_equal(stored.read(), table)
            assert stored[0].tolist() == ["a", "b", "c"]
            assert stored[1, 2] == "ü" and stored[:, 1].tolist() == ["b", "e"]
            assert np.asarray(stored, dtype=str).tolist() == table.tolist()
            with pytest.raises(IndexError):
                stored[2]
            # The numeric path is untouched.
            assert np.array_equal(sample.root["images/CT"][...], Flat.image())


class TestAttrs:
    def test_S2_5_attributes_read_as_a_mapping(self, written: Path):
        with medh5.open(written) as sample:
            attrs = sample.root.attrs
            assert isinstance(attrs, Attrs)
            assert attrs["note"] == "kept"
            assert np.array_equal(attrs["scale"], [1.5, 2.0])
            assert "note" in attrs and "absent" not in attrs
            assert attrs.get("absent", 7) == 7
            assert {"note", "scale"} <= set(attrs.keys())
            assert attrs.to_dict()["note"] == "kept"
            assert sample.root["images/CT"].attrs["modality"] == "CT"

    def test_S2_5_a_writer_handle_writes_and_deletes(self, tmp_path: Path):
        path = tmp_path / "attrs.medh5"
        with Flat.open_writer(path) as w:
            w.handle.attrs["gone"] = 1
            assert w.handle.attrs["gone"] == 1
            del w.handle.attrs["gone"]
            assert "gone" not in w.handle.attrs
            with pytest.raises(KeyError):
                del w.handle.attrs["gone"]
        with medh5.open(path) as sample:
            assert "gone" not in sample.root.attrs

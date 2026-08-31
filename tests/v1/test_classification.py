"""Classification annotations, including change labels (spec §9)."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations.classification import (
    Assertion,
    check_scope,
    encode_classification,
)
from medh5.errors import MEDH5ValidationError
from tests.v1.conftest import SHAPE


def longitudinal(path, label_set, **kwargs):
    """Two timepoints with per-visit staging and a change label."""
    with medh5.create(path, codec="portable") as w:
        w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
        w.add_timepoint("tp1", label="follow_up", days_from_baseline=92)
        w.label_set(label_set)
        for tp, frame in (("tp0", "f0"), ("tp1", "f1")):
            w.add_grid(
                f"ct_{tp}",
                shape=SHAPE,
                spacing=(1.0, 1.0, 1.0),
                timepoint=tp,
                frame_uid=frame,
            )
            w.add_image(
                f"CT_{tp}",
                np.zeros(SHAPE, dtype=np.int16),
                grid=f"ct_{tp}",
                modality="CT",
            )
        w.add_classification(
            "staging",
            {"lesion": 1.0, "vessel": 1.0},
            scope="timepoint",
            scope_ids=[0, 1],
            schemes=["Lung-RADS", "Lung-RADS"],
            scheme_values=["4A", "4B"],
            **kwargs,
        )
        w.add_classification(
            "response",
            {"liver": 1.0},
            scope="sample",
            multilabel=False,
            timepoints=["tp0", "tp1"],
        )
    return path


class TestRoundTrip:
    def test_S9_labels_and_positives(self, tmp_path, label_set):
        path = longitudinal(tmp_path / "cls.medh5", label_set)
        with medh5.open(path) as sample:
            staging = sample.annotations["staging"]
            assert staging.kind == "classification"
            assert staging.task == "classification"
            assert staging.scope == "timepoint"
            assert staging.multilabel
            assert staging.labels == {"lesion": 1.0, "vessel": 1.0}
            assert set(staging.positives) == {"lesion", "vessel"}
            assert len(staging) == 2
            assert "cls" in sample.profiles

    def test_S9_ordinal_schemes_are_stored_verbatim(self, tmp_path, label_set):
        path = longitudinal(tmp_path / "cls.medh5", label_set)
        with medh5.open(path) as sample:
            staging = sample.annotations["staging"]
            assert staging.scheme("Lung-RADS") == "4A"
            assert staging.scheme("BI-RADS") is None
            assert staging.scheme_values == ("4A", "4B")

    def test_S9_assertions_carry_their_scope_unit(self, tmp_path, label_set):
        path = longitudinal(tmp_path / "cls.medh5", label_set)
        with medh5.open(path) as sample:
            assertions = list(sample.annotations["staging"].assertions())
            assert [a.scope_id for a in assertions] == [0, 1]
            assert all(a.is_positive for a in assertions)
            grouped = sample.annotations["staging"].by_scope_id()
            assert set(grouped) == {0, 1}
            assert "Lung-RADS" in repr(assertions[0])

    def test_S9_change_label_names_the_timepoints_compared(self, tmp_path, label_set):
        path = longitudinal(tmp_path / "cls.medh5", label_set)
        with medh5.open(path) as sample:
            response = sample.annotations["response"]
            assert response.is_change_label
            assert response.timepoints == ("tp0", "tp1")
            assert response.compared_timepoints == ("tp0", "tp1")
            assert not sample.annotations["staging"].is_change_label
            assert sample.annotations.spanning() == (response,)

    def test_summary_is_json_safe(self, tmp_path, label_set):
        import json

        path = longitudinal(tmp_path / "cls.medh5", label_set)
        with medh5.open(path) as sample:
            summary = sample.annotations["response"].summary()
            json.dumps(summary)
            assert summary["change_label"] is True


class TestThreeStateSemantics:
    def test_S9_positive_negative_unknown(self, tmp_path, label_set):
        """The distinction partial labelling depends on."""
        path = tmp_path / "states.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_classification(
                "findings",
                {"lesion": 1.0, "vessel": 0.0},
                annotated_classes=["lesion", "vessel", "spleen"],
            )
        with medh5.open(path) as sample:
            findings = sample.annotations["findings"]
            assert findings.state("lesion") == "positive"
            assert findings.state("vessel") == "negative"  # asserted absent
            assert findings.state("spleen") == "negative"  # looked for, not found
            assert findings.state("liver") == "unknown"  # nobody looked
            assert findings.value("lesion") == 1.0
            assert findings.value("liver") is None

    def test_S9_explicit_zero_differs_from_absence(self, tmp_path, label_set):
        """ "0 of 4 raters" and "not assessed" must not collapse."""
        path = tmp_path / "zero.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_classification(
                "consensus", {"lesion": 0.0}, annotated_classes=["lesion"]
            )
        with medh5.open(path) as sample:
            consensus = sample.annotations["consensus"]
            assert consensus.value("lesion") == 0.0
            assert consensus.state("lesion") == "negative"
            assert consensus.positives == ()

    def test_S5_4_hierarchical_closure(self, tmp_path, label_set):
        path = tmp_path / "hier.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="ct", modality="CT")
            w.add_classification("dx", {"lesion": 1.0}, closure="implicit")
        with medh5.open(path) as sample:
            annotation = sample.annotations["dx"]
            assert annotation.closure == "implicit"
            closed = label_set.close(annotation.asserted_class_ids.tolist(), "implicit")
            assert set(closed) == {1, 3}  # lesion implies liver


class TestValidationAtWrite:
    def test_S9_single_label_rejects_two_positives_in_one_unit(
        self, tmp_path, label_set
    ):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_classification({1: 1.0, 3: 1.0}, multilabel=False)
        assert exc.value.code == "E404"

    def test_S9_single_label_allows_one_positive_per_unit(self):
        payload = encode_classification(
            {1: 1.0, 3: 1.0}, scope="slice", scope_ids=[0, 1], multilabel=False
        )
        assert payload.datasets["values"].tolist() == [1.0, 1.0]

    def test_S9_values_must_lie_in_the_unit_interval(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_classification({1: 1.5})
        assert exc.value.code == "E404"

    def test_unknown_scope_is_refused(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            check_scope("universe")
        assert exc.value.code == "E412"
        with pytest.raises(MEDH5ValidationError):
            encode_classification({1: 1.0}, scope="universe")

    def test_column_lengths_must_agree(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            encode_classification({1: 1.0, 2: 1.0}, scope_ids=[0])
        assert exc.value.code == "E405"

    def test_S9_timepoint_scope_ids_must_be_declared_indices(self, tmp_path, label_set):
        with (
            pytest.raises(MEDH5ValidationError) as exc,
            medh5.create(tmp_path / "x.medh5") as w,
        ):
            w.add_timepoint("tp0")
            w.label_set(label_set)
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE), grid="ct", modality="CT")
            w.add_classification("s", {1: 1.0}, scope="timepoint", scope_ids=[7])
        assert exc.value.code == "E409"


class TestAssertionRecord:
    def test_positive_and_negative(self):
        assert Assertion(1, 1.0).is_positive
        assert Assertion(1, 0.0).is_negative
        assert not Assertion(1, 0.0).is_positive
        assert "Assertion(1=0.5" in repr(Assertion(1, 0.5))


class TestMultiAssertionScopes:
    """§9: `scope_ids` is per assertion, so one class can be asserted many times."""

    def _multi(self, tmp_path, label_set):
        shape = (4, 8, 8)
        path = tmp_path / "cls.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.add_classification("cls", labels={3: 0.0}, scope="slice", scope_ids=[0])
        # The writer takes a Mapping, so it cannot express two assertions for
        # one class; a conforming third-party writer can, and §9 describes it.
        with h5py.File(path, "r+") as handle:
            group = handle["annotations/cls"]
            for name, value in (
                ("class_ids", np.array([3, 3], np.uint16)),
                ("values", np.array([0.0, 1.0], np.float32)),
                ("scope_ids", np.array([0, 9], np.int64)),
            ):
                del group[name]
                group.create_dataset(name, data=value)
        return path

    def test_S9_collapsing_accessors_refuse_rather_than_contradict(
        self, tmp_path, label_set
    ):
        """`value` took the first row and `labels` the last, on the same file.

        So `state()` answered "negative" for a class `positives` listed as
        positive. Neither row is more authoritative than the other.
        """
        path = self._multi(tmp_path, label_set)
        with medh5.open(path) as sample:
            annotation = sample.annotations["cls"]
            assert [
                (a.class_id, a.value, a.scope_id) for a in annotation.assertions()
            ] == [
                (3, 0.0, 0),
                (3, 1.0, 9),
            ]
            for call in (
                lambda a: a.value(3),
                lambda a: a.state(3),
                lambda a: a.labels,
                lambda a: a.positives,
            ):
                with pytest.raises(MEDH5ValidationError) as exc:
                    call(annotation)
                assert exc.value.code == "E412"

    def test_S9_a_scope_id_selects_the_assertion(self, tmp_path, label_set):
        path = self._multi(tmp_path, label_set)
        with medh5.open(path) as sample:
            annotation = sample.annotations["cls"]
            assert annotation.value(3, scope_id=0) == 0.0
            assert annotation.value(3, scope_id=9) == 1.0
            assert annotation.state(3, scope_id=0) == "negative"
            assert annotation.state(3, scope_id=9) == "positive"

    def test_an_ordinary_single_assertion_file_is_unaffected(self, tmp_path, label_set):
        shape = (4, 8, 8)
        path = tmp_path / "plain.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.add_classification("cls", labels={3: 1.0})
        with medh5.open(path) as sample:
            annotation = sample.annotations["cls"]
            assert annotation.value(3) == 1.0
            assert annotation.state(3) == "positive"
            assert annotation.positives == ("lesion",)

    def test_S9_a_multi_assertion_file_still_summarises(self, tmp_path, label_set):
        """`summary()` must describe every file, including the ambiguous ones.

        It read the collapsing `labels`, so the refusal added for the public
        accessors also took out `Sample.summary()` and `medh5 info` -- on
        exactly the multi-assertion files §9 makes ordinary and this change set
        out to support.
        """
        path = self._multi(tmp_path, label_set)
        with medh5.open(path) as sample:
            summary = sample.annotations["cls"].summary()
            # Per scope unit, because one value per class would lose an assertion.
            assert summary["labels"] == {"lesion": {"0": 0.0, "9": 1.0}}
            assert summary["assertions"] == 2
            json.dumps(sample.summary())

    def test_a_single_assertion_file_keeps_the_flat_summary_shape(
        self, tmp_path, label_set
    ):
        shape = (4, 8, 8)
        path = tmp_path / "flat.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.add_classification("cls", labels={3: 1.0})
        with medh5.open(path) as sample:
            assert sample.annotations["cls"].summary()["labels"] == {"lesion": 1.0}

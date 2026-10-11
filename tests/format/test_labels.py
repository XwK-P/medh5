"""Label sets: the DAG, closures, digests and the registry (spec §5)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from medh5 import labels
from medh5.errors import CODES, MEDH5ValidationError
from medh5.labels import (
    IGNORE_ID,
    LabelClass,
    LabelSet,
    OntologyCode,
    Relation,
    Skeleton,
    canonical_json,
    from_keys,
)


def hierarchy() -> LabelSet:
    return LabelSet(
        "anat",
        classes=[
            LabelClass(1, "abdominal_organ", "Abdominal Organ"),
            LabelClass(2, "liver", "Liver", parents=[1]),
            LabelClass(3, "liver_segment_iv", "Liver Segment IV", parents=[2]),
            LabelClass(4, "kidney", "Kidney", parents=[1]),
            LabelClass(5, "kidney_left", "Left Kidney", parents=[4], laterality="left"),
        ],
        relations=[Relation(subject=5, predicate="part_of", object=1)],
    )


class TestLabelClass:
    def test_S5_3_reserved_ids_are_refused(self):
        for reserved in (0, IGNORE_ID):
            with pytest.raises(MEDH5ValidationError) as exc:
                LabelClass(reserved, "x", "X")
            assert exc.value.code == "E303"

    @pytest.mark.parametrize("out_of_range", [-5, 70000])
    def test_S15_2_an_id_outside_the_range_is_E306_not_E303(self, out_of_range):
        """E303 is a reserved id; any other id outside 1..65534 is a class
        entry out of id range, which §15.2 calls E306."""
        with pytest.raises(MEDH5ValidationError, match="outside") as exc:
            LabelClass(out_of_range, "vessel", "Vessel")
        assert exc.value.code == "E306"

    def test_colour_must_be_rgba(self):
        with pytest.raises(MEDH5ValidationError):
            LabelClass(1, "x", "X", color=(1, 2, 3))  # type: ignore[arg-type]
        assert LabelClass(1, "x", "X", color=(1, 2, 3, 4)).color == (1, 2, 3, 4)

    def test_round_trips_through_json(self):
        entry = LabelClass(
            7,
            "lesion",
            "Lesion",
            parents=[1],
            category="lesion",
            color=(1, 2, 3, 255),
            codes=(OntologyCode("SNOMED-CT", "1234", "Lesion"),),
            properties={"is_lesion": True},
        )
        assert LabelClass.from_json(entry.to_json()) == entry


class TestClassProperties:
    """§5.2: a class's ``properties`` is the class's own dict, as 1.x's
    dataclass field was --- an edit in place is kept, and is what the class,
    and a label set holding it, are from then on."""

    def test_S5_2_an_edit_to_properties_is_kept(self):
        import copy
        import pickle

        liver = LabelClass(2, "liver", "Liver", properties={"paired": False})
        held = liver.properties
        assert type(held) is dict
        assert liver.properties is held
        held["bodypart"] = "abdomen"
        edited = {"paired": False, "bodypart": "abdomen"}
        assert liver.properties == edited
        assert liver.to_json()["properties"] == edited
        assert liver == LabelClass.from_json(liver.to_json())
        assert liver != LabelClass(2, "liver", "Liver", properties={"paired": False})
        for copied in (
            pickle.loads(pickle.dumps(liver)),
            copy.copy(liver),
            copy.deepcopy(liver),
        ):
            assert copied.properties == edited
        assert f"properties={edited!r}" in repr(liver)

    def test_S5_2_is_lesion_reads_the_properties_as_they_stand(self):
        mass = LabelClass(7, "mass", "Mass", category="organ")
        assert not mass.is_lesion
        mass.properties["is_lesion"] = True
        assert mass.is_lesion

    def test_S5_2_a_hash_outlives_an_edit(self):
        """A dict changes in place, a hash must not: a class hashes by the
        fields an edit cannot reach, so equal classes still hash alike."""
        liver = LabelClass(2, "liver", "Liver")
        before = hash(liver)
        liver.properties["paired"] = False
        assert hash(liver) == before
        same = LabelClass(2, "liver", "Liver", properties={"paired": False})
        assert liver == same
        assert hash(liver) == hash(same)

    def test_S5_2_a_property_json_cannot_hold_is_refused_where_it_is_used(self):
        liver = LabelClass(2, "liver", "Liver")
        liver.properties["callback"] = object()
        with pytest.raises(TypeError, match="not JSON serializable"):
            liver.to_json()
        with pytest.raises(TypeError, match="not JSON serializable"):
            LabelSet("anat", classes=[liver])

    def test_S5_1_a_label_set_keeps_its_classes_and_digests_them_as_they_stand(self):
        liver = LabelClass(2, "liver", "Liver")
        anat = LabelSet("anat", classes=[liver, LabelClass(1, "organ", "Organ")])
        assert anat["liver"] is liver
        assert anat[2] is liver
        before = anat.digest()
        liver.properties["paired"] = False
        assert anat.to_json()["classes"][1]["properties"] == {"paired": False}
        assert anat.digest() != before
        assert anat.digest() == LabelSet.from_json(anat.to_json()).digest()
        assert anat.subset(["liver"])["liver"].properties == {"paired": False}


class TestLabelSet:
    def test_S5_2_hierarchy_is_a_dag_not_a_tree(self):
        """§5.2: a class may have several parents; ancestors close transitively."""
        ls = LabelSet(
            "dag",
            classes=[
                LabelClass(1, "organ", "Organ"),
                LabelClass(2, "urinary", "Urinary System"),
                LabelClass(3, "kidney", "Kidney", parents=[1, 2]),
            ],
        )
        assert set(ls.ancestors("kidney")) == {1, 2}

    def test_S5_3_cycles_are_an_error(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            LabelSet(
                "cyc",
                classes=[
                    LabelClass(1, "a", "A", parents=[2]),
                    LabelClass(2, "b", "B", parents=[1]),
                ],
            )
        assert exc.value.code == "E304"

    def test_S5_4_a_hierarchy_as_deep_as_the_id_space_loads(self):
        """A parent chain through every id overflowed the stack --- a SIGSEGV
        of whatever opened the file --- because the cycle check recursed once
        per link.  Run apart, so a regression fails this test alone."""
        script = textwrap.dedent(
            """
            from medh5.labels import MAX_CLASS_ID, LabelSet
            classes = [{"id": 1, "key": "c1", "name": "C1"}] + [
                {"id": i, "key": f"c{i}", "name": f"C{i}", "parents": [i - 1]}
                for i in range(2, MAX_CLASS_ID + 1)
            ]
            deep = LabelSet.from_json({"id": "deep", "classes": classes})
            assert len(deep.ancestors(MAX_CLASS_ID)) == MAX_CLASS_ID - 1
            classes[0]["parents"] = [MAX_CLASS_ID]
            try:
                LabelSet.from_json({"id": "deep", "classes": classes})
            except Exception as exc:
                assert exc.code == "E304", exc
            else:
                raise AssertionError("the cycle was not found")
            """
        )
        done = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
        )
        assert done.returncode == 0, done.stderr

    def test_duplicate_ids_and_keys_are_rejected(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            LabelSet("d", classes=[LabelClass(1, "a", "A"), LabelClass(1, "b", "B")])
        assert exc.value.code == "E302"
        with pytest.raises(MEDH5ValidationError) as exc:
            LabelSet("d", classes=[LabelClass(1, "a", "A"), LabelClass(2, "a", "B")])
        assert exc.value.code == "E302"

    def test_unknown_parent_is_rejected(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            LabelSet("p", classes=[LabelClass(1, "a", "A", parents=[9])])
        assert exc.value.code == "E306"

    def test_S5_4_a_relation_naming_no_class_is_E306(self):
        """A relation's ends are classes of its label set: one naming another
        id is E306, which the code table gave only to a class naming an
        unknown parent."""
        with pytest.raises(MEDH5ValidationError, match="relation") as exc:
            LabelSet(
                "r",
                classes=[LabelClass(1, "a", "A")],
                relations=[Relation(subject=1, predicate="part_of", object=9)],
            )
        assert exc.value.code == "E306"
        assert "relation" in CODES["E306"].summary

    def test_S5_4_closure_explicit_never_infers(self):
        """§5.4: `explicit` MUST NOT add ancestors; `implicit` MUST."""
        ls = hierarchy()
        assert ls.close([3], "explicit") == (3,)
        assert set(ls.close([3], "implicit")) == {1, 2, 3}

    def test_closure_must_be_known(self):
        with pytest.raises(MEDH5ValidationError):
            hierarchy().close([1], "transitive")

    def test_lookup_by_id_or_key(self):
        ls = hierarchy()
        assert ls[2] is ls["liver"]
        assert ls.get("nope") is None
        with pytest.raises(KeyError):
            ls["nope"]
        assert ls.ids_for(["liver", 4]) == (2, 4)

    def test_descendants_and_relations(self):
        ls = hierarchy()
        assert set(ls.descendants("abdominal_organ")) == {2, 3, 4, 5}
        assert ls.relations_of("kidney_left", "part_of")[0].object == 1

    def test_missing_reports_unknown_ids(self):
        assert hierarchy().missing([1, 99, 100]) == (99, 100)

    def test_S5_1_digest_ignores_carriage(self):
        """Inline and ref copies of one vocabulary must digest identically."""
        ls = hierarchy()
        assert ls.digest() == ls.as_ref("medh5:/label_set").digest()
        assert ls.digest() != LabelSet("anat", classes=list(ls)[:2]).digest()

    def test_canonical_json_is_deterministic(self):
        doc = {"b": 1, "a": [3, 2], "c": "é"}
        assert canonical_json(doc) == '{"a":[3,2],"b":1,"c":"é"}'.encode()

    def test_ref_form_requires_uri_and_no_classes(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            LabelSet("x", form="ref")
        assert exc.value.code == "E305"
        with pytest.raises(MEDH5ValidationError):
            LabelSet("x", classes=[LabelClass(1, "a", "A")], form="ref", uri="u")

    def test_subset_pulls_in_ancestors(self):
        subset = hierarchy().subset(["liver_segment_iv"])
        assert set(subset.ids) == {1, 2, 3}

    def test_round_trips_through_json(self):
        ls = LabelSet(
            "sk",
            classes=[LabelClass(1, "a", "A")],
            skeletons=[Skeleton("s", (1,), ((1, 1),))],
        )
        back = LabelSet.from_json(ls.to_json())
        assert back is not None
        assert back.skeleton("s").keypoints == (1,)
        assert LabelSet.from_json(None) is None

    def test_from_keys_mints_sequential_ids(self):
        ls = from_keys(["liver", "spleen"], id="minted")
        assert ls.ids == (1, 2)
        assert ls["spleen"].name == "Spleen"

    def test_colors_omits_unset(self):
        ls = LabelSet(
            "c",
            classes=[
                LabelClass(1, "a", "A", color=(1, 2, 3, 4)),
                LabelClass(2, "b", "B"),
            ],
        )
        assert ls.colors() == {1: (1, 2, 3, 4)}


class TestRegistry:
    def test_bundled_vocabularies_load_and_validate(self):
        assert set(labels.available()) >= {
            "amos22-organs",
            "binary-foreground",
            "brats-subregions",
        }
        for name in labels.available():
            ls = labels.load(name)
            ls.check()
            assert len(ls) >= 1

    def test_unknown_vocabulary_is_a_coded_error(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            labels.load("nope")
        assert exc.value.code == "E305"

    def test_register_and_unregister(self):
        ls = LabelSet("tmp", classes=[LabelClass(1, "a", "A")])
        labels.register("tmp-vocab", ls)
        assert labels.load("tmp-vocab") is ls
        labels.unregister("tmp-vocab")
        assert "tmp-vocab" not in labels.available()

    def test_describe_reports_digests(self):
        described = labels.describe()
        assert described["amos22-organs"]["classes"] == 15
        assert described["amos22-organs"]["sha256"]

    def test_load_file(self, tmp_path):
        path = tmp_path / "v.json"
        path.write_text(
            '{"id":"f","version":"1","classes":[{"id":1,"key":"a","name":"A"}]}',
            encoding="utf-8",
        )
        assert labels.load_file(path)["a"].name == "A"

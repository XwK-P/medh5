"""Reference MEDH5 1.0 writer: proves the spec layout is self-consistent and implementable.

Follows docs/spec/medh5-1.0.md literally, with h5py and no medh5. Emits one *longitudinal* sample
-- one subject at two timepoints, multi-modal at baseline -- exercising
core+seg+det+cls+reg+curation+training+longitudinal, then checks it: every check raises when it
fails, because Appendix C.2 of the specification publishes their results.
"""

import hashlib
import itertools
import json
import math
import os

import h5py
import hdf5plugin
import numpy as np

# The current directory, not the script's: running the prototype should not drop
# a 7 MB artefact into the source tree it is being read from.  `MEDH5_PROTOTYPE_OUT`
# lets CI put it somewhere disposable.
OUT = os.environ.get("MEDH5_PROTOTYPE_OUT", os.getcwd())
PATH = os.path.join(OUT, "case_0001.medh5")
HERE = os.path.dirname(os.path.abspath(__file__))
# `docs/examples/` -> repository root is two levels, not three.  Deriving it
# from HERE rather than hard-coding the hops keeps this working if the
# directory moves again.
ROOT = os.path.dirname(os.path.dirname(HERE))
SCHEMA = os.path.join(ROOT, "crates", "medh5", "data", "medh5-sample-1.0.schema.json")
S = h5py.string_dtype()
FRAME0 = "pseudo:1.2.826.0.1.3680043.9.7.100"   # baseline study frame
FRAME1 = "pseudo:1.2.826.0.1.3680043.9.7.101"   # follow-up: a NEW frame (spec 3.4)


def blosc(c="zstd", lvl=3, filters=None):
    kw = {"cname": c, "clevel": lvl}
    if filters is not None:
        kw["filters"] = filters
    return dict(hdf5plugin.Blosc2(**kw))


# ---------------------------------------------------------------- 13.1 digests
def digest(path, arr):
    h = hashlib.sha256()
    a = np.ascontiguousarray(arr)
    h.update(path.encode()); h.update(b"\0")
    h.update(a.dtype.str.replace(">", "<").encode()); h.update(b"\0")
    h.update(",".join(map(str, a.shape)).encode()); h.update(b"\0")
    h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


def ds(parent, name, data, path, **kw):
    d = parent.create_dataset(name, data=data, **kw)
    d.attrs["digest"] = digest(path, data)
    return d


def index_ds(parent, name, data, **kw):
    """A dataset under index/: no digest, which writers MUST NOT stamp there (13.1)."""
    return parent.create_dataset(name, data=data, **kw)


# ---------------------------------------------------------------- 13.2 content id
def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# The attributes the specification defines, per object -- all that `canonical_attrs` covers.
# At the root, exactly these three; `created`, `generator` and `content_id` itself are out.
ROOT_ATTRS = ("medh5_version", "medh5_kind", "medh5_profiles")
# For each member of these groups, its section's attributes (3.2, 4, 6-9, 10). Its own `digest`
# is out: the dataset lines carry it. Every member has a line, `{}` if it carries none of them.
SPEC_ATTRS = {
    "grids": ("shape", "axis_names", "axis_kinds", "spacing", "origin", "direction", "coord_system",
              "units", "timepoint", "frame_uid", "time_values", "time_units", "chunk_hint",
              "patch_hint"),
    "images": ("grid", "modality", "value_type", "channel_names", "rescale_slope",
               "rescale_intercept", "value_units", "window_center", "window_width", "valid_mask",
               "prov", "levels", "downsample_factors", "downsample_method", "grid_levels"),
    "annotations": ("kind", "task", "grid", "timepoints", "space", "frame_uid", "class_ids",
                    "annotated_class_ids", "closure", "ignore_id", "ignore_mask", "prov", "quality",
                    "derived_from", "scope", "scope_ids", "multilabel", "normalized", "threshold",
                    "skeleton"),
    "transforms": ("kind", "from_frame", "to_frame", "from_grid", "to_grid", "units", "invertible",
                   "inverse_id", "prov", "metrics", "field_grid", "vector_space", "interpolation",
                   "extrapolation", "cp_grid", "order", "components"),
}


def plain(v):
    """An attribute value as 13.2 serialises it: a string, a boolean, a number, nested lists."""
    if isinstance(v, np.ndarray):
        return [plain(x) for x in v] if v.ndim else plain(v[()])
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def canonical_attrs(obj, names):
    """The 5.1 canonical JSON of the spec-defined attributes *obj* carries.

    `allow_nan` is json's default: a NaN attribute hashes as `NaN`, as every implementation does.
    """
    present = {k: plain(obj.attrs[k]) for k in names if k in obj.attrs}
    return json.dumps(present, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_id(f):
    """13.2: three groups of lines, each sorted on its own, concatenated in that order.

    `index/` contributes nothing. The `meta` line carries the bare hex digest; an attribute line,
    like a dataset line, carries `sha256:` before it.
    """
    datasets = []

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split("/")[0] != "index" and "digest" in obj.attrs:
            datasets.append(f"{name}\t{obj.attrs['digest']}\n")

    f.visititems(walk)
    raw = f["meta"][()]
    meta = raw if isinstance(raw, bytes) else raw.encode("utf-8")
    attrs = [f"@\tsha256:{sha256_hex(canonical_attrs(f, ROOT_ATTRS))}\n"]
    for group, names in SPEC_ATTRS.items():
        for member in f.get(group, {}):
            obj = f[f"{group}/{member}"]
            attrs.append(f"@{group}/{member}\tsha256:{sha256_hex(canonical_attrs(obj, names))}\n")
    lines = "".join(sorted(datasets)) + f"meta\t{hashlib.sha256(meta).hexdigest()}\n" + "".join(sorted(attrs))
    return "sha256:" + sha256_hex(lines)


# ---------------------------------------------------------------- 13.3 source digest
def group_digest(group):
    """An annotation's digest: over the datasets directly in its group, sorted by name."""
    lines = sorted(f"{name}\t{group[name].attrs['digest']}\n"
                   for name in group if isinstance(group[name], h5py.Dataset))
    return "sha256:" + sha256_hex("".join(lines))


# ---------------------------------------------------------------- synthetic content
rng = np.random.default_rng(7)
SH_CT, SH_PET = (160, 160, 160), (80, 80, 80)
# Follow-up was acquired with a shorter z coverage. Nothing is resampled: tp1 simply has its own
# grid, with its own shape and its own origin (spec 3.2 / 3.7).
Z0_FU, SH_CT1 = 4, (152, 160, 160)
CROP_FU = (slice(Z0_FU, Z0_FU + SH_CT1[0]), slice(None), slice(None))
zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in SH_CT], indexing="ij")


def sphere(c, r):
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) < r * r


CLASSES = [  # id, key, name, parents, category
    (1, "abdominal_organ", "Abdominal organ", [], "region"),
    (2, "liver", "Liver", [1], "organ"),
    (3, "liver_segment_iv", "Liver segment IV", [2], "organ"),
    (4, "spleen", "Spleen", [1], "organ"),
    (5, "kidney", "Kidney", [1], "organ"),
    (6, "kidney_left", "Left kidney", [5], "organ"),
    (7, "kidney_right", "Right kidney", [5], "organ"),
    (8, "aorta", "Aorta", [1], "vessel"),
    (9, "liver_lesion", "Liver lesion", [], "lesion"),
    (10, "stage_t3", "Stage T3", [], "finding"),
    (11, "stage_n1", "Stage N1", [], "finding"),
    (12, "stage_m0", "Stage M0", [], "finding"),
    (13, "partial_response", "Partial response (RECIST 1.1)", [], "response"),
]
ORGANS = {
    2: sphere((80, 70, 70), 34), 3: sphere((72, 62, 62), 14),   # segment IV inside liver -> overlap
    4: sphere((80, 110, 108), 18), 6: sphere((100, 100, 50), 14),
    7: sphere((100, 40, 50), 14), 8: sphere((80, 80, 80), 6),   # aorta crosses liver -> overlap
}
# instance_id -> (centre, radius) per timepoint. 3 and 6 resolve; 8 is new at follow-up. (spec 7.4)
LESIONS = {
    "tp0": {1: ((74, 66, 66), 6.0), 3: ((86, 74, 60), 4.0), 6: ((70, 78, 72), 3.5)},
    "tp1": {1: ((74, 66, 66), 4.0), 8: ((78, 84, 64), 3.0)},
}


def greedy_layers(masks):
    """Colour the class overlap graph; classes in a layer are mutually exclusive (spec 7.7)."""
    ids = sorted(masks)
    adj = {i: set() for i in ids}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if np.any(masks[i] & masks[j]):
                adj[i].add(j); adj[j].add(i)
    color = {}
    for i in sorted(ids, key=lambda k: -len(adj[k])):
        used = {color[n] for n in adj[i] if n in color}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    n_layers = max(color.values()) + 1
    data = np.zeros((n_layers,) + SH_CT, np.uint16)
    for i, m in masks.items():
        data[color[i]][m] = i
    width = max(sum(1 for i in ids if color[i] == lay) for lay in range(n_layers))
    table = np.zeros((n_layers, width), np.uint16)
    for lay in range(n_layers):
        members = [i for i in ids if color[i] == lay]
        table[lay, : len(members)] = members
    return data, table, ids


def synth_ct(shift):
    vol = np.where(sphere((80, 80, 80), 74), 40 + 25 * np.sin((zz + shift) / 9), -1000).astype(np.float32)
    for i, m in ORGANS.items():
        vol[m] += 15 * i
    return np.clip(vol + rng.normal(0, 8, SH_CT), -1024, 3071).astype(np.int16)


def write_layers(grp, name, masks, grid, tp, prov, quality, crop=None):
    data, table, ids = greedy_layers(masks)
    if crop is not None:
        data = np.ascontiguousarray(data[(slice(None),) + crop])
    a = grp.create_group(name)
    a.attrs.update(kind="layers", task="segmentation", grid=grid, closure="explicit",
                   class_ids=np.asarray(ids, np.uint16),
                   annotated_class_ids=np.asarray(ids, np.uint16),
                   ignore_id=np.uint16(65535), prov=prov, quality=quality)
    chunks = (1,) + tuple(min(c, s) for c, s in zip((32, 64, 64), data.shape[1:]))
    ds(a, "data", data, f"annotations/{name}/data",
       chunks=chunks, **blosc("zstd", 3, hdf5plugin.Blosc2.BITSHUFFLE))
    ds(a, "layer_class_ids", table, f"annotations/{name}/layer_class_ids")
    return group_digest(a), data.shape[0]


def write_instances(grp, name, lesions, grid, prov, quality, crop=None):
    boxes, cls, iids, offs, shapes, buf = [], [], [], [0], [], []
    for iid, (c, r) in sorted(lesions.items()):
        m = sphere(c, r)
        if crop is not None:
            m = m[crop]
        idx = np.argwhere(m)
        lo, hi = idx.min(0), idx.max(0) + 1
        packed = np.packbits(m[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))].ravel())
        boxes.append(np.stack([lo - 0.5, hi - 0.5], 1))      # spec 8.1 edge convention
        cls.append(9); iids.append(iid); shapes.append(hi - lo)
        offs.append(offs[-1] + packed.size); buf.append(packed)
    a = grp.create_group(name)
    a.attrs.update(kind="instances", task="segmentation", grid=grid, space="index", closure="explicit",
                   class_ids=np.array([9], np.uint16), annotated_class_ids=np.array([9], np.uint16),
                   prov=prov, quality=quality)
    ds(a, "boxes", np.asarray(boxes, np.float32), f"annotations/{name}/boxes")
    ds(a, "class_ids", np.asarray(cls, np.uint16), f"annotations/{name}/class_ids")
    ds(a, "instance_ids", np.asarray(iids, np.uint32), f"annotations/{name}/instance_ids")
    ds(a, "mask_offsets", np.asarray(offs, np.uint64), f"annotations/{name}/mask_offsets")
    ds(a, "mask_shapes", np.asarray(shapes, np.int32), f"annotations/{name}/mask_shapes")
    ds(a, "mask_data", np.concatenate(buf), f"annotations/{name}/mask_data", chunks=True, **blosc())
    return np.asarray(boxes, np.float32)


# ---------------------------------------------------------------- write
with h5py.File(PATH, "w") as f:
    f.attrs["medh5_version"] = "1.0"
    f.attrs["medh5_kind"] = "sample"
    f.attrs["medh5_profiles"] = np.array(
        ["core", "seg", "det", "cls", "reg", "curation", "training", "longitudinal"], dtype=S)
    f.attrs["digest_algo"] = "sha256"
    f.attrs["created"] = "2026-08-15T12:00:00Z"
    f.attrs["generator"] = "medh5-proto 1.0.0-dev"

    # --- 3.2 / 3.7 grids: one per (lattice, timepoint) --------------------
    G = f.create_group("grids")
    GRIDS = (("ct_tp0", SH_CT, (1.5, .8, .8), "tp0", FRAME0, -119.25),
             ("pet_tp0", SH_PET, (3.0, 1.6, 1.6), "tp0", FRAME0, -119.25),  # same frame -> no transform
             ("ct_tp1", SH_CT1, (1.5, .8, .8), "tp1", FRAME1,             # new frame -> registration
              -119.25 + Z0_FU * 1.5))                                     # shorter coverage, shifted origin
    for gid, shape, spacing, tp, frame, origin_z in GRIDS:
        g = G.create_group(gid)
        g.attrs["shape"] = np.asarray(shape, np.int64)
        g.attrs["axis_names"] = np.array(["z", "y", "x"], dtype=S)
        g.attrs["axis_kinds"] = np.array(["spatial"] * 3, dtype=S)
        g.attrs["spacing"] = np.asarray(spacing, np.float64)
        g.attrs["origin"] = np.array([origin_z, -63.6, -63.6], np.float64)
        g.attrs["direction"] = np.eye(3, dtype=np.float64)          # 2-D, per spec 2.5
        g.attrs["coord_system"] = "LPS"
        g.attrs["units"] = "mm"
        g.attrs["timepoint"] = tp                                   # spec 3.7 rule 2
        g.attrs["frame_uid"] = frame
        g.attrs["patch_hint"] = np.array([96, 96, 96], np.int64)

    # --- 4 images: timepoint inherited from the grid ----------------------
    I = f.create_group("images")
    for name, arr, grid, mod, units, chunks in (
        ("CT_tp0", synth_ct(0), "ct_tp0", "CT", "HU", (32, 64, 64)),
        ("CT_tp1", np.ascontiguousarray(synth_ct(3)[CROP_FU]), "ct_tp1", "CT", "HU", (32, 64, 64)),
        ("PET_tp0", np.abs(rng.normal(1.2, .4, SH_PET)).astype(np.float32), "pet_tp0", "PT", "SUVbw", (16, 32, 32)),
    ):
        d = ds(I, name, arr, f"images/{name}", chunks=chunks,
               **blosc("zstd", 3, hdf5plugin.Blosc2.SHUFFLE))
        d.attrs.update(grid=grid, modality=mod, value_type="quantitative", value_units=units,
                       prov="act_import")
        if mod == "CT":
            d.attrs.update(rescale_slope=1.0, rescale_intercept=0.0,
                           window_center=np.array([50.], np.float64),
                           window_width=np.array([400.], np.float64))

    A = f.create_group("annotations")
    # --- 7.2 layers at both timepoints ------------------------------------
    organs_digest, n_layers = write_layers(A, "organs_tp0", ORGANS, "ct_tp0", "tp0", "act_seg", "q_organs")
    write_layers(A, "organs_tp1", ORGANS, "ct_tp1", "tp1", "act_seg_fu", "q_organs_fu", crop=CROP_FU)

    # --- 7.4 instances with instance ids stable across timepoints ---------
    write_instances(A, "lesions_tp0", LESIONS["tp0"], "ct_tp0", "act_seg", "q_lesions")
    boxes_tp1 = write_instances(A, "lesions_tp1", LESIONS["tp1"], "ct_tp1", "act_seg_fu", "q_lesions",
                                crop=CROP_FU)

    # --- 8.2 boxes (model prediction at follow-up) ------------------------
    a = A.create_group("lesion_boxes_tp1")
    a.attrs.update(kind="boxes", task="detection", grid="ct_tp1", space="index", closure="explicit",
                   class_ids=np.array([9], np.uint16), annotated_class_ids=np.array([9], np.uint16),
                   prov="act_predict", quality="q_pred")
    ds(a, "boxes", boxes_tp1, "annotations/lesion_boxes_tp1/boxes")
    ds(a, "class_ids", np.full(len(boxes_tp1), 9, np.uint16), "annotations/lesion_boxes_tp1/class_ids")
    ds(a, "instance_ids", np.asarray(sorted(LESIONS["tp1"]), np.uint32),
       "annotations/lesion_boxes_tp1/instance_ids")
    ds(a, "scores", np.array([.94, .62][: len(boxes_tp1)], np.float32),
       "annotations/lesion_boxes_tp1/scores")

    # --- 9 classification: per-timepoint staging --------------------------
    a = A.create_group("staging_tp0")
    a.attrs.update(kind="classification", task="classification", scope="timepoint", multilabel=True,
                   closure="implicit", grid="ct_tp0", class_ids=np.array([10, 11, 12], np.uint16),
                   annotated_class_ids=np.array([10, 11, 12], np.uint16),
                   prov="act_seg", quality="q_organs")
    ds(a, "class_ids", np.array([10, 11], np.uint16), "annotations/staging_tp0/class_ids")
    ds(a, "values", np.array([1.0, 1.0], np.float32), "annotations/staging_tp0/values")
    ds(a, "scope_ids", np.array([0, 0], np.int64), "annotations/staging_tp0/scope_ids")

    # --- 9 classification: change across timepoints -----------------------
    a = A.create_group("response")
    a.attrs.update(kind="classification", task="classification", scope="sample", multilabel=False,
                   closure="explicit", timepoints=np.array(["tp0", "tp1"], dtype=S),
                   class_ids=np.array([13], np.uint16), annotated_class_ids=np.array([13], np.uint16),
                   prov="act_review", quality="q_response")
    ds(a, "class_ids", np.array([13], np.uint16), "annotations/response/class_ids")
    ds(a, "values", np.array([1.0], np.float32), "annotations/response/values")

    # --- 10.3 inter-timepoint registration --------------------------------
    T = f.create_group("transforms").create_group("tp0_to_tp1")
    M = np.eye(4); M[:3, 3] = [2.1, -0.8, 0.4]
    T.attrs.update(kind="affine", from_frame=FRAME0, to_frame=FRAME1,
                   from_grid="ct_tp0", to_grid="ct_tp1", units="mm", invertible=True,
                   prov="act_reg", metrics="q_reg")
    ds(T, "matrix", M, "transforms/tp0_to_tp1/matrix")

    # --- 14.3 sampling index ----------------------------------------------
    ix = f.create_group("index").create_group("organs_tp0")
    ix.attrs["source_digest"] = organs_digest                   # 13.3: the group's digest
    ix.attrs["max_coords"] = np.int64(4096)
    ix.attrs["seed"] = np.int64(0)
    counts, bbs = [], []
    fg = ix.create_group("fg_coords")
    for i in sorted(ORGANS):
        idx = np.argwhere(ORGANS[i])
        counts.append(len(idx))
        bbs.append(np.stack([idx.min(0) - .5, idx.max(0) + .5], 1))
        pick = idx[rng.choice(len(idx), min(4096, len(idx)), replace=False)].astype(np.int32)
        index_ds(fg, str(i), pick, **blosc())
    index_ds(ix, "class_ids", np.asarray(sorted(ORGANS), np.uint16))
    index_ds(ix, "voxel_counts", np.asarray(counts, np.int64))
    index_ds(ix, "class_bboxes", np.asarray(bbs, np.float32))

    # --- 2.4 the sample document -------------------------------------------
    meta = {
        "identity": {"sample_id": "DEMO-0001", "subject_id": "DEMO-0001",
                     "sex": "F", "bodypart": "abdomen"},
        "timepoints": [
            {"id": "tp0", "index": 0, "label": "baseline", "date": "2026-02-01",
             "days_from_baseline": 0, "study_uid": "pseudo:1.2.826.0.1.3680043.9.7.100",
             "series_uids": {"CT_tp0": "pseudo:…1", "PET_tp0": "pseudo:…2"},
             "subject_age_years": 61.4},
            {"id": "tp1", "index": 1, "label": "follow_up_3mo", "date": "2026-05-04",
             "days_from_baseline": 92, "study_uid": "pseudo:1.2.826.0.1.3680043.9.7.101",
             "series_uids": {"CT_tp1": "pseudo:…3"}, "subject_age_years": 61.7,
             "description": "post two cycles of chemotherapy"},
        ],
        "cohort": {"dataset_id": "demo-abdomen-longitudinal-v1", "site_id": "site-B",
                   "scanner_id": "SOMATOM-Force-042", "group_id": "DEMO-0001"},
        "label_set": {
            "id": "demo-abdomen", "version": "1.0.0", "form": "inline",
            "sha256": hashlib.sha256(b"demo-abdomen-1.0.0").hexdigest(),
            "classes": [{"id": i, "key": k, "name": n, "parents": p, "category": c,
                         "color": [(30 + i * 17) % 256, (90 + i * 29) % 256, (140 + i * 11) % 256, 255],
                         "codes": ([{"system": "SNOMED-CT", "code": "10200004", "name": "Liver"}]
                                   if k == "liver" else [])}
                        for i, k, n, p, c in CLASSES],
            "relations": [{"subject": 6, "predicate": "part_of", "object": 1}]},
        "provenance": {
            "agents": [
                {"id": "r1", "type": "person", "name": "pseudonym:RAD-07", "role": "annotator",
                 "qualification": "board-certified radiologist, 9y abdominal"},
                {"id": "r2", "type": "person", "name": "pseudonym:RAD-12", "role": "reviewer"},
                {"id": "s1", "type": "software", "name": "medh5", "version": "1.0.0"},
                {"id": "s2", "type": "software", "name": "nnU-Net", "version": "2.5.1"}],
            "activities": [
                {"id": "act_import", "type": "import", "agent": "s1", "ended": "2026-05-06T09:11:40Z",
                 "tool": "medh5 convert from-dicom",
                 "outputs": ["images/CT_tp0", "images/PET_tp0", "images/CT_tp1"],
                 "params": {"modality_lut": "applied", "studies": 2}},
                {"id": "act_seg", "type": "annotate", "agent": "r1", "started": "2026-02-05T13:02:00Z",
                 "ended": "2026-02-05T14:47:00Z", "tool": "3D Slicer 5.6.2",
                 "outputs": ["annotations/organs_tp0", "annotations/lesions_tp0", "annotations/staging_tp0"]},
                {"id": "act_seg_fu", "type": "annotate", "agent": "r1", "ended": "2026-05-07T11:20:00Z",
                 "tool": "3D Slicer 5.6.2", "inputs": ["annotations/lesions_tp0"],
                 "outputs": ["annotations/organs_tp1", "annotations/lesions_tp1"],
                 "params": {"protocol": "lesion ids carried forward from baseline"}},
                {"id": "act_predict", "type": "predict", "agent": "s2", "ended": "2026-05-06T02:00:00Z",
                 "outputs": ["annotations/lesion_boxes_tp1"], "params": {"checkpoint": "fold_all", "tta": True}},
                {"id": "act_reg", "type": "register", "agent": "s1", "ended": "2026-05-07T10:00:00Z",
                 "inputs": ["images/CT_tp0", "images/CT_tp1"], "outputs": ["transforms/tp0_to_tp1"],
                 "params": {"metric": "MI", "optimizer": "gradient_descent"}},
                {"id": "act_review", "type": "review", "agent": "r2", "ended": "2026-05-08T08:30:00Z",
                 "inputs": ["annotations/lesions_tp0", "annotations/lesions_tp1"],
                 "outputs": ["annotations/response"],
                 "params": {"criteria": "RECIST 1.1", "verdict": "partial_response"}}]},
        "quality": {
            "q_organs": {"status": "approved", "confidence": 0.94, "reviewed_by": ["r2"],
                         "agreement": [{"metric": "dice", "value": 0.913,
                                        "against": "annotations/organs_rater2",
                                        "per_class": {"2": 0.97, "3": 0.71}}],
                         "issues": [{"code": "boundary_uncertain", "severity": "info",
                                     "class_ids": [3], "note": "segment IV boundary blurred by motion"}],
                         "edit_effort_s": 640},
            "q_organs_fu": {"status": "approved", "confidence": 0.91},
            "q_lesions": {"status": "approved", "confidence": 0.88},
            "q_pred": {"status": "draft", "confidence": 0.6},
            "q_response": {"status": "approved", "confidence": 0.9, "reviewed_by": ["r2"]},
            "q_reg": {"status": "reviewed",
                      "agreement": [{"metric": "tre", "value": 1.8,
                                     "against": "annotations/landmarks_tp0"}]}},
        "splits": [{"set_id": "cv5-2026-02", "partition": "train", "fold": 2,
                    "assigned_by": "medh5 split", "assigned_at": "2026-05-10T00:00:00Z",
                    "manifest_sha256": hashlib.sha256(b"manifest").hexdigest()}],
        "acquisition": {
            "CT_tp0": {"KVP": 120, "XRayTubeCurrent": 210, "ConvolutionKernel": "B30f",
                       "ContrastPhase": "portal_venous", "SliceThickness": 1.5},
            "CT_tp1": {"KVP": 120, "XRayTubeCurrent": 195, "ConvolutionKernel": "B30f",
                       "ContrastPhase": "portal_venous", "SliceThickness": 1.5},
            "PET_tp0": {"Radiopharmaceutical": "FDG", "InjectedDose_MBq": 310}},
        "deidentification": {"method": "dicom-psi-profile", "profile": "DICOM PS3.15 E.1 basic",
                             "date_shift_days": -117, "id_mapping": "external", "performed_by": "s1",
                             "date": "2026-05-06T09:11:40Z", "burned_in_annotation_checked": True},
        "extra": {"com.example.trial": {"arm": "B"}},
    }
    f.create_dataset("meta", data=json.dumps(meta, separators=(",", ":")), dtype=S)
    f.attrs["content_id"] = content_id(f)

print(f"wrote {PATH}  ({os.path.getsize(PATH) / 1024**2:.2f} MiB)  "
      f"timepoints={len(meta['timepoints'])}  layers L={n_layers}")

# ---------------------------------------------------------------- verify
# Every check raises: Appendix C.2 publishes these results, and a check that only printed made
# the table a claim. A check here is this script's reading of the text, so CI also validates the
# file with the reference implementation, which once disagreed with every digest printed "[ok]".
import jsonschema  # noqa: E402


class CheckFailed(AssertionError):
    pass


def check(ok, what, detail=None):
    if not ok:
        raise CheckFailed(what if detail is None else f"{what}: {detail}")
    print(f"[ok] {what}")


def box_to_slice(lo, hi):
    """8.1: boxes sit at voxel edges, so [a, b] is the slice floor(a + 0.5) : floor(b + 0.5)."""
    return slice(math.floor(lo + 0.5), math.floor(hi + 0.5))


schema = json.load(open(SCHEMA))
with h5py.File(PATH, "r") as f:
    doc = json.loads(f["meta"][()])
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(doc)
    print("[ok] /meta validates against medh5-sample-1.0.schema.json")

    errs = []
    known = {c["id"] for c in doc["label_set"]["classes"]}
    acts = {a["id"] for a in doc["provenance"]["activities"]}
    agents = {a["id"] for a in doc["provenance"]["agents"]}
    tps = {t["id"]: t for t in doc["timepoints"]}
    multi_tp = len(tps) > 1

    # E108 timepoint declaration
    idxs = [t["index"] for t in doc["timepoints"]]
    if idxs != list(range(len(idxs))):
        errs.append("E108 timepoint index not dense/increasing")

    for a in doc["provenance"]["activities"]:
        if a["agent"] not in agents:
            errs.append(f"E601 activity {a['id']} -> unknown agent")

    # E106/E107 grids, W910 frame reuse across timepoints
    frame_tp = {}
    for gid in f["grids"]:
        ga = f[f"grids/{gid}"].attrs
        if multi_tp and "timepoint" not in ga:
            errs.append(f"E106 grids/{gid}: no timepoint")
        elif multi_tp and str(ga["timepoint"]) not in tps:
            errs.append(f"E107 grids/{gid}: undeclared timepoint")
        if "frame_uid" in ga and "timepoint" in ga:
            frame_tp.setdefault(str(ga["frame_uid"]), set()).add(str(ga["timepoint"]))
        D = ga["direction"]
        if D.ndim != 2:
            errs.append(f"E102 grids/{gid}: direction not 2-D")
        if not np.allclose(D @ D.T, np.eye(len(D)), atol=1e-4):
            errs.append(f"E102 grids/{gid}: not orthonormal")
    warns = [f"W910 frame {k} spans timepoints {sorted(v)}" for k, v in frame_tp.items() if len(v) > 1]

    # annotations
    inst_class = {}
    for name in f["annotations"]:
        at = f[f"annotations/{name}"].attrs
        cid, acid = set(at["class_ids"].tolist()), set(at["annotated_class_ids"].tolist())
        if not cid <= known:
            errs.append(f"E402 {name}: class ids {cid - known} not in label set")
        if not acid <= cid:
            errs.append(f"E403 {name}: annotated not subset of class_ids")
        if at.get("prov") is not None and str(at["prov"]) not in acts:
            errs.append(f"E601 {name}: dangling prov")
        if at.get("quality") is not None and str(at["quality"]) not in doc["quality"]:
            errs.append(f"E602 {name}: unknown quality key")
        if at.get("grid") is not None and str(at["grid"]) not in f["grids"]:
            errs.append(f"E101 {name}: unknown grid")
        if "timepoints" in at:                                    # E409
            for tp in at["timepoints"]:
                if (tp.decode() if isinstance(tp, bytes) else str(tp)) not in tps:
                    errs.append(f"E409 {name}: undeclared timepoint {tp}")
        grp = f[f"annotations/{name}"]
        if "instance_ids" in grp:                                 # W909
            for iid, c in zip(grp["instance_ids"][...], grp["class_ids"][...]):
                inst_class.setdefault(int(iid), set()).add(int(c))
        if "boxes" in grp:                                        # E406
            b = grp["boxes"][...]
            if np.any(b[..., 0] > b[..., 1]):
                errs.append(f"E406 {name}: lo > hi")
        if str(at["kind"]) == "layers":                           # E404
            flat = [int(v) for v in grp["layer_class_ids"][...].ravel() if v]
            if len(flat) != len(set(flat)) or set(flat) != cid:
                errs.append(f"E404 {name}: layer/class_ids invariant violated")

    warns += [f"W909 instance {i} has classes {sorted(c)}" for i, c in inst_class.items() if len(c) > 1]
    # W911 multi-timepoint sample with no transform relating two timepoints
    if multi_tp:
        linked = any(
            len({str(f[f"grids/{g}"].attrs["timepoint"]) for g in f["grids"]
                 if str(f[f"grids/{g}"].attrs["frame_uid"]) in
                 (str(f[f"transforms/{t}"].attrs["from_frame"]), str(f[f"transforms/{t}"].attrs["to_frame"]))}) > 1
            for t in f.get("transforms", {}))
        if not linked:
            warns.append("W911 no transform relates two timepoints")

    for name in f["images"]:
        d = f[f"images/{name}"]
        if tuple(d.shape) != tuple(f[f"grids/{d.attrs['grid']}"].attrs["shape"]):
            errs.append(f"E202 images/{name}: shape != grid shape")

    check(not errs, "cross-reference, geometry and encoding checks (E1xx-E6xx): clean", errs)
    check(not warns, "longitudinal warnings W909-W911: none", warns)

    # 13 integrity -----------------------------------------------------------
    bad, unstamped, stamped_index = [], [], []

    def chk(name, obj):
        if not isinstance(obj, h5py.Dataset) or name == "meta":
            return
        if name.split("/")[0] == "index":
            if "digest" in obj.attrs:
                stamped_index.append(name)
        elif "digest" not in obj.attrs:
            unstamped.append(name)
        elif digest(name, obj[...]) != obj.attrs["digest"]:
            bad.append(name)

    f.visititems(chk)
    check(not bad and not unstamped, "every dataset's digest matches its content (13.1)",
          {"mismatched": bad, "missing": unstamped})
    check(not stamped_index, "no dataset under index/ carries a digest (13.1)", stamped_index)
    check(content_id(f) == f.attrs["content_id"], f"content_id is the 13.2 Merkle root: {f.attrs['content_id'][:23]}...",
          content_id(f))
    source = group_digest(f["annotations/organs_tp0"])
    check(f["index/organs_tp0"].attrs["source_digest"] == source,
          "index/organs_tp0 source_digest is its annotation's group digest (13.3, W905)", source)

    # 3.3 geometry: index -> world -> index at every corner of every grid ------
    worst = 0.0
    for gid in f["grids"]:
        ga = f[f"grids/{gid}"].attrs
        affine = ga["direction"] * ga["spacing"]               # direction @ diag(spacing)
        corners = np.array(list(itertools.product(*[(0, n - 1) for n in ga["shape"]])), np.float64)
        world = corners @ affine.T + ga["origin"]
        back = np.linalg.solve(affine, (world - ga["origin"]).T).T
        worst = max(worst, float(np.abs(back - corners).max()))
    check(worst < 1e-9, f"index -> world -> index round trip on every grid (3.3): worst {worst:.1e} voxel")

    # 8.1 boxes and 7.4 instance masks ------------------------------------------
    s = box_to_slice(11.5, 39.5)
    check(s == slice(12, 40) and (s.start - 0.5, s.stop - 0.5) == (11.5, 39.5) and s.stop - s.start == 28,
          "box [11.5, 39.5] <-> slice(12, 40), extent 28 (8.1)", s)
    decoded, wrong = 0, []
    for name in ("lesions_tp0", "lesions_tp1"):
        g = f[f"annotations/{name}"]
        offsets, packed = g["mask_offsets"][...], g["mask_data"][...]
        for k, (box, stored) in enumerate(zip(g["boxes"][...], g["mask_shapes"][...])):
            shape = tuple(int(n) for n in stored)
            extent = tuple(sl.stop - sl.start for sl in (box_to_slice(lo, hi) for lo, hi in box))
            bits = np.unpackbits(packed[offsets[k]:offsets[k + 1]])
            mask = bits[: math.prod(shape)].reshape(shape).astype(bool)
            # Tight: every face of the box touches the object, or the box is not its extent.
            faces = all(np.take(mask, i, axis=a).any() for a in range(mask.ndim) for i in (0, -1))
            if extent != shape or not faces or bits[math.prod(shape):].any():
                wrong.append(f"{name}[{k}]: box extent {extent}, mask shape {shape}")
            decoded += 1
    check(not wrong, f"{decoded} instance masks decode to their box extents (7.4)", wrong)

    # 7.6 encoding equivalence: layers <-> bitmask, lossless per class -------
    lay = f["annotations/organs_tp0/data"]
    lci = f["annotations/organs_tp0/layer_class_ids"][...]
    layer_of = {int(c): lyr for lyr in range(lci.shape[0]) for c in lci[lyr] if c}
    cids = sorted(layer_of)
    planes = np.zeros((math.ceil(len(cids) / 64),) + lay.shape[1:], np.uint64)
    for pos, c in enumerate(cids):
        pl, bit = divmod(pos, 64)
        planes[pl][lay[layer_of[c]] == c] |= np.uint64(1) << np.uint64(bit)
    lossy = []
    for pos, c in enumerate(cids):
        pl, bit = divmod(pos, 64)
        from_bits = ((planes[pl] >> np.uint64(bit)) & np.uint64(1)).astype(bool)
        if not np.array_equal(lay[layer_of[c]] == c, from_bits):
            lossy.append(c)
    check(not lossy, f"transcode layers <-> bitmask lossless for all {len(cids)} classes (7.6)", lossy)

    # longitudinal reader checks -------------------------------------------
    ct0, ct1, pet0 = (f[f"grids/{g}"].attrs for g in ("ct_tp0", "ct_tp1", "pet_tp0"))
    check(pet0["frame_uid"] == ct0["frame_uid"] and tuple(pet0["shape"]) != tuple(ct0["shape"])
          and tuple(pet0["spacing"]) != tuple(ct0["spacing"]),
          "pet_tp0 shares ct_tp0's frame at another shape and spacing (3.4)")
    check(ct1["frame_uid"] != ct0["frame_uid"] and (ct0["shape"][0], ct1["shape"][0]) == (160, 152),
          "ct_tp1 is on its own frame, 152 slices against 160, nothing resampled (3.4, 3.7)")
    tpo = {str(f[f'grids/{g}'].attrs['timepoint']) for g in f["grids"]}
    check(tpo == set(tps) == {"tp0", "tp1"}, f"grids span the declared timepoints {sorted(tpo)} (3.7)")
    base = {int(i) for i in f["annotations/lesions_tp0/instance_ids"][...]}
    fu = {int(i) for i in f["annotations/lesions_tp1/instance_ids"][...]}
    tracked = (sorted(base & fu), sorted(base - fu), sorted(fu - base))
    check(tracked == ([1], [3, 6], [8]), "lesion tracking (7.4): persisted=[1] resolved=[3, 6] new=[8]", tracked)
    r = f["annotations/response"].attrs
    spans = [t.decode() if isinstance(t, bytes) else str(t) for t in r["timepoints"]]
    check(str(r["scope"]) == "sample" and spans == ["tp0", "tp1"],
          f"change label: scope={str(r['scope'])} timepoints={spans} (9, E409)")

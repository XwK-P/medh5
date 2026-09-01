"""Measure multi-label voxel-annotation encodings: v0.6 per-class bool vs v1 candidates.

Synthetic 200-class anatomy: 24 large mutually-exclusive "organs" + 176 small
structures (substructures/lesions) that overlap organs and sometimes each other.
"""
import json, math, os, time
import numpy as np, h5py, hdf5plugin

SHAPE = (160, 160, 160)
N_ORGAN, N_SMALL = 24, 176
C = N_ORGAN + N_SMALL
RNG = np.random.default_rng(0)
OUT = os.path.dirname(os.path.abspath(__file__))

def build():
    Z, Y, X = SHAPE
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in SHAPE], indexing="ij")
    masks = {}
    # organs: voronoi partition of a central body region -> mutually exclusive
    seeds = RNG.uniform(30, 130, size=(N_ORGAN, 3)).astype(np.float32)
    d2 = np.empty((N_ORGAN,) + SHAPE, dtype=np.float32)
    for i, (cz, cy, cx) in enumerate(seeds):
        d2[i] = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    nearest = np.argmin(d2, axis=0).astype(np.int16)
    body = ((zz - 80) ** 2 / 55**2 + (yy - 80) ** 2 / 60**2 + (xx - 80) ** 2 / 60**2) < 1.0
    for i in range(N_ORGAN):
        masks[i + 1] = (nearest == i) & body
    del d2, nearest
    # small structures: spheres, radius 3-9, scattered -> overlap organs, sometimes each other
    for j in range(N_SMALL):
        cz, cy, cx = RNG.uniform(25, 135, size=3)
        r = RNG.uniform(3, 9)
        masks[N_ORGAN + 1 + j] = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) < r * r
    return masks

def greedy_color(masks):
    """Overlap graph -> greedy coloring; classes in a layer are mutually exclusive."""
    ids = sorted(masks)
    # pairwise overlap via bbox prefilter then AND
    bbox = {}
    for i in ids:
        idx = np.argwhere(masks[i])
        bbox[i] = (idx.min(0), idx.max(0)) if idx.size else (np.zeros(3, int), -np.ones(3, int))
    adj = {i: set() for i in ids}
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            (lo1, hi1), (lo2, hi2) = bbox[a], bbox[b]
            if np.any(hi1 < lo2) or np.any(hi2 < lo1):
                continue
            lo = np.maximum(lo1, lo2); hi = np.minimum(hi1, hi2)
            sl = tuple(slice(int(l), int(h) + 1) for l, h in zip(lo, hi))
            if np.any(masks[a][sl] & masks[b][sl]):
                adj[a].add(b); adj[b].add(a)
    order = sorted(ids, key=lambda i: -len(adj[i]))
    color = {}
    for i in order:
        used = {color[n] for n in adj[i] if n in color}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    return color, adj

def blosc(cname="zstd", clevel=5, shuffle=None):
    kw = {"cname": cname, "clevel": clevel}
    if shuffle is not None:
        kw["filters"] = shuffle
    return dict(hdf5plugin.Blosc2(**kw))

def write_perclass(path, masks, chunks):
    t = time.perf_counter()
    with h5py.File(path, "w") as f:
        g = f.create_group("seg")
        for i, m in masks.items():
            g.create_dataset(str(i), data=m, chunks=chunks, **blosc("lz4hc", 8))
    return time.perf_counter() - t

def write_layers(path, masks, color, chunks):
    L = max(color.values()) + 1
    lay = np.zeros((L,) + SHAPE, dtype=np.uint16)
    for i, m in masks.items():
        lay[color[i]][m] = i
    t = time.perf_counter()
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=lay, chunks=(1,) + chunks,
                         **blosc("zstd", 5, hdf5plugin.Blosc2.BITSHUFFLE))
    return time.perf_counter() - t, L

def write_bitmask(path, masks, chunks):
    P = math.ceil(C / 64)
    bits = np.zeros((P,) + SHAPE, dtype=np.uint64)
    for pos, i in enumerate(sorted(masks)):
        p, b = divmod(pos, 64)
        bits[p][masks[i]] |= np.uint64(1) << np.uint64(b)
    t = time.perf_counter()
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=bits, chunks=(1,) + chunks,
                         **blosc("zstd", 5, hdf5plugin.Blosc2.BITSHUFFLE))
    return time.perf_counter() - t, P

def write_instances(path, masks):
    boxes, payload, offsets, ids = [], [], [0], []
    for i in sorted(masks):
        idx = np.argwhere(masks[i])
        if idx.size == 0:
            continue
        lo, hi = idx.min(0), idx.max(0) + 1
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        crop = np.packbits(masks[i][sl].ravel())
        boxes.append(np.stack([lo, hi], axis=1)); payload.append(crop)
        offsets.append(offsets[-1] + crop.size); ids.append(i)
    flat = np.concatenate(payload)
    t = time.perf_counter()
    with h5py.File(path, "w") as f:
        f.create_dataset("bbox", data=np.asarray(boxes, np.int32))
        f.create_dataset("class_ids", data=np.asarray(ids, np.uint16))
        f.create_dataset("mask_offsets", data=np.asarray(offsets, np.uint64))
        f.create_dataset("mask_data", data=flat, chunks=True, **blosc("zstd", 5))
    return time.perf_counter() - t

def read_patch_perclass(path, sl, n=20):
    ts = []
    with h5py.File(path, "r") as f:
        g = f["seg"]
        keys = sorted(g, key=int)
        for _ in range(n):
            t = time.perf_counter()
            out = np.stack([g[k][sl] for k in keys])
            ts.append(time.perf_counter() - t)
    return float(np.median(ts))

def read_patch_dense(path, sl, n=20):
    ts = []
    with h5py.File(path, "r") as f:
        d = f["data"]
        for _ in range(n):
            t = time.perf_counter()
            out = d[(slice(None),) + sl]
            ts.append(time.perf_counter() - t)
    return float(np.median(ts))

def mb(p): return os.path.getsize(p) / 1024**2

if __name__ == "__main__":
    print("building synthetic 200-class volume ...", flush=True)
    masks = build()
    occ = sum(int(m.sum()) for m in masks.values())
    print(f"shape={SHAPE} voxels={np.prod(SHAPE)/1e6:.1f}M  labelled voxel-instances={occ/1e6:.1f}M "
          f"(mean {occ/np.prod(SHAPE):.2f} labels/voxel)", flush=True)
    color, adj = greedy_color(masks)
    print(f"overlap graph: mean degree={np.mean([len(v) for v in adj.values()]):.1f}  layers={max(color.values())+1}", flush=True)
    chunks = (32, 64, 64)
    sl = (slice(48, 112), slice(48, 112), slice(48, 112))  # 64^3 patch
    res = {}
    p = f"{OUT}/perclass.h5";  res["per-class bool (v0.6)"] = (write_perclass(p, masks, chunks), mb(p), read_patch_perclass(p, sl), 200)
    p = f"{OUT}/layers.h5";    wt, L = write_layers(p, masks, color, chunks); res["layers uint16 (v1)"] = (wt, mb(p), read_patch_dense(p, sl), L)
    p = f"{OUT}/bitmask.h5";   wt, P = write_bitmask(p, masks, chunks);       res["bitmask uint64 (v1)"] = (wt, mb(p), read_patch_dense(p, sl), P)
    p = f"{OUT}/instances.h5"; wt = write_instances(p, masks);                res["instances (v1)"] = (wt, mb(p), float("nan"), len(masks))
    print(f"\n{'encoding':26s} {'write s':>8s} {'size MiB':>9s} {'64^3 all-class read ms':>23s} {'datasets/planes':>16s}")
    for k, (wt, size, rd, n) in res.items():
        print(f"{k:26s} {wt:8.2f} {size:9.2f} {rd*1000:23.2f} {n:16d}")
    json.dump({k: {"write_s": v[0], "size_mib": v[1], "read_ms": v[2], "n": v[3]} for k, v in res.items()},
              open(f"{OUT}/results.json", "w"), indent=2)

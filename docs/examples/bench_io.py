"""Codec profiles, int16-vs-float32 storage, single-class query, foreground index."""
import json, os, time
import numpy as np, h5py, hdf5plugin

OUT = os.path.dirname(os.path.abspath(__file__))
SHAPE = (192, 256, 256)
RNG = np.random.default_rng(1)

def synth_ct():
    """CT-like: smooth body + noise, HU range, int16."""
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, s, dtype=np.float32) for s in SHAPE], indexing="ij")
    body = ((y**2 + x**2) < 0.75).astype(np.float32)
    soft = 40 + 30 * np.sin(6 * z) * np.cos(5 * y)
    bone = 900 * (((y + .3)**2 + (x - .2)**2 + z**2) < 0.02)
    vol = np.where(body > 0, soft + bone, -1000.0)
    vol += RNG.normal(0, 12, SHAPE).astype(np.float32)
    return np.clip(vol, -1024, 3071)

def bench_write_read(path, data, chunks, **kw):
    t = time.perf_counter()
    with h5py.File(path, "w") as f:
        f.create_dataset("d", data=data, chunks=chunks, **kw)
    wt = time.perf_counter() - t
    size = os.path.getsize(path) / 1024**2
    sl = (slice(64, 128), slice(96, 160), slice(96, 160))
    ts = []
    with h5py.File(path, "r") as f:
        d = f["d"]
        for _ in range(30):
            t = time.perf_counter(); _ = d[sl]; ts.append(time.perf_counter() - t)
        t = time.perf_counter(); _ = d[...]; full = time.perf_counter() - t
    return wt, size, float(np.median(ts)) * 1000, full

ct = synth_ct()
raw_f32 = ct.astype(np.float32)
raw_i16 = np.rint(ct).astype(np.int16)     # HU stored natively, slope=1 intercept=0
chunks = (32, 64, 64)
print(f"volume {SHAPE} = {np.prod(SHAPE)/1e6:.1f}M voxels; raw float32 = {raw_f32.nbytes/1024**2:.1f} MiB, int16 = {raw_i16.nbytes/1024**2:.1f} MiB\n")

profiles = {
    "training  lz4  L1 +shuffle":  (raw_i16, dict(hdf5plugin.Blosc2(cname="lz4", clevel=1, filters=hdf5plugin.Blosc2.SHUFFLE))),
    "balanced  lz4hc L8 +shuffle": (raw_i16, dict(hdf5plugin.Blosc2(cname="lz4hc", clevel=8, filters=hdf5plugin.Blosc2.SHUFFLE))),
    "archive   zstd L9 +bitshuf":  (raw_i16, dict(hdf5plugin.Blosc2(cname="zstd", clevel=9, filters=hdf5plugin.Blosc2.BITSHUFFLE))),
    "portable  gzip L4 +shuffle":  (raw_i16, dict(compression="gzip", compression_opts=4, shuffle=True)),
    "f32       lz4hc L8 +shuffle": (raw_f32, dict(hdf5plugin.Blosc2(cname="lz4hc", clevel=8, filters=hdf5plugin.Blosc2.SHUFFLE))),
    "f32       zstd L9 +bitshuf":  (raw_f32, dict(hdf5plugin.Blosc2(cname="zstd", clevel=9, filters=hdf5plugin.Blosc2.BITSHUFFLE))),
}
print(f"{'profile':30s} {'write s':>8s} {'size MiB':>9s} {'ratio':>6s} {'64^3 read ms':>13s} {'full read s':>12s}")
rows = {}
for name, (data, kw) in profiles.items():
    wt, size, rd, full = bench_write_read(f"{OUT}/p.h5", data, chunks, **kw)
    ratio = data.nbytes / 1024**2 / size
    rows[name] = dict(write_s=wt, size_mib=size, ratio=ratio, patch_ms=rd, full_s=full)
    print(f"{name:30s} {wt:8.2f} {size:9.2f} {ratio:6.1f} {rd:13.2f} {full:12.2f}")

# ---- single-class query: 200 per-class datasets vs 5 layers vs 4 bitplanes ----
print("\n--- single-class 64^3 query (one class out of 200) ---")
def med(fn, n=30):
    ts = [ (lambda t0: (fn(), time.perf_counter()-t0)[1])(time.perf_counter()) for _ in range(n) ]
    return float(np.median(ts)) * 1000
sl = (slice(48, 112),) * 3
with h5py.File(f"{OUT}/perclass.h5", "r") as f:
    g = f["seg"]; k = sorted(g, key=int)[137]
    print(f"per-class bool: {med(lambda: g[k][sl]):.2f} ms  (1 of 200 datasets)")
with h5py.File(f"{OUT}/layers.h5", "r") as f:
    d = f["data"]
    print(f"layers        : {med(lambda: d[2][sl] == 138):.2f} ms  (1 layer of 5, + compare)")
with h5py.File(f"{OUT}/bitmask.h5", "r") as f:
    d = f["data"]
    print(f"bitmask       : {med(lambda: (d[2][sl] >> np.uint64(9)) & np.uint64(1)):.2f} ms  (1 plane of 4, + shift)")

# ---- foreground sampling: full-volume argwhere vs precomputed index ----
print("\n--- foreground sampling for patch centers ---")
with h5py.File(f"{OUT}/layers.h5", "r") as f:
    lay = f["data"][...]
mask = (lay[0] == 7)
t = time.perf_counter(); coords = np.argwhere(mask); t_argwhere = time.perf_counter() - t
print(f"v0.6 path : load full mask + argwhere -> {t_argwhere*1000:.1f} ms, {coords.nbytes/1024**2:.1f} MiB resident, {len(coords)} coords")
sub = coords[RNG.choice(len(coords), size=min(4096, len(coords)), replace=False)].astype(np.int32)
p = f"{OUT}/idx.h5"
with h5py.File(p, "w") as f:
    f.create_dataset("fg/7", data=sub, **dict(hdf5plugin.Blosc2(cname="zstd", clevel=5)))
t = time.perf_counter()
with h5py.File(p, "r") as f:
    _ = f["fg/7"][...]
t_idx = time.perf_counter() - t
print(f"v1 path   : read 4096-coord index    -> {t_idx*1000:.2f} ms, {sub.nbytes/1024:.1f} KiB resident, {len(sub)} coords")
json.dump(rows, open(f"{OUT}/io_results.json", "w"), indent=2)

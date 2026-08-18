"""Single-class query, codec-matched, with correct single-call HDF5 slicing."""
import math, os, time
import numpy as np, h5py, hdf5plugin
import bench_encodings as E

OUT = os.path.dirname(os.path.abspath(__file__))
masks = E.build()
color, adj = E.greedy_color(masks)
L = max(color.values()) + 1
C = len(masks)
P = math.ceil(C / 64)
chunks = (32, 64, 64)
sl = (slice(48, 112),) * 3

lay = np.zeros((L,) + E.SHAPE, np.uint16)
for i, m in masks.items():
    lay[color[i]][m] = i
bits = np.zeros((P,) + E.SHAPE, np.uint64)
pos_of = {}
for pos, i in enumerate(sorted(masks)):
    p, b = divmod(pos, 64); pos_of[i] = (p, b)
    bits[p][masks[i]] |= np.uint64(1) << np.uint64(b)

CODECS = {
    "lz4 L1 +shuffle":   dict(hdf5plugin.Blosc2(cname="lz4", clevel=1, filters=hdf5plugin.Blosc2.SHUFFLE)),
    "zstd L5 +bitshuf":  dict(hdf5plugin.Blosc2(cname="zstd", clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE)),
}
def med(fn, n=40):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1000

TARGET = sorted(masks)[137]
print(f"C={C} classes, layers L={L}, bitplanes P={P}, patch=64^3, target class id={TARGET}\n")
print(f"{'codec':20s} {'encoding':14s} {'size MiB':>9s} {'1-class ms':>11s} {'all-class ms':>13s}")
for cname, kw in CODECS.items():
    p1, p2, p3 = f"{OUT}/q_pc.h5", f"{OUT}/q_la.h5", f"{OUT}/q_bm.h5"
    with h5py.File(p1, "w") as f:
        g = f.create_group("seg")
        for i, m in masks.items():
            g.create_dataset(str(i), data=m, chunks=chunks, **kw)
    with h5py.File(p2, "w") as f:
        f.create_dataset("d", data=lay, chunks=(1,) + chunks, **kw)
    with h5py.File(p3, "w") as f:
        f.create_dataset("d", data=bits, chunks=(1,) + chunks, **kw)

    with h5py.File(p1, "r") as f:
        g = f["seg"]; keys = sorted(g, key=int)
        one = med(lambda: g[str(TARGET)][sl])
        allc = med(lambda: np.stack([g[k][sl] for k in keys]), n=10)
    print(f"{cname:20s} {'per-class':14s} {os.path.getsize(p1)/1024**2:9.2f} {one:11.2f} {allc:13.2f}")

    li = color[TARGET]
    with h5py.File(p2, "r") as f:
        d = f["d"]
        one = med(lambda: d[(li,) + sl] == TARGET)
        allc = med(lambda: d[(slice(None),) + sl], n=10)
    print(f"{cname:20s} {'layers':14s} {os.path.getsize(p2)/1024**2:9.2f} {one:11.2f} {allc:13.2f}")

    pl, b = pos_of[TARGET]
    with h5py.File(p3, "r") as f:
        d = f["d"]
        one = med(lambda: (d[(pl,) + sl] >> np.uint64(b)) & np.uint64(1))
        allc = med(lambda: d[(slice(None),) + sl], n=10)
    print(f"{cname:20s} {'bitmask':14s} {os.path.getsize(p3)/1024**2:9.2f} {one:11.2f} {allc:13.2f}")
    print()

"""Synthetic + real-ref test for genre_graft: shape, root protection, DTW monotonicity, pull-toward-genre."""
import glob, os
import numpy as np
from hml_skeleton import fk, q_from_two_vectors, PRIMARY_CHILD, PARENTS, T_POSE, IDX
import genre_graft as gg
import genre_style as gs


def synth(seed, motion, phase=0.0):
    rng = np.random.default_rng(seed)
    off = np.zeros((22, 3))
    for j, val in T_POSE.items(): off[j] = val
    T = 140
    gq = np.tile(np.array([1., 0, 0, 0]), (T, 22, 1))
    for t in range(T):
        for j in range(22):
            c = PRIMARY_CHILD[j]
            if c is None: gq[t, j] = gq[t, PARENTS[j]]; continue
            base = np.array(off[c], float)
            d = base + motion*np.linalg.norm(base)*np.array([np.sin(0.2*t+j+phase), np.cos(0.15*t+j), np.sin(0.1*t+2*j)])
            gq[t, j] = q_from_two_vectors(base, d)
    root = np.stack([np.array([0.003*t, 0, 0]) for t in range(T)])
    return fk(gq, off, root).astype(np.float32)


def ck(n, c): print(f"  [{'PASS' if c else 'FAIL'}] {n}"); return c
ok = True

visitor = synth(1, 0.18)
reference = synth(7, 0.55, phase=1.7)            # DIFFERENT motion family -> a genuinely distinct "genre"

# DTW path is monotonic
V = np.array([gg.gm.sig(visitor, a, a+gg.WIN, gg.gm.motif_bank(reference)[1]) for a in range(0, len(visitor)-gg.WIN+1, gg.STRIDE)])
R = np.array([gg.gm.sig(reference, a, a+gg.WIN, gg.gm.motif_bank(reference)[1]) for a in range(0, len(reference)-gg.WIN+1, gg.STRIDE)])
path = gg._dtw_path(V, R)
mono = all(path[k][0] <= path[k+1][0] and path[k][1] <= path[k+1][1] for k in range(len(path)-1))
ok &= ck("DTW path monotonic", mono)
ok &= ck("DTW path spans both ends", path[0] == (0, 0) and path[-1] == (len(V)-1, len(R)-1))

out, sim = gg.graft(visitor, reference, floor=0.6, gain=0.4)
ok &= ck("grafted finite", bool(np.all(np.isfinite(out))))
ok &= ck("grafted shape == visitor", out.shape == visitor.shape)
ok &= ck("root (pelvis) preserved", bool(np.allclose(out[:, IDX["pelvis"]], visitor[:, IDX["pelvis"]], atol=0.05)))
dev = float(np.mean(np.linalg.norm(out - visitor, axis=-1)))
ok &= ck(f"grafted differs from visitor (dev={dev:.3f}>0)", dev > 1e-3)

# the point of graft: LIMB ARTICULATION should move toward the real reference. Measure scale-invariantly
# (hip-centered, divided by body scale) so retargeting the reference to the visitor's size is not penalized.
limb = [j for z, js in gs.ZONES.items() if z != "root" for j in js]
def norm(clip):
    h = clip - clip[:, IDX["pelvis"]:IDX["pelvis"]+1, :]
    return h / gg._body_scale(clip)
ref_n, vis_n, out_n = norm(reference), norm(visitor), norm(out)
def nearest_ref_dist(clip_n):                    # mean over frames of nearest reference-pose limb distance
    d = 0.0
    for t in range(0, len(clip_n), 5):
        dists = np.linalg.norm((ref_n[:, limb, :] - clip_n[t, limb, :]).reshape(len(ref_n), -1), axis=1)
        d += float(dists.min())
    return d
dv, do = nearest_ref_dist(vis_n), nearest_ref_dist(out_n)
ok &= ck(f"grafted limb shape closer to real genre poses than visitor ({do:.2f} < {dv:.2f})", do < dv)

# real AIST references: graft each of the 3 genres onto a visitor, sanity only
real = {os.path.splitext(os.path.basename(p))[0]: gs.load_clip(p) for p in sorted(glob.glob("refs/*.json"))}
if real:
    vguest = next(iter(real.values()))
    for g, ref in real.items():
        o, s = gg.graft(vguest, ref, floor=0.5)
        ok &= ck(f"real graft '{g}' finite, full length ({o.shape[0]}=={vguest.shape[0]})",
                 bool(np.all(np.isfinite(o))) and o.shape[0] == vguest.shape[0])

print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")

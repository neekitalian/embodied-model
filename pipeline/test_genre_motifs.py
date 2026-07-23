"""Synthetic test for genre_motifs: similarity in range, enhance shape/identity, allocation orders."""
import numpy as np
from hml_skeleton import fk, q_from_two_vectors, PRIMARY_CHILD, PARENTS, T_POSE
import genre_motifs as gm

def synth(seed, motion, phase=0.0):
    rng = np.random.default_rng(seed)
    off = np.zeros((22, 3))
    for j, val in T_POSE.items(): off[j] = val
    T = 120
    gq = np.tile(np.array([1., 0, 0, 0]), (T, 22, 1))
    for t in range(T):
        for j in range(22):
            c = PRIMARY_CHILD[j]
            if c is None: gq[t, j] = gq[t, PARENTS[j]]; continue
            base = np.array(off[c], float)
            d = base + motion*np.linalg.norm(base)*np.array([np.sin(0.2*t+j+phase), np.cos(0.15*t+j), np.sin(0.1*t+2*j)])
            gq[t, j] = q_from_two_vectors(base, d)
    root = np.stack([np.array([0.002*t, 0, 0]) for t in range(T)])
    return fk(gq, off, root).astype(np.float32)

def ck(n, c): print(f"  [{'PASS' if c else 'FAIL'}] {n}"); return c
ok = True

visitor   = synth(1, 0.18)
ref_like  = synth(1, 0.45)          # same shape family, higher energy
ref_other = synth(7, 0.45, phase=1.7)

curve, mean = gm.sim_curve(visitor, ref_like)
ok &= ck("sim curve in [0,1]", bool(np.all(curve >= -1e-6) and np.all(curve <= 1 + 1e-6)))
ok &= ck("sim curve length == visitor", len(curve) == len(visitor))

out, m = gm.enhance(visitor, ref_like, base_alpha=0.45)
ok &= ck("enhance finite", bool(np.all(np.isfinite(out))))
ok &= ck("enhance shape (T,22,3)", out.shape[1:] == (22, 3))
dev = float(np.mean(np.linalg.norm(out[:len(out)] - visitor[:len(out)], axis=-1)))
ok &= ck(f"enhance differs from identity (dev={dev:.3f}>0)", dev > 1e-3)

alloc = gm.allocate(visitor, {"like": ref_like, "other": ref_other})
ok &= ck(f"allocation returns all genres {[(g, round(s,3)) for g,s in alloc.items()]}", set(alloc) == {"like", "other"})
ok &= ck("more-similar reference scores >= dissimilar", alloc["like"] >= alloc["other"] - 1e-3)

# alpha floor honored: sim-gated weight never below base_alpha's effect at similarity 0
z, _ = gm.enhance(visitor, ref_like, base_alpha=0.0, gain=0.0)
ok &= ck("gain=0, alpha=0 -> ~identity", float(np.mean(np.linalg.norm(z - visitor[:len(z)], axis=-1))) < 0.05)

print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")

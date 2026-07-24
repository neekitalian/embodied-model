"""Synthetic + real-ref test for exhibition.unnoticed_dance (read -> compare -> allocate -> edit)."""
import glob, os
import numpy as np
from hml_skeleton import fk, q_from_two_vectors, PRIMARY_CHILD, PARENTS, T_POSE
import exhibition as ex


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

visitor = synth(1, 0.18)
refs = {"like": synth(1, 0.45), "other": synth(7, 0.45, phase=1.7)}

# fit reuse: a model fit once matches sim_curve computed from scratch
m = ex.GenreModel.fit("like", refs["like"])
import genre_motifs as gm
c_model, _ = m.compare(visitor)
c_scratch, _ = gm.sim_curve(visitor, refs["like"])
ok &= ck("fitted model curve == from-scratch sim_curve", bool(np.allclose(c_model, c_scratch, atol=1e-6)))

edited, rep = ex.unnoticed_dance(visitor, refs)
ok &= ck("edited finite", bool(np.all(np.isfinite(edited))))
ok &= ck("edited shape (T,22,3)", edited.shape[1:] == (22, 3))
ok &= ck("edited length == visitor", len(edited) == len(visitor))
dev = float(np.mean(np.linalg.norm(edited - visitor[:len(edited)], axis=-1)))
ok &= ck(f"edit changes the movement (dev={dev:.3f}>0)", dev > 1e-3)
ok &= ck("report has all keys", {"genre", "scores", "allocation", "similarity", "frames"} <= set(rep))

# graft edit path: real genre motion spliced in, full length, root preserved
eg, rg = ex.unnoticed_dance(visitor, refs, graft=True)
ok &= ck("graft edit finite + full length", bool(np.all(np.isfinite(eg))) and len(eg) == len(visitor))
ok &= ck("graft report tagged edit=graft", rg.get("edit") == "graft")
ok &= ck(f"allocation sums to 1 ({sum(rep['allocation'].values()):.3f})", abs(sum(rep["allocation"].values()) - 1.0) < 1e-5)
ok &= ck(f"allocated genre is the most similar ({rep['genre']})", rep["genre"] == max(rep["scores"], key=rep["scores"].get))
ok &= ck("body most resembles 'like'", rep["genre"] == "like")

# pre-fit models can be passed in (skip re-fitting)
models = ex.fit_models(refs)
e2, r2 = ex.unnoticed_dance(visitor, refs, models=models)
ok &= ck("pre-fit models give identical result", bool(np.allclose(e2, edited, atol=1e-6)) and r2["genre"] == rep["genre"])

# real AIST references, if present: each genre clip should be allocated to itself
real = {os.path.splitext(os.path.basename(p))[0]: __import__("genre_style").load_clip(p)
        for p in sorted(glob.glob("refs/*.json"))}
if len(real) >= 2:
    rmodels = ex.fit_models(real)
    for g, clip in real.items():
        _, rr = ex.unnoticed_dance(clip, real, models=rmodels)
        ok &= ck(f"real ref '{g}' allocated to itself (got {rr['genre']})", rr["genre"] == g)

print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")

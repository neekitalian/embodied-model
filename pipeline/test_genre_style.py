"""Synthetic test for genre_style.transfer - shape, identity-preservation, and style change."""
import numpy as np
from hml_skeleton import fk, q_from_two_vectors, PRIMARY_CHILD, PARENTS, T_POSE, IDX
import genre_style as gs

def synth(seed, motion=0.25):
    rng = np.random.default_rng(seed)
    off = np.zeros((22, 3))
    for j, val in T_POSE.items(): off[j] = val
    T = 90
    gq = np.tile(np.array([1., 0, 0, 0]), (T, 22, 1))
    for t in range(T):
        for j in range(22):
            c = PRIMARY_CHILD[j]
            if c is None: gq[t, j] = gq[t, PARENTS[j]]; continue
            base = np.array(off[c], float)
            d = base + motion*np.linalg.norm(base)*np.array([np.sin(0.2*t+j), np.cos(0.15*t+j), np.sin(0.1*t+2*j)])
            gq[t, j] = q_from_two_vectors(base, d)
    root = np.stack([np.array([0.002*t, 0.01*np.sin(0.1*t), 0]) for t in range(T)])
    return fk(gq, off, root).astype(np.float32)

def ck(name, cond): print(f"  [{'PASS' if cond else 'FAIL'}] {name}"); return cond

visitor = synth(0, 0.20)     # calm visitor
reference = synth(7, 0.55)   # expansive genre reference
ok = True

# zones partition all 22 joints exactly once
allz = sorted(sum(gs.ZONES.values(), []))
ok &= ck("zones cover all 22 joints, no overlap", allz == list(range(22)))

styled = gs.transfer(visitor, reference)
ok &= ck("output shape (T,22,3)", styled.shape == visitor.shape)
ok &= ck("all finite", np.all(np.isfinite(styled)))

# identity preserved most in root, changed most in limbs
def zone_delta(zone):
    js = gs.ZONES[zone]
    return float(np.mean(np.linalg.norm(styled[:, js, :] - visitor[:len(styled), js, :], axis=-1)))
root_d, arm_d, leg_d = zone_delta("root"), zone_delta("left_arm"), zone_delta("left_leg")
ok &= ck(f"root changes least (identity protected): root={root_d:.3f} < arm={arm_d:.3f}", root_d < arm_d)
ok &= ck(f"limbs carry the style (arm/leg moved): arm={arm_d:.3f}>0 leg={leg_d:.3f}>0", arm_d > 1e-3 and leg_d > 1e-3)

# rhythm_align preserves length
al = gs.rhythm_align(visitor, reference)
ok &= ck("rhythm_align preserves length", len(al) == min(len(visitor), len(reference)))

# alpha=0 -> (near) identity everywhere
id0 = gs.transfer(visitor, reference, {z:0.0 for z in gs.ZONES})
ok &= ck("alpha=0 stays close to identity", float(np.mean(np.linalg.norm(id0 - visitor[:len(id0)], axis=-1))) < 0.05)

print("\nRESULT:", "ALL PASS ✓" if ok else "SOME FAILED ✗")

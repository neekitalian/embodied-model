"""Synthetic smoke test for stage12_capture_lift - no camera / mediapipe needed.
Fakes MediaPipe world landmarks and checks the lift math is sane."""
import numpy as np
import stage12_capture_lift as S

# --- fake a plausible standing skeleton in MediaPipe world frame (meters, y-DOWN, hip-centered) ---
class LM:  # mimic a mediapipe landmark
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z

def fake_landmarks(t=0.0):
    """33 landmarks; only the ones the mapper reads need to be right. y grows downward."""
    lm = [LM(0, 0, 0) for _ in range(33)]
    swing = 0.15 * np.sin(t)             # a little arm swing so 1e filter has signal
    lm[S.MP["NOSE"]]   = LM(0.0,  -0.60, 0.05)
    lm[S.MP["L_SH"]]   = LM(-0.18, -0.45, 0.0)
    lm[S.MP["R_SH"]]   = LM( 0.18, -0.45, 0.0)
    lm[S.MP["L_EL"]]   = LM(-0.30, -0.25 + swing, 0.0)
    lm[S.MP["R_EL"]]   = LM( 0.30, -0.25 - swing, 0.0)
    lm[S.MP["L_WR"]]   = LM(-0.40, -0.05 + swing, 0.0)
    lm[S.MP["R_WR"]]   = LM( 0.40, -0.05 - swing, 0.0)
    lm[S.MP["L_HIP"]]  = LM(-0.10,  0.00, 0.0)
    lm[S.MP["R_HIP"]]  = LM( 0.10,  0.00, 0.0)
    lm[S.MP["L_KN"]]   = LM(-0.11,  0.45, 0.0)
    lm[S.MP["R_KN"]]   = LM( 0.11,  0.45, 0.0)
    lm[S.MP["L_AN"]]   = LM(-0.12,  0.90, 0.0)
    lm[S.MP["R_AN"]]   = LM( 0.12,  0.90, 0.0)
    lm[S.MP["L_FOOT"]] = LM(-0.12,  0.98, 0.12)
    lm[S.MP["R_FOOT"]] = LM( 0.12,  0.98, 0.12)
    return lm

def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond

ok = True

# 1) single-frame map shape + ordering
j = S.mp33_world_to_h22(fake_landmarks())
ok &= check("output shape == (22,3)", j.shape == (22, 3))
ok &= check("all finite", np.all(np.isfinite(j)))
ok &= check("pelvis == midpoint of hips",
            np.allclose(j[S.IDX["pelvis"]], [0, 0, 0], atol=1e-6))
ok &= check("spine1<spine2<spine3 in y (toward neck)",
            j[S.IDX["spine1"]][1] > j[S.IDX["spine2"]][1] > j[S.IDX["spine3"]][1])  # y-down: neck has smaller y
ok &= check("head above neck (smaller y = higher, pre-flip)",
            j[S.IDX["head"]][1] < j[S.IDX["neck"]][1])
ok &= check("left/right shoulders on opposite x sides",
            j[S.IDX["left_shoulder"]][0] < 0 < j[S.IDX["right_shoulder"]][0])

# 2) full-sequence transform: y-up + hip-centered
seq = np.stack([S.mp33_world_to_h22(fake_landmarks(t)) for t in np.linspace(0, 6, 90)])
out = S.to_yup_hipcentered(seq)
ok &= check("transformed shape preserved (90,22,3)", out.shape == (90, 22, 3))
ok &= check("pelvis ~0 every frame (hip-centered)",
            np.allclose(out[:, S.IDX["pelvis"], :], 0, atol=1e-5))
ok &= check("head now ABOVE pelvis in y (y-up after flip)",
            np.all(out[:, S.IDX["head"], 1] > out[:, S.IDX["pelvis"], 1]))
ok &= check("ankles BELOW pelvis in y (feet down)",
            np.all(out[:, S.IDX["left_ankle"], 1] < 0) and np.all(out[:, S.IDX["right_ankle"], 1] < 0))

# 3) 1€ filter: preserves shape, reduces jitter on a noisy signal
euro = S.VectorEuro((22, 3), freq=30.0, min_cutoff=1.0, beta=0.01)
rng = np.random.default_rng(0)
noisy = seq + rng.normal(0, 0.02, seq.shape)
filt = np.stack([euro(f) for f in noisy])
ok &= check("filter preserves shape", filt.shape == seq.shape)
raw_jitter = np.mean(np.abs(np.diff(noisy, axis=0)))
filt_jitter = np.mean(np.abs(np.diff(filt, axis=0)))
ok &= check(f"filter reduces frame-to-frame jitter ({raw_jitter:.4f} -> {filt_jitter:.4f})",
            filt_jitter < raw_jitter)

print("\nRESULT:", "ALL PASS ✓" if ok else "SOME FAILED ✗")

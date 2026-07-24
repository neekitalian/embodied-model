"""
Motif GRAFT (Scheme B): splice the reference's REAL genre motion into the visitor where they align.

Unlike genre_style.transfer (a geometric proxy that only reshapes the visitor's own trajectory), graft
transplants the ACTUAL limb articulation from the genre reference clip, so the output genuinely looks
like the genre - the pirouette / bounce / extension the reference dancer performed, not a scaled version
of the visitor's own motion.

Pipeline:
  1. MATCH   encode sliding windows of visitor and reference (Laban+FFT descriptor, z-scored in the
             reference's own feature space) and DTW-align them -> a MONOTONIC frame map r(t), so grafted
             motion follows the visitor's phrasing smoothly instead of teleporting between poses.
  2. ALIGN   per frame, take the real reference pose reference[r(t)], hip-center it, rotate it about the
             vertical to face the same way as the visitor, and scale it to the visitor's body size.
  3. GRAFT   keep the visitor's ROOT (pelvis/spine1) so locomotion + identity of place stays theirs;
             blend the limbs visitor <-> aligned-reference by a per-frame weight that rises with
             similarity (floor = the alpha slider). Where you resemble the genre, you get its real moves.
  4. SMOOTH  1e-filter to erase graft seams.

Matching uses the interpretable Laban descriptor (numpy + browser-portable, no torch), so recognition
allocation can still come from the trained model while the graft itself runs anywhere.

Deps: numpy. Reuses genre_style.py (ZONES, one_euro_smooth) + genre_motifs.py (motif bank / descriptor).
"""
import argparse, glob, os
import numpy as np
import genre_style as gs
import genre_motifs as gm
from hml_skeleton import IDX

WIN, STRIDE = 24, 12
_HIP, _CHEST = IDX["pelvis"], IDX["spine3"]


def _yaw(pose):
    o = pose[_CHEST] - pose[_HIP]
    return float(np.arctan2(o[0], o[2] + 1e-8))


def _rot_y(p, ang):
    """Rotate points (...,3) about the vertical (y) axis by ang."""
    c, s = np.cos(ang), np.sin(ang)
    out = np.asarray(p, dtype=np.float32).copy()
    x, z = p[..., 0], p[..., 2]
    out[..., 0] = x * c + z * s
    out[..., 2] = -x * s + z * c
    return out


def _dtw_path(V, R):
    """Monotonic DTW alignment of window-feature rows V (Nv,D) and R (Nr,D) (unit-norm). Returns [(vi,rj)]."""
    Nv, Nr = len(V), len(R)
    C = 1.0 - V @ R.T                                   # cosine distance (rows unit-norm)
    D = np.full((Nv + 1, Nr + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, Nv + 1):
        ci = C[i - 1]
        for j in range(1, Nr + 1):
            D[i, j] = ci[j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    i, j, path = Nv, Nr, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        if step == 0: i, j = i - 1, j - 1
        elif step == 1: i -= 1
        else: j -= 1
    path.reverse()
    return path


def _body_scale(clip):
    """Mean joint distance from the hip - a translation/rotation-invariant size proxy."""
    return float(np.mean(np.linalg.norm(clip - clip[:, _HIP:_HIP + 1, :], axis=-1))) + 1e-6


def graft(visitor, reference, win=WIN, stride=STRIDE, floor=0.5, gain=0.5, cap=0.95, protect_root=True, curve=None):
    """Return (grafted_clip, mean_similarity). Full visitor length; root kept; limbs grafted by similarity.
    Pass `curve` (per-frame gate 0..1, e.g. from the trained model) to override the built-in Laban curve."""
    visitor = np.asarray(visitor, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    T = len(visitor)
    bank, stats = gm.motif_bank(reference, win, stride)               # feature space = reference's own

    v_starts = list(range(0, max(1, T - win + 1), stride))
    r_starts = list(range(0, max(1, len(reference) - win + 1), stride))
    V = np.array([gm.sig(visitor, a, a + win, stats) for a in v_starts], dtype=np.float32)
    R = np.array([gm.sig(reference, a, a + win, stats) for a in r_starts], dtype=np.float32)

    # 1. MATCH: DTW -> per-visitor-window reference center, then per-frame reference index r(t)
    path = _dtw_path(V, R)
    vmap = {}
    for vi, rj in path:
        vmap.setdefault(vi, []).append(rj)
    v_centers = [a + win / 2.0 for a in v_starts]
    r_for_v = [float(np.mean([r_starts[rj] + win / 2.0 for rj in vmap.get(vi, [0])])) for vi in range(len(v_starts))]
    if len(v_centers) >= 2:
        r_frame = np.interp(np.arange(T), v_centers, r_for_v)
    else:
        r_frame = np.full(T, r_for_v[0] if r_for_v else 0.0)
    for t in range(1, T):                                            # smooth the frame map (no teleporting)
        r_frame[t] = 0.15 * r_frame[t] + 0.85 * r_frame[t - 1]
    r_idx = np.clip(np.round(r_frame).astype(int), 0, len(reference) - 1)

    curve_lab, mean = gm.curve_from_bank(visitor, bank, stats, win, stride)
    if curve is None:
        curve = curve_lab
    else:
        curve = np.asarray(curve, dtype=np.float32)
    sc = _body_scale(visitor) / _body_scale(reference)

    # smoothed per-frame yaw delta so the grafted limbs face the way the visitor faces
    vyaw = np.unwrap(np.array([_yaw(visitor[t]) for t in range(T)]))
    ryaw = np.unwrap(np.array([_yaw(reference[r_idx[t]]) for t in range(T)]))
    dyaw = vyaw - ryaw
    for t in range(1, T):
        dyaw[t] = 0.1 * dyaw[t] + 0.9 * dyaw[t - 1]

    # 2+3. ALIGN + GRAFT
    out = visitor.copy()
    for t in range(T):
        rf = reference[r_idx[t]] - reference[r_idx[t], _HIP]          # hip-center the reference pose
        rf = _rot_y(rf, dyaw[t]) * sc                                 # face like visitor + scale to body
        hip = visitor[t, _HIP]
        w = min(cap, floor + gain * curve[t])
        for zone, joints in gs.ZONES.items():
            if protect_root and zone == "root":
                continue                                             # keep locomotion / identity of place
            for j in joints:
                out[t, j] = (1 - w) * visitor[t, j] + w * (hip + rf[j])

    out = gs.one_euro_smooth(out, alpha=0.5)                          # 4. SMOOTH seams
    return out, float(mean)


def main():
    ap = argparse.ArgumentParser(description="Motif graft: splice real genre motion into the visitor.")
    ap.add_argument("--visitor", required=True)
    ap.add_argument("--reference", help="genre reference clip")
    ap.add_argument("--genre", help="resolve a reference from --refs-dir instead")
    ap.add_argument("--refs-dir", default="refs")
    ap.add_argument("--alpha", type=float, default=0.5, help="graft floor (identity<->genre)")
    ap.add_argument("--out", default="grafted.npy")
    a = ap.parse_args()
    a.visitor, a.refs_dir, a.out = map(os.path.expanduser, (a.visitor, a.refs_dir, a.out))
    visitor = gs.load_clip(a.visitor)
    if a.reference:
        ref = gs.load_clip(os.path.expanduser(a.reference))
    else:
        hits = sorted(glob.glob(os.path.join(a.refs_dir, f"*{(a.genre or '').lower()}*.json")))
        if not hits:
            raise SystemExit(f"no reference for genre '{a.genre}' in {a.refs_dir}")
        ref = gs.load_clip(hits[0])
    out, sim = graft(visitor, ref, floor=a.alpha)
    np.save(a.out, out.astype(np.float32))
    print(f"[graft] visitor {visitor.shape} + reference {ref.shape} -> {out.shape} "
          f"(similarity {sim:.3f}) -> {a.out}  (preview: python view_clip.py {a.out})")


if __name__ == "__main__":
    main()

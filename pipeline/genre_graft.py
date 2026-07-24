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

WIN, STRIDE = 24, 6                                  # finer stride -> higher-resolution matching / transitions
_HIP, _CHEST = IDX["pelvis"], IDX["spine3"]

# per-zone GENRE strength: how much each anatomical zone carries the genre vs stays you. Root fully you
# (locomotion / identity of place); torso mostly you; limbs carry the genre. Raises identity preservation
# while keeping the genre legible where it reads most - the arms and legs.
ZONE_GENRE = {"root": 0.0, "spine": 0.35, "left_arm": 1.0, "right_arm": 1.0, "left_leg": 0.85, "right_leg": 0.85}


def _smooth_zp(x, a=0.4):
    """Zero-phase smoothing (forward then backward EMA): removes the lag a one-directional filter adds,
    so transitions stay crisp and centered instead of smeared/delayed. Works on 1-D curves and (T,22,3)."""
    x = np.asarray(x, dtype=np.float32).copy()
    for t in range(1, len(x)):
        x[t] = a * x[t] + (1 - a) * x[t - 1]
    for t in range(len(x) - 2, -1, -1):
        x[t] = a * x[t] + (1 - a) * x[t + 1]
    return x


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


def _align_reference(visitor, reference, win=WIN, stride=STRIDE, curve=None):
    """DTW-align the reference to the visitor and return the per-frame HIP-CENTERED reference pose
    (faced like the visitor, scaled to their body) + the per-frame similarity curve + its mean.
    aligned[t] is what you graft the visitor's limbs toward at frame t."""
    visitor = np.asarray(visitor, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    T = len(visitor)
    bank, stats = gm.motif_bank(reference, win, stride)               # feature space = reference's own

    v_starts = list(range(0, max(1, T - win + 1), stride))
    r_starts = list(range(0, max(1, len(reference) - win + 1), stride))
    V = np.array([gm.sig(visitor, a, a + win, stats) for a in v_starts], dtype=np.float32)
    R = np.array([gm.sig(reference, a, a + win, stats) for a in r_starts], dtype=np.float32)

    path = _dtw_path(V, R)                                            # monotonic frame map, no teleporting
    vmap = {}
    for vi, rj in path:
        vmap.setdefault(vi, []).append(rj)
    v_centers = [a + win / 2.0 for a in v_starts]
    r_for_v = [float(np.mean([r_starts[rj] + win / 2.0 for rj in vmap.get(vi, [0])])) for vi in range(len(v_starts))]
    r_frame = np.interp(np.arange(T), v_centers, r_for_v) if len(v_centers) >= 2 else np.full(T, r_for_v[0] if r_for_v else 0.0)
    r_frame = _smooth_zp(r_frame, 0.25)                              # zero-phase: smooth, no lag
    r_idx = np.clip(np.round(r_frame).astype(int), 0, len(reference) - 1)

    curve_lab, mean = gm.curve_from_bank(visitor, bank, stats, win, stride)
    curve = curve_lab if curve is None else np.asarray(curve, dtype=np.float32)
    sc = _body_scale(visitor) / _body_scale(reference)

    vyaw = np.unwrap(np.array([_yaw(visitor[t]) for t in range(T)]))
    ryaw = np.unwrap(np.array([_yaw(reference[r_idx[t]]) for t in range(T)]))
    dyaw = _smooth_zp(vyaw - ryaw, 0.15)

    aligned = np.zeros((T, 22, 3), dtype=np.float32)
    for t in range(T):
        rf = reference[r_idx[t]] - reference[r_idx[t], _HIP]          # hip-center
        aligned[t] = _rot_y(rf, dyaw[t]) * sc                        # face like visitor + scale to body
    return aligned, np.asarray(curve, dtype=np.float32), float(mean)


def _compose(visitor, hip_targets, strength, zone_genre):
    """Blend visitor <-> (hip + hip-centered target) per zone, per frame, then zero-phase smooth.
    strength (T,) is the overall genre weight; zone_genre scales it per anatomical zone."""
    T = len(visitor)
    out = np.asarray(visitor, dtype=np.float32).copy()
    for t in range(T):
        hip = visitor[t, _HIP]
        for zone, joints in gs.ZONES.items():
            zg = zone_genre.get(zone, 1.0)
            if zg <= 0.0:
                continue                                             # zone stays fully the visitor
            w = strength[t] * zg
            for j in joints:
                out[t, j] = (1 - w) * visitor[t, j] + w * (hip + hip_targets[t, j])
    return _smooth_zp(out, 0.75)                                      # light zero-phase seam smoothing (keeps genre poses crisp)


def graft(visitor, reference, win=WIN, stride=STRIDE, floor=0.5, gain=0.5, cap=0.95, curve=None, zone_genre=None):
    """Single-genre graft. Full visitor length; per-zone identity preservation; smooth transitions.
    Pass `curve` (per-frame gate 0..1, e.g. from the trained model) to override the built-in Laban curve."""
    visitor = np.asarray(visitor, dtype=np.float32)
    zone_genre = ZONE_GENRE if zone_genre is None else zone_genre
    aligned, curve, mean = _align_reference(visitor, reference, win, stride, curve)
    strength = _smooth_zp(np.clip(floor + gain * curve, 0.0, cap), 0.5)
    return _compose(visitor, aligned, strength, zone_genre), float(mean)


def graft_mix(visitor, references, curves=None, win=WIN, stride=STRIDE, floor=0.45, gain=0.5, cap=0.9, zone_genre=None):
    """GENRE MIX graft: blend the aligned real motion of ALL genres, weighted PER FRAME by how much the
    body resembles each (softmax-free, just normalized similarity). A body that reads 51% ballet / 45%
    jazz literally dances a ballet-jazz blend, more ballet where it leans ballet and more jazz where it
    leans jazz. Root/torso identity preserved via zone_genre; transitions zero-phase smoothed.

    references : {genre: clip}.  curves : optional {genre: per-frame sim} (e.g. from the trained model).
    Returns (mixed_clip, {genre: mean_similarity})."""
    visitor = np.asarray(visitor, dtype=np.float32)
    zone_genre = ZONE_GENRE if zone_genre is None else zone_genre
    genres = list(references)
    aligned, curve = {}, {}
    means = {}
    for g in genres:
        aligned[g], curve[g], means[g] = _align_reference(visitor, references[g], win, stride,
                                                           (curves or {}).get(g))
    T = len(visitor)
    C = np.stack([curve[g] for g in genres])                         # (G, T) per-frame similarity
    mix = C / (C.sum(axis=0, keepdims=True) + 1e-8)                  # (G, T) per-frame genre weights
    max_sim = C.max(axis=0)                                          # (T,) how genre-like overall
    strength = _smooth_zp(np.clip(floor + gain * max_sim, 0.0, cap), 0.5)
    mixed = np.zeros((T, 22, 3), dtype=np.float32)                   # per-frame blended genre target
    for gi, g in enumerate(genres):
        mixed += mix[gi][:, None, None] * aligned[g]
    return _compose(visitor, mixed, strength, zone_genre), means


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

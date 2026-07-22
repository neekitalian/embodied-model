"""
End-to-end MVP orchestrator (local).  Drop into momask-codes-main/ as `scripts/run_mvp.py`.

Flow:  visitor webcam clip (Stage 1+2)  ->  pick a target dance clip whose body-characteristics
match the visitor (Stage: genre match)  ->  your zone-alpha feature blend  ->  reconstructed
joints saved as .npy for Unity (Stage 3).

!!! ASSUMPTIONS — I could not read your repo from this session, so these calls are written to the
    signatures you *described*. Verify against semantic_spectrum/ and scripts/run_zone_blend.py,
    and fix the three ADAPT markers below. Everything else is glue you can keep.
"""
import argparse
import glob
import os
import numpy as np

# --- your existing Phase 2 modules (do NOT rewrite) ---
from semantic_spectrum.zones import get_zones                 # get_zones(mode) -> {zone: [joint_idx,...]}
from semantic_spectrum.zone_features import extract_zone_features  # ADAPT: confirm function name
from semantic_spectrum.blend import feature_blend, reconstruct     # feature_blend(M_o, M_t, alpha); reconstruct(...)


def load_joints(path):
    """Load a (T,22,3) clip. .npy from Stage 1, or an AIST++ SMPL .pkl (joints key)."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    if path.endswith(".pkl"):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        # ADAPT #1: AIST++ .pkl layout varies. Common keys: 'smpl_poses','smpl_trans',
        # or precomputed 'keypoints3d'. If you already have a HumanML3D-22 export step,
        # call it here instead. Placeholder assumes a (T,J,3) joints array under one of these:
        for k in ("keypoints3d", "joints3d", "joints"):
            if k in d:
                return np.asarray(d[k], dtype=np.float32)[:, :22, :]
        raise KeyError(f"{path}: no joints array found; keys={list(d.keys())}")
    raise ValueError(f"Unsupported clip: {path}")


def body_signature(joints, zones):
    """Coarse per-zone feature vector used to match a dance genre to the visitor's body."""
    feats = extract_zone_features(joints, zones)          # ADAPT #2: returns per-zone features
    # flatten zone features into one vector for nearest-clip matching
    return np.concatenate([np.ravel(v) for v in (feats.values() if isinstance(feats, dict) else feats)])


def pick_target(visitor, library_dir, zones):
    """Nearest target clip in the library by body-signature distance (genre match by body chars)."""
    v = body_signature(visitor, zones)
    best, best_d = None, np.inf
    for path in sorted(glob.glob(os.path.join(library_dir, "*.npy")) +
                       glob.glob(os.path.join(library_dir, "*.pkl"))):
        try:
            t = load_joints(path)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")
            continue
        s = body_signature(t, zones)
        n = min(len(v), len(s))
        d = float(np.linalg.norm(v[:n] - s[:n]))
        if d < best_d:
            best, best_d, best_t = path, d, t
    if best is None:
        raise RuntimeError(f"No usable target clips in {library_dir}")
    print(f"[match] {os.path.basename(best)}  (dist={best_d:.3f})")
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visitor", default="visitor_clip.npy", help="Stage 1 output")
    ap.add_argument("--library", default="target_library/", help="dir of target dance clips")
    ap.add_argument("--alpha", type=float, default=0.6, help="identity-preservation blend weight")
    ap.add_argument("--zone-mode", default="default")
    ap.add_argument("--out", default="blended_for_unity.npy")
    a = ap.parse_args()

    zones = get_zones(a.zone_mode)
    visitor = load_joints(a.visitor)
    print(f"[in] visitor {visitor.shape}")

    target = pick_target(visitor, a.library, zones)

    # ADAPT #3: alpha may be a scalar or a per-zone dict in your feature_blend. Scalar shown.
    blended_feats = feature_blend(visitor, target, a.alpha)
    out = reconstruct(blended_feats)                       # autoregressive joint reconstruction
    out = np.asarray(out, dtype=np.float32)
    np.save(a.out, out)
    print(f"[out] {out.shape} -> {a.out}  (feed to Unity Stage 3)")


if __name__ == "__main__":
    main()

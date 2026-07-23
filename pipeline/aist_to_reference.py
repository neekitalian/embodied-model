"""
AIST++ -> genre reference clip (HumanML3D-22 JSON) for genre_style.py.

AIST++ gives SMPL params per sequence. HumanML3D-22 joints ARE the first 22 SMPL joints in the
SAME order, so once you have SMPL joint positions the mapping is just [:22]. This script does the
normalization (hip-center, y-up, downsample, trim) and writes the reference JSON that
genre_style.py --reference consumes.

Run LOCALLY on ~/Downloads/aist/ (this data is non-commercial → gallery/research track only).

Genre -> AIST label:  Jazz=gJS (Street Jazz) · Ballet=gJB (Ballet Jazz) · Hip-hop=gLH/gMH (Hip-hop)

Deps: numpy. SMPL forward (ADAPT) needs your existing SMPL/HumanML3D utils or `aist_plusplus` + smplx.
"""
import argparse, glob, json, os
import numpy as np

# SMPL joint order 0..23 == HumanML3D 0..21 for the first 22 (22,23 are hands -> dropped)
SMPL_TO_H22 = list(range(22))


def smpl_joints_from_pkl(path):
    """
    ADAPT: return SMPL joint positions (T,24,3) for an AIST++ sequence.
    AIST++ .pkl has 'smpl_poses' (T,72 axis-angle), 'smpl_trans' (T,3), 'smpl_scaling'.
    Getting joints requires an SMPL forward pass. Three ways, in order of convenience:
      (a) if your pkl already carries a joints array ('keypoints3d'/'joints3d' as SMPL-24), use it;
      (b) reuse the SMPL forward you already have in momask-codes-main / HumanML3D preprocessing;
      (c) pip install smplx + the SMPL neutral model, run forward with smpl_poses/trans.
    The placeholder below handles (a); wire (b)/(c) for the general case.
    """
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    for k in ("keypoints3d", "joints3d", "smpl_joints"):
        if k in d:
            j = np.asarray(d[k], dtype=np.float32)
            if j.ndim == 3 and j.shape[1] >= 24:
                return j[:, :24, :]
    raise NotImplementedError(
        f"{os.path.basename(path)}: no precomputed SMPL joints found (keys={list(d.keys())}). "
        "Wire an SMPL forward pass (see ADAPT in smpl_joints_from_pkl).")


def normalize(joints24, src_fps=60, out_fps=30, seconds=None, up="y", scale=None):
    """SMPL-24 -> HumanML3D-22, hip-centered, y-up, downsampled."""
    j = joints24[:, SMPL_TO_H22, :].astype(np.float32)          # (T,22,3) same joint order
    # unit scale: AIST++ world is often mm; auto-detect large magnitudes
    if scale is None:
        scale = 0.001 if np.nanmax(np.abs(j)) > 50 else 1.0
    j = j * scale
    # axis to y-up (SMPL/AIST world varies; expose a flag)
    if up == "z":   j = j[:, :, [0, 2, 1]]; j[..., 1] *= 1
    if up == "-y":  j[..., 1] *= -1
    # downsample
    if out_fps and out_fps < src_fps:
        step = max(1, round(src_fps / out_fps))
        j = j[::step]
    if seconds:
        j = j[: int(seconds * out_fps)]
    # hip-center each frame (pelvis = joint 0), matching the portal / genre_style convention
    j = j - j[:, 0:1, :]
    return j


def convert(pkl, out, **kw):
    joints24 = smpl_joints_from_pkl(pkl)
    ref = normalize(joints24, **kw)
    json.dump({"format": "HumanML3D-22", "axes": "y-up, hip-centered",
               "fps": kw.get("out_fps", 30), "frames": len(ref), "joints": ref.tolist()}, open(out, "w"))
    print(f"[aist] {os.path.basename(pkl)} -> {ref.shape} -> {out}")
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="AIST++ sequence .pkl (or a glob)")
    ap.add_argument("--out", default="reference.json")
    ap.add_argument("--out-fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=6.0, help="trim length (a representative phrase)")
    ap.add_argument("--up", default="y", choices=["y", "z", "-y"])
    ap.add_argument("--scale", type=float, default=None)
    a = ap.parse_args()
    paths = glob.glob(a.pkl)
    if not paths: raise SystemExit(f"no match: {a.pkl}")
    convert(paths[0], a.out, out_fps=a.out_fps, seconds=a.seconds, up=a.up, scale=a.scale)
    print("Then:  python genre_style.py --visitor visitor_clip.json --reference", a.out, "--out styled.npy")


if __name__ == "__main__":
    main()

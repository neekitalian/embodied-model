"""
Video -> genre reference clip (HumanML3D-22 JSON) for genre_style.py.

Run a dance video (e.g. an AIST Dance DB clip) through MediaPipe pose and export the same
HumanML3D-22 JSON the portal produces. Use it to batch-convert genre clips locally, then feed
them to genre_style.py as --reference. Reuses the capture+lift from stage12_capture_lift.py.

AIST DB videos are non-commercial → gallery/research track only.
Pick clips by genre from the filename:  gJS=Street Jazz · gJB=Ballet Jazz · gLH/gMH=Hip-hop.

Deps: pip install mediapipe opencv-python numpy
Usage: python video_to_reference.py --video gJB_sBM_c01_d05_mJB0_ch01.mp4 --out ballet_ref.json --seconds 6
       python video_to_reference.py --glob "aist/gLH_*.mp4" --out-dir refs/   # batch
"""
import argparse, glob, json, os
import numpy as np
from stage12_capture_lift import mp33_world_to_h22, to_yup_hipcentered, VectorEuro


def process(video_path, out, out_fps=30, seconds=None):
    import cv2, mediapipe as mp
    pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / out_fps))
    euro = VectorEuro((22, 3), freq=out_fps, min_cutoff=1.0, beta=0.01)
    frames, i = [], 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if res.pose_world_landmarks:
                    frames.append(euro(mp33_world_to_h22(res.pose_world_landmarks.landmark)))
                if seconds and len(frames) >= seconds * out_fps:
                    break
            i += 1
    finally:
        cap.release(); pose.close()
    if len(frames) < 2:
        raise RuntimeError(f"{video_path}: no full-body pose detected")
    seq = to_yup_hipcentered(np.stack(frames)).astype(np.float32)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"format": "HumanML3D-22", "axes": "y-up, hip-centered",
               "fps": out_fps, "frames": len(seq), "joints": seq.tolist()}, open(out, "w"))
    print(f"[video->ref] {os.path.basename(video_path)} -> {seq.shape} -> {out}")
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="single video file")
    ap.add_argument("--glob", help="glob of videos for batch mode")
    ap.add_argument("--out", default="reference.json", help="output (single mode)")
    ap.add_argument("--out-dir", default="refs", help="output dir (batch mode)")
    ap.add_argument("--out-fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=6.0, help="trim to a representative phrase")
    a = ap.parse_args()
    a.glob = os.path.expanduser(a.glob) if a.glob else a.glob      # so "~/Downloads/..." works
    a.video = os.path.expanduser(a.video) if a.video else a.video
    a.out, a.out_dir = os.path.expanduser(a.out), os.path.expanduser(a.out_dir)
    if a.glob:
        os.makedirs(a.out_dir, exist_ok=True)
        hits = sorted(glob.glob(a.glob, recursive=True))
        if not hits:
            raise SystemExit(f"no videos match {a.glob} - run: ls the folder to check the path/prefix")
        for v in hits:
            out = os.path.join(a.out_dir, os.path.splitext(os.path.basename(v))[0] + ".json")
            try: process(v, out, a.out_fps, a.seconds)
            except Exception as e: print(f"  skip {os.path.basename(v)}: {e}")
    elif a.video:
        process(a.video, a.out, a.out_fps, a.seconds)
        print("Then:  python genre_style.py --visitor visitor_clip.json --reference", a.out, "--out styled.npy")
    else:
        ap.error("give --video or --glob")


if __name__ == "__main__":
    main()

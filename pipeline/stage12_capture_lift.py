"""
Stage 1 + 2 - Webcam capture -> MediaPipe BlazePose (world 3D) -> HumanML3D 22-joint skeleton.

Drop this into momask-codes-main/ (e.g. as `pipeline/stage12_capture_lift.py`).
It is deliberately self-contained: it does NOT import your semantic_spectrum modules,
so you can verify capture + lift on their own before wiring in Stage 3 blending.

Deps:  pip install mediapipe opencv-python numpy
Usage: python stage12_capture_lift.py --seconds 6 --out visitor_clip.npy
       (press 'q' to stop early; press 'c' once in a rough T-pose to set a facing reference)

Output: np.float32 array of shape (T, 22, 3) in HumanML3D joint order, y-up, hip-centered.
"""
import argparse
import math
import numpy as np

# --- MediaPipe Pose (33) landmark indices we use ---
MP = dict(NOSE=0, L_SH=11, R_SH=12, L_EL=13, R_EL=14, L_WR=15, R_WR=16,
          L_HIP=23, R_HIP=24, L_KN=25, R_KN=26, L_AN=27, R_AN=28,
          L_HEEL=29, R_HEEL=30, L_FOOT=31, R_FOOT=32)

# --- HumanML3D / SMPL 22-joint order (body only, no hands) ---
H22 = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
       "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
       "neck", "left_collar", "right_collar", "head", "left_shoulder",
       "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
IDX = {name: i for i, name in enumerate(H22)}


class OneEuroFilter:
    """Per-scalar 1€ filter - kills MediaPipe jitter without the lag of a moving average."""
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


class VectorEuro:
    """1€ filter over an arbitrary-shaped array (one filter per element)."""
    def __init__(self, shape, **kw):
        self.filters = [OneEuroFilter(**kw) for _ in range(int(np.prod(shape)))]
        self.shape = shape

    def __call__(self, arr):
        flat = arr.reshape(-1)
        out = np.array([f(v) for f, v in zip(self.filters, flat)], dtype=np.float32)
        return out.reshape(self.shape)


def mp33_world_to_h22(lm):
    """
    Map MediaPipe pose_world_landmarks (33 x [x,y,z], meters, hip-centered) to HumanML3D 22.
    Direct joints are copied; the spine chain / collars / head are derived by interpolation.
    Good enough for an MVP; swap in a learned lifter (MotionAGFormer) later for cleaner 3D.
    """
    def p(i):
        return np.array([lm[i].x, lm[i].y, lm[i].z], dtype=np.float32)

    j = np.zeros((22, 3), dtype=np.float32)
    l_hip, r_hip = p(MP["L_HIP"]), p(MP["R_HIP"])
    l_sh, r_sh = p(MP["L_SH"]), p(MP["R_SH"])
    pelvis = (l_hip + r_hip) / 2.0
    neck = (l_sh + r_sh) / 2.0

    j[IDX["pelvis"]] = pelvis
    j[IDX["left_hip"]] = l_hip
    j[IDX["right_hip"]] = r_hip
    j[IDX["left_knee"]] = p(MP["L_KN"])
    j[IDX["right_knee"]] = p(MP["R_KN"])
    j[IDX["left_ankle"]] = p(MP["L_AN"])
    j[IDX["right_ankle"]] = p(MP["R_AN"])
    j[IDX["left_foot"]] = p(MP["L_FOOT"])
    j[IDX["right_foot"]] = p(MP["R_FOOT"])

    # spine chain: evenly interpolate pelvis -> neck
    j[IDX["spine1"]] = pelvis + 0.25 * (neck - pelvis)
    j[IDX["spine2"]] = pelvis + 0.50 * (neck - pelvis)
    j[IDX["spine3"]] = pelvis + 0.75 * (neck - pelvis)
    j[IDX["neck"]] = neck

    j[IDX["left_shoulder"]] = l_sh
    j[IDX["right_shoulder"]] = r_sh
    j[IDX["left_elbow"]] = p(MP["L_EL"])
    j[IDX["right_elbow"]] = p(MP["R_EL"])
    j[IDX["left_wrist"]] = p(MP["L_WR"])
    j[IDX["right_wrist"]] = p(MP["R_WR"])

    # collars: partway from neck toward each shoulder
    j[IDX["left_collar"]] = neck + 0.4 * (l_sh - neck)
    j[IDX["right_collar"]] = neck + 0.4 * (r_sh - neck)

    # head: extrapolate above neck through the nose direction
    nose = p(MP["NOSE"])
    j[IDX["head"]] = neck + 1.3 * (nose - neck)
    return j


def to_yup_hipcentered(seq):
    """
    MediaPipe world frame is x-right, y-DOWN, z-toward-camera. HumanML3D is y-UP.
    Flip Y (and Z to keep a right-handed frame), then re-center each frame on the pelvis.
    """
    seq = seq.copy()
    seq[..., 1] *= -1.0
    seq[..., 2] *= -1.0
    pelvis = seq[:, IDX["pelvis"], :][:, None, :]
    seq = seq - pelvis  # hip-centered; handle global root translation downstream
    return seq


def capture(seconds=6.0, out="visitor_clip.npy", camera=0, show=True):
    import cv2
    import mediapipe as mp

    pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    euro = VectorEuro((22, 3), freq=fps, min_cutoff=1.0, beta=0.01)

    frames, n_target = [], int(seconds * fps)
    print(f"[capture] recording ~{seconds}s at {fps:.0f} fps - 'q' to stop, 'c' to mark T-pose")
    try:
        while len(frames) < n_target:
            ok, img = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks:
                j = mp33_world_to_h22(res.pose_world_landmarks.landmark)
                frames.append(euro(j))
            if show:
                cv2.putText(img, f"{len(frames)}/{n_target}", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Stage 1 capture", img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        pose.close()

    if not frames:
        raise RuntimeError("No pose detected - check lighting / that a full body is in frame.")
    seq = to_yup_hipcentered(np.stack(frames))  # (T,22,3)
    np.save(out, seq.astype(np.float32))
    print(f"[capture] saved {seq.shape} -> {out}")
    return seq


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default="visitor_clip.npy")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    capture(a.seconds, a.out, a.camera, show=not a.no_show)

"""
Build the Round 2 perceptual-study stimulus set for the Research Portal.

For each of the 4 source actions it: copies the source video into research/sources/, extracts the
skeleton (MediaPipe -> HumanML3D-22), renders the original skeleton video, then for every genre x
allocation level grafts the real genre motion at that CONSTANT allocation and renders a stimulus video.
Finally it writes research/stimuli.json (the manifest the Research Portal reads).

  4 actions x 3 genres x 4 levels = 48 stimulus videos (+ 4 originals).

The allocation level is the graft's identity<->genre weight held CONSTANT across the clip (gain = 0), so
0.20 / 0.40 / 0.60 / 0.80 is a clean experimental manipulation rather than a per-frame-adaptive amount.

Run on your Mac (needs the pipeline deps + ffmpeg for mp4):
  pip install mediapipe opencv-python matplotlib
  # ffmpeg:  brew install ffmpeg   (without it, videos fall back to .gif)
  python make_stimuli.py --videos ~/Desktop/Videos
  python make_stimuli.py --videos ~/Desktop/Videos --force   # regenerate everything

Then commit research/sources + research/stimuli + research/stimuli.json so Vercel serves them.
"""
import argparse, json, os, shutil, subprocess
import numpy as np

import genre_style as gs
from genre_graft import graft

# (key, source-file-on-your-Desktop, clean-name-in-app, label, type, social, repetition)
ACTIONS = [
    ("waving",  "waving.mp4",         "waving.mp4",             "waving hello",         "social / repetitive",        "social",     "repetitive"),
    ("shaking", "shaking hands.mp4",  "shaking_hands.mp4",      "shaking hands",        "social / non-repetitive",    "social",     "non-repetitive"),
    ("walking", "walk_riku_cam1.mp4", "walking.mp4",            "walking",              "non-social / repetitive",    "non-social", "repetitive"),
    ("ladder",  "sorting.mp4",        "working_on_ladder.mp4",  "working on a ladder",  "non-social / non-repetitive","non-social", "non-repetitive"),
]
GENRES = [("ballet", "ballet"), ("jazz", "jazz"), ("hip-hop", "hiphop")]
LEVELS = [0.20, 0.40, 0.60, 0.80]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(ROOT, "research", "sources")
STIM_DIR = os.path.join(ROOT, "research", "stimuli")
REFS = os.path.join(HERE, "refs")


def render(motion, out_path, fps=30):
    """Render a HumanML3D-22 clip as a clean 2D skeleton video (dark bg), matching the portal look.
    Writes mp4 via ffmpeg; falls back to gif if ffmpeg is unavailable (note: .gif is gitignored)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from hml_skeleton import BONES
    seq = np.asarray(motion, dtype=np.float32)
    P = seq[:, :, [0, 1]]                                            # front view: x horizontal, y up
    lo, hi = P.reshape(-1, 2).min(0), P.reshape(-1, 2).max(0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2 + 1e-3
    fig = plt.figure(figsize=(4, 5.2)); fig.patch.set_facecolor("#0b0b10")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    def frame(i):
        ax.clear(); ax.set_facecolor("#0b0b10"); ax.axis("off")
        p = P[i]
        for a_, b_ in BONES:
            ax.plot([p[a_, 0], p[b_, 0]], [p[a_, 1], p[b_, 1]], c="#6ad07f", lw=3, solid_capstyle="round")
        ax.scatter(p[:, 0], p[:, 1], s=16, c="#ffd24d", zorder=3)
        ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad); ax.set_aspect("equal")
        return []

    anim = FuncAnimation(fig, frame, frames=len(seq), interval=1000/fps, blit=False)
    try:
        anim.save(out_path, writer=FFMpegWriter(fps=fps, codec="libx264", bitrate=1400))
        plt.close(fig); return out_path
    except Exception as e:
        gif = os.path.splitext(out_path)[0] + ".gif"
        print(f"    (ffmpeg unavailable: {e}; wrote {os.path.basename(gif)} instead - install ffmpeg for mp4)")
        anim.save(gif, writer=PillowWriter(fps=fps)); plt.close(fig); return gif


def copy_web(src, dst):
    """Transcode a source video to browser-friendly H.264 (yuv420p + faststart) so it plays inline on
    the web. Falls back to a raw copy if ffmpeg is missing (some codecs like HEVC then won't play)."""
    try:
        subprocess.run(["ffmpeg", "-y", "-i", src, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-movflags", "+faststart", dst],
                       check=True, capture_output=True)
    except Exception:
        shutil.copyfile(src, dst)


def extract(video_path, out_json, seconds):
    import video_to_reference as v2r
    v2r.process(video_path, out_json, out_fps=30, seconds=seconds)
    return gs.load_clip(out_json)


def build_manifest():
    stimuli = []
    for key, _src, clean, label, atype, social, rep in ACTIONS:
        for gdisp, gid in GENRES:
            for lv in LEVELS:
                pct = int(round(lv * 100)); sid = f"{key}_{gid}_{pct}"
                stimuli.append({
                    "stimulus_id": sid, "source_video_filename": clean,
                    "source_video_path": f"/research/sources/{clean}",
                    "original_video_path": f"/research/sources/{key}_original.mp4",
                    "original_json_path": f"/research/sources/{key}_original.json",
                    "action_key": key, "action_label": label, "action_type": atype,
                    "social_dimension": social, "repetition_dimension": rep,
                    "genre": gdisp, "allocation_level": lv, "allocation_pct": pct,
                    "output_video_path": f"/research/stimuli/{sid}.mp4",
                    "output_json_path": f"/research/stimuli/{sid}.json",
                })
    return {"study": "Unnoticed Dance - Round 2 perceptual study",
            "actions": [{"key": a[0], "source": a[2], "orig_source": a[1], "label": a[3],
                         "type": a[4], "social": a[5], "repetition": a[6]} for a in ACTIONS],
            "genres": [g[0] for g in GENRES], "levels": LEVELS, "n_stimuli": len(stimuli), "stimuli": stimuli}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="~/Desktop/Videos", help="folder holding the 4 source videos")
    ap.add_argument("--seconds", type=float, default=None, help="trim each source to N seconds (optional)")
    ap.add_argument("--force", action="store_true", help="regenerate outputs that already exist")
    a = ap.parse_args()
    vdir = os.path.expanduser(a.videos)
    os.makedirs(SRC_DIR, exist_ok=True); os.makedirs(STIM_DIR, exist_ok=True)
    refs = {g: gs.load_clip(os.path.join(REFS, f"{g}.json")) for g, _ in GENRES}

    for key, src, clean, label, *_ in ACTIONS:
        src_path = os.path.join(vdir, src)
        if not os.path.exists(src_path):
            raise SystemExit(f"missing source video: {src_path}  (put the 4 videos in {vdir})")
        print(f"[{key}] {src} -> {clean}")
        copy_web(src_path, os.path.join(SRC_DIR, clean))            # transcode to web-safe H.264 under a clean name

        orig_json = os.path.join(SRC_DIR, f"{key}_original.json")
        orig_vid = os.path.join(SRC_DIR, f"{key}_original.mp4")
        if a.force or not os.path.exists(orig_json):
            visitor = extract(src_path, orig_json, a.seconds)
            render(visitor, orig_vid)
        else:
            visitor = gs.load_clip(orig_json)
        print(f"    original: {visitor.shape}")

        for gdisp, gid in GENRES:
            for lv in LEVELS:
                pct = int(round(lv * 100)); sid = f"{key}_{gid}_{pct}"
                out_json = os.path.join(STIM_DIR, f"{sid}.json")
                out_vid = os.path.join(STIM_DIR, f"{sid}.mp4")
                if not a.force and (os.path.exists(out_vid) or os.path.exists(os.path.splitext(out_vid)[0] + ".gif")):
                    continue
                styled, _ = graft(visitor, refs[gdisp], floor=lv, gain=0.0, cap=1.0)   # constant allocation = lv
                json.dump({"format": "HumanML3D-22", "fps": 30, "frames": len(styled),
                           "joints": styled.astype(np.float32).tolist()}, open(out_json, "w"))
                render(styled, out_vid)
                print(f"    {sid}  (level {lv:.2f})")

    json.dump(build_manifest(), open(os.path.join(ROOT, "research", "stimuli.json"), "w"), indent=1)
    print(f"[done] wrote research/stimuli.json (48 stimuli). Commit research/ so Vercel serves it.")


if __name__ == "__main__":
    main()

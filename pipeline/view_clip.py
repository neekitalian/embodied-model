"""
Eyeball a captured/blended clip as a 3D stick figure — BEFORE Unity exists.
Position-based (no rotation math), so it shows exactly what's in the .npy.

Deps:  pip install matplotlib numpy
Usage: python view_clip.py visitor_clip.npy               # interactive
       python view_clip.py visitor_clip.npy --save out.gif # write a gif to share
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from hml_skeleton import BONES, JOINT_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--save", default=None, help="write a .gif instead of showing a window")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="draw every Nth frame (speed)")
    a = ap.parse_args()

    if a.save:
        matplotlib.use("Agg")

    seq = np.load(a.clip).astype(np.float32)[::a.stride]   # (T,22,3)
    assert seq.ndim == 3 and seq.shape[1:] == (22, 3), f"expected (T,22,3), got {seq.shape}"
    T = len(seq)

    # fixed equal-aspect bounds over the whole clip
    lo, hi = seq.reshape(-1, 3).min(0), seq.reshape(-1, 3).max(0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2 + 1e-3

    fig = plt.figure(figsize=(5, 7))
    ax = fig.add_subplot(111, projection="3d")

    def frame(i):
        ax.clear()
        p = seq[i]
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=14, c="#5fa9d8")     # note: y-up -> plot y as vertical
        for a_, b_ in BONES:
            ax.plot([p[a_, 0], p[b_, 0]], [p[a_, 2], p[b_, 2]], [p[a_, 1], p[b_, 1]],
                    c="#e8e8ee", lw=2)
        ax.set_xlim(ctr[0]-rad, ctr[0]+rad)
        ax.set_ylim(ctr[2]-rad, ctr[2]+rad)
        ax.set_zlim(ctr[1]-rad, ctr[1]+rad)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"frame {i*a.stride}/{T*a.stride}")
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)")
        return []

    anim = FuncAnimation(fig, frame, frames=T, interval=1000/a.fps, blit=False)
    if a.save:
        from matplotlib.animation import PillowWriter
        anim.save(a.save, writer=PillowWriter(fps=a.fps))
        print(f"[view] wrote {a.save} ({T} frames)")
    else:
        plt.show()


if __name__ == "__main__":
    main()

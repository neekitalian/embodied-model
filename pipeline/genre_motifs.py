"""
Genre MOTIF matching -> similarity-gated expressiveness + genre allocation.

The step the visitor asked for, grounded in this project's methodology:
  * a genre's SIGNATURE MOVEMENTS = a motif bank: sliding windows of its AIST reference, each encoded
    as a per-zone feature vector (velocity / expansiveness / vertical-bias), UNIT-NORMALIZED so we match
    the *shape* of the movement (which zones lead, expansiveness ratios) independent of raw energy.
    -> this is the "motion-to-motion selection on the visitor's own features" from the top-of-chat doc.
  * RECOGNITION / FITTING: slide the same encoder over the visitor, take the best cosine match per window
    -> a per-frame similarity curve = where your motion already resembles that genre.
  * EXPRESSIVENESS: crossfade identity <-> full-genre transfer with a per-frame weight that rises with
    similarity, so the output looks MORE like the genre exactly where you already lean that way, and stays
    you elsewhere. Across genres the boost is RELATIVE (softmax-ish), so the genre you most resemble is
    pushed hardest -> also gives genre ALLOCATION ("the genre your body dances").

This is the Motion-Puzzle (per-zone) + retrieval + zone-alpha design, one ADAPT away from the real
semantic_spectrum encoder (swap `sig()` for its per-zone latent).

Deps: numpy. Reuses genre_style.py (ZONES, zone_style_features, transfer).
"""
import argparse, glob, json, os
import numpy as np
import genre_style as gs


def sig(clip, a, b):
    """Unit-normalized per-zone feature vector for window [a,b)."""
    v = []
    for js in gs.ZONES.values():
        v.extend(gs.zone_style_features(clip[a:b], js).tolist())
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def motif_bank(ref, win=24, stride=12):
    if len(ref) < win:
        return [sig(ref, 0, len(ref))]
    return [sig(ref, a, a + win) for a in range(0, len(ref) - win + 1, stride)]


def sim_curve(visitor, ref, win=24, stride=12):
    """Per-frame similarity (0..1) of the visitor to the genre's motif bank, plus its mean."""
    bank = motif_bank(ref, win, stride)
    T = len(visitor)
    centers, vals = [], []
    last = max(1, T - win + 1)
    for a in range(0, last, stride):
        vs = sig(visitor, a, a + win)
        m = max(float(vs @ b) for b in bank)          # cosine (both unit-norm)
        centers.append(a + win / 2.0); vals.append(max(0.0, m))
    if len(centers) >= 2:
        curve = np.interp(np.arange(T), centers, vals).astype(np.float32)
    else:
        curve = np.full(T, vals[0] if vals else 0.0, dtype=np.float32)
    for t in range(1, T):                              # smooth
        curve[t] = 0.3 * curve[t] + 0.7 * curve[t - 1]
    return curve, float(curve.mean())


def enhance(visitor, ref, base_alpha=0.45, gain=0.6, cap=0.9):
    """Similarity-gated crossfade identity <-> full-genre transfer. Returns (styled_clip, mean_similarity)."""
    styled = gs.transfer(visitor, ref, {z: cap for z in gs.ZONES})
    curve, mean = sim_curve(visitor, ref)
    T = min(len(visitor), len(styled))
    out = np.asarray(visitor[:T], dtype=np.float32).copy()
    for t in range(T):
        w = min(cap, base_alpha + gain * curve[t])
        out[t] = (1 - w) * visitor[t] + w * styled[t]
    return out, mean


def allocate(visitor, refs):
    """refs: {genre: clip}. Returns {genre: mean_similarity} (which genre the body most resembles)."""
    return {g: sim_curve(visitor, r)[1] for g, r in refs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visitor", required=True)
    ap.add_argument("--refs-dir", default="refs")
    ap.add_argument("--genre", help="enhance this genre; omit to just --report allocation")
    ap.add_argument("--alpha", type=float, default=0.45, help="similarity floor")
    ap.add_argument("--out", default="enhanced.npy")
    ap.add_argument("--report", action="store_true", help="print per-genre similarity (allocation)")
    a = ap.parse_args()
    a.visitor = os.path.expanduser(a.visitor); a.refs_dir = os.path.expanduser(a.refs_dir)
    visitor = gs.load_clip(a.visitor)
    refs = {os.path.splitext(os.path.basename(p))[0]: gs.load_clip(p)
            for p in sorted(glob.glob(os.path.join(a.refs_dir, "*.json")) + glob.glob(os.path.join(a.refs_dir, "*.npy")))}
    if a.report or not a.genre:
        scores = allocate(visitor, refs)
        tot = sum(scores.values()) or 1
        print("[allocation] the genre your body most resembles:")
        for g, s in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"  {g:12s} similarity {s:.3f}  ({100*s/tot:4.1f}% of total)")
    if a.genre:
        ref = refs.get(a.genre) or next((r for n, r in refs.items() if a.genre.lower() in n.lower()), None)
        if ref is None:
            raise SystemExit(f"no reference for '{a.genre}' in {a.refs_dir}")
        out, mean = enhance(visitor, ref, base_alpha=a.alpha)
        np.save(a.out, out.astype(np.float32))
        print(f"[enhance] {a.genre}: similarity {mean:.3f} -> {out.shape} -> {a.out}  (preview: python view_clip.py {a.out})")


if __name__ == "__main__":
    main()

"""
Genre MOTIF matching -> similarity-gated expressiveness + genre allocation.

The step the visitor asked for, grounded in this project's methodology and now sharpened with the
feature set from Hamscher et al., "Dance Style Classification using Laban-Inspired and Frequency-Domain
Motion Features" (arXiv 2511.20469), evaluated on the same AIST genres we use (Street Jazz gJS, Ballet
Jazz gJB, Middle Hip-hop gMH):

  * a genre's SIGNATURE MOVEMENTS = a motif bank: sliding windows of its AIST reference, each encoded as
    a LABAN + FREQUENCY descriptor (see _laban_desc):
      - SHAPE  : hip-relative extremity distances e_hands / e_feet / e_cross (mean + spread), kept in real
                 units so expansiveness (ballet-open vs hip-hop-tucked) is a feature, not noise.
      - SPACE  : torso yaw phi = atan2(o_x, o_z) from the chest-hip vector -> turning / orientation.
      - EFFORT : per-zone velocity + acceleration -> sharpness (hip-hop hits) vs legato (ballet).
      - RHYTHM : FFT dominant frequency + spectral energy of the extremity signals -> the genre's beat
                 fingerprint (the periodicity the old geometric proxy ignored entirely).
    Descriptors are STANDARDIZED per reference (z-scored against that genre's own window statistics) so no
    raw-unit dimension dominates the cosine, while expansiveness still shows up as large z-scores.
  * RECOGNITION / FITTING: slide the same encoder over the visitor, z-score with the genre's stats, take
    the best cosine match per window -> a per-frame similarity curve = where your motion already resembles
    that genre's signature moves.
  * EXPRESSIVENESS: crossfade identity <-> full-genre transfer with a per-frame weight that rises with
    similarity, so the output looks MORE like the genre exactly where you already lean that way, and stays
    you elsewhere. Across genres the boost is RELATIVE, so the genre you most resemble is pushed hardest
    -> also gives genre ALLOCATION ("the genre your body dances").

This is the Motion-Puzzle (per-zone) + retrieval + zone-alpha design with an interpretable, literature-
backed encoder in place of the hand-built geometric proxy -- one ADAPT away from a learned semantic_spectrum
(swap _laban_desc for its per-zone latent).

Deps: numpy. Reuses genre_style.py (ZONES, transfer) and the HumanML3D-22 joint map.
"""
import argparse, glob, os
import numpy as np
import genre_style as gs
from hml_skeleton import IDX

WIN, STRIDE = 24, 12
# joint shortcuts on the HumanML3D-22 skeleton
_HIP, _CHEST = IDX["pelvis"], IDX["spine3"]
_LW, _RW = IDX["left_wrist"], IDX["right_wrist"]
_LF, _RF = IDX["left_foot"], IDX["right_foot"]


def _laban_desc(clip, a, b):
    """Raw Laban + frequency descriptor for window [a,b). Returns a 1-D feature vector (un-standardized).
    Layout: 6 shape + 2 space + 2*|zones| effort + 4 frequency."""
    W = np.asarray(clip[a:b], dtype=np.float32)
    n = len(W)
    hip = W[:, _HIP, :]
    R = W - hip[:, None, :]                                   # hip-relative (paper: features about the hip)

    # SHAPE -- extremity distances, real units (mean + spread)
    hands = np.linalg.norm(R[:, _RW] - R[:, _LW], axis=-1)
    feet = np.linalg.norm(R[:, _RF] - R[:, _LF], axis=-1)
    cross = 0.5 * (np.linalg.norm(R[:, _RW] - R[:, _LF], axis=-1)
                   + np.linalg.norm(R[:, _LW] - R[:, _RF], axis=-1))
    shape = [hands.mean(), hands.std(), feet.mean(), feet.std(), cross.mean(), cross.std()]

    # SPACE -- torso yaw phi = atan2(o_x, o_z), o = chest - hip (R already hip-relative)
    o = R[:, _CHEST, :]
    yaw = np.unwrap(np.arctan2(o[:, 0], o[:, 2] + 1e-8)) if n > 1 else np.zeros(1)
    yaw_rate = np.diff(yaw) if n > 1 else np.zeros(1)
    space = [float(yaw.std()), float(np.abs(yaw_rate).mean())]

    # EFFORT -- per-zone velocity + acceleration
    effort = []
    for js in gs.ZONES.values():
        z = R[:, js, :]
        vel = np.linalg.norm(np.diff(z, axis=0), axis=-1).mean() if n > 1 else 0.0
        acc = np.linalg.norm(np.diff(z, n=2, axis=0), axis=-1).mean() if n > 2 else 0.0
        effort.extend([float(vel), float(acc)])

    # RHYTHM -- FFT dominant frequency (normalized bin) + spectral energy of the extremity signals
    freq = []
    for s in (hands, feet):
        s = s - s.mean()
        if n >= 4:
            F = np.abs(np.fft.rfft(s)); F[0] = 0.0
            dom = (np.argmax(F) / (len(F) - 1)) if len(F) > 1 and F.sum() > 1e-8 else 0.0
            energy = float((F ** 2).sum())
        else:
            dom, energy = 0.0, 0.0
        freq.extend([dom, energy])

    return np.array(shape + space + effort + freq, dtype=np.float32)


def _standardizer(descs):
    """Mean/std over a set of raw descriptors -> a (mu, inv_sigma) pair for z-scoring."""
    M = np.mean(descs, axis=0)
    S = np.std(descs, axis=0)
    S[S < 1e-6] = 1.0
    return M, 1.0 / S


def _zscore(desc, stats):
    mu, inv = stats
    return (desc - mu) * inv


def sig(clip, a, b, stats=None):
    """Descriptor for window [a,b). With `stats` (mu, inv_sigma) the vector is z-scored and L2-normalized
    for cosine matching; without, the raw Laban+FFT vector is returned."""
    d = _laban_desc(clip, a, b)
    if stats is None:
        return d
    v = _zscore(d, stats)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-8 else v


def motif_bank(ref, win=WIN, stride=STRIDE, salient=True, keep=0.5):
    """Genre signature-move bank. Returns (bank, stats): bank = list of z-scored+unit-norm descriptors,
    stats = (mu, inv_sigma) for the reference's own windows. salient=True keeps only the most DISTINCTIVE
    windows -- those whose descriptor sits farthest from the genre mean (largest z-score norm) -- so the
    bank is the genre's signature moves, not neutral filler."""
    if len(ref) < win:
        raw = [_laban_desc(ref, 0, len(ref))]
    else:
        raw = [_laban_desc(ref, a, a + win) for a in range(0, len(ref) - win + 1, stride)]
    stats = _standardizer(raw)
    bank = []
    for d in raw:
        v = _zscore(d, stats)
        nrm = np.linalg.norm(v)
        bank.append(v / nrm if nrm > 1e-8 else v)
    if not salient or len(bank) <= 3:
        return bank, stats
    # distinctiveness = distance from the genre mean; z-scored mean is ~0, so the z-score norm ranks it
    dist = np.array([float(np.linalg.norm(_zscore(d, stats))) for d in raw])
    k = max(3, int(round(keep * len(bank))))
    keep_idx = sorted(np.argsort(dist)[::-1][:k])
    return [bank[i] for i in keep_idx], stats


def curve_from_bank(visitor, bank, stats, win=WIN, stride=STRIDE):
    """Per-frame similarity (0..1) of the visitor to a PRE-FIT motif bank, plus its mean.
    Lets a caller fit a genre once (motif_bank) and reuse it -- see exhibition.GenreModel."""
    T = len(visitor)
    centers, vals = [], []
    last = max(1, T - win + 1)
    for a in range(0, last, stride):
        vs = sig(visitor, a, a + win, stats)                 # z-scored with the genre's own stats
        m = max(float(vs @ b) for b in bank)                 # cosine (both unit-norm)
        centers.append(a + win / 2.0); vals.append(max(0.0, m))
    if len(centers) >= 2:
        curve = np.interp(np.arange(T), centers, vals).astype(np.float32)
    else:
        curve = np.full(T, vals[0] if vals else 0.0, dtype=np.float32)
    for t in range(1, T):                                    # smooth
        curve[t] = 0.3 * curve[t] + 0.7 * curve[t - 1]
    return curve, float(curve.mean())


def sim_curve(visitor, ref, win=WIN, stride=STRIDE):
    """Per-frame similarity (0..1) of the visitor to the genre's motif bank, plus its mean."""
    bank, stats = motif_bank(ref, win, stride)
    return curve_from_bank(visitor, bank, stats, win, stride)


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

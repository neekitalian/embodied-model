"""
unnoticed_dance() -- the single exhibition entry point.

One call runs the whole installation flow on a visitor's captured motion:

  1. READ      the visitor's daily action (a HumanML3D-22 clip from the portal / camera)
  2. COMPARE   it against each genre's DANCE FEATURES (the Laban + FFT motif bank in genre_motifs)
  3. ALLOCATE  the genre + per-frame feature weights the body most resembles
  4. EDIT      re-express the movement with the winning genre's FITTED model, gated by similarity so
               identity is protected and genre character rises only where the body already leans that way

The "model" is GenreModel, FIT from each reference clip. `fit()` is where learning happens: from the
reference data we derive
  * the COMPARISON basis -- a motif bank of signature-move descriptors + per-genre z-score statistics,
  * the EDIT basis       -- the reference's own per-zone style (expansiveness / vertical bias / effort),
                            which genre_style.transfer transplants onto the visitor.
Today this is lightweight, data-derived learning over the 3 AIST references (no labels needed).
GenreModel.fit is the single seam to swap in a trained classifier / learned embedding once a labeled
daily-plus-dance dataset exists -- the four-step flow above does not change.

Deps: numpy. Reuses genre_style.py + genre_motifs.py.

CLI:
  python exhibition.py --visitor clip.json --refs-dir refs --out edited.npy
"""
import argparse, glob, json, os
import numpy as np
import genre_style as gs
import genre_motifs as gm


class GenreModel:
    """A per-genre model FIT from its reference dance clip(s)."""

    def __init__(self, name, reference, bank, stats):
        self.name = name
        self.reference = np.asarray(reference, dtype=np.float32)
        self.bank = bank                       # signature-move descriptors (unit vectors)
        self.stats = stats                     # (mu, inv_sigma) z-score stats for this genre

    @classmethod
    def fit(cls, name, reference, win=gm.WIN, stride=gm.STRIDE):
        """LEARN the genre from its reference: signature-move bank + z-score statistics."""
        ref = np.asarray(reference, dtype=np.float32)
        bank, stats = gm.motif_bank(ref, win, stride)
        return cls(name, ref, bank, stats)

    def compare(self, visitor, win=gm.WIN, stride=gm.STRIDE):
        """COMPARE: per-frame similarity curve of the visitor to this genre's signature moves + its mean."""
        return gm.curve_from_bank(visitor, self.bank, self.stats, win, stride)

    def edit(self, visitor, base_alpha=0.45, gain=0.6, cap=0.9):
        """EDIT: identity <-> genre crossfade, weighted per frame by the learned similarity.
        The genre target comes from THIS model's reference style (genre_style.transfer)."""
        visitor = np.asarray(visitor, dtype=np.float32)
        styled = gs.transfer(visitor, self.reference, {z: cap for z in gs.ZONES})
        curve, mean = self.compare(visitor)
        T = min(len(visitor), len(styled))
        out = visitor[:T].copy()
        for t in range(T):
            w = min(cap, base_alpha + gain * curve[t])       # boost where the body already resembles the genre
            out[t] = (1 - w) * visitor[t] + w * styled[t]
        return out, mean


def fit_models(references, win=gm.WIN, stride=gm.STRIDE):
    """Fit one GenreModel per reference clip. references: {genre: clip}."""
    return {g: GenreModel.fit(g, r, win, stride) for g, r in references.items()}


def unnoticed_dance(visitor, references, base_alpha=0.45, gain=0.6, cap=0.9, models=None):
    """The exhibition function: read -> compare -> allocate -> edit.

    visitor    : HumanML3D-22 clip (T,22,3), the daily action.
    references : {genre: clip} dance references (or pass pre-fit `models` to skip fitting).
    Returns (edited_clip, report) where report = {genre, scores, allocation, similarity, frames}.
    """
    visitor = np.asarray(visitor, dtype=np.float32)          # 1. READ
    if models is None:
        models = fit_models(references)                      #    (learn genre models from the dance data)

    curves = {g: m.compare(visitor) for g, m in models.items()}   # 2. COMPARE with dance features
    scores = {g: float(curves[g][1]) for g in models}
    total = sum(scores.values()) or 1.0
    allocation = {g: scores[g] / total for g in scores}      # 3. ALLOCATE (which genre the body dances)
    genre = max(scores, key=scores.get)

    edited, sim = models[genre].edit(visitor, base_alpha, gain, cap)   # 4. EDIT with the fitted model
    report = {"genre": genre, "scores": scores, "allocation": allocation,
              "similarity": float(sim), "frames": int(len(edited))}
    return edited, report


def _load_refs(refs_dir):
    refs = {}
    for p in sorted(glob.glob(os.path.join(refs_dir, "*.json")) + glob.glob(os.path.join(refs_dir, "*.npy"))):
        refs[os.path.splitext(os.path.basename(p))[0]] = gs.load_clip(p)
    return refs


def main():
    ap = argparse.ArgumentParser(description="Unnoticed Dance: read -> compare -> allocate -> edit.")
    ap.add_argument("--visitor", required=True, help="visitor clip (.json from portal / video, or .npy)")
    ap.add_argument("--refs-dir", default="refs", help="folder of genre reference clips")
    ap.add_argument("--alpha", type=float, default=0.45, help="identity<->genre floor (0..1)")
    ap.add_argument("--out", default="edited.npy", help="edited clip output (.npy)")
    a = ap.parse_args()
    a.visitor, a.refs_dir, a.out = map(os.path.expanduser, (a.visitor, a.refs_dir, a.out))

    visitor = gs.load_clip(a.visitor)
    refs = _load_refs(a.refs_dir)
    if not refs:
        raise SystemExit(f"no genre references in {a.refs_dir}")

    edited, rep = unnoticed_dance(visitor, refs, base_alpha=a.alpha)
    np.save(a.out, edited.astype(np.float32))

    print("[unnoticed_dance] read -> compare -> allocate -> edit")
    print(f"  1. READ      visitor {visitor.shape}")
    print(f"  2/3. COMPARE + ALLOCATE (the genre your body most resembles):")
    for g, s in sorted(rep["scores"].items(), key=lambda kv: -kv[1]):
        star = "  <- danced" if g == rep["genre"] else ""
        print(f"       {g:12s} similarity {s:.3f}  ({100*rep['allocation'][g]:4.1f}%)" + star)
    print(f"  4. EDIT      as {rep['genre']} (mean similarity {rep['similarity']:.3f}) "
          f"-> {edited.shape} -> {a.out}")
    print(f"  Preview: python view_clip.py {a.out}   |   Stream: python run_local.py --visitor {a.visitor} "
          f"--genre {rep['genre']} --enhance --stream")


if __name__ == "__main__":
    main()

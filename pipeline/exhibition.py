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


def unnoticed_dance(visitor, references, base_alpha=0.45, gain=0.6, cap=0.9, models=None, graft=False):
    """The exhibition function: read -> compare -> allocate -> edit.

    visitor    : HumanML3D-22 clip (T,22,3), the daily action.
    references : {genre: clip} dance references (or pass pre-fit `models` to skip fitting).
    graft      : if True, the EDIT splices the reference's REAL genre motion (genre_graft) into the
                 visitor where they align - looks genuinely like the genre - instead of the geometric
                 transfer; gated by the same similarity curve. Root is kept either way.
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

    if graft == "mix":                                       # 4. EDIT: GENRE MIX - blend all genres per frame
        import genre_graft
        edited, means = genre_graft.graft_mix(visitor, references, curves={g: curves[g][0] for g in models},
                                              floor=base_alpha, gain=gain)
        sim, edit = means[genre], "graft-mix"
    elif graft:                                              # single-genre graft of the real reference motion
        import genre_graft
        edited, sim = genre_graft.graft(visitor, references[genre], floor=base_alpha,
                                        gain=gain, cap=max(cap, 0.95), curve=curves[genre][0])
        edit = "graft"
    else:
        edited, sim = models[genre].edit(visitor, base_alpha, gain, cap)
        edit = "transfer"
    report = {"genre": genre, "scores": scores, "allocation": allocation,
              "similarity": float(sim), "frames": int(len(edited)), "edit": edit}
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
    ap.add_argument("--model", help="trained proto encoder checkpoint (.pt from train_proto.py); "
                                    "uses the LEARNED embedding for compare/allocate (needs torch)")
    ap.add_argument("--protos-dir", help="labeled clips folder (data/<genre>/*) to build each genre's "
                                         "prototype from ALL its clips instead of the single reference")
    ap.add_argument("--graft", action="store_true", help="EDIT by grafting the reference's real genre "
                                                         "motion (genre_graft) instead of geometric transfer")
    ap.add_argument("--graft-mix", action="store_true", help="EDIT by grafting a PER-FRAME MIX of all "
                                                             "genres weighted by similarity (your personal blend)")
    ap.add_argument("--out", default="edited.npy", help="edited clip output (.npy)")
    a = ap.parse_args()
    a.visitor, a.refs_dir, a.out = map(os.path.expanduser, (a.visitor, a.refs_dir, a.out))

    visitor = gs.load_clip(a.visitor)
    refs = _load_refs(a.refs_dir)
    if not refs:
        raise SystemExit(f"no genre references in {a.refs_dir}")

    # standing model: default to the best validated checkpoint when present (and torch is available)
    if not a.model:
        import importlib.util
        if importlib.util.find_spec("torch"):
            for cand in ("proto_pretrain2.pt", "proto_encoder.pt"):
                if os.path.exists(cand):
                    a.model = cand
                    if not a.protos_dir and os.path.isdir("data"):
                        a.protos_dir = "data"
                    break

    models = None
    if a.model:
        from proto_infer import build_learned_models
        proto_clips = None
        if a.protos_dir:
            pd = os.path.expanduser(a.protos_dir)
            proto_clips = {os.path.basename(d): [gs.load_clip(p) for p in
                           sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*.npy")))]
                           for d in sorted(glob.glob(os.path.join(pd, "*"))) if os.path.isdir(d)}
            proto_clips = {g: c for g, c in proto_clips.items() if c}
        models = build_learned_models(refs, os.path.expanduser(a.model), proto_clips=proto_clips)
        print(f"[unnoticed_dance] using LEARNED prototypical encoder: {a.model}"
              + (f" (prototypes from {a.protos_dir})" if proto_clips else ""))

    graft_mode = "mix" if a.graft_mix else a.graft
    edited, rep = unnoticed_dance(visitor, refs, base_alpha=a.alpha, models=models, graft=graft_mode)
    np.save(a.out, edited.astype(np.float32))

    print("[unnoticed_dance] read -> compare -> allocate -> edit")
    print(f"  1. READ      visitor {visitor.shape}")
    print(f"  2/3. COMPARE + ALLOCATE (the genre your body most resembles):")
    for g, s in sorted(rep["scores"].items(), key=lambda kv: -kv[1]):
        star = "  <- danced" if g == rep["genre"] else ""
        print(f"       {g:12s} similarity {s:.3f}  ({100*rep['allocation'][g]:4.1f}%)" + star)
    print(f"  4. EDIT      as {rep['genre']} via {rep['edit']} (mean similarity {rep['similarity']:.3f}) "
          f"-> {edited.shape} -> {a.out}")
    print(f"  Preview: python view_clip.py {a.out}   |   Stream: python run_local.py --visitor {a.visitor} "
          f"--genre {rep['genre']} --enhance --stream")


if __name__ == "__main__":
    main()

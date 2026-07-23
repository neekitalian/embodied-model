"""
Use a trained prototypical ST-GCN encoder in the exhibition pipeline.

build_learned_models(references, ckpt) returns {genre: LearnedGenreModel}, each exposing the same
.compare(visitor) / .edit(visitor) interface as exhibition.GenreModel -- so it drops straight into
exhibition.unnoticed_dance(visitor, references, models=...):

    from proto_infer import build_learned_models
    import exhibition
    models = build_learned_models(refs, "proto_encoder.pt")
    edited, report = exhibition.unnoticed_dance(visitor, refs, models=models)

What changes vs. the hand-feature path: COMPARE + ALLOCATE now use the LEARNED embedding (windows encoded
by the ST-GCN, similarity = softmax over distance to each genre's prototype) instead of the fixed
Laban+FFT cosine. The EDIT stage is byte-for-byte the same (genre_style.transfer, identity preserved) --
recognition informs the edit; it never replaces it.

torch is imported lazily so importing exhibition.py stays torch-free; you only need torch when you opt in
to the learned models. Falls back cleanly with a clear message if the checkpoint or torch is missing.
"""
import numpy as np
import genre_style as gs

WIN, STRIDE = 24, 12


class _Ctx:
    """Shared encoder + per-genre prototypes (built once, referenced by every LearnedGenreModel)."""
    def __init__(self, encoder, genres, protos, win, stride, tau, torch):
        self.encoder, self.genres, self.protos = encoder, genres, protos     # protos: (K, D) tensor
        self.win, self.stride, self.tau, self.torch = win, stride, tau, torch

    def embed_windows(self, clip):
        """Encode sliding windows -> (centers, embeddings tensor (M, D))."""
        torch = self.torch
        T = len(clip); wins, centers = [], []
        last = max(1, T - self.win + 1)
        for a in range(0, last, self.stride):
            w = clip[a:a + self.win]
            if len(w) < self.win:                                            # pad tail
                w = np.concatenate([w, np.repeat(w[-1:], self.win - len(w), axis=0)], axis=0)
            wins.append(w); centers.append(a + self.win / 2.0)
        x = torch.from_numpy(np.stack(wins).astype(np.float32))
        with torch.no_grad():
            z = self.encoder(x)
        return centers, z

    def genre_prob_curves(self, clip):
        """Per-genre probability curve over the clip: softmax over prototype distance, interpolated + smoothed."""
        torch = self.torch
        centers, z = self.embed_windows(clip)
        d2 = torch.cdist(z, self.protos) ** 2                                # (M, K)
        probs = torch.softmax(-d2 / self.tau, dim=1).cpu().numpy()           # (M, K)
        T = len(clip); curves = {}
        for ki, g in enumerate(self.genres):
            vals = probs[:, ki]
            if len(centers) >= 2:
                c = np.interp(np.arange(T), centers, vals).astype(np.float32)
            else:
                c = np.full(T, vals[0] if len(vals) else 0.0, dtype=np.float32)
            for t in range(1, T):
                c[t] = 0.3 * c[t] + 0.7 * c[t - 1]
            curves[g] = c
        return curves


class LearnedGenreModel:
    """Same interface as exhibition.GenreModel, backed by the learned encoder + prototypes."""
    def __init__(self, name, reference, ctx):
        self.name = name
        self.reference = np.asarray(reference, dtype=np.float32)
        self.ctx = ctx

    def compare(self, visitor):
        curve = self.ctx.genre_prob_curves(visitor)[self.name]
        return curve, float(curve.mean())

    def edit(self, visitor, base_alpha=0.45, gain=0.6, cap=0.9):
        visitor = np.asarray(visitor, dtype=np.float32)
        styled = gs.transfer(visitor, self.reference, {z: cap for z in gs.ZONES})
        curve, mean = self.compare(visitor)
        T = min(len(visitor), len(styled))
        out = visitor[:T].copy()
        for t in range(T):
            w = min(cap, base_alpha + gain * curve[t])
            out[t] = (1 - w) * visitor[t] + w * styled[t]
        return out, mean


def build_learned_models(references, ckpt="proto_encoder.pt", win=WIN, stride=STRIDE):
    """Load the trained encoder, build prototypes from `references`, return {genre: LearnedGenreModel}."""
    import torch                                                             # lazy: only when opting in
    from proto_model import STGCNEncoder
    blob = torch.load(ckpt, map_location="cpu")
    enc = STGCNEncoder(embed_dim=blob["embed_dim"])
    enc.load_state_dict(blob["state_dict"]); enc.eval()
    tau = blob.get("tau", 1.0); win = blob.get("win", win)

    genres = [g for g in blob["genres"] if g in references] or list(references)
    ctx = _Ctx(enc, genres, None, win, stride, tau, torch)
    # prototype per genre = mean embedding of that genre's reference windows
    protos = []
    for g in genres:
        _, z = ctx.embed_windows(np.asarray(references[g], dtype=np.float32))
        protos.append(torch.nn.functional.normalize(z.mean(0, keepdim=True), dim=-1))
    ctx.protos = torch.cat(protos, 0)                                       # (K, D)
    return {g: LearnedGenreModel(g, references[g], ctx) for g in genres}

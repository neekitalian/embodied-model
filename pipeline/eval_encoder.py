"""
Decide WHICH trained encoder to trust: held-out clip classification.

For each checkpoint: split data/<genre>/ clips per genre (test_frac held out), build prototypes from
the TRAIN clips' window embeddings, classify each HELD-OUT clip by its mean window probability.
Reports accuracy + per-genre breakdown per model -- real generalization, not training accuracy.

  python eval_encoder.py --data data --models proto_scratch.pt proto_encoder.pt
"""
import argparse, glob, os
import numpy as np
import torch

import genre_style as gs
from proto_model import STGCNEncoder, proto_logits

WIN, STRIDE = 24, 12


def load_labeled(data_dir):
    data = {}
    for d in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(d):
            continue
        clips = [gs.load_clip(p) for p in sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*.npy")))]
        if clips:
            data[os.path.basename(d)] = clips
    return data


def embed_clip(enc, clip):
    """All sliding-window embeddings of a clip, (M, D)."""
    T = len(clip); wins = []
    for a in range(0, max(1, T - WIN + 1), STRIDE):
        w = clip[a:a + WIN]
        if len(w) < WIN:
            w = np.concatenate([w, np.repeat(w[-1:], WIN - len(w), axis=0)], axis=0)
        wins.append(w)
    x = torch.from_numpy(np.stack(wins).astype(np.float32))
    with torch.no_grad():
        return enc(x)


def evaluate(ckpt, data, test_frac, seed):
    blob = torch.load(ckpt, map_location="cpu")
    enc = STGCNEncoder(embed_dim=blob["embed_dim"])
    enc.load_state_dict(blob["state_dict"]); enc.eval()
    rng = np.random.default_rng(seed)
    genres = sorted(data)

    train, test = {}, {}
    for g in genres:
        idx = rng.permutation(len(data[g]))
        n_test = max(1, int(round(test_frac * len(data[g]))))
        test[g] = [data[g][i] for i in idx[:n_test]]
        train[g] = [data[g][i] for i in idx[n_test:]] or test[g]

    protos = torch.cat([
        torch.nn.functional.normalize(
            torch.cat([embed_clip(enc, c) for c in train[g]], 0).mean(0, keepdim=True), dim=-1)
        for g in genres], 0)

    per = {g: [0, 0] for g in genres}
    for g in genres:
        for c in test[g]:
            z = embed_clip(enc, c)
            probs = torch.softmax(proto_logits(z, protos, blob.get("tau", 1.0)), dim=1).mean(0)
            pred = genres[int(probs.argmax())]
            per[g][1] += 1
            if pred == g:
                per[g][0] += 1
    correct = sum(v[0] for v in per.values()); total = sum(v[1] for v in per.values())
    return correct / max(1, total), per, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="data/<genre>/*.json labeled clips")
    ap.add_argument("--models", nargs="+", required=True, help="checkpoints to compare")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=5, help="average over this many random splits")
    a = ap.parse_args()
    data = load_labeled(os.path.expanduser(a.data))
    if len(data) < 2:
        raise SystemExit(f"need >=2 genres in {a.data}")
    print(f"[eval] genres={{g:len(v) for g,v in ...}}".replace("{g:len(v) for g,v in ...}", str({g: len(v) for g, v in data.items()})))
    print(f"[eval] held-out clip classification, averaged over {a.seeds} splits (test_frac={a.test_frac})\n")

    results = []
    for ckpt in a.models:
        ckpt = os.path.expanduser(ckpt)
        accs, per_sum = [], {g: [0, 0] for g in sorted(data)}
        for s in range(a.seeds):
            acc, per, _ = evaluate(ckpt, data, a.test_frac, seed=s)
            accs.append(acc)
            for g in per:
                per_sum[g][0] += per[g][0]; per_sum[g][1] += per[g][1]
        mean, std = float(np.mean(accs)), float(np.std(accs))
        results.append((ckpt, mean))
        print(f"{os.path.basename(ckpt):24s} held-out acc {mean:.3f} +/- {std:.3f}")
        for g, (c, t) in per_sum.items():
            print(f"    {g:12s} {c}/{t}  ({c/max(1,t):.2f})")
        print()
    best = max(results, key=lambda r: r[1])
    print(f"[eval] winner: {os.path.basename(best[0])}  ({best[1]:.3f}) -> use this with exhibition.py --model")


if __name__ == "__main__":
    main()

"""
Pretrain the prototypical ST-GCN encoder on MANY dance clips (e.g. the full AIST Dance DB),
then fine-tune on the 3 exhibition genres with train_proto.py --init.

Rung 2 of the performance ladder: the encoder learns general dance-movement structure from all
10 AIST genres, so few-shot adaptation to jazz/ballet/hip-hop starts from a much better metric
than training on 3 clips alone. AIST is non-commercial -> anything pretrained here stays on the
gallery/research track.

Data: a folder of HumanML3D-22 clips (.json/.npy) produced by video_to_reference.py. Genre labels
come from either subfolders (data/<genre>/*.json) or AIST filename codes (gJS_..., gMH_...):
  gBR break  gPO pop  gLO lock  gMH hiphop-middle  gLH hiphop-la
  gHO house  gWA waack  gKR krump  gJS jazz-street  gJB jazz-ballet

Per-genre clips are split train/val; the checkpoint with the best val episode accuracy is kept.
Runs on Mac MPS (overnight for the full DB) or any single cloud GPU (under an hour) -- the same
script, just faster hardware.

  python pretrain_aist.py --data aist_clips --epochs 3000 --out aist_encoder.pt
  python train_proto.py --data refs --init aist_encoder.pt --epochs 200 --out proto_encoder.pt
"""
import argparse, glob, os
import numpy as np
import torch
from torch import optim

import genre_style as gs
from proto_model import STGCNEncoder, prototypes, proto_loss, proto_logits
from train_proto import WIN, augment, sample_window

AIST_CODES = {"gBR": "break", "gPO": "pop", "gLO": "lock", "gMH": "hiphop-middle",
              "gLH": "hiphop-la", "gHO": "house", "gWA": "waack", "gKR": "krump",
              "gJS": "jazz-street", "gJB": "jazz-ballet"}


def load_dataset(data_dir):
    """{genre: [clips]} from subfolders, or from AIST filename codes in a flat folder."""
    data = {}
    subdirs = [d for d in sorted(glob.glob(os.path.join(data_dir, "*"))) if os.path.isdir(d)]
    if subdirs:
        for d in subdirs:
            clips = [gs.load_clip(p) for p in sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*.npy")))]
            if clips:
                data[os.path.basename(d)] = clips
        return data
    for p in sorted(glob.glob(os.path.join(data_dir, "*.json")) + glob.glob(os.path.join(data_dir, "*.npy"))):
        code = os.path.basename(p)[:3]
        g = AIST_CODES.get(code, code)
        data.setdefault(g, []).append(gs.load_clip(p))
    return data


def split(data, val_frac=0.2, rng=None):
    """Per-genre train/val split by CLIP (never by window, so val is truly unseen movement)."""
    rng = rng or np.random.default_rng(0)
    tr, va = {}, {}
    for g, clips in data.items():
        idx = rng.permutation(len(clips))
        n_val = max(1, int(round(val_frac * len(clips)))) if len(clips) > 1 else 0
        va_i, tr_i = idx[:n_val], idx[n_val:]
        if len(tr_i) == 0:                       # 1-clip genre: train on it, no val
            tr_i, va_i = idx, idx[:0]
        tr[g] = [clips[i] for i in tr_i]
        if len(va_i):
            va[g] = [clips[i] for i in va_i]
    return tr, va


def episode(data, genres, k_shot, q_query, rng, device, train=True):
    xs, ys, xq, yq = [], [], [], []
    for ki, g in enumerate(genres):
        clips = data[g]
        for _ in range(k_shot):
            w = sample_window(clips[rng.integers(len(clips))], rng)
            xs.append(augment(w, rng) if train else np.asarray(w, np.float32)); ys.append(ki)
        for _ in range(q_query):
            w = sample_window(clips[rng.integers(len(clips))], rng)
            xq.append(augment(w, rng) if train else np.asarray(w, np.float32)); yq.append(ki)
    t = lambda a: torch.from_numpy(np.stack(a).astype(np.float32)).to(device)
    return t(xs), torch.tensor(ys, device=device), t(xq), torch.tensor(yq, device=device)


def eval_episodes(enc, data, genres, k_shot, q_query, rng, device, n=20):
    enc.eval(); accs = []
    with torch.no_grad():
        for _ in range(n):
            gsub = [g for g in genres if g in data]
            xs, ys, xq, yq = episode(data, gsub, k_shot, q_query, rng, device, train=False)
            z = enc(torch.cat([xs, xq], 0))
            zs, zq = z[:len(xs)], z[len(xs):]
            pred = proto_logits(zq, prototypes(zs, ys, len(gsub))).argmax(1)
            accs.append((pred == yq).float().mean().item())
    enc.train()
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="clips folder: data/<genre>/* or flat AIST-coded files")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--k-shot", type=int, default=3)
    ap.add_argument("--q-query", type=int, default=5)
    ap.add_argument("--n-way", type=int, default=5, help="genres sampled per episode")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="aist_encoder.pt")
    a = ap.parse_args()
    a.data, a.out = os.path.expanduser(a.data), os.path.expanduser(a.out)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)

    data = load_dataset(a.data)
    if len(data) < 2:
        raise SystemExit(f"need >=2 genres in {a.data}, found {sorted(data)}")
    train_d, val_d = split(data, a.val_frac, rng)
    print(f"[pretrain_aist] device={device}  genres={len(data)}  "
          f"clips train={sum(map(len, train_d.values()))} val={sum(map(len, val_d.values()))}")
    for g in sorted(data):
        print(f"    {g:16s} train {len(train_d.get(g, [])):3d}  val {len(val_d.get(g, [])):3d}")

    enc = STGCNEncoder(embed_dim=a.embed_dim).to(device)
    opt = optim.Adam(enc.parameters(), lr=a.lr)
    genres = sorted(train_d)
    best = -1.0
    enc.train()
    for ep in range(1, a.epochs + 1):
        gsub = list(rng.choice(genres, size=min(a.n_way, len(genres)), replace=False))
        xs, ys, xq, yq = episode(train_d, gsub, a.k_shot, a.q_query, rng, device)
        z = enc(torch.cat([xs, xq], 0))
        zs, zq = z[:len(xs)], z[len(xs):]
        loss = proto_loss(zq, yq, prototypes(zs, ys, len(gsub)))
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % max(1, a.epochs // 30) == 0 or ep == 1:
            va = eval_episodes(enc, val_d if val_d else train_d, genres, a.k_shot, a.q_query, rng, device)
            tag = ""
            if va > best:
                best = va
                torch.save({"state_dict": enc.state_dict(), "genres": genres,
                            "embed_dim": a.embed_dim, "win": WIN, "tau": 1.0,
                            "pretrained_on": "AIST (non-commercial, gallery/research track)"}, a.out)
                tag = "  -> saved (best)"
            print(f"  ep {ep:5d}/{a.epochs}  loss {loss.item():.4f}  val-episode-acc {va:.3f}{tag}")
    print(f"[pretrain_aist] best val episode acc {best:.3f} -> {a.out}")
    print(f"Fine-tune: python train_proto.py --data refs --init {a.out} --epochs 200 --out proto_encoder.pt")


if __name__ == "__main__":
    main()

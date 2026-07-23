"""
Episodic few-shot trainer for the prototypical ST-GCN encoder (proto_model.STGCNEncoder).

Run on your Mac (needs torch). Learns an embedding where a genre's windows cluster near that genre's
prototype, from only a handful of clips per genre -- the few-shot regime that fits our data reality.

Data layout (either works):
  refs/<genre>.json                      -> one reference clip per genre (our current setup)
  data/<genre>/*.json  or  *.npy         -> several labeled clips per genre (better; drop clips here)

Because 1 clip/genre is tiny, each episode augments on the fly (temporal crop, vertical-axis rotation,
mirror, scale, jitter) so the encoder sees varied views of each genre -- the bone/root perturbation idea
from YNU-Dance's augmentation, adapted to joint positions. With more real clips per genre, generalization
improves; with 1 clip it still learns a usable metric but is only as diverse as the augmentation.

  python train_proto.py --data refs --epochs 400 --out proto_encoder.pt
"""
import argparse, glob, json, os
import numpy as np
import torch
from torch import optim

import genre_style as gs
from proto_model import STGCNEncoder, prototypes, proto_loss
from skeleton_graph import NUM_NODES

WIN = 24


def load_labeled(data_dir):
    """Return {genre: [clip (T,22,3), ...]}. Supports flat refs/<genre>.json or data/<genre>/*.{json,npy}."""
    data = {}
    subdirs = [d for d in sorted(glob.glob(os.path.join(data_dir, "*"))) if os.path.isdir(d)]
    if subdirs:
        for d in subdirs:
            g = os.path.basename(d)
            clips = [gs.load_clip(p) for p in sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*.npy")))]
            if clips:
                data[g] = clips
    else:
        for p in sorted(glob.glob(os.path.join(data_dir, "*.json")) + glob.glob(os.path.join(data_dir, "*.npy"))):
            data[os.path.splitext(os.path.basename(p))[0]] = [gs.load_clip(p)]
    return data


def augment(win, rng):
    """Augment one window (T,22,3): vertical-axis rotation, mirror, scale, jitter (identity-agnostic)."""
    w = np.asarray(win, dtype=np.float32).copy()
    th = rng.uniform(-0.4, 0.4)                                   # yaw about the vertical (y) axis
    c, s = np.cos(th), np.sin(th)
    x, z = w[..., 0].copy(), w[..., 2].copy()
    w[..., 0], w[..., 2] = c * x + s * z, -s * x + c * z
    if rng.random() < 0.5:                                       # left/right mirror
        w[..., 0] = -w[..., 0]
    w *= rng.uniform(0.9, 1.1)                                   # global scale
    w += rng.normal(0, 0.01, w.shape).astype(np.float32)         # small jitter
    return w


def sample_window(clip, rng):
    T = len(clip)
    if T <= WIN:
        pad = np.repeat(clip[-1:], WIN - T + 1, axis=0)
        clip = np.concatenate([clip, pad], axis=0); T = len(clip)
    a = rng.integers(0, T - WIN + 1)
    return clip[a:a + WIN]


def episode(data, genres, k_shot, q_query, rng, device):
    """Build one N-way (k_shot+q_query) episode -> support/query tensors + query labels."""
    xs, ys, xq, yq = [], [], [], []
    for ki, g in enumerate(genres):
        clips = data[g]
        for _ in range(k_shot):
            xs.append(augment(sample_window(clips[rng.integers(len(clips))], rng), rng)); ys.append(ki)
        for _ in range(q_query):
            xq.append(augment(sample_window(clips[rng.integers(len(clips))], rng), rng)); yq.append(ki)
    t = lambda a: torch.from_numpy(np.stack(a).astype(np.float32)).to(device)
    return t(xs), torch.tensor(ys, device=device), t(xq), torch.tensor(yq, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="refs", help="refs/<genre>.json or data/<genre>/*.{json,npy}")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--k-shot", type=int, default=3)
    ap.add_argument("--q-query", type=int, default=5)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="warm-start from a pretrained encoder checkpoint (pretrain_aist.py)")
    ap.add_argument("--out", default="proto_encoder.pt")
    a = ap.parse_args()
    a.data, a.out = os.path.expanduser(a.data), os.path.expanduser(a.out)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)

    data = load_labeled(a.data)
    genres = sorted(data)
    if len(genres) < 2:
        raise SystemExit(f"need >=2 genres in {a.data}, found {genres}")
    print(f"[train_proto] device={device}  genres={genres}  clips={{g:len(v) for g,v in ...}}"
          .replace("{g:len(v) for g,v in ...}", str({g: len(v) for g, v in data.items()})))

    enc = STGCNEncoder(embed_dim=a.embed_dim).to(device)
    if a.init:
        blob = torch.load(os.path.expanduser(a.init), map_location="cpu")
        if blob.get("embed_dim", a.embed_dim) != a.embed_dim:
            raise SystemExit(f"--embed-dim {a.embed_dim} != checkpoint's {blob['embed_dim']} (pass matching --embed-dim)")
        enc.load_state_dict(blob["state_dict"])
        print(f"[train_proto] warm-started from {a.init}")
    opt = optim.Adam(enc.parameters(), lr=a.lr)
    enc.train()
    for ep in range(1, a.epochs + 1):
        xs, ys, xq, yq = episode(data, genres, a.k_shot, a.q_query, rng, device)
        zc = enc(torch.cat([xs, xq], 0))
        zs, zq = zc[:len(xs)], zc[len(xs):]
        loss = proto_loss(zq, yq, prototypes(zs, ys, len(genres)), tau=a.tau)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % max(1, a.epochs // 20) == 0 or ep == 1:
            with torch.no_grad():
                from proto_model import proto_logits
                acc = (proto_logits(zq, prototypes(zs, ys, len(genres)), a.tau).argmax(1) == yq).float().mean().item()
            print(f"  ep {ep:4d}/{a.epochs}  loss {loss.item():.4f}  episode-acc {acc:.3f}")

    torch.save({"state_dict": enc.state_dict(), "genres": genres,
                "embed_dim": a.embed_dim, "win": WIN, "tau": a.tau}, a.out)
    print(f"[train_proto] saved encoder -> {a.out}  (use it: exhibition via proto_infer.build_learned_models)")


if __name__ == "__main__":
    main()

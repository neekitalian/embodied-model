"""
Export the trained prototypical encoder for the WEBSITE: ONNX model + prototypes JSON.

Produces pipeline/model/proto_encoder.onnx and pipeline/model/protos.json. Commit both; Vercel serves
them as static files and portal.html runs the encoder IN THE VISITOR'S BROWSER via onnxruntime-web,
so the live site shows the trained verdict with no server and no Python.

  python export_onnx.py --model proto_pretrain2.pt --data data --refs-dir refs
  git add model
  git commit -m "Ship trained encoder to the website"
  git push origin main

The checkpoint is AIST-pretrained -> gallery/research track (non-commercial), stamped into the JSON.
Deps: torch (+ onnx; onnxruntime optional for the verification pass).
"""
import argparse, glob, json, os
import numpy as np
import torch

import genre_style as gs
from proto_model import STGCNEncoder

WIN, STRIDE = 24, 12


def windows(clip, win=WIN, stride=STRIDE):
    T = len(clip); out = []
    for a in range(0, max(1, T - win + 1), stride):
        w = clip[a:a + win]
        if len(w) < win:
            w = np.concatenate([w, np.repeat(w[-1:], win - len(w), axis=0)], axis=0)
        out.append(w)
    return np.stack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="proto_pretrain2.pt")
    ap.add_argument("--data", default="data", help="data/<genre>/ clips for multi-clip prototypes")
    ap.add_argument("--refs-dir", default="refs", help="fallback single-reference clips")
    ap.add_argument("--out-dir", default="model")
    a = ap.parse_args()
    a.model, a.data, a.refs_dir, a.out_dir = map(os.path.expanduser, (a.model, a.data, a.refs_dir, a.out_dir))
    os.makedirs(a.out_dir, exist_ok=True)

    blob = torch.load(a.model, map_location="cpu")
    enc = STGCNEncoder(embed_dim=blob["embed_dim"])
    enc.load_state_dict(blob["state_dict"]); enc.eval()

    # 1. ONNX export (dynamic batch of windows, shape (N, WIN, 22, 3))
    onnx_path = os.path.join(a.out_dir, "proto_encoder.onnx")
    dummy = torch.randn(2, WIN, 22, 3)
    torch.onnx.export(enc, dummy, onnx_path, opset_version=17,
                      input_names=["windows"], output_names=["embedding"],
                      dynamic_axes={"windows": {0: "batch"}, "embedding": {0: "batch"}})
    print(f"[export] encoder -> {onnx_path}  ({os.path.getsize(onnx_path)/1024:.0f} KB)")

    # 2. prototypes: mean embedding per genre over ALL data/<genre>/ clips (fallback: refs/<genre>.json)
    #    genre keys use the website's reference names (jazz / ballet / hip-hop)
    genre_clips = {}
    for d in sorted(glob.glob(os.path.join(a.data, "*"))):
        if os.path.isdir(d):
            clips = [gs.load_clip(p) for p in sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*.npy")))]
            if clips:
                genre_clips[os.path.basename(d)] = clips
    for p in sorted(glob.glob(os.path.join(a.refs_dir, "*.json"))):
        genre_clips.setdefault(os.path.splitext(os.path.basename(p))[0], [gs.load_clip(p)])
    if not genre_clips:
        raise SystemExit(f"no clips found in {a.data} or {a.refs_dir}")

    genres, protos = [], []
    with torch.no_grad():
        for g in sorted(genre_clips):
            zs = [enc(torch.from_numpy(windows(np.asarray(c, np.float32)))) for c in genre_clips[g]]
            proto = torch.nn.functional.normalize(torch.cat(zs, 0).mean(0, keepdim=True), dim=-1)
            genres.append(g); protos.append(proto[0].numpy().tolist())
            print(f"[export] prototype {g}: {sum(len(z) for z in zs)} windows from {len(genre_clips[g])} clips")

    pj = {"genres": genres, "protos": protos, "tau": float(blob.get("tau", 1.0)),
          "win": WIN, "stride": STRIDE, "embed_dim": int(blob["embed_dim"]),
          "license": "encoder pretrained on AIST Dance DB clips (non-commercial) - gallery/research use"}
    protos_path = os.path.join(a.out_dir, "protos.json")
    json.dump(pj, open(protos_path, "w"))
    print(f"[export] prototypes -> {protos_path}  ({os.path.getsize(protos_path)/1024:.0f} KB)")

    # 3. verification: onnxruntime output must match torch (skipped gracefully if ort not installed)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        x = np.random.randn(3, WIN, 22, 3).astype(np.float32)
        z_ort = sess.run(None, {"windows": x})[0]
        with torch.no_grad():
            z_torch = enc(torch.from_numpy(x)).numpy()
        err = float(np.abs(z_ort - z_torch).max())
        print(f"[verify] onnxruntime vs torch max abs diff: {err:.2e}  ({'OK' if err < 1e-4 else 'MISMATCH - do not ship'})")
    except ImportError:
        print("[verify] onnxruntime not installed - skipped (pip install onnxruntime to verify locally)")

    print("\nShip it:  git add model && git commit -m \"Ship trained encoder to the website\" && git push origin main")


if __name__ == "__main__":
    main()

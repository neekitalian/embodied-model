"""Test for the prototypical ST-GCN track.

Without torch (e.g. this repo's cloud sandbox): validates the torch-free parts -- syntax of the torch
files, the graph, and that proto_infer's LearnedGenreModel keeps the exhibition.GenreModel interface.
With torch (your Mac): also runs a real forward pass, prototype math, and a tiny overfit episode to
prove the encoder + prototypical loss actually learn.

  python test_proto_model.py
"""
import importlib.util, py_compile
import numpy as np

def ck(n, c): print(f"  [{'PASS' if c else 'FAIL'}] {n}"); return c
ok = True

# torch-free checks -----------------------------------------------------------
for f in ("proto_model.py", "train_proto.py", "proto_infer.py", "pretrain_aist.py"):
    try:
        py_compile.compile(f, doraise=True); ok &= ck(f"{f} compiles", True)
    except py_compile.PyCompileError as e:
        ok &= ck(f"{f} compiles ({e})", False)

import skeleton_graph as sg
H = sg.normalized_adjacency()
ok &= ck("graph normalized adjacency ok", H.shape == (22, 22) and bool(np.all(np.isfinite(H))))

# interface parity: LearnedGenreModel must expose what exhibition.unnoticed_dance uses
import ast
tree = ast.parse(open("proto_infer.py").read())
cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "LearnedGenreModel")
methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
ok &= ck("LearnedGenreModel has compare + edit (GenreModel interface)", {"compare", "edit"} <= methods)

HAVE_TORCH = importlib.util.find_spec("torch") is not None
if not HAVE_TORCH:
    print("  [SKIP] torch not installed here -- forward pass + overfit checks run on the Mac")
else:
    import torch
    from proto_model import STGCNEncoder, prototypes, proto_loss, proto_logits
    torch.manual_seed(0)
    enc = STGCNEncoder(embed_dim=32, channels=(16, 32))
    x = torch.randn(6, 24, 22, 3)
    z = enc(x)
    ok &= ck("forward pass shape (6,32)", z.shape == (6, 32))
    ok &= ck("embeddings unit-norm", bool(torch.allclose(z.norm(dim=-1), torch.ones(6), atol=1e-4)))
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    P = prototypes(z, labels, 2)
    ok &= ck("prototypes shape (2,32), unit-norm", P.shape == (2, 32)
             and bool(torch.allclose(P.norm(dim=-1), torch.ones(2), atol=1e-4)))
    # tiny overfit: two synthetic 'genres' (different frequencies) must become separable
    def clip(freq, seed):
        r = np.random.default_rng(seed)
        t = np.arange(24)[:, None, None]
        base = r.normal(0, 0.05, (1, 22, 3))
        return (base + 0.3 * np.sin(2 * np.pi * freq * t / 24 + r.uniform(0, 6.28))).astype(np.float32)
    xs = torch.from_numpy(np.stack([clip(2, i) for i in range(8)] + [clip(6, i + 50) for i in range(8)]))
    ys = torch.tensor([0] * 8 + [1] * 8)
    opt = torch.optim.Adam(enc.parameters(), lr=5e-3)
    for _ in range(60):
        zz = enc(xs)
        loss = proto_loss(zz[1::2], ys[1::2], prototypes(zz[::2], ys[::2], 2))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        zz = enc(xs)
        acc = (proto_logits(zz[1::2], prototypes(zz[::2], ys[::2], 2)).argmax(1) == ys[1::2]).float().mean().item()
    ok &= ck(f"overfit episode separates 2 synthetic genres (acc={acc:.2f}>=0.85)", acc >= 0.85)

print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")

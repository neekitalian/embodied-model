"""Pure-numpy test for skeleton_graph (no torch): adjacency + symmetric normalization."""
import numpy as np
from hml_skeleton import PARENTS
import skeleton_graph as sg


def ck(n, c): print(f"  [{'PASS' if c else 'FAIL'}] {n}"); return c
ok = True

A = sg.adjacency()
ok &= ck("adjacency shape (22,22)", A.shape == (22, 22))
ok &= ck("adjacency symmetric", bool(np.allclose(A, A.T)))
ok &= ck("no self loops in raw adjacency", bool(np.all(np.diag(A) == 0)))
# node 9 (spine3): parent 6, children 12/13/14
ok &= ck("spine3 neighbors = {6,12,13,14}", set(np.nonzero(A[9])[0].tolist()) == {6, 12, 13, 14})
# every bone (child,parent) is an edge
edges_ok = all(A[j, p] == 1 and A[p, j] == 1 for j, p in enumerate(PARENTS) if p is not None and p >= 0)
ok &= ck("every PARENTS bone is an edge", edges_ok)

H = sg.normalized_adjacency()
ok &= ck("normalized shape (22,22)", H.shape == (22, 22))
ok &= ck("normalized symmetric", bool(np.allclose(H, H.T, atol=1e-6)))
ok &= ck("normalized self-loops positive (diag>0)", bool(np.all(np.diag(H) > 0)))
ok &= ck("normalized finite", bool(np.all(np.isfinite(H))))
# symmetric-normalized: H = D^-1/2 (A+I) D^-1/2, so H[i,i] = 1/deg_i with self loop
Ai = sg.adjacency() + np.eye(22)
deg = Ai.sum(1)
ok &= ck("diag == 1/deg (self-loop normalization)", bool(np.allclose(np.diag(H), 1.0 / deg, atol=1e-6)))
ok &= ck("off-diag H[i,j] == A/ sqrt(di*dj)", bool(np.allclose(H[9, 6], 1.0 / np.sqrt(deg[9] * deg[6]), atol=1e-6)))

print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")

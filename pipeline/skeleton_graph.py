"""
Skeleton graph for the HumanML3D-22 body -> normalized adjacency for the ST-GCN encoder.

Pure numpy (no torch) so it is testable in any environment; proto_model.py consumes the matrix it
returns. The graph is the physical bone connectivity (PARENTS), made symmetric, self-looped, and
symmetrically normalized  A_hat = D^-1/2 (A + I) D^-1/2  -- the standard GCN propagation matrix
(Kipf & Welling), which is what ST-GCN's spatial step multiplies features by.
"""
import numpy as np
from hml_skeleton import PARENTS, JOINT_NAMES

NUM_NODES = len(JOINT_NAMES)          # 22


def adjacency(parents=PARENTS, num_nodes=NUM_NODES):
    """Binary symmetric bone adjacency (no self loops)."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for j, p in enumerate(parents):
        if p is not None and p >= 0:
            A[j, p] = 1.0
            A[p, j] = 1.0
    return A


def normalized_adjacency(parents=PARENTS, num_nodes=NUM_NODES):
    """Symmetrically-normalized propagation matrix A_hat = D^-1/2 (A + I) D^-1/2."""
    A = adjacency(parents, num_nodes) + np.eye(num_nodes, dtype=np.float32)   # add self loops
    deg = A.sum(axis=1)                                                       # row degree
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D = np.diag(d_inv_sqrt)
    return (D @ A @ D).astype(np.float32)


if __name__ == "__main__":
    A = normalized_adjacency()
    print(f"A_hat {A.shape}  symmetric={np.allclose(A, A.T)}  trace={A.trace():.3f}")

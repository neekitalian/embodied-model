"""
Prototypical residual ST-GCN for dance-genre recognition (YNU-Dance / ProtoST spirit, sized for us).

Design (see the chat where we picked this over a heavy 3D-ResNet video model):
  * a RESIDUAL spatio-temporal GCN encoder maps a skeleton window (T,22,3) -> an L2-normalized embedding.
    Each block = spatial graph conv (features x normalized adjacency) + temporal conv, with a residual
    connection (this is the "residual network" applied to the SKELETON, not RGB video, so it runs
    real-time on a Mac and matches our MediaPipe -> HumanML3D-22 stream).
  * a PROTOTYPICAL head: each genre's prototype = the mean embedding of its reference clip(s). A query
    window is scored by (negative) squared distance to each prototype -> softmax over genres. This is
    the few-shot design that works with only a handful of clips per genre -- our exact situation.

Why prototypical and not a plain classifier: with ~1 reference clip per genre a softmax classifier
overfits instantly; prototypes + episodic training generalize from tiny support sets. It also drops
straight into exhibition.GenreModel.fit -- prototypes replace the motif bank, distance replaces cosine,
and the identity-preserving EDIT stage does not change.

Deps: torch (train/run on your Mac; this repo's cloud sandbox has no torch). The skeleton graph and the
pipeline glue are torch-free and unit-tested separately (skeleton_graph.py / proto_infer.py).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skeleton_graph import normalized_adjacency, NUM_NODES


class STGCNBlock(nn.Module):
    """Residual spatio-temporal graph-conv block. x: (N, C_in, T, V) -> (N, C_out, T', V)."""

    def __init__(self, c_in, c_out, t_kernel=9, t_stride=1):
        super().__init__()
        self.gcn = nn.Conv2d(c_in, c_out, kernel_size=1)                     # 1x1: channel mix before graph agg
        pad = (t_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=(t_kernel, 1), stride=(t_stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(c_out),
        )
        if c_in == c_out and t_stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=(t_stride, 1)), nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        h = self.gcn(x)                                                      # (N, C_out, T, V)
        h = torch.einsum("nctv,vw->nctw", h, A)                             # graph aggregation over joints
        h = self.tcn(h)
        return self.relu(h + res)


class STGCNEncoder(nn.Module):
    """Skeleton window (N, T, V, C=3) -> L2-normalized embedding (N, embed_dim)."""

    def __init__(self, in_channels=3, embed_dim=128, channels=(64, 64, 128), t_kernel=9):
        super().__init__()
        A = torch.from_numpy(normalized_adjacency())                        # (V, V), fixed graph
        self.register_buffer("A", A)
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_NODES)
        blocks, c_prev = [], in_channels
        for i, c in enumerate(channels):
            t_stride = 2 if i == len(channels) - 1 else 1                    # one temporal downsample near the end
            blocks.append(STGCNBlock(c_prev, c, t_kernel=t_kernel, t_stride=t_stride))
            c_prev = c
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(c_prev, embed_dim)

    def forward(self, x):
        # x: (N, T, V, C) -> (N, C, T, V)
        N, T, V, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.data_bn(x.reshape(N, C * V, T)).reshape(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        for blk in self.blocks:
            x = blk(x, self.A)
        x = x.mean(dim=(2, 3))                                               # global average pool over T, V
        z = self.head(x)
        return F.normalize(z, dim=-1)                                        # unit sphere -> stable distances


def prototypes(embeddings, labels, num_classes):
    """Mean embedding per class. embeddings (M, D), labels (M,) -> (num_classes, D)."""
    D = embeddings.shape[1]
    protos = embeddings.new_zeros((num_classes, D))
    for k in range(num_classes):
        m = labels == k
        if m.any():
            protos[k] = embeddings[m].mean(dim=0)
    return F.normalize(protos, dim=-1)


def proto_logits(query, protos, tau=1.0):
    """Negative squared-distance logits of queries to prototypes. query (Q,D), protos (K,D) -> (Q,K)."""
    d2 = torch.cdist(query, protos) ** 2
    return -d2 / tau


def proto_loss(query, q_labels, protos, tau=1.0):
    """Prototypical cross-entropy (Snell et al.)."""
    return F.cross_entropy(proto_logits(query, protos, tau), q_labels)


if __name__ == "__main__":
    enc = STGCNEncoder()
    n_params = sum(p.numel() for p in enc.parameters())
    x = torch.randn(4, 24, NUM_NODES, 3)
    z = enc(x)
    print(f"encoder params={n_params/1e3:.0f}k  in={tuple(x.shape)} -> embed={tuple(z.shape)}  "
          f"unit-norm={torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-4)}")

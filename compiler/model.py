"""Compact 1D-CNN trained in pure NumPy (no DL framework dependency).

Topology (channel-last, valid padding):
    Conv1D(k1, stride 2) -> ReLU -> Conv1D(k2, stride 2) -> ReLU -> Dense -> ArgMax

Weight layout matches the C runtime exactly:
    conv  W[out_ch, k * in_ch]   (window flattened k-major, channel-minor)
    dense W[out, in]             (input is the flattened [len][ch] activation)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .dataset import NUM_CLASSES, SEQ_LEN


@dataclass(frozen=True)
class Hyper:
    c1: int = 8
    c2: int = 16
    k1: int = 5
    k2: int = 3
    stride: int = 2
    epochs: int = 40
    lr: float = 0.01
    batch: int = 64
    seed: int = 0

    def dims(self):
        l0 = SEQ_LEN
        l1 = (l0 - self.k1) // self.stride + 1
        l2 = (l1 - self.k2) // self.stride + 1
        return l0, l1, l2

    def as_dict(self):
        return asdict(self)


def im2col(x, k, s):
    n, length, c = x.shape
    lo = (length - k) // s + 1
    cols = np.empty((n, lo, k * c), dtype=x.dtype)
    for j in range(k):
        cols[:, :, j * c:(j + 1) * c] = x[:, j:j + s * (lo - 1) + 1:s, :]
    return cols


def col2im(dcols, length, c, k, s):
    n, lo, _ = dcols.shape
    dx = np.zeros((n, length, c), dtype=dcols.dtype)
    for j in range(k):
        dx[:, j:j + s * (lo - 1) + 1:s, :] += dcols[:, :, j * c:(j + 1) * c]
    return dx


def init_params(hp: Hyper):
    rng = np.random.default_rng(hp.seed)
    _, _, l2 = hp.dims()

    def he(shape, fan_in):
        return (rng.normal(0, np.sqrt(2.0 / fan_in), shape)).astype(np.float32)

    return {
        "w1": he((hp.c1, hp.k1 * 1), hp.k1), "b1": np.zeros(hp.c1, np.float32),
        "w2": he((hp.c2, hp.k2 * hp.c1), hp.k2 * hp.c1), "b2": np.zeros(hp.c2, np.float32),
        "w3": he((NUM_CLASSES, l2 * hp.c2), l2 * hp.c2), "b3": np.zeros(NUM_CLASSES, np.float32),
    }


def forward(p, hp: Hyper, x):
    """FP32 forward. Returns logits and a cache with pre-activation tensors."""
    x = x.astype(np.float32)
    cols1 = im2col(x, hp.k1, hp.stride)
    a1 = cols1 @ p["w1"].T + p["b1"]
    h1 = np.maximum(a1, 0)
    cols2 = im2col(h1, hp.k2, hp.stride)
    a2 = cols2 @ p["w2"].T + p["b2"]
    h2 = np.maximum(a2, 0)
    flat = h2.reshape(len(x), -1)
    logits = flat @ p["w3"].T + p["b3"]
    cache = dict(x=x, cols1=cols1, a1=a1, h1=h1, cols2=cols2, a2=a2, h2=h2, flat=flat, logits=logits)
    return logits, cache


def _backward(p, hp, c, y):
    n = len(y)
    prob = np.exp(c["logits"] - c["logits"].max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    loss = float(-np.log(prob[np.arange(n), y] + 1e-9).mean())
    d = prob
    d[np.arange(n), y] -= 1
    d /= n
    g = {}
    g["w3"] = d.T @ c["flat"]
    g["b3"] = d.sum(0)
    dh2 = (d @ p["w3"]).reshape(c["h2"].shape)
    da2 = dh2 * (c["a2"] > 0)
    g["w2"] = np.einsum("nlo,nlk->ok", da2, c["cols2"])
    g["b2"] = da2.sum((0, 1))
    dcols2 = da2 @ p["w2"]
    dh1 = col2im(dcols2, c["h1"].shape[1], hp.c1, hp.k2, hp.stride)
    da1 = dh1 * (c["a1"] > 0)
    g["w1"] = np.einsum("nlo,nlk->ok", da1, c["cols1"])
    g["b1"] = da1.sum((0, 1))
    return loss, g


def train(hp: Hyper, x, y):
    """Mini-batch Adam. Returns FP32 params."""
    p = init_params(hp)
    rng = np.random.default_rng(hp.seed + 99)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(a) for k, a in p.items()}
    t = 0
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(y)
    for ep in range(hp.epochs):
        lr = hp.lr * (0.5 * (1 + np.cos(np.pi * ep / hp.epochs)) * 0.95 + 0.05)
        perm = rng.permutation(n)
        for i in range(0, n, hp.batch):
            idx = perm[i:i + hp.batch]
            _, cache = forward(p, hp, x[idx])
            _, g = _backward(p, hp, cache, y[idx])
            t += 1
            for k in p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** t)
                vh = v[k] / (1 - b2 ** t)
                p[k] = (p[k] - lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
    return p


def predict(p, hp, x):
    logits, _ = forward(p, hp, x)
    return logits.argmax(1)


def accuracy(p, hp, x, y):
    return float((predict(p, hp, x) == y).mean())

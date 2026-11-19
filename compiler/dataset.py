"""Synthetic embedded-sensor anomaly dataset (single-channel 1D windows).

Four classes emulate what a vibration / current sensor on an MCU would see:

  0 normal       - sinusoid + AR(1) coloured noise
  1 spike        - normal + 1-2 impulsive glitches
  2 burst        - normal + a short high-frequency bearing-fault-like burst
  3 level_shift  - normal + a sudden DC step (sensor drift / load change)

The generator is fully deterministic for a given seed.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

CLASSES = ("normal", "spike", "burst", "level_shift")
SEQ_LEN = 64
NUM_CLASSES = len(CLASSES)


def _one(rng: np.random.Generator, label: int) -> np.ndarray:
    t = np.arange(SEQ_LEN) / SEQ_LEN
    f = rng.uniform(1.0, 3.0)
    ph = rng.uniform(0, 2 * np.pi)
    amp = rng.uniform(0.6, 1.0)
    x = amp * np.sin(2 * np.pi * f * t + ph)
    x = x + signal.lfilter([1.0], [1.0, -0.5], rng.normal(0, 0.18, SEQ_LEN))
    if label == 1:
        n = int(rng.integers(1, 3))
        pos = rng.choice(SEQ_LEN, n, replace=False)
        x[pos] += rng.choice([-1.0, 1.0], n) * rng.uniform(0.9, 2.0, n)
    elif label == 2:
        ln = int(rng.integers(12, 21))
        s = int(rng.integers(6, SEQ_LEN - ln - 4))
        fb = rng.uniform(12, 20)
        x[s:s + ln] += rng.uniform(0.5, 0.9) * np.sin(2 * np.pi * fb * t[:ln])
    elif label == 3:
        pos = int(rng.integers(16, 49))
        x[pos:] += rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 1.5)
    return x


def make_dataset(n_per_class: int, seed: int):
    """Returns (x[N, SEQ_LEN, 1] float32, y[N] int64), shuffled."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for label in range(NUM_CLASSES):
        for _ in range(n_per_class):
            xs.append(_one(rng, label))
            ys.append(label)
    x = np.asarray(xs, dtype=np.float32)[:, :, None]
    y = np.asarray(ys, dtype=np.int64)
    p = rng.permutation(len(y))
    return x[p], y[p]


def train_test(n_train_per_class=400, n_test_per_class=100, seed=1234):
    xtr, ytr = make_dataset(n_train_per_class, seed)
    xte, yte = make_dataset(n_test_per_class, seed + 1)
    return xtr, ytr, xte, yte

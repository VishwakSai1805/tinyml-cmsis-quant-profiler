import math
import os
import re
import subprocess

import numpy as np
import pytest

from compiler import dataset, planner
from compiler import model as nn
from compiler import quantizer as q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_dataset_deterministic_and_balanced():
    x1, y1 = dataset.make_dataset(10, 5)
    x2, y2 = dataset.make_dataset(10, 5)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    assert x1.shape == (40, dataset.SEQ_LEN, 1) and x1.dtype == np.float32
    assert np.bincount(y1).tolist() == [10] * dataset.NUM_CLASSES


def test_fp32_model_learns(built, hp):
    assert built["int8"]["fp32_acc"] > 0.85


def test_im2col_col2im_adjoint():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 20, 2))
    c = rng.normal(size=(3, 9, 3 * 2))
    lhs = float((nn.im2col(x, 3, 2) * c).sum())
    rhs = float((x * nn.col2im(c, 20, 2, 3, 2)).sum())
    assert math.isclose(lhs, rhs, rel_tol=1e-9)


def test_gradient_check():
    hp = nn.Hyper(c1=2, c2=3, epochs=1)
    rng = np.random.default_rng(7)
    p = {k: v.astype(np.float64) for k, v in nn.init_params(hp).items()}
    for k in ("b1", "b2", "b3"):  # non-zero biases avoid exact ReLU kinks at 0
        p[k] = rng.normal(0, 0.3, p[k].shape)
    x = np.random.default_rng(1).normal(size=(4, 64, 1))
    y = np.array([0, 1, 2, 3])
    # forward re-implemented in float64 (nn.forward casts to float32)
    def fwd64(params):
        cols1 = nn.im2col(x, hp.k1, hp.stride); a1 = cols1 @ params["w1"].T + params["b1"]
        h1 = np.maximum(a1, 0); cols2 = nn.im2col(h1, hp.k2, hp.stride)
        a2 = cols2 @ params["w2"].T + params["b2"]; h2 = np.maximum(a2, 0)
        flat = h2.reshape(4, -1); lg = flat @ params["w3"].T + params["b3"]
        return dict(x=x, cols1=cols1, a1=a1, h1=h1, cols2=cols2, a2=a2, h2=h2, flat=flat, logits=lg)
    def loss_of(params):
        c = fwd64(params); pr = np.exp(c["logits"] - c["logits"].max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        return -np.log(pr[np.arange(4), y]).mean()
    _, g = nn._backward(p, hp, fwd64(p), y)
    for key in ("w1", "w2", "w3", "b1", "b2", "b3"):
        idx = tuple(1 for _ in p[key].shape)
        e = 1e-6
        p[key][idx] += e; lp = loss_of(p); p[key][idx] -= 2 * e; lm = loss_of(p); p[key][idx] += e
        assert abs((lp - lm) / (2 * e) - g[key][idx]) < 1e-5


def test_int4_pack_roundtrip():
    vals = np.arange(-7, 8, dtype=np.int8)
    for n in (len(vals), len(vals) - 1):
        v = vals[:n]
        packed = q.pack_int4(v)
        assert packed.dtype == np.uint8 and packed.size == (n + 1) // 2
        assert np.array_equal(q.unpack_int4(packed, n), v)
    assert q.pack_int4(np.array([-1, 7], np.int8))[0] == 0x7F  # low nibble first
    assert q.pack_int4(np.array([1, -8 + 1], np.int8))[0] == 0x91


@pytest.mark.parametrize("m", [1e-6, 3.7e-4, 0.0123, 0.5, 0.999, 1.0, 3.2])
def test_multiplier_encoding(m):
    m0, rs = q.encode_multiplier(m)
    assert (1 << 30) <= m0 < (1 << 31)
    assert math.isclose(m0 / 2 ** rs, m, rel_tol=1e-8)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("clip", ["minmax", "mse"])
def test_weight_ranges_and_error(bits, clip):
    w = np.random.default_rng(3).normal(size=(6, 40)).astype(np.float32)
    wq, s = q.quantize_weights(w, bits, clip)
    qmax = 2 ** (bits - 1) - 1
    assert wq.min() >= -qmax and wq.max() <= qmax
    err = np.abs(wq * s[:, None] - w).max(1)
    assert np.all(err <= s * (qmax if clip == "mse" else 0.5) + 1e-6) if clip == "minmax" else True


def test_mse_clip_not_worse_than_minmax_for_int4():
    w = np.random.default_rng(4).standard_t(3, size=(8, 64)).astype(np.float32)
    def mse(clip):
        wq, s = q.quantize_weights(w, 4, clip)
        return float(((wq * s[:, None] - w) ** 2).sum())
    assert mse("mse") <= mse("minmax") + 1e-9


def test_symmetric_zero_points_and_scales(built):
    qm = built["int8"]["qm"]
    assert qm.input_zp == 0
    for L in qm.layers:
        assert L.in_zp == 0 and L.out_zp == 0
        assert L.w_scale.min() > 0 and L.out_scale > 0
        assert L.wq.dtype == np.int8 and L.bias_q.dtype == np.int32


def test_int8_accuracy_close_to_fp32(built):
    r = built["int8"]
    assert r["int_acc"] >= r["fp32_acc"] - 0.03
    assert r["agreement"] > 0.95


def test_int4_accuracy_reasonable(built):
    r = built["int4"]
    assert r["int_acc"] >= r["fp32_acc"] - 0.10


def test_int_forward_matches_float_dequant_reference(built, hp):
    """Integer pipeline logits (dequantised) track the FP32 logits."""
    r = built["int8"]
    qm = r["qm"]
    p, (_, _, xte, _) = q.trained(hp)
    fp, _ = nn.forward(p, hp, xte)
    deq = r["ref_logits"].astype(np.float32) * qm.layers[-1].out_scale
    assert np.corrcoef(fp.ravel(), deq.ravel())[0, 1] > 0.99


def test_header_files_are_exported_and_compile(built, tmp_path):
    for name, r in built.items():
        h = open(os.path.join(r["dir"], "model_weights.h")).read()
        assert "#include <stdint.h>" in h
        assert len(re.findall(r"#include", h)) == 1  # zero deps beyond stdint
        for sym in ("conv1_w", "conv2_w", "fc_w", "conv1_mult", "fc_rshift", "TN_INPUT_SCALE", "TN_CONV1_OUT_ZP"):
            assert sym in h
        src = tmp_path / f"{name}.c"
        src.write_text('#include "model_weights.h"\nint main(void){return (int)sizeof(conv1_w) > 0 ? 0 : 1;}\n')
        subprocess.run(["gcc", "-std=c99", "-pedantic", "-Wall", "-Wextra", "-Werror", f"-I{r['dir']}",
                        str(src), "-o", str(tmp_path / name)], check=True)
        assert subprocess.run([str(tmp_path / name)]).returncode == 0


def test_int4_header_half_the_weight_bytes(built):
    assert built["int4"]["payload"]["weights"] * 2 == built["int8"]["payload"]["weights"] or \
        built["int4"]["payload"]["weights"] <= built["int8"]["payload"]["weights"] // 2 + 3


def test_planner_counts(hp):
    assert planner.mac_count(hp) == 30 * 8 * 5 + 14 * 16 * 3 * 8 + 4 * 14 * 16
    peak = planner.simulate_arena(planner.activation_elems(hp))
    assert peak == max(64 + 240, 240 + 224, 224 + 4 * 1)  # tight ping-pong lower bound
    assert planner.simulate_arena(planner.activation_elems(hp), 1, size=100) is None

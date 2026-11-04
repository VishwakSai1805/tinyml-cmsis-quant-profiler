"""Post-Training Quantization (PTQ) FP32 -> symmetric INT8 / packed INT4,
plus a bit-exact integer reference and zero-dependency C header export.

Scheme (TFLite/CMSIS-NN style)
------------------------------
* Weights: symmetric, per-output-channel, zero-point 0.
    INT8 -> [-127, 127];  INT4 -> [-7, 7] packed two per byte (low nibble first).
* Activations: symmetric per-tensor INT8, zero-point 0 (calibrated on data).
* Bias: INT32 with scale s_in * s_w[oc].
* Requantisation: acc_i32 -> round(acc * M[oc]) with M = s_in*s_w/s_out encoded
  as a Q31 multiplier and a right shift:  (acc * m0 + 2^(rs-1)) >> rs.
  Everything is integer-only, so Python (`int_forward`) and the C engine are
  bit-exact.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import model as nn
from .dataset import NUM_CLASSES, SEQ_LEN, train_test
from .planner import layer_table

INT8_MIN, INT8_MAX = -128, 127


@dataclass
class QuantConfig:
    weight_bits: int = 8          # 8 or 4
    clip: str = "minmax"          # "minmax" | "mse" (MSE-optimal weight clipping)
    act_percentile: float = 100.0  # activation calibration percentile

    def tag(self):
        return f"w{self.weight_bits}a8-{self.clip}"


@dataclass
class QLayer:
    name: str
    kind: str                     # "conv1d" | "dense"
    in_len: int
    in_ch: int
    out_ch: int
    k: int
    stride: int
    out_len: int
    wbits: int
    wq: np.ndarray                # int8 [out_ch, k*in_ch] (values within wbits range)
    w_scale: np.ndarray           # float32 [out_ch]
    bias_q: np.ndarray            # int32 [out_ch]
    mult: np.ndarray              # int32 [out_ch] Q31 multiplier
    rshift: np.ndarray            # uint8 [out_ch] total right shift
    in_scale: float
    in_zp: int
    out_scale: float
    out_zp: int
    relu_after: bool


@dataclass
class QuantizedModel:
    name: str
    layers: list
    input_scale: float
    input_zp: int
    input_len: int
    input_ch: int
    num_classes: int
    cfg: QuantConfig
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------- weights --
def _quant_channel(w: np.ndarray, qmax: int, clip: str):
    amax = float(np.abs(w).max())
    if amax == 0.0:
        return np.zeros_like(w, dtype=np.int8), 1.0
    best_alpha = amax
    if clip == "mse":
        best_err = None
        for frac in np.linspace(0.3, 1.0, 36):
            alpha = amax * frac
            s = alpha / qmax
            err = float(np.sum((w - np.clip(np.rint(w / s), -qmax, qmax) * s) ** 2))
            if best_err is None or err < best_err:
                best_err, best_alpha = err, alpha
    s = best_alpha / qmax
    return np.clip(np.rint(w / s), -qmax, qmax).astype(np.int8), s


def quantize_weights(w: np.ndarray, bits: int, clip: str):
    qmax = (1 << (bits - 1)) - 1
    q = np.empty(w.shape, np.int8)
    sc = np.empty(w.shape[0], np.float32)
    for o in range(w.shape[0]):
        q[o], s = _quant_channel(w[o].astype(np.float64), qmax, clip)
        sc[o] = s
    return q, sc


def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack signed 4-bit values, flat element i -> byte i//2, low nibble first."""
    flat = q.reshape(-1).astype(np.int8)
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, np.int8)])
    nib = (flat.astype(np.int16) & 0xF).astype(np.uint8)
    return (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    lo = (packed & 0xF).astype(np.int16)
    hi = (packed >> 4).astype(np.int16)
    nib = np.empty(packed.size * 2, np.int16)
    nib[0::2], nib[1::2] = lo, hi
    return (((nib ^ 8) - 8)[:n]).astype(np.int8)


def encode_multiplier(m: float):
    """M ~= m0 * 2^-rs, m0 in [2^30, 2^31)."""
    if m <= 0:
        return 0, 31
    frac, exp = math.frexp(m)          # m = frac * 2^exp, frac in [0.5, 1)
    m0 = int(round(frac * (1 << 31)))
    if m0 == 1 << 31:
        m0 //= 2
        exp += 1
    rs = 31 - exp
    if not 1 <= rs <= 62:
        raise ValueError(f"requant shift {rs} out of range for M={m}")
    return m0, rs


# -------------------------------------------------------------- quantizer ---
def _calib_scale(a: np.ndarray, pct: float, positive_only: bool) -> float:
    v = np.maximum(a, 0) if positive_only else np.abs(a)
    m = float(np.percentile(v, pct)) if pct < 100 else float(v.max())
    return max(m, 1e-8) / 127.0


def quantize_model(p, hp: nn.Hyper, calib_x, cfg: QuantConfig, name="tinycnn") -> QuantizedModel:
    _, cache = nn.forward(p, hp, calib_x)
    pct = cfg.act_percentile
    s_in = _calib_scale(cache["x"], pct, False)
    # ReLU follows both convs, so negative values are irrelevant -> positive-only range.
    s1 = _calib_scale(cache["a1"], pct, True)
    s2 = _calib_scale(cache["a2"], pct, True)
    s3 = _calib_scale(cache["logits"], 100.0, False)
    scales_in = [s_in, s1, s2]
    scales_out = [s1, s2, s3]
    layers = []
    for i, (lname, kind, in_len, in_ch, out_ch, k, stride, out_len) in enumerate(layer_table(hp)):
        w = p[f"w{i + 1}"]
        b = p[f"b{i + 1}"]
        wq, ws = quantize_weights(w, cfg.weight_bits, cfg.clip)
        sin, sout = scales_in[i], scales_out[i]
        bq = np.rint(b.astype(np.float64) / (sin * ws.astype(np.float64))).astype(np.int64)
        bq = np.clip(bq, -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
        mult = np.zeros(out_ch, np.int32)
        rsh = np.zeros(out_ch, np.uint8)
        for o in range(out_ch):
            m0, rs = encode_multiplier(sin * float(ws[o]) / sout)
            mult[o], rsh[o] = m0, rs
        layers.append(QLayer(lname, kind, in_len, in_ch, out_ch, k, stride, out_len, cfg.weight_bits,
                             wq, ws, bq, mult, rsh, sin, 0, sout, 0, relu_after=(i < 2)))
    return QuantizedModel(name, layers, s_in, 0, SEQ_LEN, 1, NUM_CLASSES, cfg, {"hyper": hp.as_dict()})


# ------------------------------------------------------ integer reference ---
def quantize_input(qm: QuantizedModel, x: np.ndarray) -> np.ndarray:
    q = np.rint(x.astype(np.float64) / qm.input_scale) + qm.input_zp
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8).reshape(len(x), -1)


def requantize(acc: np.ndarray, mult: np.ndarray, rshift: np.ndarray, out_zp: int) -> np.ndarray:
    a = acc.astype(np.int64) * mult.astype(np.int64)
    rs = rshift.astype(np.int64)
    r = (a + (np.int64(1) << (rs - 1))) >> rs
    return np.clip(r + out_zp, INT8_MIN, INT8_MAX).astype(np.int8)


def int_forward(qm: QuantizedModel, xq: np.ndarray) -> np.ndarray:
    """Bit-exact model of runtime/tiny_nn.c. xq: int8 [N, input_len]. Returns int8 logits."""
    cur = xq.astype(np.int32)
    n = len(xq)
    for L in qm.layers:
        wl = L.k * L.in_ch
        a = cur.reshape(n, L.in_len, L.in_ch) - L.in_zp
        cols = np.empty((n, L.out_len, wl), np.int64)
        for j in range(L.k):
            cols[:, :, j * L.in_ch:(j + 1) * L.in_ch] = a[:, j:j + L.stride * (L.out_len - 1) + 1:L.stride, :]
        acc = cols @ L.wq.astype(np.int64).T + L.bias_q.astype(np.int64)
        out = requantize(acc, L.mult, L.rshift, L.out_zp).astype(np.int32)
        if L.relu_after:
            out = np.maximum(out, L.out_zp)
        cur = out.reshape(n, -1)
    return cur.astype(np.int8)


def int_predict(qm, xq):
    return int_forward(qm, xq).argmax(1)  # numpy argmax = first max, same as C


def logit_checksum(logits: np.ndarray) -> int:
    cs = 0
    for v in logits.reshape(-1):
        cs = (cs * 31 + (int(v) & 0xFF)) & 0xFFFFFFFF
    return cs


# ----------------------------------------------------------------- export ---
def _fmt_array(vals, per_line=16, hexfmt=False):
    items = [f"0x{int(v) & 0xFF:02X}" if hexfmt else str(int(v)) for v in vals]
    lines = []
    for i in range(0, len(items), per_line):
        lines.append("    " + ", ".join(items[i:i + per_line]) + ",")
    return "\n".join(lines)


HEADER_PRELUDE = """\
#ifndef {guard}
#define {guard}
/* Generated by compiler/quantizer.py -- DO NOT EDIT.
 * Model : {name}   quantisation: {tag}
 * Zero-dependency C99 header (only <stdint.h>).
 * Tensor layout is channel-last: activation[pos][ch]; conv weights [oc][k][ic];
 * INT4 weights are packed two per byte, LOW nibble first, two's complement.
 * Requantisation: out = clamp(((acc * mult[oc]) + (1 << (rshift[oc]-1))) >> rshift[oc]) + out_zp
 */
#include <stdint.h>

#ifndef TN_MODEL_SECTION
#  ifdef TN_USE_SECTIONS
#    define TN_MODEL_SECTION __attribute__((section(".model_rodata")))
#  else
#    define TN_MODEL_SECTION
#  endif
#endif
"""


def export_header(qm: QuantizedModel, path: str) -> dict:
    """Write model_weights.h. Returns raw payload byte accounting."""
    guard = "TN_MODEL_WEIGHTS_H"
    out = [HEADER_PRELUDE.format(guard=guard, name=qm.name, tag=qm.cfg.tag())]
    out.append(f"#define TN_MODEL_NAME \"{qm.name}-{qm.cfg.tag()}\"")
    out.append(f"#define TN_INPUT_LEN {qm.input_len}")
    out.append(f"#define TN_INPUT_CH {qm.input_ch}")
    out.append(f"#define TN_NUM_CLASSES {qm.num_classes}")
    out.append(f"#define TN_INPUT_SCALE {qm.input_scale!r}f")
    out.append(f"#define TN_INPUT_ZP {qm.input_zp}")
    out.append(f"#define TN_WEIGHT_BITS {qm.layers[0].wbits}")
    out.append("")
    payload = {"weights": 0, "bias": 0, "mult": 0, "rshift": 0}
    for L in qm.layers:
        P = L.name.upper()
        for key, val in [("IN_LEN", L.in_len), ("IN_CH", L.in_ch), ("OUT_CH", L.out_ch), ("K", L.k),
                         ("STRIDE", L.stride), ("OUT_LEN", L.out_len), ("WBITS", L.wbits),
                         ("IN_ZP", L.in_zp), ("OUT_ZP", L.out_zp)]:
            out.append(f"#define TN_{P}_{key} {val}")
        out.append(f"#define TN_{P}_IN_SCALE {L.in_scale!r}f")
        out.append(f"#define TN_{P}_OUT_SCALE {L.out_scale!r}f")
        out.append("")
        n = L.name
        if L.wbits == 8:
            out.append(f"static const int8_t {n}_w[{L.wq.size}] TN_MODEL_SECTION = {{\n{_fmt_array(L.wq.reshape(-1))}\n}};")
            payload["weights"] += L.wq.size
        else:
            pk = pack_int4(L.wq)
            out.append(f"static const uint8_t {n}_w[{pk.size}] TN_MODEL_SECTION = {{ /* {L.wq.size} x int4 */\n{_fmt_array(pk, hexfmt=True)}\n}};")
            payload["weights"] += pk.size
        out.append(f"static const int32_t {n}_b[{L.out_ch}] TN_MODEL_SECTION = {{\n{_fmt_array(L.bias_q, 8)}\n}};")
        out.append(f"static const int32_t {n}_mult[{L.out_ch}] TN_MODEL_SECTION = {{\n{_fmt_array(L.mult, 8)}\n}};")
        out.append(f"static const uint8_t {n}_rshift[{L.out_ch}] TN_MODEL_SECTION = {{\n{_fmt_array(L.rshift)}\n}};")
        out.append("")
        payload["bias"] += 4 * L.out_ch
        payload["mult"] += 4 * L.out_ch
        payload["rshift"] += L.out_ch
    out.append(f"#endif /* {guard} */\n")
    with open(path, "w") as f:
        f.write("\n".join(out))
    payload["total"] = sum(payload.values())
    return payload


def export_test_vectors(qm, xq, y, fp32_pred, ref_logits, path):
    n = len(y)
    ref_pred = ref_logits.argmax(1)
    out = ["#ifndef TN_TEST_VECTORS_H", "#define TN_TEST_VECTORS_H",
           "/* Generated by compiler/quantizer.py -- held-out test set, INT8-quantised inputs. */",
           "#include <stdint.h>",
           "#ifndef TN_TESTDATA_SECTION",
           "#  ifdef TN_USE_SECTIONS",
           "#    define TN_TESTDATA_SECTION __attribute__((section(\".testdata\")))",
           "#  else", "#    define TN_TESTDATA_SECTION", "#  endif", "#endif",
           f"#define TN_TEST_N {n}",
           f"#define TN_REF_LOGIT_CHECKSUM {logit_checksum(ref_logits)}u",
           f"static const int8_t tn_test_x[{n * qm.input_len}] TN_TESTDATA_SECTION = {{\n{_fmt_array(xq.reshape(-1))}\n}};",
           f"static const uint8_t tn_test_y[{n}] TN_TESTDATA_SECTION = {{\n{_fmt_array(y)}\n}};",
           f"static const uint8_t tn_test_fp32_pred[{n}] TN_TESTDATA_SECTION = {{\n{_fmt_array(fp32_pred)}\n}};",
           f"static const uint8_t tn_test_ref_pred[{n}] TN_TESTDATA_SECTION = {{\n{_fmt_array(ref_pred)}\n}};",
           "#endif\n"]
    with open(path, "w") as f:
        f.write("\n".join(out))


# --------------------------------------------------------------- pipeline ---
_TRAIN_CACHE: dict = {}


def trained(hp: nn.Hyper):
    """Train (memoised per process) and return (params, data)."""
    if hp not in _TRAIN_CACHE:
        xtr, ytr, xte, yte = train_test()
        _TRAIN_CACHE[hp] = (nn.train(hp, xtr, ytr), (xtr, ytr, xte, yte))
    return _TRAIN_CACHE[hp]


def build(hp: nn.Hyper, cfg: QuantConfig, out_dir: str | None = None):
    """Train -> PTQ -> integer eval -> (optionally) export. Returns a result dict."""
    p, (xtr, ytr, xte, yte) = trained(hp)
    qm = quantize_model(p, hp, xtr[:512], cfg)
    xq = quantize_input(qm, xte)
    logits = int_forward(qm, xq)
    fp_pred = nn.predict(p, hp, xte)
    res = {
        "hyper": hp.as_dict(), "cfg": cfg.tag(), "qm": qm, "params": p,
        "fp32_acc": float((fp_pred == yte).mean()),
        "int_acc": float((logits.argmax(1) == yte).mean()),
        "agreement": float((logits.argmax(1) == fp_pred).mean()),
        "ref_logits": logits, "xq": xq, "yte": yte, "fp_pred": fp_pred,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        res["payload"] = export_header(qm, os.path.join(out_dir, "model_weights.h"))
        export_test_vectors(qm, xq, yte, fp_pred, logits, os.path.join(out_dir, "test_vectors.h"))
        with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
            json.dump({"name": qm.name, "cfg": cfg.tag(), "hyper": hp.as_dict(),
                       "fp32_acc": res["fp32_acc"], "int_acc": res["int_acc"],
                       "input_scale": qm.input_scale,
                       "layers": [{"name": L.name, "w_scale": L.w_scale.tolist(),
                                   "in_scale": L.in_scale, "out_scale": L.out_scale} for L in qm.layers]},
                      f, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train + PTQ + export C headers")
    ap.add_argument("--out", default="build/model")
    ap.add_argument("--bits", type=int, choices=(4, 8), default=8)
    ap.add_argument("--clip", choices=("minmax", "mse"), default="minmax")
    ap.add_argument("--c1", type=int, default=8)
    ap.add_argument("--c2", type=int, default=16)
    ap.add_argument("--k1", type=int, default=5)
    ap.add_argument("--k2", type=int, default=3)
    a = ap.parse_args(argv)
    r = build(nn.Hyper(c1=a.c1, c2=a.c2, k1=a.k1, k2=a.k2), QuantConfig(a.bits, a.clip), a.out)
    print(f"FP32 acc {r['fp32_acc']:.4f} | {r['cfg']} acc {r['int_acc']:.4f} | exported -> {a.out} {r['payload']}")


if __name__ == "__main__":
    main()

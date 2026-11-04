"""Milestone 2: C runtime (host + Cortex-M3/QEMU) vs the Python integer reference."""
import os
import re
import subprocess

import numpy as np
import pytest

from compiler import planner, profiler
from compiler import quantizer as q
from conftest import ROOT, needs_arm


# ------------------------------------------------------------ host build ---
@pytest.mark.parametrize("name", ["int8", "int4"])
def test_host_bit_exact_vs_python(name, built, host_results):
    h = host_results[name]
    assert h["returncode"] == 0, h["stderr"]
    assert h["REF_MISMATCH"] == 0
    assert h["CHECKSUM_OK"] == 1
    assert h["SAMPLES"] == len(built[name]["yte"])


@pytest.mark.parametrize("name", ["int8", "int4"])
def test_host_accuracy_matches_python(name, built, host_results):
    r, h = built[name], host_results[name]
    n = h["SAMPLES"]
    assert h["ACC_INT"] == round(r["int_acc"] * n)
    assert h["ACC_FP32"] == round(r["fp32_acc"] * n)


@pytest.mark.parametrize("name", ["int8", "int4"])
def test_arena_peak_and_macs_match_planner(name, built, host_results):
    hp = None
    from compiler import model as nn
    hp = nn.Hyper()
    h = host_results[name]
    assert h["ARENA_PEAK"] == planner.simulate_arena(planner.activation_elems(hp))
    assert h["MACS"] == planner.mac_count(hp)
    assert h["MACS_VARIANT"] == 0


def test_arena_too_small_reports_error_and_exact_boundary(built, tmp_path):
    d = built["int8"]["dir"]
    peak = 464
    ok = profiler.run_host(d, str(tmp_path / "ok"), arena_size=peak)
    assert ok["returncode"] == 0 and ok["ARENA_PEAK"] == peak
    bad = profiler.run_host(d, str(tmp_path / "bad"), arena_size=peak - 4)
    assert bad["returncode"] != 0 and bad.get("ERROR") == 1


# ----------------------------------------------------------- no malloc -----
def test_runtime_sources_never_call_heap_functions():
    src = open(os.path.join(ROOT, "runtime", "tiny_nn.c")).read()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    assert "#include" in src and "<stdlib.h>" not in src
    for fn in ("malloc", "calloc", "realloc", "free", "alloca"):
        assert not re.search(rf"\b{fn}\s*\(", src), fn


@needs_arm
def test_runtime_object_has_no_libc_symbols(tmp_path):
    r = profiler._run(["make", "-B", "runtime-obj", f"OUT={tmp_path}"])
    assert r.returncode == 0, r.stderr
    syms = profiler.symbols(str(tmp_path / "tiny_nn_m3.o"))
    undefined = [ln for ln in profiler._run(["arm-none-eabi-nm", "-u", str(tmp_path / "tiny_nn_m3.o")]).stdout.split()]
    assert undefined == [], f"runtime depends on external symbols: {undefined}"
    assert not any(re.search(r"malloc|calloc|realloc|free|memcpy|printf", s) for s in syms)


def test_c99_strict_compile_of_runtime(tmp_path):
    r = subprocess.run(["gcc", "-std=c99", "-pedantic", "-Wall", "-Wextra", "-Werror", "-Wconversion", "-Wno-sign-conversion",
                        "-c", os.path.join(ROOT, "runtime", "tiny_nn.c"), "-o", str(tmp_path / "t.o")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ----------------------------------------------------------- unit kernels --
UNIT_C = r"""
#include <stdio.h>
#include "tiny_nn.h"
int main(void) {
    static uint8_t buf[64] __attribute__((aligned(8)));
    tn_arena a; int8_t r[5] = {-3, 0, 5, -128, 127}, m[4] = {1, 9, 9, 2};
    void *p0, *p1;
    tn_arena_init(&a, buf, 64);
    p0 = tn_arena_alloc(&a, 0, 10);   /* -> 12 */
    p1 = tn_arena_alloc(&a, 1, 20);   /* -> 20 */
    if ((uint8_t*)p0 != buf || (uint8_t*)p1 != buf + 44) return 1;
    if (a.peak != 32) return 2;
    if (tn_arena_alloc(&a, 0, 33) != 0) return 3;       /* 32 + 36 > 64 */
    if (tn_arena_alloc(&a, 0, 32) == 0) return 4;       /* exactly fits */
    tn_arena_release(&a, 0); tn_arena_reset(&a);
    if (a.peak != 64) return 5;
    tn_relu(r, 5, 0);
    if (r[0] != 0 || r[2] != 5 || r[3] != 0 || r[4] != 127) return 6;
    if (tn_argmax(m, 4) != 1) return 7;                 /* first max wins */
    puts("OK"); return 0;
}
"""


def test_arena_relu_argmax_units(tmp_path):
    src = tmp_path / "u.c"
    src.write_text(UNIT_C)
    subprocess.run(["gcc", "-std=c99", "-Wall", "-Werror", f"-I{ROOT}/runtime", str(src),
                    f"{ROOT}/runtime/tiny_nn.c", "-o", str(tmp_path / "u")], check=True)
    r = subprocess.run([str(tmp_path / "u")], capture_output=True, text=True)
    assert r.returncode == 0, f"unit check failed at step {r.returncode}"


CONV_C = r"""
#include <stdio.h>
#include "tiny_nn.h"
/* 1 in-ch, k=3, stride 2, 2 out-ch, INT4 weights with an odd row length (nibble straddle). */
static const uint8_t w4[3] = { 0x21, 0xF3, 0x7E };      /* elems: 1,2,3,-1,-2,7 (+pad) -> rows [1,2,3],[-1,-2,7] */
static const int32_t bias[2] = { 0, 0 };
static const int32_t mult[2] = { 1 << 30, 1 << 30 };    /* M = 0.5 */
static const uint8_t rsh[2]  = { 31, 31 };
int main(void) {
    tn_linear p = { 7, 1, 2, 3, 2, 4, 0, 0, w4, bias, mult, rsh };
    int8_t in[7] = { 10, 20, 30, 40, 50, 60, 70 }, out[6];
    tn_ctx c; c.macs = 0;
    tn_conv1d(&c, &p, in, out);
    /* pos0 window 10,20,30 ; pos1 30,40,50 ; pos2 50,60,70
       oc0 w=[1,2,3]: 140,260,380 -> *0.5 = 70,130->127(sat),190->127
       oc1 w=[-1,-2,7]: 160,  (-30-80+350)=240, (-50-120+490)=320 -> 80,120,127 */
    printf("%d %d %d %d %d %d macs=%u\n", out[0], out[1], out[2], out[3], out[4], out[5], c.macs);
    return 0;
}
"""


def test_int4_conv_odd_row_length_known_answer(tmp_path):
    src = tmp_path / "c.c"
    src.write_text(CONV_C)
    subprocess.run(["gcc", "-std=c99", "-Wall", "-Werror", f"-I{ROOT}/runtime", str(src),
                    f"{ROOT}/runtime/tiny_nn.c", "-o", str(tmp_path / "c")], check=True)
    r = subprocess.run([str(tmp_path / "c")], capture_output=True, text=True)
    assert r.stdout.strip() == "70 80 127 120 127 127 macs=18", r.stdout


# ------------------------------------------------------------ cortex-m3 ----
@needs_arm
@pytest.mark.parametrize("name", ["int8", "int4"])
def test_qemu_cortex_m3_matches_host_exactly(name, host_results, qemu_results):
    qr, h = qemu_results[name], host_results[name]
    assert qr["returncode"] == 0, qr["stderr"]
    for k in ("ARENA_PEAK", "MACS", "ACC_INT", "ACC_FP32", "REF_MISMATCH", "LOGIT_CHECKSUM", "CHECKSUM_OK"):
        assert qr[k] == h[k], k


@needs_arm
def test_qemu_flash_footprint_from_elf(built, qemu_results):
    for name in ("int8", "int4"):
        qr = qemu_results[name]
        assert qr["model_rodata"] == built[name]["payload"]["total"]  # weights+bias+mult+rshift, exactly
        assert 300 < qr["runtime_text"] < 3000
    assert qemu_results["int4"]["model_rodata"] < qemu_results["int8"]["model_rodata"]
    assert qemu_results["int4"]["static_sram"] >= 4096  # static arena sits in .bss


@needs_arm
def test_quantization_accuracy_degradation_on_target(qemu_results):
    for name, tol in (("int8", 0.03), ("int4", 0.10)):
        qr = qemu_results[name]
        deg = (qr["ACC_FP32"] - qr["ACC_INT"]) / qr["SAMPLES"]
        assert deg <= tol, (name, deg)

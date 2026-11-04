"""Drives the C benchmark (host gcc and ARM Cortex-M3 under QEMU) and collects
exact footprint numbers from the linked ELF.

Measured (not estimated):
  arena_peak      bytes, high-water mark recorded by the runtime's allocator
  macs            MACs executed in one inference, counted by the runtime
  model_rodata    bytes of weights+bias+requant params, from the ELF section table
  runtime_text    bytes of Flash code for runtime/tiny_nn.c (-Os, Cortex-M3)
  static_sram     .bss + .data of the whole firmware image (includes the arena)
  accuracy        int-engine vs FP32 predictions on the exported held-out set
"""
from __future__ import annotations

import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def parse_kv(text: str) -> dict:
    out = {}
    for m in re.finditer(r"^([A-Z0-9_]+)=(.*)$", text, re.M):
        v = m.group(2).strip()
        out[m.group(1)] = int(v) if v.lstrip("-").isdigit() else v
    out["done"] = "DONE" in text
    return out


def elf_sections(elf: str) -> dict:
    r = _run(["arm-none-eabi-size", "-A", elf], check=True)
    sec = {}
    for line in r.stdout.splitlines():
        m = re.match(r"^(\.\S+)\s+(\d+)\s+(\d+)", line)
        if m:
            sec[m.group(1)] = int(m.group(2))
    return sec


def object_text_bytes(obj: str) -> int:
    r = _run(["arm-none-eabi-size", obj], check=True)
    return int(r.stdout.splitlines()[1].split()[0])


def symbols(obj: str) -> list:
    r = _run(["arm-none-eabi-nm", obj], check=True)
    return [ln.split()[-1] for ln in r.stdout.splitlines() if ln.strip()]


def run_host(model_dir: str, out_dir: str, arena_size: int = 4096) -> dict:
    r = _run(["make", "-B", "host", f"MODEL_DIR={model_dir}", f"OUT={out_dir}", f"ARENA_SIZE={arena_size}"])
    res = parse_kv(r.stdout)
    res["returncode"], res["stderr"] = r.returncode, r.stderr
    return res


def run_qemu(model_dir: str, out_dir: str, arena_size: int = 4096) -> dict:
    args = [f"MODEL_DIR={model_dir}", f"OUT={out_dir}", f"ARENA_SIZE={arena_size}"]
    r = _run(["make", "-B", "qemu"] + args)
    res = parse_kv(r.stdout + "\n" + r.stderr)  # semihosting writes to QEMU's stderr
    res["returncode"], res["stderr"] = r.returncode, r.stderr
    elf = os.path.join(ROOT, out_dir, "bench_m3.elf") if not os.path.isabs(out_dir) else os.path.join(out_dir, "bench_m3.elf")
    if os.path.exists(elf):
        sec = elf_sections(elf)
        res["model_rodata"] = sec.get(".model_rodata", 0)
        res["static_sram"] = sec.get(".bss", 0) + sec.get(".data", 0)
        res["firmware_text"] = sec.get(".text", 0)
    ro = _run(["make", "-B", "runtime-obj"] + args)
    obj = os.path.join(out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir), "tiny_nn_m3.o")
    if ro.returncode == 0 and os.path.exists(obj):
        res["runtime_text"] = object_text_bytes(obj)
        res["runtime_obj"] = obj
    return res

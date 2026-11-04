#!/usr/bin/env python3
"""Hyperparameter x quantisation x arena-budget sweep -> reports/pareto_benchmark.md

For each architecture (conv widths x first-kernel size):
  * train FP32 for 3 seeds; PTQ to INT8 / INT4(minmax) / INT4(MSE clip);
    accuracy = mean +- std over seeds (bit-exact integer Python reference).
  * seed 0 is exported to C and run on a Cortex-M3 under QEMU: arena peak, MACs,
    Flash .model_rodata and engine .text are MEASURED from the target/ELF.
  * each arena budget is validated on target: the smallest fitting budget must
    run clean and the next smaller one must fail with TN_ERR_ARENA.
FP32 rows are analytic (no FP32 kernel is shipped): weights*4 B, 4 B activations.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from compiler import model as nn            # noqa: E402
from compiler import planner, profiler      # noqa: E402
from compiler import quantizer as q         # noqa: E402

WIDTHS = [(4, 8), (8, 16), (16, 32), (32, 64)]
KERNELS = [3, 5]
SEEDS = [0, 1, 2]
QUANTS = [("INT8", q.QuantConfig(8, "minmax")),
          ("INT4", q.QuantConfig(4, "minmax")),
          ("INT4-MSE", q.QuantConfig(4, "mse"))]
BUDGETS = [512, 1024, 2048, 4096, 8192]
BUILD = os.path.join(ROOT, "build", "sweep")


def evaluate(c1, c2, k1):
    """Accuracy stats over seeds for FP32 and every quantiser."""
    acc = {"FP32": []}
    for n, _ in QUANTS:
        acc[n] = []
    seed0 = None
    for s in SEEDS:
        hp = nn.Hyper(c1=c1, c2=c2, k1=k1, seed=s)
        for n, cfg in QUANTS:
            r = q.build(hp, cfg)
            acc[n].append(r["int_acc"])
            if n == "INT8":
                acc["FP32"].append(r["fp32_acc"])
        if s == 0:
            seed0 = hp
    return seed0, acc


def on_target(hp, qname, cfg):
    tag = f"c{hp.c1}_{hp.c2}_k{hp.k1}_{qname}"
    d = os.path.join(BUILD, tag)
    r = q.build(hp, cfg, os.path.join(d, "model"))
    res = profiler.run_qemu(os.path.join(d, "model"), os.path.join(d, "big"), 32768)
    assert res.get("done") and res["returncode"] == 0 and res["REF_MISMATCH"] == 0, (tag, res)
    peak = res["ARENA_PEAK"]
    assert peak == planner.simulate_arena(planner.activation_elems(hp)), tag
    assert res["MACS"] == planner.mac_count(hp), tag
    fit = {}
    for b in BUDGETS:
        pred = peak <= b
        # validate the boundary on target: only the smallest fitting and the largest failing budget
        fit[b] = pred
    fitting = [b for b in BUDGETS if fit[b]]
    failing = [b for b in BUDGETS if not fit[b]]
    checks = []
    if fitting:
        ok = profiler.run_qemu(os.path.join(d, "model"), os.path.join(d, "fit"), fitting[0])
        checks.append(("fit", fitting[0], ok.get("done") and ok["returncode"] == 0 and ok["ARENA_PEAK"] == peak))
    if failing:
        bad = profiler.run_qemu(os.path.join(d, "model"), os.path.join(d, "fail"), failing[-1])
        checks.append(("fail", failing[-1], bad.get("ERROR") == 1))
    res["budget_checks"] = checks
    res["fit"] = fit
    res["payload"] = r["payload"]
    return res


def pareto_mask(points):
    """points: list of (err, flash, arena); returns non-dominated flags (all minimised)."""
    flags = []
    for i, p in enumerate(points):
        dom = any(all(a <= b for a, b in zip(o, p)) and any(a < b for a, b in zip(o, p))
                  for j, o in enumerate(points) if j != i)
        flags.append(not dom)
    return flags


def main():
    t0 = time.time()
    archs = [(c1, c2, k1) for (c1, c2) in WIDTHS for k1 in KERNELS]
    rows = []
    jobs = []
    for (c1, c2, k1) in archs:
        hp0, acc = evaluate(c1, c2, k1)
        print(f"[train] c=({c1},{c2}) k1={k1}  " + "  ".join(f"{n}={np.mean(v):.4f}" for n, v in acc.items()), flush=True)
        label = f"{c1}-{c2}/k{k1}"
        params = planner.param_count(hp0)
        fp_arena = planner.simulate_arena(planner.activation_elems(hp0), 4)
        rows.append(dict(arch=label, prec="FP32", acc_mean=float(np.mean(acc["FP32"])), acc_std=float(np.std(acc["FP32"])),
                         flash=params * 4, code=None, arena=fp_arena, macs=planner.mac_count(hp0), measured=False,
                         params=params, fit={b: fp_arena <= b for b in BUDGETS}))
        for n, cfg in QUANTS:
            jobs.append((hp0, n, cfg, label, params, acc[n]))
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [(j, ex.submit(on_target, j[0], j[1], j[2])) for j in jobs]
        for (hp0, n, cfg, label, params, a), f in futs:
            res = f.result()
            rows.append(dict(arch=label, prec=n, acc_mean=float(np.mean(a)), acc_std=float(np.std(a)),
                             flash=res["model_rodata"], code=res["runtime_text"], arena=res["ARENA_PEAK"],
                             macs=res["MACS"], measured=True, params=params, fit=res["fit"],
                             target_acc_seed0=res["ACC_INT"] / res["SAMPLES"], fp32_acc_seed0=res["ACC_FP32"] / res["SAMPLES"],
                             budget_checks=res["budget_checks"], static_sram=res["static_sram"]))
    flags = pareto_mask([(1 - r["acc_mean"], r["flash"], r["arena"]) for r in rows])
    for r, f in zip(rows, flags):
        r["pareto"] = f
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "sweep_results.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    write_report(rows, time.time() - t0)
    write_svg(rows)
    bad = [r for r in rows if r.get("budget_checks") and not all(c[2] for c in r["budget_checks"])]
    assert not bad, bad
    print(f"done in {time.time() - t0:.0f}s; {sum(flags)} Pareto-optimal of {len(rows)}")


PREC_ORDER = {"FP32": 0, "INT8": 1, "INT4": 2, "INT4-MSE": 3}


def fmt_b(n):
    return f"{n:,}"


def write_report(rows, secs):
    rows = sorted(rows, key=lambda r: (r["flash"], r["arena"], -r["acc_mean"]))
    L = []
    L.append("# Pareto benchmark: FP32 vs INT8 vs INT4\n")
    L.append("_Generated by `scripts/run_sweep.py` - do not edit by hand._\n")
    L.append("## Method\n")
    L.append("* **Task**: 4-class sensor anomaly detection on 64-sample windows (normal / spike / burst / level-shift), "
             "synthetic data, 1,600 train / 400 held-out test windows.")
    L.append("* **Models**: `Conv1D(k1, s2) -> ReLU -> Conv1D(3, s2) -> ReLU -> Dense -> ArgMax`, widths swept over "
             "`c1-c2` in {4-8, 8-16, 16-32, 32-64} and first kernel `k1` in {3, 5}.")
    L.append("* **Precisions**: `FP32` (reference), `INT8` (W8A8), `INT4` (W4A8, min-max per-channel), "
             "`INT4-MSE` (W4A8, MSE-optimal clipping). Activations are always INT8; INT32 accumulators.")
    L.append("* **Accuracy** is mean +- std over 3 training seeds (integer Python reference, proven bit-exact against the target).")
    L.append("* **Footprints for INT rows are measured**, for the seed-0 model on a QEMU Cortex-M3 (`lm3s6965evb`, "
             "`-Os`, `arm-none-eabi-gcc`): arena peak from the runtime's allocator, MACs counted by the runtime, "
             "Flash `.model_rodata` from the linked ELF section table, engine code from `size` on `tiny_nn.o`.")
    L.append("* **FP32 rows are analytic**: Flash = 4 B x parameters (weights + bias only, no FP32 kernel is shipped), "
             "arena = same double-ended plan with 4 B activations.")
    L.append("* **Pareto objectives (all 3 simultaneously)**: maximise accuracy, minimise model Flash, minimise arena peak. "
             "A row is on the frontier if no other row is at least as good on all three and strictly better on one.")
    L.append("* QEMU is functional, not cycle-accurate, so no latency numbers are reported.\n")

    L.append("## Pareto frontier\n")
    L.append("| Arch (c1-c2/k1) | Precision | Accuracy % (mean+-std) | Model Flash (B) | Arena peak (B) | MACs/inf | Min arena budget |")
    L.append("|---|---|---|---|---|---|---|")
    for r in [r for r in rows if r["pareto"]]:
        mb = next((b for b in BUDGETS if r["fit"][b]), None)
        L.append(f"| {r['arch']} | **{r['prec']}** | {100 * r['acc_mean']:.2f} +- {100 * r['acc_std']:.2f} | {fmt_b(r['flash'])} | "
                 f"{fmt_b(r['arena'])} | {fmt_b(r['macs'])} | {fmt_b(mb) + ' B' if mb else '> ' + fmt_b(BUDGETS[-1]) + ' B'} |")
    L.append("")
    L.append("## Full sweep (sorted by model Flash)\n")
    L.append("`*` = Pareto-optimal. `m` = measured on target, `a` = analytic.\n")
    L.append("| | Arch | Prec | Acc % | dAcc vs FP32 (pp) | Model Flash (B) | Engine .text (B) | Arena peak (B) | MACs/inf | Params | src |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    fp = {r["arch"]: r["acc_mean"] for r in rows if r["prec"] == "FP32"}
    for r in rows:
        d = 100 * (r["acc_mean"] - fp[r["arch"]])
        L.append(f"| {'*' if r['pareto'] else ''} | {r['arch']} | {r['prec']} | {100 * r['acc_mean']:.2f} +- {100 * r['acc_std']:.2f} | "
                 f"{d:+.2f} | {fmt_b(r['flash'])} | {fmt_b(r['code']) if r['code'] else '-'} | {fmt_b(r['arena'])} | "
                 f"{fmt_b(r['macs'])} | {fmt_b(r['params'])} | {'m' if r['measured'] else 'a'} |")
    L.append("")
    L.append("## Arena-size sweep: which model fits which static SRAM arena?\n")
    L.append("`Y` = fits (INT rows: boundary verified on target - smallest fitting budget runs clean, next smaller returns `TN_ERR_ARENA`).\n")
    L.append("| Arch | Prec | Arena peak (B) | " + " | ".join(f"{b} B" for b in BUDGETS) + " |")
    L.append("|---|---|---|" + "---|" * len(BUDGETS))
    for r in sorted(rows, key=lambda r: (r["arena"], r["arch"], PREC_ORDER[r["prec"]])):
        if r["prec"] == "INT4-MSE":
            continue  # identical shapes/arena to INT4
        L.append(f"| {r['arch']} | {r['prec']} | {fmt_b(r['arena'])} | " + " | ".join("Y" if r["fit"][b] else "-" for b in BUDGETS) + " |")
    L.append("")
    L.append("## Quantisation degradation (mean over architectures)\n")
    L.append("| Precision | Mean dAcc vs FP32 (pp) | Worst dAcc (pp) | Flash vs FP32 | Arena vs FP32 |")
    L.append("|---|---|---|---|---|")
    for prec in ("INT8", "INT4", "INT4-MSE"):
        ds, fr, ar = [], [], []
        for r in rows:
            if r["prec"] != prec:
                continue
            f = next(x for x in rows if x["arch"] == r["arch"] and x["prec"] == "FP32")
            ds.append(100 * (r["acc_mean"] - f["acc_mean"]))
            fr.append(r["flash"] / f["flash"])
            ar.append(r["arena"] / f["arena"])
        L.append(f"| {prec} | {np.mean(ds):+.2f} | {min(ds):+.2f} | {np.mean(fr):.2f}x | {np.mean(ar):.2f}x |")
    L.append("")
    L.append("## Caveats\n")
    L.append("* Synthetic data; absolute accuracies are not transferable to real sensors. Relative trends are the point.")
    L.append("* 400 test windows -> one window = 0.25 pp; differences below ~1 pp are within noise (see std column).")
    L.append("* INT rows' Flash includes per-channel bias / multiplier / shift metadata; FP32 Flash is weights+bias only.")
    L.append(f"* Sweep wall-clock: {secs:.0f}s.\n")
    with open(os.path.join(ROOT, "reports", "pareto_benchmark.md"), "w") as f:
        f.write("\n".join(L))


def write_svg(rows):
    W, H, ml, mr, mt, mb = 720, 420, 64, 150, 24, 52
    xs = [r["flash"] for r in rows]
    ys = [100 * r["acc_mean"] for r in rows]
    lx0, lx1 = np.log10(min(xs)) - 0.1, np.log10(max(xs)) + 0.1
    y0, y1 = np.floor(min(ys)) - 1, np.ceil(max(ys)) + 1
    px = lambda v: ml + (np.log10(v) - lx0) / (lx1 - lx0) * (W - ml - mr)
    py = lambda v: mt + (1 - (v - y0) / (y1 - y0)) * (H - mt - mb)
    col = {"FP32": "#888780", "INT8": "#185FA5", "INT4": "#BA7517", "INT4-MSE": "#1D9E75"}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    for t in range(int(y0), int(y1) + 1, 2):
        o.append(f'<line x1="{ml}" x2="{W - mr}" y1="{py(t):.1f}" y2="{py(t):.1f}" stroke="#e5e5e5"/>'
                 f'<text x="{ml - 8}" y="{py(t) + 4:.1f}" text-anchor="end" fill="#444">{t}</text>')
    for e in range(int(np.ceil(lx0)), int(np.floor(lx1)) + 1):
        o.append(f'<line x1="{px(10 ** e):.1f}" x2="{px(10 ** e):.1f}" y1="{mt}" y2="{H - mb}" stroke="#e5e5e5"/>'
                 f'<text x="{px(10 ** e):.1f}" y="{H - mb + 16}" text-anchor="middle" fill="#444">{10 ** e:,}</text>')
    o.append(f'<text x="{(ml + W - mr) / 2}" y="{H - 10}" text-anchor="middle" fill="#222">Model Flash (bytes, log scale)</text>')
    o.append(f'<text transform="translate(16,{(mt + H - mb) / 2}) rotate(-90)" text-anchor="middle" fill="#222">Held-out accuracy (%)</text>')
    for r in rows:
        x, y = px(r["flash"]), py(100 * r["acc_mean"])
        rad = 4 + 2.2 * np.log2(max(r["arena"], 64) / 64) ** 0.9
        st = ' stroke="#111" stroke-width="2"' if r["pareto"] else ' stroke="white" stroke-width="1"'
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{col[r["prec"]]}" fill-opacity="0.8"{st}>'
                 f'<title>{r["arch"]} {r["prec"]}: {100 * r["acc_mean"]:.2f}% / {r["flash"]} B / arena {r["arena"]} B</title></circle>')
    lx = W - mr + 16
    for i, (k, c) in enumerate(col.items()):
        o.append(f'<circle cx="{lx}" cy="{mt + 12 + i * 20}" r="6" fill="{c}"/><text x="{lx + 12}" y="{mt + 16 + i * 20}">{k}</text>')
    o.append(f'<circle cx="{lx}" cy="{mt + 100}" r="6" fill="none" stroke="#111" stroke-width="2"/><text x="{lx + 12}" y="{mt + 104}">Pareto-optimal</text>')
    o.append(f'<text x="{lx - 8}" y="{mt + 128}" fill="#444">marker size ~ arena peak</text>')
    o.append("</svg>")
    with open(os.path.join(ROOT, "reports", "pareto.svg"), "w") as f:
        f.write("\n".join(o))


if __name__ == "__main__":
    main()

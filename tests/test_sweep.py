"""Milestone 3: sweep logic + integrity of the committed benchmark artefacts."""
import json
import os
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_sweep  # noqa: E402

from compiler import model as nn, planner  # noqa: E402


def test_pareto_mask_semantics():
    pts = [(0.1, 100, 10), (0.2, 50, 10), (0.1, 100, 20), (0.3, 200, 30), (0.1, 100, 10)]
    # idx2 and idx3 are dominated; duplicates (0 and 4) do not dominate each other
    assert run_sweep.pareto_mask(pts) == [True, True, False, False, True]


@pytest.fixture(scope="module")
def rows():
    p = os.path.join(ROOT, "reports", "sweep_results.json")
    if not os.path.exists(p):
        pytest.skip("run scripts/run_sweep.py first")
    return json.load(open(p))


def test_sweep_covers_grid(rows):
    assert len(rows) == len(run_sweep.WIDTHS) * len(run_sweep.KERNELS) * (1 + len(run_sweep.QUANTS))
    assert {r["prec"] for r in rows} == {"FP32", "INT8", "INT4", "INT4-MSE"}


def test_measured_numbers_match_planner(rows):
    for r in rows:
        c, k = r["arch"].split("/k")
        c1, c2 = map(int, c.split("-"))
        hp = nn.Hyper(c1=c1, c2=c2, k1=int(k))
        assert r["macs"] == planner.mac_count(hp)
        bpe = 4 if r["prec"] == "FP32" else 1
        assert r["arena"] == planner.simulate_arena(planner.activation_elems(hp), bpe)
        if r["measured"]:
            assert all(ok for _, _, ok in r["budget_checks"]), r["arch"]


def test_int_beats_fp32_on_memory(rows):
    by = {(r["arch"], r["prec"]): r for r in rows}
    for (arch, prec), r in by.items():
        if prec != "FP32":
            assert r["flash"] < by[(arch, "FP32")]["flash"]
            assert r["arena"] * 4 == by[(arch, "FP32")]["arena"]


def test_report_and_chart_exist_with_sections():
    md = open(os.path.join(ROOT, "reports", "pareto_benchmark.md")).read()
    for h in ("## Pareto frontier", "## Full sweep", "## Arena-size sweep", "## Quantisation degradation", "## Caveats"):
        assert h in md
    assert os.path.exists(os.path.join(ROOT, "reports", "pareto.svg"))

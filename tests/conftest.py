import os
import shutil
import pytest

from compiler import model as nn
from compiler import quantizer as q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def hp():
    return nn.Hyper()


@pytest.fixture(scope="session")
def built(tmp_path_factory, hp):
    """int8 + int4 builds exported to disk (session scoped: training runs once)."""
    base = tmp_path_factory.mktemp("models")
    out = {}
    for name, cfg in {"int8": q.QuantConfig(8), "int4": q.QuantConfig(4, "mse")}.items():
        d = str(base / name)
        r = q.build(hp, cfg, d)
        r["dir"] = d
        out[name] = r
    return out


def has_tool(name):
    return shutil.which(name) is not None


needs_arm = pytest.mark.skipif(not (has_tool("arm-none-eabi-gcc") and has_tool("qemu-system-arm")),
                               reason="ARM toolchain / QEMU not installed")


@pytest.fixture(scope="session")
def host_results(built, tmp_path_factory):
    from compiler import profiler
    out = {}
    for name, r in built.items():
        out[name] = profiler.run_host(r["dir"], str(tmp_path_factory.mktemp("host_" + name)))
    return out


@pytest.fixture(scope="session")
def qemu_results(built, tmp_path_factory):
    from compiler import profiler
    out = {}
    for name, r in built.items():
        out[name] = profiler.run_qemu(r["dir"], str(tmp_path_factory.mktemp("qemu_" + name)))
    return out

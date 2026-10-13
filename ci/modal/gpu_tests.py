"""Run the GPU test suite on a real GPU, via Modal (https://modal.com).

The build CI (.github/workflows/build.yml) compiles on GPU-less runners, so
every `requires_gpu` test skips there. This script runs that suite on a GPU,
and *only* runs it: nothing is compiled on Modal. It installs the wheels the
build jobs already produced (lightsim2grid + gpusim2grid, CUDA 13, FP64,
sm_75) into a prebuilt image and calls pytest:

    G2G_WHEELS=path/to/wheels modal run ci/modal/gpu_tests.py
    G2G_WHEELS=... modal run ci/modal/gpu_tests.py --pytest-args "-x -k contingency"

G2G_WHEELS is a directory holding one lightsim2grid and one gpusim2grid
wheel (the `lightsim2grid-dist` and `gpusim2grid-wheel-cu13-fp64` artifacts
of a build run). The image below never changes with the code, so Modal
builds it once and every later run goes straight to the tests; the wheels
and tests/python are mounted into the container when it starts.

Configuration (read at image-definition time, baked in with `.env()` so the
module re-imported inside the container sees the same values):

    G2G_CUDA       nvidia/cuda runtime image version   (default 13.4.2)
    G2G_PRECISION  fp64 | fp32, checked against the wheel (default fp64)
    G2G_GPU        Modal GPU type                      (default T4)
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

CUDA = os.environ.get("G2G_CUDA", "13.4.2")
PRECISION = os.environ.get("G2G_PRECISION", "fp64")
GPU = os.environ.get("G2G_GPU", "T4")
CUDA_MAJOR = CUDA.split(".")[0]
# Must match the Python the wheels were built with (ubuntu22.04's python3.10).
PY = "3.10"

image = (
    # The runtime image ships cuSPARSE/cuBLAS/cuSOLVER, no compiler needed.
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-runtime-ubuntu22.04", add_python=PY)
    .pip_install(f"nvidia-cudss-cu{CUDA_MAJOR}>=0.8", "numpy", "scipy", "pytest",
                 "pytest-timeout", "pandapower", "pypowsybl")
    # Enables the DLPack / differentiable tests (importorskip("torch") otherwise).
    .pip_install("torch", index_url="https://download.pytorch.org/whl/"
                 + {"12": "cu126", "13": "cu130"}[CUDA_MAJOR])
    .env({"G2G_CUDA": CUDA, "G2G_PRECISION": PRECISION, "G2G_GPU": GPU})
)
# Local paths only exist on the machine calling `modal run`: inside the
# container this file is /root/gpu_tests.py, where parents[2] does not exist.
if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
    WHEELS = os.environ.get("G2G_WHEELS", str(REPO / "ci" / "modal" / "wheels"))
    # Mounted at container start (no copy=True): changing them never rebuilds
    # the image above.
    image = (
        image.add_local_dir(WHEELS, "/wheels")
        .add_local_dir(REPO / "tests" / "python", "/root/tests/python",
                       ignore=["**/__pycache__"])
    )

app = modal.App("gpusim2grid-gpu-tests")


@app.function(image=image, gpu=GPU, timeout=20 * 60)
def run_tests(pytest_args: str) -> int:
    subprocess.run(["nvidia-smi"], check=True)
    wheels = sorted(str(p) for p in Path("/wheels").glob("*.whl"))
    print("installing", *wheels, sep="\n  ", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *wheels],
                   check=True)

    import importlib
    cudss = list(importlib.import_module(f"nvidia.cu{CUDA_MAJOR}").__path__)[0]
    env = dict(os.environ,
               LD_LIBRARY_PATH=f"{cudss}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))

    # Refuse a green run where every requires_gpu test just skipped, or a
    # wheel of the wrong precision.
    check = (
        "import sys; sys.path.insert(0, 'tests/python')\n"
        "from conftest import _cuda_device_count\n"
        "assert _cuda_device_count() > 0, 'no CUDA device visible (driver too old?)'\n"
        "from gpusim2grid import _gpusim2grid as m\n"
        f"assert bool(m.is_fp32) == {PRECISION == 'fp32'}, 'wrong precision'\n"
        "print('gpusim2grid is_fp32 =', m.is_fp32)\n"
    )
    subprocess.run([sys.executable, "-c", check], cwd="/root", env=env, check=True)

    # --timeout turns a hang into a failure; --durations shows where time goes.
    cmd = [sys.executable, "-m", "pytest", "tests/python", "-rs",
           "-p", "no:cacheprovider", "--timeout=300", "--durations=25",
           *shlex.split(pytest_args)]
    return subprocess.run(cmd, cwd="/root", env=env).returncode


@app.local_entrypoint()
def main(pytest_args: str = ""):
    rc = run_tests.remote(pytest_args)
    if rc != 0:
        sys.exit(rc)

"""Run the GPU test suite on a real GPU, via Modal (https://modal.com).

The build CI (.github/workflows/build.yml) compiles on GPU-less runners, so
every `requires_gpu` test skips there. This script builds the same stack in
an image on Modal and runs pytest on an actual GPU:

    pip install modal && modal token new      # once
    modal run ci/modal/gpu_tests.py           # CUDA 13, FP64, T4, full suite
    modal run ci/modal/gpu_tests.py --pytest-args "-m 'not slow' -x"

The configuration is read from environment variables at *image definition*
time and baked into the image with `.env()`, so the module re-imported inside
the container sees the same values:

    G2G_CUDA       nvidia/cuda image tag version       (default 13.4.2)
    G2G_PRECISION  fp64 | fp32                         (default fp64)
    G2G_GPU        Modal GPU type, see GPU_ARCH below  (default T4)
    LS2G_SHA       lightsim2grid commit or ref         (default master)

Image layers are cached by Modal: the lightsim2grid layer is rebuilt only
when LS2G_SHA changes; a change to this repository only redoes the last one.
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
LS2G_SHA = os.environ.get("LS2G_SHA", "master")

# The project leaves CMAKE_CUDA_ARCHITECTURES unset (nvcc's default arch plus
# PTX JIT); pin native SASS for the GPU the tests actually run on.
GPU_ARCH = {"T4": "75", "A10G": "86", "L4": "89", "L40S": "89",
            "A100": "80", "A100-80GB": "80", "H100": "90"}
ARCH = GPU_ARCH[GPU]
CUDA_MAJOR = CUDA.split(".")[0]
PREC_ENV = {"fp64": "CUDA_REAL_DOUBLE", "fp32": "CUDA_REAL_FLOAT"}[PRECISION]

PY = "3.12"
SITE = f"/usr/local/lib/python{PY}/site-packages"
# nvidia-cudss-cuXX installs under <site-packages>/nvidia/cuXX; it ships no
# CMake config, ci/cmake/cudss is the shim the build CI uses too.
CUDSS_ROOT = f"{SITE}/nvidia/cu{CUDA_MAJOR}"
SRC = "/root/gpusim2grid"
REPO = Path(__file__).resolve().parents[2]

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-devel-ubuntu22.04", add_python=PY)
    # The Modal base environment exports CXX=clang++, which the CUDA image does
    # not ship; build with the image's gcc, as the GitHub build jobs do.
    .env({"CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++"})
    .apt_install("git", "cmake", "build-essential")
    .pip_install(f"nvidia-cudss-cu{CUDA_MAJOR}>=0.8", "scikit-build-core", "pybind11",
                 "numpy", "scipy", "pytest", "pandapower", "pypowsybl",
                 "setuptools", "wheel")
    # Enables the DLPack / differentiable tests (importorskip("torch") otherwise).
    .pip_install("torch", index_url="https://download.pytorch.org/whl/"
                 + {"12": "cu126", "13": "cu130"}[CUDA_MAJOR])
    # lightsim2grid from source: gpusim2grid's C++ bridge builds against it.
    .run_commands(
        "git init -q /opt/ls2g && cd /opt/ls2g"
        " && git remote add origin https://github.com/Grid2op/lightsim2grid"
        f" && git fetch -q --depth 1 origin {LS2G_SHA} && git checkout -q FETCH_HEAD"
        " && git submodule update --init --depth 1 SuiteSparse eigen"
        " && CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip install --no-build-isolation ."
    )
    .env({
        "G2G_CUDA": CUDA, "G2G_PRECISION": PRECISION, "G2G_GPU": GPU,
        "LS2G_SHA": LS2G_SHA,
        PREC_ENV: "1",
        "CUDSS_WHEEL_ROOT": CUDSS_ROOT,
        "cudss_DIR": f"{SRC}/ci/cmake/cudss",
        "CMAKE_ARGS": f"-DCMAKE_CUDA_ARCHITECTURES={ARCH}"
                      " -DGPUSIM2GRID_SUITESPARSE_DIR=/opt/ls2g/SuiteSparse",
        # Keep the nvidia/cuda image's own entries (driver library path).
        "LD_LIBRARY_PATH": f"{CUDSS_ROOT}/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
    })
    # Repository last, so a code change only rebuilds this layer.
    .add_local_dir(REPO, SRC, copy=True,
                   ignore=[".git", "build", "dist", "_skbuild", "**/__pycache__",
                           "**/*.egg-info", "**/*.so"])
    .run_commands(f"cd {SRC} && CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)"
                  " pip install -v --no-build-isolation .")
)

app = modal.App("gpusim2grid-gpu-tests")


@app.function(image=image, gpu=GPU, timeout=3600)
def run_tests(pytest_args: str) -> int:
    subprocess.run(["nvidia-smi"], check=True)

    # Guard against a green run where every requires_gpu test just skipped.
    sys.path.insert(0, f"{SRC}/tests/python")
    from conftest import _cuda_device_count
    if _cuda_device_count() == 0:
        raise RuntimeError("no CUDA device visible (driver too old for "
                           f"CUDA {CUDA}?) -- see nvidia-smi output above")

    from gpusim2grid import _gpusim2grid as m
    print(f"gpusim2grid: CUDA {CUDA}, {PRECISION}, sm_{ARCH} on {GPU}, "
          f"is_fp32={m.is_fp32}", flush=True)
    if bool(m.is_fp32) != (PRECISION == "fp32"):
        raise RuntimeError("extension compiled with the wrong precision")

    # From the source tree so conftest.py/pytest.ini apply; the installed
    # package still wins (src/ layout, the tree has no importable gpusim2grid).
    cmd = [sys.executable, "-m", "pytest", "tests/python", "-rs",
           "-p", "no:cacheprovider", *shlex.split(pytest_args)]
    return subprocess.run(cmd, cwd=SRC).returncode


@app.local_entrypoint()
def main(pytest_args: str = ""):
    rc = run_tests.remote(pytest_args)
    if rc != 0:
        sys.exit(rc)

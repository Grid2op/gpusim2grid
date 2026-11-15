# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The CUDA build requirements the in-tree backend picks, and the checks of
``import gpusim2grid`` (no GPU needed)."""

import os
import sys

import pytest

from gpusim2grid import _build_checks as bc

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def backend():
    pytest.importorskip("scikit_build_core")
    sys.path.insert(0, os.path.join(_REPO, "_build_backend"))
    try:
        import gpusim2grid_build
        yield gpusim2grid_build
    finally:
        sys.path.pop(0)


def _fake_nvcc(versions):
    def nvcc_major(path):
        return versions[path]
    return nvcc_major


def _no_which(_):
    return None


def test_default_is_cuda13_toolkit_from_pypi(backend):
    reqs = backend.cuda_requirements(env={}, which=_no_which, nvcc_major=_fake_nvcc({}))
    assert reqs == [backend.CMAKE_REQ, backend.CUDA_TOOLKIT_REQ[13], backend.CUDSS_REQ[13]]


def test_nvcc_on_path_alone_does_not_choose(backend):
    reqs = backend.cuda_requirements(env={}, which=lambda _: "/usr/bin/nvcc",
                                     nvcc_major=_fake_nvcc({"/usr/bin/nvcc": 12}))
    assert reqs == [backend.CMAKE_REQ, backend.CUDA_TOOLKIT_REQ[13], backend.CUDSS_REQ[13]]


def test_cudacxx_12_takes_cudss_cu12_only(backend):
    env = {"CUDACXX": "/opt/cuda-12.4/bin/nvcc"}
    reqs = backend.cuda_requirements(env=env, which=_no_which,
                                     nvcc_major=_fake_nvcc({env["CUDACXX"]: 12}))
    assert reqs == [backend.CMAKE_REQ, backend.CUDSS_REQ[12]]


def test_toolkit_root_13_takes_no_pypi_toolkit(backend):
    env = {"CUDAToolkit_ROOT": "/opt/cuda-13"}
    reqs = backend.cuda_requirements(env=env, which=_no_which,
                                     nvcc_major=_fake_nvcc({"/opt/cuda-13/bin/nvcc": 13}))
    assert reqs == [backend.CMAKE_REQ, backend.CUDSS_REQ[13]]


def test_explicit_12_uses_the_nvcc_on_path(backend):
    reqs = backend.cuda_requirements(env={"GPUSIM2GRID_CUDA": "12"},
                                     which=lambda _: "/usr/bin/nvcc",
                                     nvcc_major=_fake_nvcc({"/usr/bin/nvcc": 12}))
    assert reqs == [backend.CMAKE_REQ, backend.CUDSS_REQ[12]]


def test_explicit_12_without_a_cuda12_compiler_is_refused(backend):
    with pytest.raises(backend.CudaSelectionError, match="PyPI has no CUDA 12 compiler"):
        backend.cuda_requirements(env={"GPUSIM2GRID_CUDA": "12"}, which=_no_which,
                                  nvcc_major=_fake_nvcc({}))
    with pytest.raises(backend.CudaSelectionError, match="nvcc on PATH is CUDA 13"):
        backend.cuda_requirements(env={"GPUSIM2GRID_CUDA": "12"},
                                  which=lambda _: "/usr/bin/nvcc",
                                  nvcc_major=_fake_nvcc({"/usr/bin/nvcc": 13}))


def test_explicit_disagreeing_with_cudacxx_is_refused(backend):
    env = {"GPUSIM2GRID_CUDA": "12", "CUDACXX": "/opt/cuda-13/bin/nvcc"}
    with pytest.raises(backend.CudaSelectionError, match="is CUDA 13"):
        backend.cuda_requirements(env=env, which=_no_which,
                                  nvcc_major=_fake_nvcc({env["CUDACXX"]: 13}))
    with pytest.raises(backend.CudaSelectionError, match="expected one of"):
        backend.cuda_requirements(env={"GPUSIM2GRID_CUDA": "11"}, which=_no_which,
                                  nvcc_major=_fake_nvcc({}))


def test_cudss_from_elsewhere(backend):
    reqs = backend.cuda_requirements(env={"GPUSIM2GRID_CUDSS_FROM_PYPI": "0"}, which=_no_which,
                                     nvcc_major=_fake_nvcc({}))
    assert reqs == [backend.CMAKE_REQ, backend.CUDA_TOOLKIT_REQ[13]]


def test_backend_requirements_match_the_extras(backend):
    tomllib = pytest.importorskip("tomllib")
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    for major in (12, 13):
        name, spec = backend.CUDSS_REQ[major].split(">=")
        assert any(r.startswith(name + ">=" + spec.split(",")[0]) for r in extras[f"cuda{major}"])


# --- the import-time checks -------------------------------------------------

CU12_CUDSS = ("/x/nvidia/cu12/lib/libcudss.so.1", ["libcublas.so.12", "libnvrtc.so.12", "libc.so.6"])
CU13_CUDSS = ("/x/nvidia/cu13/lib/libcudss.so.1", ["libcublas.so.13", "libnvrtc.so.13", "libc.so.6"])
CU12_CUSPARSE = ("/x/nvidia/cusparse/lib/libcusparse.so.12", ["libnvJitLink.so.12", "libm.so.6"])
CU13_CUSPARSE = ("/x/nvidia/cu13/lib/libcusparse.so.12", ["libm.so.6"])


def test_matching_libraries_pass():
    assert bc.cuda_major_mismatches(13, [CU13_CUDSS, CU13_CUSPARSE]) == []
    assert bc.cuda_major_mismatches(12, [CU12_CUDSS, CU12_CUSPARSE]) == []


def test_other_major_is_reported():
    msgs = bc.cuda_major_mismatches(13, [CU12_CUDSS, CU13_CUSPARSE])
    assert len(msgs) == 1 and "CUDA 12" in msgs[0] and "libcudss" in msgs[0]
    msgs = bc.cuda_major_mismatches(13, [CU13_CUDSS, CU12_CUSPARSE])
    assert len(msgs) == 1 and "libnvJitLink.so.12" in msgs[0]
    assert len(bc.cuda_major_mismatches(12, [CU13_CUDSS, CU13_CUSPARSE])) == 1


def test_driver_check():
    assert bc.driver_mismatch(13, 0) is None          # no driver: GPU-less CI
    assert bc.driver_mismatch(13, 13000) is None
    assert bc.driver_mismatch(12, 13000) is None
    assert "12.8" in bc.driver_mismatch(13, 12080)


def test_lightsim2grid_check():
    assert bc.lightsim2grid_mismatch("1.3.1", "1.3.1") is None
    assert bc.lightsim2grid_mismatch("", "1.3.1") is None   # built without the bridge
    assert "1.3.2" in bc.lightsim2grid_mismatch("1.3.1", "1.3.2")


def test_this_process_loaded_the_build_major():
    """The libraries the extension actually loaded are of its CUDA major (the
    check `import gpusim2grid` just passed, on real data)."""
    info = pytest.importorskip("gpusim2grid._build_info")
    from gpusim2grid import _gpusim2grid
    libs = _gpusim2grid._loaded_library_needs(["libcudss", "libcusparse"])
    names = [os.path.basename(p) for p, _ in libs]
    assert any(n.startswith("libcudss") for n in names), libs
    assert any(n.startswith("libcusparse") for n in names), libs
    cudss_needed = next(nd for p, nd in libs if os.path.basename(p).startswith("libcudss"))
    assert f"libcublas.so.{info.CUDA_MAJOR}" in cudss_needed, cudss_needed
    assert bc.cuda_major_mismatches(info.CUDA_MAJOR, libs) == []

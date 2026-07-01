"""Shared pytest fixtures for CroqTile tests."""

import os
import subprocess
import tempfile
import shutil

import pytest

from croqtile.runtime import (
    find_croqtile_bin, find_nvcc,
    compile_co_to_cuda, compile_co_to_script,
    run_compilation_script,
    find_runtime_include, find_cutlass_include,
)

CROQTILE_BIN = find_croqtile_bin()
HAS_CROQTILE = CROQTILE_BIN is not None
NVCC_BIN = find_nvcc()
HAS_NVCC = NVCC_BIN is not None
HAS_RUNTIME = find_runtime_include() is not None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: end-to-end compilation test")
    config.addinivalue_line(
        "markers", "gpu: requires GPU + nvcc for execution")


@pytest.fixture
def compile_co():
    """Fixture that compiles .co source -> CUDA source (no nvcc).

    Returns a function: compile_co(source: str) -> cuda_source: str
    """
    if not HAS_CROQTILE:
        pytest.skip("croqtile compiler not found")

    temp_files = []

    def _compile(source: str) -> str:
        with tempfile.NamedTemporaryFile(
                suffix=".co", mode="w", delete=False) as f:
            f.write(source)
            co_path = f.name
        temp_files.append(co_path)

        result = subprocess.run(
            [CROQTILE_BIN, "-es", "-t", "cute", co_path],
            capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"croqtile compilation failed:\n"
                f"--- .co source ---\n{source}\n"
                f"--- stderr ---\n{result.stderr}")
        return result.stdout

    yield _compile

    for path in temp_files:
        try:
            os.unlink(path)
        except OSError:
            pass


@pytest.fixture
def compile_and_run():
    """Fixture that compiles .co -> CUDA -> executable and runs on GPU.

    Uses the -gs (generate-script) mode which bundles all runtime
    headers and nvcc flags into a self-contained script.

    Returns a function: compile_and_run(co_source, arch="sm_86") -> stdout
    """
    if not HAS_CROQTILE:
        pytest.skip("croqtile compiler not found")
    if not HAS_NVCC:
        pytest.skip("nvcc not found -- install CUDA toolkit")

    temp_files = []

    def _compile_and_run(co_source: str, *,
                         arch: str = "sm_86",
                         target: str = "cute",
                         timeout: int = 120) -> str:
        script_path = compile_co_to_script(
            co_source, arch=arch, target=target,
            croqtile_bin=CROQTILE_BIN)
        temp_files.append(script_path)
        stdout, rc = run_compilation_script(
            script_path, mode="execute", timeout=timeout)
        if rc != 0:
            raise RuntimeError(
                f"Execution failed (exit {rc}):\nstdout: {stdout}")
        return stdout

    yield _compile_and_run

    for path in temp_files:
        try:
            os.unlink(path)
        except OSError:
            pass

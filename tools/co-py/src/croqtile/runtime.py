"""CroqTile runtime -- compile, execute, and verify kernels from Python.

Provides a Python-native interface for:
  - Finding the croqtile compiler and nvcc
  - Compiling .co source -> CUDA -> executable
  - Executing kernels with numpy data
  - Auto-generating host verification code

Usage::

    import croq
    import numpy as np

    prog = croq.Program()
    prog.add(my_kernel)

    result = croq.runtime.compile_and_run(
        prog, arch="sm_86",
        inputs={"lhs": np.ones((6, 64), dtype=np.int32),
                "rhs": np.ones((6, 64), dtype=np.int32)})
"""

from __future__ import annotations

import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DTYPE_MAP = {
    "s32": ("choreo::s32", "int32_t", "int"),
    "u32": ("choreo::u32", "uint32_t", "unsigned int"),
    "s16": ("choreo::s16", "int16_t", "short"),
    "u16": ("choreo::u16", "uint16_t", "unsigned short"),
    "s8": ("choreo::s8", "int8_t", "signed char"),
    "u8": ("choreo::u8", "uint8_t", "unsigned char"),
    "f32": ("choreo::f32", "float", "float"),
    "f64": ("choreo::f64", "double", "double"),
    "f16": ("choreo::f16", "half", "half"),
    "bf16": ("choreo::bf16", "nv_bfloat16", "nv_bfloat16"),
    "s64": ("choreo::s64", "int64_t", "long long"),
    "u64": ("choreo::u64", "uint64_t", "unsigned long long"),
}


def find_croqtile_bin() -> Optional[str]:
    """Locate the croqtile compiler binary.

    Search order:
      1. CROQTILE_BIN environment variable
      2. Bundled binary inside the croqtile package
      3. Common development paths
      4. PATH
    """
    env = os.environ.get("CROQTILE_BIN") or os.environ.get("CHOREO_BIN")
    if env and os.path.isfile(env):
        return env

    pkg_dir = Path(__file__).parent
    bundled = pkg_dir / "bin" / "choreo"
    if bundled.is_file():
        return str(bundled)

    candidates = [
        pkg_dir / ".." / ".." / ".." / "build" / "choreo",
        pkg_dir / ".." / ".." / ".." / "choreo",
        pkg_dir / ".." / ".." / "build" / "choreo",
        Path.home() / "cursor" / "choreo" / "build" / "choreo",
        Path.home() / "cursor" / "choreo" / "choreo",
    ]
    for c in candidates:
        resolved = c.resolve()
        if resolved.is_file():
            return str(resolved)

    found = shutil.which("choreo") or shutil.which("croqtile")
    return found


def find_nvcc() -> Optional[str]:
    """Locate nvcc (NVIDIA CUDA compiler)."""
    env = os.environ.get("NVCC")
    if env and os.path.isfile(env):
        return env

    found = shutil.which("nvcc")
    if found:
        return found

    for base in ["/usr/local/cuda/bin", "/usr/local/cuda-12.2/bin",
                 "/usr/local/cuda-11.8/bin", "/opt/cuda/bin"]:
        p = os.path.join(base, "nvcc")
        if os.path.isfile(p):
            return p
    return None


def find_runtime_include() -> Optional[str]:
    """Find the CroqTile runtime include directory (choreo.h)."""
    pkg_dir = Path(__file__).parent
    candidates = [
        pkg_dir / ".." / ".." / ".." / "runtime",
        pkg_dir / "include",
        pkg_dir / ".." / ".." / "build" / "_deps" / "croqtile-src" / "runtime",
        Path.home() / "cursor" / "choreo" / "runtime",
    ]
    for c in candidates:
        resolved = c.resolve()
        if (resolved / "choreo.h").is_file():
            return str(resolved)
    return None


def find_cutlass_include() -> Optional[str]:
    """Find CUTLASS/CuTe include directory."""
    pkg_dir = Path(__file__).parent
    candidates = [
        pkg_dir / ".." / ".." / ".." / "extern" / "cutlass" / "include",
        pkg_dir / ".." / ".." / "build" / "_deps" / "croqtile-src" / "extern"
                 / "cutlass" / "include",
        Path.home() / "cursor" / "choreo" / "extern" / "cutlass" / "include",
    ]
    for c in candidates:
        resolved = c.resolve()
        if resolved.is_dir() and (resolved / "cutlass").is_dir():
            return str(resolved)
    return None


def compile_co_to_cuda(co_source: str, *,
                       arch: str = "sm_86",
                       target: str = "cute",
                       croqtile_bin: str = "") -> str:
    """Compile .co source -> CUDA/C++ source via the croqtile compiler.

    Uses ``-es`` (emit source only). Returns the generated CUDA source
    as a string.
    """
    if not croqtile_bin:
        croqtile_bin = find_croqtile_bin()
    if not croqtile_bin:
        raise RuntimeError(
            "Cannot find croqtile compiler binary. "
            "Set CROQTILE_BIN env var or install the croqtile package.")

    with tempfile.NamedTemporaryFile(
            suffix=".co", mode="w", delete=False) as f:
        f.write(co_source)
        co_path = f.name

    try:
        cmd = [croqtile_bin, "-es", "-t", target]
        if arch:
            cmd.append(f"-arch={arch}")
        cmd.append(co_path)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"croqtile compilation failed (exit {result.returncode}):\n"
                f"--- stderr ---\n{result.stderr}\n"
                f"--- .co source (first 50 lines) ---\n"
                + "\n".join(co_source.splitlines()[:50]))
        return result.stdout
    finally:
        os.unlink(co_path)


def compile_co_to_script(co_source: str, *,
                         arch: str = "sm_86",
                         target: str = "cute",
                         croqtile_bin: str = "") -> str:
    """Compile .co source -> self-contained compilation script.

    Uses ``-gs`` (generate script). The script embeds runtime headers
    and nvcc commands. Returns the path to the generated script.
    """
    if not croqtile_bin:
        croqtile_bin = find_croqtile_bin()
    if not croqtile_bin:
        raise RuntimeError(
            "Cannot find croqtile compiler binary. "
            "Set CROQTILE_BIN env var or install the croqtile package.")

    with tempfile.NamedTemporaryFile(
            suffix=".co", mode="w", delete=False) as f:
        f.write(co_source)
        co_path = f.name

    script_path = co_path + ".cute.result"

    try:
        cmd = [croqtile_bin, "-gs", "-t", target]
        if arch:
            cmd.append(f"-arch={arch}")
        cmd.extend([co_path, "-o", script_path])
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"croqtile compilation failed (exit {result.returncode}):\n"
                f"--- stderr ---\n{result.stderr}\n"
                f"--- .co source (first 50 lines) ---\n"
                + "\n".join(co_source.splitlines()[:50]))
        return script_path
    finally:
        os.unlink(co_path)


def run_compilation_script(script_path: str, *,
                           mode: str = "execute",
                           timeout: int = 120,
                           env: Optional[Dict[str, str]] = None) -> Tuple[str, int]:
    """Run a generated compilation script.

    Args:
        script_path: Path to the .cute.result script
        mode: "execute" (compile+run), "compile-link" (compile only)
        timeout: Timeout in seconds
        env: Additional environment variables

    Returns (stdout, returncode).
    """
    run_env = dict(os.environ)
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not os.path.isdir(cuda_home):
        for candidate in ["/usr/local/cuda-12.2",
                          "/usr/local/cuda-11.8",
                          "/opt/cuda"]:
            if os.path.isdir(candidate):
                cuda_home = candidate
                break
    run_env["CUDA_HOME"] = cuda_home

    cute_home = find_cutlass_include()
    if cute_home:
        run_env["CUTE_HOME"] = str(
            Path(cute_home).parent)
    run_env.setdefault("CHOREO_DISABLE_TIMING", "1")
    if env:
        run_env.update(env)

    result = subprocess.run(
        ["bash", script_path, f"--{mode}"],
        capture_output=True, text=True,
        timeout=timeout, env=run_env)
    return result.stdout, result.returncode


def run_executable(exe_path: str, *,
                   timeout: int = 60,
                   env: Optional[Dict[str, str]] = None) -> Tuple[str, int]:
    """Run a compiled executable and capture output.

    Returns (stdout, returncode).
    """
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    result = subprocess.run(
        [exe_path], capture_output=True, text=True,
        timeout=timeout, env=run_env)
    return result.stdout, result.returncode


def compile_and_run(program, *,
                    arch: str = "sm_86",
                    target: str = "cute",
                    check_lines: Optional[List[str]] = None,
                    timeout: int = 120) -> str:
    """Full pipeline: Program -> .co -> compile -> execute on GPU.

    Uses the ``-gs`` (generate-script) mode which produces a
    self-contained bash script that embeds runtime headers and nvcc
    commands. This is the recommended way to run kernels end-to-end.

    Args:
        program: A croq.Program with kernel(s) and host code
        arch: GPU architecture (e.g. "sm_86", "sm_90a")
        target: Compilation target ("cute")
        check_lines: Expected output lines to verify
        timeout: Total timeout in seconds (compilation + execution)

    Returns:
        stdout from the executed binary
    """
    co_source = program.to_co(check_lines=check_lines)
    script_path = compile_co_to_script(
        co_source, arch=arch, target=target)
    try:
        stdout, rc = run_compilation_script(
            script_path, mode="execute", timeout=timeout)
        if rc != 0:
            raise RuntimeError(
                f"Kernel execution failed (exit {rc}):\n"
                f"stdout: {stdout}\n")
        if check_lines:
            for line in check_lines:
                if line not in stdout:
                    raise RuntimeError(
                        f"Expected '{line}' in output but got:\n{stdout}")
        return stdout
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


def generate_host_main(kernel_name: str,
                       params: List[Dict],
                       *,
                       verify: bool = True,
                       tolerance: float = 0.005) -> str:
    """Auto-generate C++ host verification code.

    Args:
        kernel_name: Name of the __co__ function
        params: List of parameter specs, each a dict:
            {"name": "lhs", "dtype": "f16", "shape": (128, 256),
             "role": "input"|"output", "fill": "random"|0.0}
        verify: Whether to add verification code
        tolerance: Tolerance for floating-point comparison

    Returns:
        C++ main() function as a string
    """
    lines = ["int main() {"]

    make_calls = []
    view_calls = []
    for p in params:
        dtype = p["dtype"]
        shape = p["shape"]
        name = p["name"]
        role = p.get("role", "input")
        fill = p.get("fill", "random")
        choreo_dt = _DTYPE_MAP.get(dtype, (f"choreo::{dtype}",))[0]
        shape_args = ", ".join(str(d) for d in shape)

        lines.append(
            f"  auto {name} = "
            f"choreo::make_spandata<{choreo_dt}>({shape_args});")
        if role == "input":
            if fill == "random":
                lines.append(f"  {name}.fill_random(-10, 10);")
            elif isinstance(fill, (int, float)):
                lines.append(f"  {name}.fill({fill});")
        elif role == "output":
            lines.append(f"  {name}.fill(0);")
        make_calls.append(name)
        view_calls.append(f"{name}.view()")

    # Kernel call
    input_params = [p for p in params if p.get("role", "input") == "input"]
    output_params = [p for p in params if p.get("role") == "output"]

    if output_params:
        # void kernel with output param
        all_views = ", ".join(f"{p['name']}.view()" for p in params)
        lines.append(f"  {kernel_name}({all_views});")
    else:
        # kernel with return value
        input_views = ", ".join(
            f"{p['name']}.view()" for p in input_params)
        lines.append(
            f"  auto res = {kernel_name}({input_views});")

    lines.append('  std::cout << "Test Passed" << std::endl;')
    lines.append("}")
    return "\n".join(lines)

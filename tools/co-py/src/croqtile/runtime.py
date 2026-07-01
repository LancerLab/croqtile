"""croqtile-python runtime -- compile, execute, and verify kernels from Python.

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
    """Find the Croqtile runtime include directory (choreo.h)."""
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


# ---- numpy-based data I/O for kernel execution ----------------------------

_NUMPY_DTYPE_MAP = {
    "s32": "int32_t", "u32": "uint32_t",
    "s16": "int16_t", "u16": "uint16_t",
    "s8": "int8_t", "u8": "uint8_t",
    "s64": "int64_t", "u64": "uint64_t",
    "f32": "float", "f64": "double",
    "f16": "half", "bf16": "nv_bfloat16",
}

_NUMPY_TO_CHOREO_DTYPE = {
    "int32": "s32", "uint32": "u32",
    "int16": "s16", "uint16": "u16",
    "int8": "s8", "uint8": "u8",
    "int64": "s64", "uint64": "u64",
    "float32": "f32", "float64": "f64",
    "float16": "f16",
}


def _numpy_dtype_to_choreo(np_dtype) -> str:
    """Convert numpy dtype name to Choreo dtype string."""
    name = str(np_dtype)
    if name in _NUMPY_TO_CHOREO_DTYPE:
        return _NUMPY_TO_CHOREO_DTYPE[name]
    raise ValueError(f"Unsupported numpy dtype: {np_dtype}")


def generate_host_main_numpy(kernel_name: str,
                             params: List[Dict],
                             output_file: str) -> str:
    """Generate C++ host main() that reads inputs from binary files and
    writes output to a binary file.

    Args:
        kernel_name: Name of the __co__ function
        params: List of parameter specs:
            {"name": str, "dtype": str, "shape": tuple,
             "role": "input"|"output", "bin_path": str}
        output_file: Path to write the kernel output (binary)

    Returns:
        C++ main() source string
    """
    includes = [
        '#include <fstream>',
        '#include <iostream>',
        '#include <vector>',
        '#include <cstring>',
    ]
    lines = includes + ["", "int main() {"]

    for p in params:
        dtype = p["dtype"]
        shape = p["shape"]
        name = p["name"]
        role = p.get("role", "input")
        ctype = _NUMPY_DTYPE_MAP.get(dtype, dtype)
        choreo_dt = _DTYPE_MAP.get(dtype, (f"choreo::{dtype}",))[0]
        shape_args = ", ".join(str(d) for d in shape)
        numel = 1
        for d in shape:
            numel *= d

        lines.append(
            f"  auto {name} = "
            f"choreo::make_spandata<{choreo_dt}>({shape_args});")

        if role == "input" and "bin_path" in p:
            lines.append(f"  {{")
            lines.append(
                f'    std::ifstream ifs("{p["bin_path"]}", '
                f'std::ios::binary);')
            lines.append(
                f"    ifs.read(reinterpret_cast<char*>({name}.data()),"
                f" {numel} * sizeof({ctype}));")
            lines.append(f"  }}")

    input_params = [p for p in params if p.get("role", "input") == "input"]
    output_params = [p for p in params if p.get("role") == "output"]

    if output_params:
        all_views = ", ".join(f"{p['name']}.view()" for p in params)
        lines.append(f"  {kernel_name}({all_views});")
        out_p = output_params[0]
        out_name = out_p["name"]
    else:
        input_views = ", ".join(
            f"{p['name']}.view()" for p in input_params)
        lines.append(
            f"  auto res = {kernel_name}({input_views});")
        out_name = "res"

    out_dtype = (output_params[0]["dtype"] if output_params
                 else input_params[0]["dtype"])
    out_ctype = _NUMPY_DTYPE_MAP.get(out_dtype, out_dtype)
    out_shape = (output_params[0]["shape"] if output_params
                 else input_params[0]["shape"])
    out_numel = 1
    for d in out_shape:
        out_numel *= d

    lines.append(f"  {{")
    lines.append(
        f'    std::ofstream ofs("{output_file}", std::ios::binary);')
    lines.append(
        f"    ofs.write(reinterpret_cast<const char*>({out_name}.data()),"
        f" {out_numel} * sizeof({out_ctype}));")
    lines.append(f"  }}")
    lines.append('  std::cout << "IO_DONE" << std::endl;')
    lines.append("}")
    return "\n".join(lines)


def run_with_numpy(program, inputs: Dict, *,
                   output_shape=None,
                   output_dtype=None,
                   arch: str = "sm_86",
                   target: str = "cute",
                   timeout: int = 120):
    """Execute a kernel with numpy array inputs and get numpy output.

    Args:
        program: A croq.Program with a single kernel (no host code)
        inputs: Dict mapping parameter names to numpy arrays
        output_shape: Shape of output (inferred from kernel return type
                      if not specified)
        output_dtype: numpy dtype of output (inferred if not specified)
        arch: GPU architecture
        target: Compilation target
        timeout: Execution timeout

    Returns:
        numpy array with kernel output

    Example::

        import croq
        import numpy as np

        @croq.co
        def ele_add(lhs: croq.s32[6, 64],
                    rhs: croq.s32[6, 64]) -> croq.s32[6, 64]:
            output = croq.declare(croq.s32[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add)

        a = np.random.randint(-10, 10, (6, 64), dtype=np.int32)
        b = np.random.randint(-10, 10, (6, 64), dtype=np.int32)
        result = croq.runtime.run_with_numpy(
            prog, {"lhs": a, "rhs": b}, arch="sm_86")
        np.testing.assert_array_equal(result, a + b)
    """
    import numpy as np

    tmpdir = tempfile.mkdtemp(prefix="croqtile_np_")
    try:
        params = []
        kernel_name = None

        for name, arr in inputs.items():
            bin_path = os.path.join(tmpdir, f"{name}.bin")
            arr.tofile(bin_path)
            choreo_dtype = _numpy_dtype_to_choreo(arr.dtype)
            params.append({
                "name": name,
                "dtype": choreo_dtype,
                "shape": arr.shape,
                "role": "input",
                "bin_path": bin_path,
            })

        if output_shape is None:
            first_arr = next(iter(inputs.values()))
            output_shape = first_arr.shape
        if output_dtype is None:
            first_arr = next(iter(inputs.values()))
            output_dtype = first_arr.dtype

        out_bin = os.path.join(tmpdir, "output.bin")

        kernel_name = program._kernel_name if hasattr(
            program, '_kernel_name') else None
        if kernel_name is None:
            co_src = program.to_co()
            for line in co_src.split('\n'):
                if line.strip().startswith('__co__'):
                    parts = line.split('(')[0].split()
                    kernel_name = parts[-1]
                    break

        if not kernel_name:
            raise ValueError("Cannot determine kernel name from program")

        host_main = generate_host_main_numpy(
            kernel_name, params, out_bin)
        program.add(host_main)

        co_source = program.to_co(check_lines=["IO_DONE"])
        script_path = compile_co_to_script(
            co_source, arch=arch, target=target)

        stdout, rc = run_compilation_script(
            script_path, mode="execute", timeout=timeout)
        if rc != 0:
            raise RuntimeError(
                f"Kernel execution failed (exit {rc}):\n{stdout}")

        if not os.path.exists(out_bin):
            raise RuntimeError(
                f"Output file not produced. stdout:\n{stdout}")

        result = np.fromfile(out_bin, dtype=output_dtype)
        return result.reshape(output_shape)
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmpdir, ignore_errors=True)

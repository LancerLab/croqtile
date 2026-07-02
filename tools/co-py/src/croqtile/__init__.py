"""
CroqTile: Python bindings for the CroqTile data-movement DSL compiler.

    import croq

    @croq.co
    def ele_add(lhs: croq.s32[2,3,64], rhs: croq.s32[2,3,64]) -> croq.s32[6,64]:
        output = croq.declare(croq.s32[6,64], "output")
        for p, q in croq.parallel(p=6, q=64):
            output[p, q] = lhs[p, q] + rhs[p, q]
        return output

    prog = croq.Program()
    prog.add(ele_add)
    print(prog.dump_ast())
"""

__version__ = "0.1.0"

from croqtile import _core  # noqa: F401

# Type aliases (support [] syntax: croq.s32[6, 64])
from croqtile.types import (  # noqa: F401
    f64, f32, tf32, f16, bf16,
    f8_e4m3, f8_e5m2,
    f6_e2m3, f6_e3m2, f4_e2m1,
    s64, u64, s32, u32, s16, u16, s8, u8,
    boolean,
    stream,
    GLOBAL, SHARED, LOCAL,
)

# Builder -- primary API
from croqtile.builder import (  # noqa: F401
    Var,
    TensorType,
    MDSpan,
    ITuple,
    _DType,
    Program,
    co,
    parallel,
    foreach,
    foreach_staged,
    with_in,
    pb,
    fe,
    fs,
    wi,
    declare,
    declare_int,
    assign,
    wait,
    host,
    select,
    cdiv,
    swap,
    sync,
    FULL,
    BLOCK,
    GROUP,
    THREAD,
    Global,
    device_if,
    dma,
    tma,
    mma,
    compile_to_source,
    dump_ast,
    # Math builtins
    sqrt, rsqrt, sin, cos, tan, sinh, cosh, tanh,
    asin, acos, atan, exp, expm1, log, log1p,
    ceil, floor, round_, sign, gelu, sigmoid, softplus, isfinite,
    pow, atan2,
    println,
)

# Runtime -- compilation + execution pipeline
from croqtile import runtime  # noqa: F401
from croqtile.runtime import execute, detect_gpu_arch  # noqa: F401

# Backward-compat (deprecated, will be removed)
from croqtile.builder import (  # noqa: F401
    Expr,
    Tensor,
    Param,
    Function,
    BlockBuilder,
    var,
    return_,
    assign_at,
)


def sdk_version() -> str:
    """Return the linked CroqTile SDK version."""
    return _core.version()

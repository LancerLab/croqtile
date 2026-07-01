"""
Type aliases and storage qualifiers.

Each type is a _DType instance that supports [] for shaped tensors:
    croq.s32          -> scalar
    croq.s32[6, 64]   -> shaped tensor
"""

from croqtile._core import BaseType, Storage
from croqtile.builder import _DType

# Numeric types -- each supports [] for shapes
f64 = _DType(BaseType.F64)
f32 = _DType(BaseType.F32)
tf32 = _DType(BaseType.TF32)
f16 = _DType(BaseType.F16)
bf16 = _DType(BaseType.BF16)
f8_e4m3 = _DType(BaseType.F8_E4M3)
f8_e5m2 = _DType(BaseType.F8_E5M2)
f6_e2m3 = _DType(BaseType.F6_E2M3)
f6_e3m2 = _DType(BaseType.F6_E3M2)
f4_e2m1 = _DType(BaseType.F4_E2M1)

s64 = _DType(BaseType.S64)
u64 = _DType(BaseType.U64)
s32 = _DType(BaseType.S32)
u32 = _DType(BaseType.U32)
s16 = _DType(BaseType.S16)
u16 = _DType(BaseType.U16)
s8 = _DType(BaseType.S8)
u8 = _DType(BaseType.U8)

boolean = _DType(BaseType.BOOL)
stream = _DType(BaseType.STREAM)

# Storage qualifiers
GLOBAL = Storage.GLOBAL
SHARED = Storage.SHARED
LOCAL = Storage.LOCAL

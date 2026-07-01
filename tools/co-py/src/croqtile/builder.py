"""
CroqTile builder -- Pythonic API for constructing CroqTile AST programs.

Design principles:
  - All variables are Var objects -- no string lookups.
  - ``for`` loops define scoped blocks with zero name duplication.
  - Python's ``return`` works naturally inside ``@croq.co``.
  - Indexed assignment ``output[p, q] = expr`` emits AST statements.

Scoped constructs:
  - ``for p, q in croq.parallel(p=6, q=64):``
  - ``for m, k in croq.foreach(m=128, k=256):``
  - ``for k in croq.with_in(k=16):``
  - ``@croq.co`` decorator traces a function into a CroqFunction.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Sequence, Tuple

from croqtile import _core

BaseType = _core.BaseType
Storage = _core.Storage


# =========================================================================== #
#  Thread-local scope stack
# =========================================================================== #

_tls = threading.local()


def _scope_stack() -> list:
    if not hasattr(_tls, "stack"):
        _tls.stack = []
    return _tls.stack


def _current_scope() -> "_Scope":
    stack = _scope_stack()
    if not stack:
        raise RuntimeError(
            "No active croq scope -- are you inside a @croq.co function?")
    return stack[-1]


def _push_scope(scope: "_Scope"):
    _scope_stack().append(scope)


def _pop_scope() -> "_Scope":
    return _scope_stack().pop()


def _emit(node):
    """Append a statement node to the current scope."""
    _current_scope().append(node)


def _source_emit(line: str):
    """Record a .co source line in the current scope."""
    try:
        _current_scope().append_source(line)
    except RuntimeError:
        pass


def _source_emit_block(header: str, body_src: list):
    """Record a scoped block in the current scope's source."""
    try:
        _current_scope().append_source({"header": header, "body": body_src})
    except RuntimeError:
        pass


class _Scope:
    __slots__ = ("_stmts", "_src")

    def __init__(self):
        self._stmts = _core.MultiNodes()
        self._src: list = []

    def append(self, node):
        self._stmts.append(node)

    def append_source(self, item):
        """Append a source line (str) or block (dict with header+body)."""
        self._src.append(item)

    def build(self) -> _core.MultiNodes:
        return self._stmts

    def source(self) -> list:
        return self._src


# =========================================================================== #
#  Var
# =========================================================================== #

class Var:
    """A symbolic expression node with Pythonic operators.

    All values flowing through CroqPy are Var objects -- function parameters,
    declared variables, loop indices, and expression results.

    Each Var carries a ``_co`` string -- its ``.co`` source representation --
    alongside the C++ AST node.  This enables round-trip emission of ``.co``
    source from a CroqPy program without walking the opaque AST.
    """

    __slots__ = ("_node", "_name", "_co")

    def __init__(self, node, name: str = "", *, co: str = ""):
        self._node = node
        self._name = name
        self._co = co or name

    @property
    def node(self):
        return self._node

    @property
    def name(self) -> str:
        return self._name

    @property
    def co(self) -> str:
        return self._co

    @classmethod
    def lit(cls, value) -> "Var":
        """Create a literal Var from an int or float."""
        if isinstance(value, float):
            s = f"{value}f" if value == int(value) else str(value)
            return cls(_core.make_float_expr(value), co=s)
        return cls(_core.make_int_expr(int(value)), co=str(int(value)))

    def __add__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("+", self._node, o._node),
                   co=f"{self._co} + {o._co}")

    def __radd__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("+", o._node, self._node),
                   co=f"{o._co} + {self._co}")

    def __sub__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("-", self._node, o._node),
                   co=f"{self._co} - {o._co}")

    def __rsub__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("-", o._node, self._node),
                   co=f"{o._co} - {self._co}")

    def __mul__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("*", self._node, o._node),
                   co=f"{self._co} * {o._co}")

    def __rmul__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("*", o._node, self._node),
                   co=f"{o._co} * {self._co}")

    def __truediv__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("/", self._node, o._node),
                   co=f"{self._co} / {o._co}")

    def __mod__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("%", self._node, o._node),
                   co=f"{self._co} % {o._co}")

    def __neg__(self):
        return Var(_core.make_binary_expr("-", _core.make_int_expr(0),
                                          self._node),
                   co=f"-{self._co}")

    def __lt__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("<", self._node, o._node),
                   co=f"{self._co} < {o._co}")

    def __gt__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr(">", self._node, o._node),
                   co=f"{self._co} > {o._co}")

    def __le__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("<=", self._node, o._node),
                   co=f"{self._co} <= {o._co}")

    def __ge__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr(">=", self._node, o._node),
                   co=f"{self._co} >= {o._co}")

    def __eq__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("==", self._node, o._node),
                   co=f"{self._co} == {o._co}")

    def __ne__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("!=", self._node, o._node),
                   co=f"{self._co} != {o._co}")

    def __getitem__(self, key) -> "Var":
        if not isinstance(key, tuple):
            key = (key,)
        result = self
        for idx in key:
            result = Var(_core.make_elemof_expr(
                result._node, _to_var(idx)._node))
        keys_co = ", ".join(_var_co(k) for k in key)
        result._co = f"{self._co}.at({keys_co})"
        return result

    def __setitem__(self, key, value):
        if not self._name:
            raise RuntimeError(
                "Cannot assign to an unnamed Var -- "
                "only named variables support indexed assignment")
        if not isinstance(key, tuple):
            key = (key,)
        mv = _core.MultiValues()
        for idx in key:
            mv.append(_to_var(idx)._node)
        da = _core.DataAccess(self._name, mv)
        _emit(_core.Assignment(da, _unwrap(value)))
        keys_co = ", ".join(_var_co(k) for k in key)
        _source_emit(
            f"{self._name}.at({keys_co}) = {_var_co(value)};")

    def __iadd__(self, other):
        """Support ``v += expr`` -> returns ``v + other`` for rebinding.

        When used via subscript (``output[i] += val``), Python calls
        __getitem__, __iadd__, then __setitem__ -- producing the correct
        AST assignment automatically.
        """
        o = _coerce(other)
        return Var(_core.make_binary_expr("+", self._node, o._node),
                   co=f"{self._co} + {o._co}")

    def __isub__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("-", self._node, o._node),
                   co=f"{self._co} - {o._co}")

    def __imul__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("*", self._node, o._node),
                   co=f"{self._co} * {o._co}")

    # -- @ operator: hierarchical index composition (CroqTile's #)

    def __matmul__(self, other):
        o = _coerce(other)
        return Var(_core.make_binary_expr("#", self._node, o._node),
                   co=f"{self._co}#{o._co}")

    # -- .data: access the data member of a DMA future

    @property
    def data(self) -> "Var":
        co_name = f"{self._co}.data" if self._co else ""
        return Var(_core.make_binary_expr(
            "dataof", self._node, _core.make_int_expr(0)),
            self._name + ".data" if self._name else "",
            co=co_name)

    # -- .chunkat(i, j): tiled chunk access

    def chunkat(self, *indices) -> "Var":
        idx_nodes = [_unwrap(i) for i in indices]
        ca = _core.make_chunkat(self._name, idx_nodes)
        idx_co = ", ".join(_var_co(i) for i in indices)
        return Var(ca, self._name, co=f"{self._co}.chunkat({idx_co})")

    # -- .subspan(M, K).at(bm, bk): subspan indexing

    def subspan(self, *dims):
        return _SubSpanBuilder(self, dims)

    # -- .view(shape).from_(offsets): view access

    def view(self, *shape):
        """Create a view with the given shape.

        Use with .from_() to specify offsets:
            lhs.view(1, X, croq.FULL).from_(p, 3*x, 0)

        Note: use ``from_`` (trailing underscore) because ``from``
        is a Python keyword.
        """
        return _ViewBuilder(self, shape)

    # -- .span_as(M, N): reshape view

    def span_as(self, *dims) -> "Var":
        """Create a span_as view: ``lhs.span_as(6, 64)``."""
        dims_co = ", ".join(_var_co(d) for d in dims)
        name = f"{self._name}_as"
        decl_line = f"{name} = {self._co}.span_as({dims_co});"
        _source_emit(decl_line)
        # AST: variable alias (declare + assign)
        return Var(_core.make_id_expr(name), name, co=name)

    def __repr__(self):
        return f"Var({self._name or self._node.type_name()})"

    def __hash__(self):
        return id(self)

    def __bool__(self):
        raise TypeError(
            "Cannot use Var in a boolean context. "
            "Use croq.select() for conditional expressions.")


# =========================================================================== #
#  Types -- croq.s32[M, N]
# =========================================================================== #

class _DType:
    """Wraps a BaseType. Supports ``croq.s32[6, 64]`` for shaped tensors."""

    __slots__ = ("_bt",)

    def __init__(self, bt: BaseType):
        self._bt = bt

    def __getitem__(self, shape) -> "TensorType":
        if not isinstance(shape, tuple):
            shape = (shape,)
        return TensorType(self._bt, list(shape))

    def __repr__(self):
        return f"dtype({self._bt.name.lower()})"

    @property
    def base(self) -> BaseType:
        return self._bt


class TensorType:
    """A shaped tensor type: dtype + shape."""

    __slots__ = ("dtype", "shape")

    def __init__(self, dtype: BaseType, shape: List[int]):
        self.dtype = dtype
        self.shape = shape

    def to_data_type(self) -> _core.DataType:
        return _core.DataType(self.dtype, self.shape)

    def __repr__(self):
        dims = ", ".join(str(d) for d in self.shape)
        return f"{self.dtype.name.lower()}[{dims}]"


class MDSpan:
    """Multi-dimensional span descriptor.

    Represents CroqTile's MDSpan type -- a multi-dimensional view into data
    with a known rank and optional shape.

        ms = croq.MDSpan(croq.f16, rank=2)
        ms = croq.MDSpan(croq.f16, shape=[128, 256])
    """

    __slots__ = ("dtype", "rank", "shape")

    def __init__(self, dtype, *, rank: int = 0,
                 shape: Optional[List[int]] = None):
        if isinstance(dtype, _DType):
            self.dtype = dtype._bt
        elif isinstance(dtype, BaseType):
            self.dtype = dtype
        else:
            raise TypeError(f"Expected a dtype, got {type(dtype)}")
        if shape is not None:
            self.shape = list(shape)
            self.rank = len(self.shape)
        else:
            self.shape = None
            self.rank = rank

    def to_data_type(self) -> _core.DataType:
        if self.shape:
            return _core.DataType(self.dtype, self.shape)
        return _core.DataType(self.dtype)

    def __repr__(self):
        if self.shape:
            dims = ", ".join(str(d) for d in self.shape)
            return f"MDSpan({self.dtype.name.lower()}[{dims}])"
        return f"MDSpan({self.dtype.name.lower()}, rank={self.rank})"


class ITuple:
    """Index tuple type -- a bounded multi-dimensional index.

    Represents CroqTile's ITuple/BoundedITuple type used for parallel
    decomposition indices and tiling factors.

        it = croq.ITuple(6, 64)       # 2D index tuple with bounds [6, 64]
        it = croq.ITuple(128)          # 1D index with bound 128
    """

    __slots__ = ("bounds",)

    def __init__(self, *bounds: int):
        self.bounds = list(bounds)

    @property
    def rank(self) -> int:
        return len(self.bounds)

    def __repr__(self):
        dims = ", ".join(str(b) for b in self.bounds)
        return f"ITuple({dims})"


class _StorageAnnotation:
    """Wraps a type annotation with storage qualifier for function params.

        @croq.co
        def kern(output: croq.Global(croq.f16[M, N])):
            ...
    """
    __slots__ = ("_type", "_storage")

    def __init__(self, type_hint, storage: _core.Storage):
        self._type = type_hint
        self._storage = storage


def Global(type_hint):
    """Mark a parameter as global storage: ``croq.Global(croq.f16[M, N])``."""
    return _StorageAnnotation(type_hint, _core.Storage.GLOBAL)


def _to_data_type(t) -> _core.DataType:
    if isinstance(t, TensorType):
        return t.to_data_type()
    if isinstance(t, MDSpan):
        return t.to_data_type()
    if isinstance(t, _DType):
        return _core.DataType(t._bt)
    if isinstance(t, _core.DataType):
        return t
    raise TypeError(
        f"Expected a type (e.g. croq.s32 or croq.s32[6,64]), got {type(t)}")


# =========================================================================== #
#  Statement emitters
# =========================================================================== #

def declare(type_spec, name: str, *,
            storage: Optional[Storage] = None,
            init=None) -> Var:
    """Declare a variable in the current scope.

        output = croq.declare(croq.s32[6, 64], "output")
        buf    = croq.declare(croq.f16[128, 64], "buf", storage=croq.SHARED)
        acc    = croq.declare(croq.s32[8, 128], "acc", storage=croq.LOCAL, init=0)
    """
    dt = _to_data_type(type_spec)
    if storage is not None and init is not None:
        decl = _core.NamedVariableDecl(name, dt, storage, int(init))
    elif storage is not None:
        decl = _core.NamedVariableDecl(name, dt, storage)
    else:
        decl = _core.NamedVariableDecl(name, dt)
    _emit(decl)
    # Source recording
    ts = _type_to_co(type_spec)
    sp = f"{_storage_to_co(storage)} " if storage else ""
    init_s = f"{{{init}}}" if init is not None else ""
    _source_emit(f"{sp}{ts} {name}{init_s};")
    return Var(_core.make_id_expr(name), name)


def wait(*futures: Var):
    """Emit wait statement(s) for async DMA/TMA futures."""
    mv = _core.MultiValues()
    for f in futures:
        mv.append(_unwrap(f))
    _emit(_core.make_wait(mv))
    names = ", ".join(_var_co(f) for f in futures)
    _source_emit(f"wait {names};")


def host(code: str):
    """Embed raw C++ host code in the program."""
    _emit(_core.CppSourceCode(code))


def select(cond: Var, true_val, false_val) -> Var:
    """Ternary selection: ``croq.select(cond, a, b)`` -> CroqTile ``(cond) ? a : b``."""
    cond_node = _unwrap(cond)
    if not isinstance(cond_node, _core.Expr):
        cond_node = _core.make_binary_expr("!=", cond_node,
                                           _core.make_int_expr(0))
    co = f"({_var_co(cond)}) ? {_var_co(true_val)} : {_var_co(false_val)}"
    return Var(_core.make_ternary_expr(
        cond_node, _unwrap(true_val), _unwrap(false_val)), co=co)


def cdiv(a, b) -> Var:
    """Ceiling division: ``croq.cdiv(a, b)`` -> CroqTile ``cdiv(a, b)``."""
    return Var(_core.make_binary_expr("cdiv", _unwrap(a), _unwrap(b)),
               co=f"cdiv({_var_co(a)}, {_var_co(b)})")


def swap(a: Var, b: Var):
    """Emit swap statement: ``croq.swap(a, b)`` -> CroqTile ``swap(a, b)``."""
    _emit(_core.make_swap(_unwrap(a), _unwrap(b)))
    _source_emit(f"swap {_var_co(a)}, {_var_co(b)};")


def sync(kind: str = "shared"):
    """Emit synchronization barrier.

        croq.sync("shared")   -> CroqTile ``sync.shared;``
        croq.sync()           -> same (shared is default)
    """
    _emit(_core.make_sync(kind))
    _source_emit(f"sync.{kind};")


# ---- Math builtins ----

def _make_unary_builtin(name: str):
    """Create a unary math builtin: croq.sqrt(x) -> __sqrt(x)."""
    def fn(x) -> Var:
        return Var(_core.make_call_expr(name, [_unwrap(x)]),
                   co=f"{name}({_var_co(x)})")
    fn.__name__ = name.lstrip("_")
    fn.__doc__ = f"Math builtin: ``{name}(x)``."
    return fn


def _make_binary_builtin(name: str):
    """Create a binary math builtin: croq.pow(x, y) -> __pow(x, y)."""
    def fn(x, y) -> Var:
        return Var(_core.make_call_expr(name, [_unwrap(x), _unwrap(y)]),
                   co=f"{name}({_var_co(x)}, {_var_co(y)})")
    fn.__name__ = name.lstrip("_")
    fn.__doc__ = f"Math builtin: ``{name}(x, y)``."
    return fn


sqrt = _make_unary_builtin("__sqrt")
rsqrt = _make_unary_builtin("__rsqrt")
sin = _make_unary_builtin("__sin")
cos = _make_unary_builtin("__cos")
tan = _make_unary_builtin("__tan")
sinh = _make_unary_builtin("__sinh")
cosh = _make_unary_builtin("__cosh")
tanh = _make_unary_builtin("__tanh")
asin = _make_unary_builtin("__asin")
acos = _make_unary_builtin("__acos")
atan = _make_unary_builtin("__atan")
exp = _make_unary_builtin("__exp")
expm1 = _make_unary_builtin("__expm1")
log = _make_unary_builtin("__log")
log1p = _make_unary_builtin("__log1p")
ceil = _make_unary_builtin("__ceil")
floor = _make_unary_builtin("__floor")
round_ = _make_unary_builtin("__round")
sign = _make_unary_builtin("__sign")
gelu = _make_unary_builtin("__gelu")
sigmoid = _make_unary_builtin("__sigmoid")
softplus = _make_unary_builtin("__softplus")
isfinite = _make_unary_builtin("__isfinite")

pow = _make_binary_builtin("__pow")
atan2 = _make_binary_builtin("__atan2")


def println(*args):
    """Device print: ``croq.println(a, b)`` -> CroqTile ``println(a, b)``."""
    _emit(_core.make_println([_unwrap(a) for a in args]))
    args_co = ", ".join(_var_co(a) for a in args)
    _source_emit(f"println({args_co});")


def declare_int(name: str, value) -> Var:
    """Declare an integer variable: ``croq.declare_int("MMA_M", 16)``."""
    dt = _core.DataType(_core.BaseType.S32)
    init_node = _unwrap(value)
    decl = _core.NamedVariableDecl(name, dt)
    _emit(decl)
    _emit(_core.Assignment(name, init_node))
    _source_emit(f"int {name} = {_var_co(value)};")
    return Var(_core.make_id_expr(name), name)


def assign(target: Var, value):
    """Emit a plain assignment: ``croq.assign(mc, mc + bias)``.

    CroqTile: ``mc += bias;``  ->  CroqPy: ``croq.assign(mc, mc + bias)``

    Works for any named Var. For indexed assignment, use
    ``output[i, j] = expr`` instead.
    """
    if not target._name:
        raise RuntimeError(
            "Cannot assign to an unnamed Var -- "
            "target must be a named variable")
    _emit(_core.Assignment(target._name, _unwrap(value)))
    _source_emit(f"{target._name} = {_var_co(value)};")


# Full-slice sentinel for chunkat: output.chunkat(croq._, iv)
class _FullSlice:
    """Sentinel for full-dimension slice in chunkat/subspan."""
    pass

FULL = _FullSlice()


class _DeviceIfContext:
    """Context manager for device if-statements.

        with croq.device_if(cond):
            out[p] = a[p]

        with croq.device_if(cond, else_=True) as (if_block, else_block):
            # not yet supported -- use two device_ifs with negated condition
    """

    def __init__(self, cond):
        self._cond = cond
        self._scope = None

    def __enter__(self):
        self._scope = _Scope()
        _push_scope(self._scope)
        return self

    def __exit__(self, *exc):
        _pop_scope()
        body = self._scope.build()
        src = self._scope.source()
        cond_node = _unwrap(self._cond)
        if not isinstance(cond_node, _core.Expr):
            cond_node = _core.make_binary_expr("!=", cond_node,
                                               _core.make_int_expr(0))
        _emit(_core.make_if(cond_node, body))
        _source_emit_block(f"if ({_var_co(self._cond)})", src)
        return False


def device_if(cond: Var):
    """Device-side conditional block.

        with croq.device_if(p < 64):
            output[p] = lhs[p] + rhs[p]

    Maps to CroqTile: ``if (cond) { ... }``
    """
    return _DeviceIfContext(cond)


# =========================================================================== #
#  _SubSpanBuilder -- fluent var.subspan(M, K).at(bm, bk)
# =========================================================================== #

class _SubSpanAtVar(Var):
    """Var wrapping a subspan-at result.

    Supports further ``.chunkat()`` chaining:
        var.subspan(M, K).at(stage, 0).chunkat(_, iv_warp)
    """

    __slots__ = ("_ss_name", "_ss_shape", "_ss_at")

    def __init__(self, node, name, shape_nodes, at_nodes):
        super().__init__(node, name)
        self._ss_name = name
        self._ss_shape = shape_nodes
        self._ss_at = at_nodes

    def chunkat(self, *indices) -> Var:
        idx_nodes = [_unwrap(i) for i in indices]
        ca = _core.make_chunkat_subspan(
            self._ss_name, self._ss_shape, self._ss_at, idx_nodes)
        return Var(ca, self._ss_name)


class _SubSpanBuilder:
    """Intermediate object for ``var.subspan(M, K).at(bm, bk)``.

    Also supports optional ``step()``:
        var.subspan(M, K).step(M, K).at(bm, bk)
    """

    __slots__ = ("_var", "_dims", "_steps")

    def __init__(self, var: Var, dims, steps=None):
        self._var = var
        self._dims = dims
        self._steps = steps

    def step(self, *steps) -> "_SubSpanBuilder":
        """Add step/stride factors: ``var.subspan(M, K).step(M, K)``."""
        return _SubSpanBuilder(self._var, self._dims, steps)

    def at(self, *indices) -> _SubSpanAtVar:
        shape_nodes = [_unwrap(d) for d in self._dims]
        at_nodes = [_unwrap(i) for i in indices]
        step_nodes = ([_unwrap(s) for s in self._steps]
                      if self._steps else [])
        ca = _core.make_chunkat_subspan(
            self._var._name, shape_nodes, at_nodes, [], step_nodes)
        dims_co = ", ".join(_var_co(d) for d in self._dims)
        at_co = ", ".join(_var_co(i) for i in indices)
        step_co = ""
        if self._steps:
            step_co = ".step(" + ", ".join(
                _var_co(s) for s in self._steps) + ")"
        co = (f"{self._var._co}.subspan({dims_co})"
              f"{step_co}.at({at_co})")
        result = _SubSpanAtVar(
            ca, self._var._name, shape_nodes, at_nodes)
        result._co = co
        return result


# =========================================================================== #
#  _ViewBuilder -- fluent var.view(1, X, _).from_(p, 3*x, 0)
# =========================================================================== #

class _ViewBuilder:
    """Intermediate object for ``var.view(shape).from_(offsets)``.

    CroqTile syntax: lhs.view(1, X, _).from(p, 3*x, 0)
    Python: lhs.view(1, X, croq.FULL).from_(p, 3*x, 0)

    Note: ``from_`` instead of ``from`` (Python keyword).
    """

    __slots__ = ("_var", "_shape")

    def __init__(self, var: Var, shape):
        self._var = var
        self._shape = shape

    def from_(self, *offsets) -> Var:
        """Materialize the view+from as a ChunkAt with View operation."""
        shape_nodes = [_unwrap(s) for s in self._shape]
        offset_nodes = [_unwrap(o) for o in offsets]
        ca = _core.make_chunkat_view(
            self._var._name, shape_nodes, offset_nodes)
        shape_co = ", ".join(_var_co(s) for s in self._shape)
        offsets_co = ", ".join(_var_co(o) for o in offsets)
        co = f"{self._var._co}.view({shape_co}).from({offsets_co})"
        return Var(ca, self._var._name, co=co)


# =========================================================================== #
#  DMA namespace -- croq.dma.copy(...), croq.dma.copy_async(...)
# =========================================================================== #

class _DMANamespace:
    """Accessed as ``croq.dma``.

        f = croq.dma.copy(input, to=croq.SHARED)
        f = croq.dma.copy_async(input, to=croq.SHARED)
        croq.wait(f)
        croq.dma.copy(f.data, to=output)
        f = croq.dma.any()
    """

    _storage_names = {
        _core.Storage.SHARED: "shared",
        _core.Storage.LOCAL: "local",
        _core.Storage.GLOBAL: "global",
    }
    _counter = 0

    @classmethod
    def _gen_name(cls, prefix: str = "dma") -> str:
        cls._counter += 1
        return f"_{prefix}_{cls._counter}"

    def copy(self, src: Var, dst=None, *, to=None, name: str = "",
             swizzle: int = 0) -> Var:
        """Emit sync DMA copy. Returns a future Var.

        Two calling forms:
            croq.dma.copy(src, to=croq.SHARED)
            croq.dma.copy(src, dst, swizzle=128)
        """
        if to is None and dst is None:
            raise TypeError("dma.copy requires either dst or to=")
        target = dst if dst is not None else to
        from_node = _unwrap(src)
        if isinstance(target, _core.Storage):
            to_node = _core.make_select(self._storage_names[target])
        else:
            to_node = _unwrap(target)
        if not name:
            name = self._gen_name()
        node = _core.make_dma_copy(
            name, from_node, to_node, False, False, swizzle)
        _emit(node)
        swiz = f".swiz<{swizzle}>" if swizzle else ""
        if isinstance(target, _core.Storage):
            _source_emit(
                f"{name} = dma.copy{swiz} {_var_co(src)} "
                f"=> {self._storage_names[target]};")
        else:
            _source_emit(
                f"dma.copy{swiz} {_var_co(src)} => {_var_co(target)};")
        return Var(node, name, co=name)

    def copy_async(self, src: Var, dst=None, *, to=None,
                   name: str = "", swizzle: int = 0) -> Var:
        """Emit async DMA copy. Returns a future Var."""
        if to is None and dst is None:
            raise TypeError("dma.copy_async requires either dst or to=")
        target = dst if dst is not None else to
        from_node = _unwrap(src)
        if isinstance(target, _core.Storage):
            to_node = _core.make_select(self._storage_names[target])
        else:
            to_node = _unwrap(target)
        if not name:
            name = self._gen_name()
        node = _core.make_dma_copy(
            name, from_node, to_node, True, False, swizzle)
        _emit(node)
        swiz = f".swiz<{swizzle}>" if swizzle else ""
        if isinstance(target, _core.Storage):
            _source_emit(
                f"{name} = dma.copy.async{swiz} {_var_co(src)} "
                f"=> {self._storage_names[target]};")
        else:
            _source_emit(
                f"dma.copy.async{swiz} {_var_co(src)} "
                f"=> {_var_co(target)};")
        return Var(node, name, co=name)

    def any(self, name: str = "") -> Var:
        """Emit a DMA.any placeholder future."""
        if not name:
            name = self._gen_name("any")
        node = _core.make_dma_any(name)
        _emit(node)
        _source_emit(f"{name} = dma.any;")
        return Var(node, name, co=name)


dma = _DMANamespace()


# =========================================================================== #
#  TMA namespace -- croq.tma.copy(...), croq.tma.copy_async(...)
# =========================================================================== #

class _TMANamespace:
    """Accessed as ``croq.tma``.

        f = croq.tma.copy_async(src, to=lhs_s, swizzle=128)
        croq.wait(f)
        croq.tma.copy(output_s, to=output_ref)
    """

    _storage_names = {
        _core.Storage.SHARED: "shared",
        _core.Storage.LOCAL: "local",
        _core.Storage.GLOBAL: "global",
    }
    _counter = 0

    @classmethod
    def _gen_name(cls, prefix: str = "tma") -> str:
        cls._counter += 1
        return f"_{prefix}_{cls._counter}"

    def copy(self, src: Var, dst=None, *, to=None, name: str = "",
             swizzle: int = 0) -> Var:
        if to is None and dst is None:
            raise TypeError("tma.copy requires either dst or to=")
        target = dst if dst is not None else to
        from_node = _unwrap(src)
        if isinstance(target, _core.Storage):
            to_node = _core.make_select(self._storage_names[target])
        else:
            to_node = _unwrap(target)
        if not name:
            name = self._gen_name()
        node = _core.make_dma_copy(
            name, from_node, to_node, False, True, swizzle)
        _emit(node)
        swiz = f".swiz<{swizzle}>" if swizzle else ""
        if isinstance(target, _core.Storage):
            _source_emit(
                f"{name} = tma.copy{swiz} {_var_co(src)} "
                f"=> {self._storage_names[target]};")
        else:
            _source_emit(
                f"tma.copy{swiz} {_var_co(src)} => {_var_co(target)};")
        return Var(node, name, co=name)

    def copy_async(self, src: Var, dst=None, *, to=None, name: str = "",
                   swizzle: int = 0) -> Var:
        if to is None and dst is None:
            raise TypeError("tma.copy_async requires either dst or to=")
        target = dst if dst is not None else to
        from_node = _unwrap(src)
        if isinstance(target, _core.Storage):
            to_node = _core.make_select(self._storage_names[target])
        else:
            to_node = _unwrap(target)
        if not name:
            name = self._gen_name()
        node = _core.make_dma_copy(
            name, from_node, to_node, True, True, swizzle)
        _emit(node)
        swiz = f".swiz<{swizzle}>" if swizzle else ""
        if isinstance(target, _core.Storage):
            _source_emit(
                f"{name} = tma.copy.async{swiz} {_var_co(src)} "
                f"=> {self._storage_names[target]};")
        else:
            _source_emit(
                f"tma.copy.async{swiz} {_var_co(src)} "
                f"=> {_var_co(target)};")
        return Var(node, name, co=name)

    def any(self, name: str = "") -> Var:
        """Emit a TMA.any placeholder future."""
        if not name:
            name = self._gen_name("tma_any")
        node = _core.make_dma_any(name)
        _emit(node)
        _source_emit(f"{name} = tma.any;")
        return Var(node, name, co=name)


tma = _TMANamespace()


# =========================================================================== #
#  MMA namespace -- croq.mma.fill(), .load(), .exec(), .store(), .commit()
# =========================================================================== #

class _MMANamespace:
    """Accessed as ``croq.mma``.

        mc = croq.mma.fill(0.0)
        croq.mma.fill(mc, 0.0)
        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
        croq.mma.exec(mc, ma, mb, method="row.col")
        croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))
        croq.mma.commit()
    """

    ROW_COL = "row.col"
    ROW_ROW = "row.row"
    COL_COL = "col.col"
    COL_ROW = "col.row"

    _counter = 0

    @classmethod
    def _gen_name(cls, prefix: str) -> str:
        cls._counter += 1
        return f"_mma_{prefix}_{cls._counter}"

    @staticmethod
    def _unwrap_dtype(dtype):
        if dtype is None:
            return _core.BaseType.F32
        if isinstance(dtype, _DType):
            return dtype._bt
        return dtype

    def fill(self, value_or_var, value=None, *, dtype=None) -> Var:
        bt = self._unwrap_dtype(dtype)
        dt_suffix = f".{bt.name.lower()}" if dtype else ""
        if isinstance(value_or_var, Var) and value is not None:
            node = _core.make_mma_fill(
                float(value), value_or_var._name, bt)
            _emit(node)
            _source_emit(
                f"mma.fill{dt_suffix} {value_or_var._name}, {value};")
            return value_or_var
        if isinstance(value_or_var, FragRef) and value is not None:
            node = _core.make_mma_fill(
                float(value), value_or_var._name, bt)
            _emit(node)
            _source_emit(
                f"mma.fill{dt_suffix} {_var_co(value_or_var)}, {value};")
            return value_or_var
        name = self._gen_name("acc")
        node = _core.make_mma_fill(
            float(value_or_var), "", bt)
        _emit(node)
        _source_emit(f"{name} = mma.fill{dt_suffix} {value_or_var};")
        return Var(_core.make_id_expr(name), name)

    def load(self, chunkat_var: Var, *, swizzle: int = 0) -> Var:
        ca_node = chunkat_var._node
        name = self._gen_name("frag")
        node = _core.make_mma_load(ca_node, swizzle)
        _emit(node)
        swiz = f".swiz<{swizzle}>" if swizzle else ""
        _source_emit(f"{name} = mma.load{swiz} {chunkat_var._co};")
        return Var(_core.make_id_expr(name), name)

    def exec(self, acc: Var, lhs: Var, rhs: Var, *,
             method: str = "row.col") -> Var:
        acc_expr = _to_expr(acc)
        lhs_expr = _to_expr(lhs)
        rhs_expr = _to_expr(rhs)
        node = _core.make_mma_exec(method, acc_expr, lhs_expr, rhs_expr)
        _emit(node)
        _source_emit(
            f"mma.{method} {_var_co(acc)}, {_var_co(lhs)}, {_var_co(rhs)};")
        return acc

    def store(self, src: Var, target: Var, *, transpose: bool = False):
        src_expr = _to_expr(src)
        target_node = target._node
        if isinstance(target_node, _core.ChunkAt):
            node = _core.make_mma_store(src_expr, target_node, transpose)
        elif target._name:
            node = _core.make_mma_store_var(
                src_expr, target._name, transpose)
        else:
            raise TypeError(
                "mma.store target must be a chunkat expression or named Var")
        _emit(node)
        _source_emit(f"mma.store {_var_co(src)}, {target._co};")

    def commit(self):
        _emit(_core.make_mma_commit())
        _source_emit("mma.commit;")

    def frag(self, name: str, shape: list, init=0.0, *,
             dtype=None) -> "FragArray":
        """Declare a frag array: ``mc = croq.mma.frag("mc", [2, 3], 0.0)``

        CroqTile: ``frag mc[2][3] {0.0}``
        Access elements: ``mc[0][1]`` returns a FragRef for MMA ops.
        """
        bt = self._unwrap_dtype(dtype)
        node = _core.make_mma_fill(float(init), "", bt)
        _emit(node)
        return FragArray(name, shape, bt)


class FragArray:
    """A named fragment array for MMA operations.

    Supports indexing: ``mc[0][1]`` -> FragRef for use in mma.exec/store.
    """

    __slots__ = ("_name", "_shape", "_bt")

    def __init__(self, name: str, shape: list, bt):
        self._name = name
        self._shape = shape
        self._bt = bt

    def __getitem__(self, key) -> "FragRef":
        if isinstance(key, int):
            return FragRef(self._name, [key], self._shape, self._bt)
        raise TypeError(f"FragArray index must be int, got {type(key)}")


class FragRef:
    """Reference to a specific element of a frag array: mc[0][1]."""

    __slots__ = ("_name", "_indices", "_shape", "_bt")

    def __init__(self, name, indices, shape, bt):
        self._name = name
        self._indices = list(indices)
        self._shape = shape
        self._bt = bt

    def __getitem__(self, key) -> "FragRef":
        if isinstance(key, (int, Var)):
            new_indices = self._indices + [key]
            return FragRef(self._name, new_indices, self._shape, self._bt)
        raise TypeError(f"FragRef index must be int or Var, got {type(key)}")

    @property
    def node(self):
        return _core.make_id_expr(self._name)

    @property
    def name(self) -> str:
        return self._name


mma = _MMANamespace()


# =========================================================================== #
#  _ScopedBlock -- for-loop protocol for zero-duplication scoping
# =========================================================================== #

class _ScopedBlock:
    """Iterator that yields loop vars once, then finalizes the AST.

    Python's ``for`` loop: __iter__->push scope, __next__#1->yield vars,
    body executes, __next__#2->pop scope & finalize, StopIteration.
    """

    def __init__(self, vars_tuple, finalize_fn):
        self._vars = vars_tuple
        self._finalize = finalize_fn
        self._yielded = False
        self._scope = None

    def __iter__(self):
        self._scope = _Scope()
        _push_scope(self._scope)
        return self

    def __next__(self):
        if self._yielded:
            _pop_scope()
            self._finalize(self._scope.build(), self._scope.source())
            raise StopIteration
        self._yielded = True
        return self._vars[0] if len(self._vars) == 1 else self._vars

    def __del__(self):
        if self._scope is not None and not self._yielded:
            try:
                _pop_scope()
            except (IndexError, RuntimeError):
                pass


# =========================================================================== #
#  parallel / foreach / with_in
# =========================================================================== #

BLOCK = "block"
GROUP = "group"
THREAD = "thread"


def parallel(*args, scope=None, **bindings):
    """Nested parallel-by scopes with optional hardware scope.

        for p, q in croq.parallel(p=6, q=64):
            ...
        for p in croq.parallel(p=6, scope=croq.BLOCK):
            ...
        # Dynamic bounds with Var expressions:
        for m in croq.parallel(m=croq.cdiv(M, WARP_M), scope=croq.BLOCK):
            ...
    """
    loop_vars = tuple(Var(_core.make_id_expr(n), n) for n in bindings)
    items = list(bindings.items())

    def finalize(body, src):
        for vname, bound in reversed(items):
            wrapper = _core.MultiNodes()
            if isinstance(bound, Var):
                wrapper.append(_core.make_parallel_by_expr(
                    vname, _to_expr(bound), body, scope or ""))
            else:
                wrapper.append(_core.make_parallel_by(
                    vname, int(bound), body, scope or ""))
            body = wrapper
        _emit(body)
        # Source: parallel {vars} by [bounds] : scope
        scope_s = f" : {scope}" if scope else ""
        if len(items) == 1:
            n, b = items[0]
            _source_emit_block(
                f"parallel {n} by {_var_co(b)}{scope_s}", src)
        else:
            names = ", ".join(n for n, _ in items)
            bounds = ", ".join(_var_co(b) for _, b in items)
            _source_emit_block(
                f"parallel {{{names}}} by [{bounds}]{scope_s}", src)

    return _ScopedBlock(loop_vars, finalize)


def foreach(**bindings):
    """Foreach loop scope.

        for m, n, k in croq.foreach(m=128, n=256, k=256):
            out[m, n] = out[m, n] + lhs[m, k] * rhs[k, n]

        # Dynamic bounds with Var expressions:
        for k in croq.foreach(k=croq.cdiv(K, TILE_K)):
            ...
    """
    loop_vars = tuple(Var(_core.make_id_expr(n), n) for n in bindings)
    items = list(bindings.items())
    has_var = any(isinstance(v, Var) for _, v in items)

    def finalize(body, src):
        if has_var:
            expr_items = [
                (n, _to_expr(v) if isinstance(v, Var)
                 else _core.make_int_expr(int(v)))
                for n, v in items
            ]
            _emit(_core.make_foreach_expr(expr_items, body))
        else:
            int_items = [(n, int(v)) for n, v in items]
            _emit(_core.make_foreach(int_items, body))
        # Source: foreach {vars} in [bounds]
        if len(items) == 1:
            n, b = items[0]
            _source_emit_block(f"foreach {n} in [{_var_co(b)}]", src)
        else:
            names = ", ".join(n for n, _ in items)
            bounds = ", ".join(_var_co(b) for _, b in items)
            _source_emit_block(
                f"foreach {{{names}}} in [{bounds}]", src)

    return _ScopedBlock(loop_vars, finalize)


def foreach_staged(var: Var, *, start: int = 1):
    """Staged foreach: iterates from ``start`` using the enclosing with-in bound.

    CroqTile syntax: ``foreach k(1:) { ... }``
    CroqPy:
        with croq.with_in(k=K//MMA_K) as k:
            ...
            for _ in croq.foreach_staged(k, start=1):
                ...
    """
    name = var._name
    loop_var = Var(_core.make_id_expr(name), name)

    def finalize(body, src):
        _emit(_core.make_foreach_staged(name, start, body))
        _source_emit_block(f"foreach {name}({start}:)", src)

    return _ScopedBlock((loop_var,), finalize)


# Short alias
fs = foreach_staged


class _WithInContext:
    """Context manager for with-in blocks.

    Yields Var objects directly -- no dict lookup needed:

        with croq.with_in(tile_k=16) as tile_k:
            ...

        with croq.with_in(tile_m=8, tile_n=8) as (tile_m, tile_n):
            ...
    """

    def __init__(self, bindings: dict):
        self._bindings = bindings
        self._scope = None
        self._vars = tuple(
            Var(_core.make_id_expr(n), n) for n in bindings)

    def __enter__(self):
        self._scope = _Scope()
        _push_scope(self._scope)
        return self._vars[0] if len(self._vars) == 1 else self._vars

    def __exit__(self, *exc):
        _pop_scope()
        body = self._scope.build()
        src = self._scope.source()
        self._emit_with_in(body)
        self._emit_with_in_source(src)
        return False

    def _emit_with_in(self, body):
        items = list(self._bindings.items())
        has_var = any(isinstance(v, Var) for _, v in items)
        if has_var:
            expr_items = [
                (n, _to_expr(v) if isinstance(v, Var)
                 else _core.make_int_expr(int(v)))
                for n, v in items
            ]
            _emit(_core.make_with_in_expr(expr_items, body))
        else:
            _emit(_core.make_with_in(
                [(n, int(v)) for n, v in items], body))

    def _emit_with_in_source(self, src):
        items = list(self._bindings.items())
        if len(items) == 1:
            n, b = items[0]
            _source_emit_block(f"with {n} in [{_var_co(b)}]", src)
        else:
            names = ", ".join(n for n, _ in items)
            bounds = ", ".join(_var_co(b) for _, b in items)
            _source_emit_block(
                f"with {{{names}}} in [{bounds}]", src)

    # Also support for-loop pattern for backward compat
    def __iter__(self):
        block = _ScopedBlock(self._vars, self._make_finalize())
        return iter(block)

    def _make_finalize(self):
        def finalize(body, src):
            self._emit_with_in(body)
            self._emit_with_in_source(src)
        return finalize


def with_in(**bindings):
    """With-in scope (software pipelining / staging).

    Primary usage (with...as):

        with croq.with_in(tile_k=16) as tile_k:
            ...

        with croq.with_in(tile_m=8, tile_n=8) as (tile_m, tile_n):
            ...

    Also supports for-loop:

        for tile_k in croq.with_in(tile_k=16):
            ...
    """
    return _WithInContext(bindings)


# Short aliases
pb = parallel
fe = foreach
wi = with_in


# =========================================================================== #
#  @co -- the decorator
# =========================================================================== #

class _FunctionBuilder:
    def __init__(self, func_node: _core.ChoreoFunction, name: str,
                 *, source_lines: list = None, ret_co: str = "auto",
                 params_co: list = None):
        self._func = func_node
        self.name = name
        self._source_lines = source_lines or []
        self._ret_co = ret_co
        self._params_co = params_co or []

    def build(self) -> _core.ChoreoFunction:
        return self._func

    def to_co(self) -> str:
        """Generate .co source for this function."""
        params = ", ".join(self._params_co)
        header = f"__co__ {self._ret_co} {self.name}({params})"
        body_text = _serialize_source(self._source_lines, indent=1)
        return_line = ""
        if self._source_lines:
            last = self._source_lines[-1]
            if isinstance(last, str) and last.startswith("return "):
                pass
        return f"{header} {{\n{body_text}\n}}"


def co(fn=None, *, ret=None):
    """Trace a Python function into a CroqFunction AST.

        @croq.co
        def ele_add(lhs: croq.s32[2,3,64], rhs: croq.s32[2,3,64]) -> croq.s32[6,64]:
            output = croq.declare(croq.s32[6,64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

    Parameters become Var objects. Python's ``return`` emits the AST Return node.
    """
    def decorator(func):
        return _trace_function(func, ret_override=ret)
    if fn is not None:
        return _trace_function(fn, ret_override=ret)
    return decorator


def _trace_function(func, *, ret_override=None):
    import inspect
    sig = inspect.signature(func)
    hints = func.__annotations__

    ret_hint = ret_override or hints.get("return")
    if ret_hint is None:
        ret_dt = _core.DataType(_core.BaseType.VOID)
        ret_co = "void"
    else:
        ret_dt = _to_data_type(ret_hint)
        ret_co = _type_to_co(ret_hint)

    param_list = _core.ParamList()
    params_co = []
    var_args = {}
    for pname in sig.parameters:
        hint = hints.get(pname)
        if hint is None:
            raise TypeError(
                f"Parameter '{pname}' of @croq.co function "
                f"must have a type annotation")
        if isinstance(hint, _StorageAnnotation):
            dt = _to_data_type(hint._type)
            param_list.add(_core.Parameter(dt, pname, hint._storage))
            sp = _storage_to_co(hint._storage)
            params_co.append(
                f"{sp} {_type_to_co(hint._type)} {pname}")
        else:
            dt = _to_data_type(hint)
            param_list.add(_core.Parameter(dt, pname))
            params_co.append(f"{_type_to_co(hint)} {pname}")
        var_args[pname] = Var(_core.make_id_expr(pname), pname)

    scope = _Scope()
    _push_scope(scope)
    try:
        result = func(**var_args)
    finally:
        _pop_scope()

    body = scope.build()
    source_lines = list(scope.source())

    if isinstance(result, Var):
        body.append(_core.Return(result._node))
        source_lines.append(f"return {result._co};")

    func_node = _core.make_function(func.__name__, ret_dt, param_list, body)
    return _FunctionBuilder(func_node, func.__name__,
                            source_lines=source_lines,
                            ret_co=ret_co,
                            params_co=params_co)


# =========================================================================== #
#  Program
# =========================================================================== #

class Program:
    """Top-level program container.

        prog = croq.Program()
        prog.add(ele_add)
        print(prog.dump_ast())
    """

    def __init__(self):
        self._items: list = []
        self._defines: list = []

    def define(self, name: str, value):
        """Add a #define directive: ``prog.define("M", 128)``."""
        self._defines.append((name, value))
        return self

    def add(self, item):
        """Add a @croq.co function or host code string."""
        if isinstance(item, str):
            self._items.append(("host", item))
        elif isinstance(item, _FunctionBuilder):
            self._items.append(("func", item))
        else:
            raise TypeError(f"Cannot add {type(item)} to Program")
        return self

    def build(self) -> _core.Program:
        prog = _core.Program()
        for kind, item in self._items:
            if kind == "func":
                prog.append(item.build())
            else:
                prog.append(_core.CppSourceCode(item))
        return prog

    def dump_ast(self) -> str:
        return _core.dump_ast(self.build())

    def to_co(self, *, run_directive: str = "",
              check_lines: Optional[List[str]] = None) -> str:
        """Generate complete .co source file.

        Args:
            run_directive: Optional RUN directive
                (e.g. "croqtile -gs -t cute %s -o %s.cute.result && ...")
            check_lines: Optional CHECK lines
                (e.g. ["Test Passed"])
        """
        parts = []
        if run_directive:
            parts.append(f"// REQUIRES: TARGET-GPU")
            parts.append(f"// RUN: {run_directive}")
            parts.append("")
        for name, value in self._defines:
            parts.append(f"#define {name} {value}")
        if self._defines:
            parts.append("")
        for kind, item in self._items:
            if kind == "func":
                parts.append(item.to_co())
                parts.append("")
            else:
                parts.append(item.strip())
                parts.append("")
        if check_lines:
            for line in check_lines:
                parts.append(f"// CHECK: {line}")
        return "\n".join(parts)


# =========================================================================== #
#  Compilation
# =========================================================================== #

def compile_to_source(program: Program, *,
                      target: str = "cute",
                      arch: str = "",
                      croqtile_bin: str = "") -> str:
    """Compile a CroqTile program through the croqtile compiler.

    Generates a ``.co`` file from the program, then invokes the ``croqtile``
    compiler binary to compile it.  Returns the generated CUDA/C++ source.
    """
    from croqtile.runtime import compile_co_to_cuda, find_croqtile_bin

    if not croqtile_bin:
        croqtile_bin = find_croqtile_bin()
    if not croqtile_bin:
        raise RuntimeError(
            "Cannot find croqtile compiler binary. "
            "Set CROQTILE_BIN env var.")

    co_source = program.to_co()
    return compile_co_to_cuda(
        co_source, arch=arch, target=target,
        croqtile_bin=croqtile_bin)


def dump_ast(program: Program) -> str:
    return program.dump_ast()


# =========================================================================== #
#  Kept for backward compat but not recommended
# =========================================================================== #

Tensor = TensorType
Expr = Var


def var(name: str) -> Var:
    """Reference a variable by name. Prefer using Var objects directly."""
    return Var(_core.make_id_expr(name), name)


def return_(expr=None):
    """Explicit return. Prefer Python's ``return`` inside @croq.co."""
    if expr is None:
        _emit(_core.Return(_core.make_id_expr("void")))
    else:
        _emit(_core.Return(_unwrap(expr)))


def assign_at(name: str, indices: Sequence, value):
    """Emit indexed assignment. Prefer ``var[i, j] = expr``."""
    mv = _core.MultiValues()
    for idx in indices:
        mv.append(_to_var(idx)._node)
    da = _core.DataAccess(name, mv)
    _emit(_core.Assignment(da, _unwrap(value)))


class Param:
    __slots__ = ("name", "tensor", "storage")

    def __init__(self, name: str, tensor, storage=None):
        if isinstance(tensor, TensorType):
            self.tensor = tensor
        elif isinstance(tensor, _DType):
            self.tensor = TensorType(tensor._bt, [])
        else:
            self.tensor = tensor
        self.name = name
        self.storage = storage

    def to_parameter(self) -> _core.Parameter:
        dt = self.tensor.to_data_type()
        if self.storage is not None:
            return _core.Parameter(dt, self.name, self.storage)
        return _core.Parameter(dt, self.name)


class Function:
    """Imperative function builder. Prefer @croq.co."""

    def __init__(self, name, ret_type, params):
        self.name = name
        self._ret_type = ret_type
        self._params = list(params)
        self._body = _core.MultiNodes()

    def _append(self, node):
        self._body.append(node)

    def declare(self, name, tensor, *, storage=None, init=None):
        dt = _to_data_type(tensor)
        if storage is not None and init is not None:
            self._append(_core.NamedVariableDecl(name, dt, storage, int(init)))
        elif storage is not None:
            self._append(_core.NamedVariableDecl(name, dt, storage))
        else:
            self._append(_core.NamedVariableDecl(name, dt))
        return Var(_core.make_id_expr(name), name)

    def var(self, name):
        return Var(_core.make_id_expr(name), name)

    def assign_at(self, name, indices, value):
        mv = _core.MultiValues()
        for idx in indices:
            mv.append(_to_var(idx)._node)
        self._append(_core.Assignment(_core.DataAccess(name, mv),
                                       _unwrap(value)))

    def parallel_by(self, bindings):
        return BlockBuilder(self, bindings)

    def return_(self, expr):
        self._append(_core.Return(_unwrap(expr)))

    def build(self):
        dt = _to_data_type(self._ret_type)
        pl = _core.ParamList()
        for p in self._params:
            pl.add(p.to_parameter())
        return _core.make_function(self.name, dt, pl, self._body)


class BlockBuilder:
    def __init__(self, func, bindings):
        self._func = func
        self._bindings = bindings
        self._stmts = _core.MultiNodes()

    def var(self, name):
        return self._func.var(name)

    def assign_at(self, name, indices, value):
        mv = _core.MultiValues()
        for idx in indices:
            mv.append(_to_var(idx)._node)
        self._stmts.append(_core.Assignment(
            _core.DataAccess(name, mv), _unwrap(value)))

    def finalize(self):
        body = self._stmts
        for vname, bound in reversed(self._bindings[1:]):
            wrapper = _core.MultiNodes()
            wrapper.append(_core.make_parallel_by(vname, bound, body))
            body = wrapper
        self._func._body.append(
            _core.make_parallel_by(
                self._bindings[0][0], self._bindings[0][1], body))


# =========================================================================== #
#  Internals
# =========================================================================== #

def _unwrap(obj):
    if isinstance(obj, Var):
        return obj._node
    if isinstance(obj, _core.Node):
        return obj
    if isinstance(obj, _FullSlice):
        return _core.make_id_expr("_")
    if isinstance(obj, int):
        return _core.make_int_expr(obj)
    if isinstance(obj, float):
        return _core.make_float_expr(obj)
    raise TypeError(f"Cannot convert {type(obj)} to AST node")


def _coerce(obj) -> "Var":
    """Convert a value to a Var with proper .co representation."""
    if isinstance(obj, Var):
        return obj
    if isinstance(obj, FragRef):
        idx = "".join(f"[{i}]" for i in obj._indices)
        name = f"{obj._name}{idx}"
        return Var(_core.make_id_expr(obj._name), name, co=name)
    if isinstance(obj, _FullSlice):
        return Var(_core.make_id_expr("_"), "_", co="_")
    if isinstance(obj, int):
        return Var.lit(obj)
    if isinstance(obj, float):
        return Var.lit(obj)
    raise TypeError(f"Cannot coerce {type(obj)} to Var")


def _var_co(obj) -> str:
    """Get .co representation of any value."""
    if isinstance(obj, Var):
        return obj._co
    if isinstance(obj, FragRef):
        idx = "".join(f"[{i}]" for i in obj._indices)
        return f"{obj._name}{idx}"
    if isinstance(obj, _FullSlice):
        return "_"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        s = f"{obj}f" if obj == int(obj) else str(obj)
        return s
    return str(obj)


def _type_to_co(type_spec) -> str:
    """Convert a type spec to .co type string."""
    if isinstance(type_spec, TensorType):
        bt = type_spec.dtype.name.lower()
        dims = ", ".join(str(d) for d in type_spec.shape)
        return f"{bt} [{dims}]"
    if isinstance(type_spec, _DType):
        return type_spec._bt.name.lower()
    if isinstance(type_spec, MDSpan):
        bt = type_spec.dtype.name.lower()
        if type_spec.shape:
            dims = ", ".join(str(d) for d in type_spec.shape)
            return f"{bt} [{dims}]"
        return bt
    if isinstance(type_spec, _core.DataType):
        return "auto"
    return "auto"


_STORAGE_CO = {
    Storage.SHARED: "shared",
    Storage.LOCAL: "local",
    Storage.GLOBAL: "global",
}


def _storage_to_co(storage) -> str:
    return _STORAGE_CO.get(storage, "")


def _serialize_source(items: list, indent: int = 0) -> str:
    """Convert a source IR (list of strs / dicts) to indented .co text."""
    lines = []
    prefix = "  " * indent
    for item in items:
        if isinstance(item, str):
            lines.append(prefix + item)
        elif isinstance(item, dict):
            header = item["header"]
            body = item.get("body", [])
            if body:
                lines.append(prefix + header + " {")
                lines.append(_serialize_source(body, indent + 1))
                lines.append(prefix + "}")
            else:
                lines.append(prefix + header)
    return "\n".join(lines)


def _to_expr(obj) -> _core.Expr:
    """Convert to an Expr node (needed for MMA exec/store)."""
    if isinstance(obj, Var):
        node = obj._node
        if isinstance(node, _core.Expr):
            return node
        if obj._name:
            return _core.make_id_expr(obj._name)
    if isinstance(obj, FragRef):
        return _core.make_id_expr(obj._name)
    if isinstance(obj, _core.Expr):
        return obj
    if isinstance(obj, int):
        return _core.make_int_expr(obj)
    if isinstance(obj, float):
        return _core.make_float_expr(obj)
    raise TypeError(f"Cannot convert {type(obj)} to Expr")


def _to_var(obj) -> Var:
    if isinstance(obj, Var):
        return obj
    if isinstance(obj, str):
        return Var(_core.make_id_expr(obj), obj)
    if isinstance(obj, int):
        return Var(_core.make_int_expr(obj))
    if isinstance(obj, float):
        return Var(_core.make_float_expr(obj))
    raise TypeError(f"Cannot convert {type(obj)} to Var")

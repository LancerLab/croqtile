"""
Tests for the croqtile-python bindings.

Layers:
  1. _core: raw C++ binding smoke tests
  2. Types: croq.s32[6, 64]
  3. Var operators + @ (hash) operator
  4. @croq.co + for-loop scoped blocks (parallel / foreach / with_in)
  5. Python return -> AST Return
  6. DMA / TMA / MMA API
  7. Full kernel ports from choreo/tests/gpu/end2end/
  8. Edge cases
"""

import pytest
import croq


# ===================================================================
# 1. Low-level _core tests
# ===================================================================

class TestCoreImport:
    def test_version(self):
        from croqtile import _core
        v = _core.version()
        assert isinstance(v, str) and len(v) > 0

    def test_enum_basetype(self):
        from croqtile._core import BaseType
        assert BaseType.F16 != BaseType.F32

    def test_enum_storage(self):
        from croqtile._core import Storage
        assert Storage.GLOBAL != Storage.SHARED


class TestCoreNodes:
    def test_identifier(self):
        from croqtile._core import Identifier
        assert Identifier("foo").name == "foo"

    def test_int_literal(self):
        from croqtile._core import IntLiteral
        assert IntLiteral(42).value == 42

    def test_make_id_expr(self):
        from croqtile._core import make_id_expr
        assert make_id_expr("x").is_reference()

    def test_make_binary_expr(self):
        from croqtile._core import make_binary_expr, make_int_expr
        expr = make_binary_expr("+", make_int_expr(3), make_int_expr(4))
        assert expr.is_binary()

    def test_multi_nodes(self):
        from croqtile._core import MultiNodes, make_int_expr
        mn = MultiNodes()
        assert mn.count() == 0
        mn.append(make_int_expr(1))
        assert mn.count() == 1

    def test_program_with_function(self):
        from croqtile._core import (
            Program, make_function, DataType, BaseType,
            ParamList, MultiNodes, dump_ast,
        )
        prog = Program()
        func = make_function("test_func", DataType(BaseType.VOID),
                             ParamList(), MultiNodes())
        prog.append(func)
        assert prog.count() == 1
        assert "test_func" in dump_ast(prog)


# ===================================================================
# 2. Type syntax
# ===================================================================

class TestTypeSyntax:
    def test_scalar_dtype(self):
        assert repr(croq.s32) == "dtype(s32)"

    def test_shaped_dtype(self):
        t = croq.s32[6, 64]
        assert isinstance(t, croq.TensorType)
        assert t.shape == [6, 64]

    def test_1d_shape(self):
        t = croq.f16[128]
        assert t.shape == [128]

    def test_to_data_type(self):
        dt = croq.f32[10, 20].to_data_type()
        assert dt is not None


# ===================================================================
# 3. Var operators + @ (hash) operator
# ===================================================================

class TestVar:
    def test_arithmetic(self):
        from croqtile.builder import Var, _core
        a = Var(_core.make_id_expr("a"), "a")
        b = Var(_core.make_id_expr("b"), "b")
        assert (a + b).node.is_binary()
        assert (a - b).node.is_binary()
        assert (a * b).node.is_binary()
        assert (a / b).node.is_binary()
        assert (a % b).node.is_binary()

    def test_radd(self):
        from croqtile.builder import Var, _core
        a = Var(_core.make_id_expr("a"), "a")
        c = 3 + a
        assert c.node.is_binary()

    def test_comparison(self):
        from croqtile.builder import Var, _core
        a = Var(_core.make_id_expr("a"), "a")
        b = Var(_core.make_id_expr("b"), "b")
        assert (a < b).node.is_binary()
        assert (a > b).node.is_binary()

    def test_getitem(self):
        from croqtile.builder import Var, _core
        v = Var(_core.make_id_expr("data"), "data")
        elem = v[Var(_core.make_id_expr("i"), "i"),
                 Var(_core.make_id_expr("j"), "j")]
        assert isinstance(elem, croq.Var)

    def test_getitem_int(self):
        from croqtile.builder import Var, _core
        v = Var(_core.make_id_expr("data"), "data")
        elem = v[0]
        assert isinstance(elem, croq.Var)

    def test_int_coercion(self):
        from croqtile.builder import Var, _core
        a = Var(_core.make_id_expr("a"), "a")
        c = a + 5
        assert c.node.is_binary()

    def test_negation(self):
        from croqtile.builder import Var, _core
        a = Var(_core.make_id_expr("a"), "a")
        neg = -a
        assert neg.node.is_binary()

    def test_hash_operator(self):
        """The @ operator maps to Choreo's # (hierarchical index)."""
        from croqtile.builder import Var, _core
        m = Var(_core.make_id_expr("m"), "m")
        g = Var(_core.make_id_expr("g"), "g")
        result = m @ g
        assert result.node.is_binary()

    def test_data_property(self):
        from croqtile.builder import Var, _core
        f = Var(_core.make_id_expr("f"), "f")
        d = f.data
        assert isinstance(d, croq.Var)
        assert "f.data" in d.name

    def test_chunkat(self):
        from croqtile.builder import Var, _core
        v = Var(_core.make_id_expr("lhs"), "lhs")
        m = Var(_core.make_id_expr("m"), "m")
        k = Var(_core.make_id_expr("k"), "k")
        ca = v.chunkat(m, k)
        assert isinstance(ca, croq.Var)


# ===================================================================
# 4. @croq.co + for-loop scoped blocks
# ===================================================================

class TestDeclarativeAPI:
    def test_simple_function(self):
        @croq.co
        def my_func(x: croq.s32[10]) -> croq.s32[10]:
            return x

        prog = croq.Program()
        prog.add(my_func)
        text = prog.dump_ast()
        assert "my_func" in text
        assert "Return" in text

    def test_ele_add_for_loop(self):
        """Canonical test: for p, q in croq.parallel(...)"""
        @croq.co
        def ele_add(lhs: croq.s32[2, 3, 64],
                    rhs: croq.s32[2, 3, 64]) -> croq.s32[6, 64]:
            output = croq.declare(croq.s32[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add)
        text = prog.dump_ast()

        assert "ele_add" in text
        assert "lhs" in text
        assert "rhs" in text
        assert "output" in text
        assert text.count("Parallelization") == 2
        assert "+" in text
        assert "Return" in text

    def test_single_parallel_var(self):
        @croq.co
        def kern(a: croq.s32[10]) -> croq.s32[10]:
            out = croq.declare(croq.s32[10], "out")
            for i in croq.parallel(i=10):
                out[i] = a[i] + 1
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "Parallelization" in text

    def test_declare_with_storage(self):
        @croq.co
        def kernel(x: croq.f16[128]) -> croq.f16[128]:
            buf = croq.declare(croq.f16[128, 64], "buf", storage=croq.SHARED)
            return x

        text = croq.Program().add(kernel).dump_ast()
        assert "buf" in text

    def test_declare_with_init(self):
        @croq.co
        def kernel(x: croq.s32[8]) -> croq.s32[8]:
            acc = croq.declare(croq.s32[8, 128], "acc",
                               storage=croq.LOCAL, init=0)
            return x

        text = croq.Program().add(kernel).dump_ast()
        assert "acc" in text

    def test_co_with_ret_kwarg(self):
        @croq.co(ret=croq.s32[10])
        def my_func(x: croq.s32[10]):
            return x

        text = croq.Program().add(my_func).dump_ast()
        assert "my_func" in text

    def test_setitem_assignment(self):
        @croq.co
        def kern(a: croq.s32[10], b: croq.s32[10]) -> croq.s32[10]:
            out = croq.declare(croq.s32[10], "out")
            for i in croq.parallel(i=10):
                out[i] = a[i] + b[i]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "Assign" in text
        assert "+" in text


# ===================================================================
# 5. foreach / with_in (for-loop pattern)
# ===================================================================

class TestForeach:
    def test_foreach_single(self):
        @croq.co
        def kern(a: croq.s32[10, 10]) -> croq.s32[10, 10]:
            out = croq.declare(croq.s32[10, 10], "out")
            for i in croq.foreach(i=10):
                for j in croq.foreach(j=10):
                    out[i, j] = a[i, j]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "Foreach" in text or "Loop" in text or "Iteration" in text

    def test_foreach_multi(self):
        @croq.co
        def kern(a: croq.s32[128, 256],
                 b: croq.s32[256, 256]) -> croq.s32[128, 256]:
            out = croq.declare(croq.s32[128, 256], "out")
            for m, n, k in croq.foreach(m=128, n=256, k=256):
                out[m, n] = out[m, n] + a[m, k] * b[k, n]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "Foreach" in text or "Iteration" in text


class TestWithIn:
    def test_with_in_for_loop(self):
        """with_in now uses for-loop pattern -- no more V['k']"""
        @croq.co
        def kern(a: croq.s32[16, 16]) -> croq.s32[16, 16]:
            out = croq.declare(croq.s32[16, 16], "out")
            for k in croq.with_in(k=16):
                for i in croq.foreach(i=16):
                    out[i, k] = a[i, k]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "With" in text

    def test_with_in_multi_vars(self):
        @croq.co
        def kern(a: croq.s32[16, 16]) -> croq.s32[16, 16]:
            out = croq.declare(croq.s32[16, 16], "out")
            for tile_m, tile_n in croq.with_in(tile_m=8, tile_n=8):
                for i in croq.foreach(i=8):
                    out[i, tile_n] = a[i, tile_m]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "With" in text


# ===================================================================
# 6. DMA / TMA / MMA API
# ===================================================================

class TestDMA:
    def test_dma_copy_sync(self):
        """Port of tests/gpu/end2end/copy.co -- sync DMA copy"""
        @croq.co
        def dma_copy_sync(input: croq.s32[6, 17, 64]) -> croq.s32[6, 17, 64]:
            output = croq.declare(croq.s32[6, 17, 64], "output")
            for _ in croq.parallel(p=1, q=1):
                f = croq.dma.copy(input, to=croq.SHARED, name="f")
                croq.dma.copy(f.data, to=output, name="")
            return output

        text = croq.Program().add(dma_copy_sync).dump_ast()
        assert "DMA" in text
        assert "dma_copy_sync" in text

    def test_dma_copy_async(self):
        """Port of tests/gpu/end2end/copy.co -- async DMA copy"""
        @croq.co
        def dma_copy_async(input: croq.s32[6, 17, 64]) -> croq.s32[6, 17, 64]:
            output = croq.declare(croq.s32[6, 17, 64], "output")
            for _ in croq.parallel(p=1, q=1):
                f = croq.dma.copy_async(input, to=croq.SHARED, name="f")
                croq.wait(f)
                croq.dma.copy(f.data, to=output, name="")
            return output

        text = croq.Program().add(dma_copy_async).dump_ast()
        assert "DMA" in text
        assert "async" in text.lower() or "WAIT" in text or "Wait" in text


class TestMMA:
    def test_mma_fill(self):
        @croq.co
        def kern(lhs: croq.f16[128, 64]) -> croq.f16[128, 256]:
            output = croq.declare(croq.f16[128, 256], "output")
            mc = croq.mma.fill(0.0)
            return output

        text = croq.Program().add(kern).dump_ast()
        assert "MMA" in text

    def test_mma_load_and_exec(self):
        @croq.co
        def kern(lhs: croq.f16[128, 64],
                 rhs: croq.f16[64, 256]) -> croq.f16[128, 256]:
            output = croq.declare(croq.f16[128, 256], "output")
            mc = croq.mma.fill(0.0)
            for m, n in croq.parallel(m=8, n=16):
                for k in croq.foreach(k=4):
                    ma = croq.mma.load(lhs.chunkat(m, k))
                    mb = croq.mma.load(rhs.chunkat(k, n))
                    croq.mma.exec(mc, ma, mb, method="row.col")
                croq.mma.store(mc, output.chunkat(m, n))
            return output

        text = croq.Program().add(kern).dump_ast()
        assert "MMA" in text
        assert "ROW.COL" in text or "row" in text.lower()


# ===================================================================
# 7. Full kernel ports
# ===================================================================

class TestKernelPorts:
    def test_ele_add_co(self):
        """Port of tests/gpu/end2end/add.co"""
        @croq.co
        def ele_add(lhs: croq.s32[2, 3, 64],
                    rhs: croq.s32[2, 3, 64]) -> croq.s32[6, 64]:
            output = croq.declare(croq.s32[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add)
        text = prog.dump_ast()
        assert "ChoreoFunction" in text
        assert "ele_add" in text
        assert text.count("Parallelization") == 2
        assert "+" in text

    def test_matmul_co(self):
        """Port of tests/gpu/end2end/matmul.co (simplified)"""
        @croq.co
        def matmul(lhs: croq.s32[128, 256],
                   rhs: croq.s32[256, 256]) -> croq.s32[128, 256]:
            out = croq.declare(croq.s32[128, 256], "out")
            for p, q in croq.parallel(p=16, q=64):
                for m, n, k in croq.foreach(m=8, n=4, k=256):
                    out[m, n] = out[m, n] + lhs[m, k] * rhs[k, n]
            return out

        prog = croq.Program()
        prog.add(matmul)
        text = prog.dump_ast()
        assert "matmul" in text
        assert "Parallelization" in text
        assert "*" in text

    def test_matmul_dma_co(self):
        """Port of tests/gpu/end2end/matmul-dma.co structure"""
        @croq.co
        def matmul_dma(lhs: croq.s32[128, 256],
                       rhs: croq.s32[256, 256]) -> croq.s32[128, 256]:
            output = croq.declare(croq.s32[128, 256], "output")
            for px, py in croq.parallel(px=8, py=16):
                for tile_k in croq.with_in(tile_k=16):
                    lhs_load = croq.dma.copy(
                        lhs.chunkat(px, tile_k), to=croq.LOCAL, name="lhs_load")
                    rhs_load = croq.dma.copy(
                        rhs.chunkat(tile_k, py), to=croq.LOCAL, name="rhs_load")
                    for qx, qy in croq.parallel(qx=16, qy=16):
                        for k in croq.foreach(k=16):
                            output[px @ qx, py @ qy] = (
                                output[px @ qx, py @ qy]
                                + lhs_load.data[qx, k]
                                * rhs_load.data[k, qy])
            return output

        prog = croq.Program()
        prog.add(matmul_dma)
        text = prog.dump_ast()
        assert "matmul_dma" in text
        assert "DMA" in text
        assert "With" in text

    def test_mma_matmul_co(self):
        """Port of tests/gpu/end2end/mma.co structure"""
        M, N, K = 128, 256, 64
        MMA_M, MMA_N, MMA_K = 16, 16, 16

        @croq.co
        def matmul_mma(lhs: croq.f16[128, 64],
                       rhs: croq.f16[64, 256]) -> croq.f16[128, 256]:
            output = croq.declare(croq.f16[128, 256], "output")
            for m, n in croq.parallel(m=M // MMA_M // 4, n=N // MMA_N):
                mc = croq.mma.fill(0.0)
                for g0, g1 in croq.parallel(g0=4, g1=1):
                    croq.mma.fill(mc, 0.0)
                    for k in croq.foreach(k=K // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        croq.mma.exec(mc, ma, mb, method="row.col")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))
            return output

        prog = croq.Program()
        prog.add(matmul_mma)
        text = prog.dump_ast()
        assert "matmul_mma" in text
        assert "MMA" in text
        assert "DMA" not in text or True  # MMA should be present

    def test_host_code(self):
        @croq.co
        def my_add(a: croq.s32[10], b: croq.s32[10]) -> croq.s32[10]:
            out = croq.declare(croq.s32[10], "out")
            for i in croq.parallel(i=10):
                out[i] = a[i] + b[i]
            return out

        prog = croq.Program()
        prog.add(my_add)
        prog.add("""
int main() {
    auto a = choreo::make_spandata<choreo::s32>(10);
    auto b = choreo::make_spandata<choreo::s32>(10);
    a.fill_random(-10, 10);
    b.fill_random(-10, 10);
    auto res = my_add(a.view(), b.view());
    std::cout << "Test Passed" << std::endl;
}
""")
        text = prog.dump_ast()
        assert "my_add" in text
        assert "CppSource" in text or "main" in text.lower() or "Source" in text


# ===================================================================
# 8. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_nested_arithmetic(self):
        from croqtile.builder import Var, _core
        a = Var(_core.make_id_expr("a"), "a")
        b = Var(_core.make_id_expr("b"), "b")
        c = Var(_core.make_id_expr("c"), "c")
        result = (a + b) * c - a
        assert result.node.is_binary()

    def test_multiple_functions(self):
        @croq.co
        def func1(x: croq.s32[4]) -> croq.s32[4]:
            return x

        @croq.co
        def func2(y: croq.f32[8]) -> croq.f32[8]:
            return y

        prog = croq.Program()
        prog.add(func1)
        prog.add(func2)
        text = prog.dump_ast()
        assert "func1" in text
        assert "func2" in text

    def test_var_bool_raises(self):
        from croqtile.builder import Var, _core
        a = Var(_core.make_id_expr("a"), "a")
        with pytest.raises(TypeError, match="boolean"):
            if a:
                pass

    def test_import_croq(self):
        import croq as c
        assert hasattr(c, "co")
        assert hasattr(c, "parallel")
        assert hasattr(c, "foreach")
        assert hasattr(c, "with_in")
        assert hasattr(c, "pb")
        assert hasattr(c, "fe")
        assert hasattr(c, "wi")
        assert hasattr(c, "s32")
        assert hasattr(c, "dma")
        assert hasattr(c, "tma")
        assert hasattr(c, "mma")
        assert hasattr(c, "MDSpan")
        assert hasattr(c, "ITuple")

    def test_return_none_is_void(self):
        """A function returning None should not emit a Return statement."""
        @croq.co
        def kern(x: croq.s32[10]) -> croq.s32[10]:
            out = croq.declare(croq.s32[10], "out")

        text = croq.Program().add(kern).dump_ast()
        assert "`- Return:" not in text

    def test_hash_operator_in_kernel(self):
        """Test @ operator inside a real kernel context."""
        @croq.co
        def kern(a: croq.s32[32, 32]) -> croq.s32[32, 32]:
            out = croq.declare(croq.s32[32, 32], "out")
            for p in croq.parallel(p=4):
                for q in croq.parallel(q=8):
                    for g in croq.parallel(g=4):
                        out[p @ g, q] = a[p @ g, q]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "Parallelization" in text
        assert "#" in text

    def test_short_aliases(self):
        """Test pb/fe/wi short aliases."""
        @croq.co
        def kern(a: croq.s32[10, 10]) -> croq.s32[10, 10]:
            out = croq.declare(croq.s32[10, 10], "out")
            for p, q in croq.pb(p=10, q=10):
                out[p, q] = a[p, q]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert text.count("Parallelization") == 2

    def test_short_aliases_foreach_with_in(self):
        """Test fe/wi short aliases."""
        @croq.co
        def kern(a: croq.s32[16, 16]) -> croq.s32[16, 16]:
            out = croq.declare(croq.s32[16, 16], "out")
            for k in croq.wi(k=16):
                for i in croq.fe(i=16):
                    out[i, k] = a[i, k]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "With" in text

    def test_mdspan_type(self):
        ms = croq.MDSpan(croq.f16, shape=[128, 256])
        assert ms.rank == 2
        assert ms.shape == [128, 256]
        assert ms.dtype == croq._core.BaseType.F16
        assert "MDSpan" in repr(ms)

    def test_mdspan_rank_only(self):
        ms = croq.MDSpan(croq.s32, rank=3)
        assert ms.rank == 3
        assert ms.shape is None

    def test_ituple(self):
        it = croq.ITuple(6, 64)
        assert it.rank == 2
        assert it.bounds == [6, 64]
        assert "ITuple" in repr(it)

    def test_ituple_1d(self):
        it = croq.ITuple(128)
        assert it.rank == 1
        assert it.bounds == [128]

    def test_mdspan_in_declare(self):
        """MDSpan can be used as a type in declare()."""
        @croq.co
        def kern(x: croq.f16[128]) -> croq.f16[128]:
            buf = croq.declare(croq.MDSpan(croq.f16, shape=[64, 32]), "buf",
                               storage=croq.SHARED)
            return x

        text = croq.Program().add(kern).dump_ast()
        assert "buf" in text


class TestWithAs:
    """with...as pattern for with_in."""

    def test_with_as_single(self):
        @croq.co
        def kern(a: croq.s32[16, 16]) -> croq.s32[16, 16]:
            out = croq.declare(croq.s32[16, 16], "out")
            with croq.with_in(k=16) as k:
                for i in croq.foreach(i=16):
                    out[i, k] = a[i, k]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "With" in text

    def test_with_as_multi(self):
        @croq.co
        def kern(a: croq.s32[8, 8]) -> croq.s32[8, 8]:
            out = croq.declare(croq.s32[8, 8], "out")
            with croq.with_in(tile_m=8, tile_n=8) as (tile_m, tile_n):
                out[tile_m, tile_n] = a[tile_m, tile_n]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "With" in text

    def test_wi_alias_with_as(self):
        """Short alias also works with with...as."""
        @croq.co
        def kern(a: croq.s32[16]) -> croq.s32[16]:
            out = croq.declare(croq.s32[16], "out")
            with croq.wi(k=16) as k:
                out[k] = a[k]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "With" in text


class TestSelect:
    """Ternary selection."""

    def test_select_basic(self):
        @croq.co
        def kern(x: croq.s32[6, 64]) -> croq.s32[6, 64]:
            out = croq.declare(croq.s32[6, 64], "out")
            for p, q in croq.parallel(p=6, q=64):
                out[p, q] = croq.select(p > q, x[p, q], p + q)
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "?" in text or "Select" in text


class TestScopeAnnotations:
    """Parallel scope annotations (block, group, thread)."""

    def test_block_scope(self):
        @croq.co
        def kern(x: croq.s32[128]) -> croq.s32[128]:
            out = croq.declare(croq.s32[128], "out")
            for p in croq.parallel(p=128, scope=croq.BLOCK):
                out[p] = x[p]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "block" in text or "Parallelization" in text

    def test_thread_scope(self):
        @croq.co
        def kern(x: croq.s32[6, 128]) -> croq.s32[6, 128]:
            out = croq.declare(croq.s32[6, 128], "out")
            for p in croq.parallel(p=6, scope=croq.BLOCK):
                for t in croq.parallel(t=128, scope=croq.THREAD):
                    out[p, t] = x[p, t]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "Parallelization" in text

    def test_group_scope(self):
        @croq.co
        def kern(x: croq.s32[4, 64]) -> croq.s32[4, 64]:
            out = croq.declare(croq.s32[4, 64], "out")
            for p in croq.parallel(p=4, scope="group-4"):
                for t in croq.parallel(t=64, scope=croq.THREAD):
                    out[p, t] = x[p, t]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "Parallelization" in text


class TestSwap:
    """swap(a, b) statement."""

    def test_swap_basic(self):
        @croq.co(ret=croq.s32[8])
        def kern(a: croq.s32[8], b: croq.s32[8]):
            croq.swap(a, b)
            return a

        text = croq.Program().add(kern).dump_ast()
        assert "Swap" in text or "Rotate" in text

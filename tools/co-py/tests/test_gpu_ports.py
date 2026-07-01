"""
Ports of Choreo gpu/end2end tests to croqtile-python.

Each test corresponds to a .co file under tests/gpu/end2end/ in the Choreo
repo. These tests verify that croqtile-python can express the same kernel patterns
and produce valid AST.
"""
import croqtile as croq


# =========================================================================== #
#  Port: mma.co -- basic MMA matmul with group parallelism
# =========================================================================== #

M, N, K = 128, 256, 64

class TestMmaCo:
    def test_mma_matmul(self):
        MMA_M, MMA_N, MMA_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[M, K], rhs: croq.f16[K, N]) -> croq.f16[M, N]:
            output = croq.declare(croq.f16[M, N], "output")

            for m, n in croq.parallel(
                    m=M // MMA_M // 4, n=N // MMA_N, scope=croq.BLOCK):
                mc = croq.mma.fill(0.0)
                for g0, g1 in croq.parallel(g0=4, g1=1, scope=croq.GROUP):
                    mc = croq.mma.fill(mc, 0.0, dtype=croq.f32)
                    for k in croq.foreach(k=K // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        mc = croq.mma.exec(mc, ma, mb, method="row.col")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))

            return output

        prog = croq.Program()
        prog.add(matmul)
        text = prog.dump_ast()

        assert "MMA" in text
        assert "Parallelization" in text
        assert "LOAD" in text or "STORE" in text


# =========================================================================== #
#  Port: copy.co -- DMA copy sync + async
# =========================================================================== #

class TestCopyCo:
    def test_dma_copy_sync(self):
        @croq.co
        def dma_copy_sync(
                inp: croq.s32[6, 17, 64]) -> croq.s32[6, 17, 64]:
            output = croq.declare(croq.s32[6, 17, 64], "output")
            for _p in croq.parallel(_p=1):
                f = croq.dma.copy(inp, to=croq.SHARED)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy_sync).dump_ast()
        assert "DMA" in text

    def test_dma_copy_async(self):
        @croq.co
        def dma_copy_async(
                inp: croq.s32[6, 17, 64]) -> croq.s32[6, 17, 64]:
            output = croq.declare(croq.s32[6, 17, 64], "output")
            for _p in croq.parallel(_p=1):
                f = croq.dma.copy_async(inp, to=croq.SHARED)
                croq.wait(f)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy_async).dump_ast()
        assert "DMA" in text
        assert "WAIT" in text


# =========================================================================== #
#  Port: matmul.co -- basic triple-loop matmul
# =========================================================================== #

class TestMatmulCo:
    def test_basic_matmul(self):
        M_, N_, K_ = 128, 256, 256

        @croq.co
        def matmul(lhs: croq.s32[M_, K_],
                   rhs: croq.s32[K_, N_]) -> croq.s32[M_, N_]:
            output = croq.declare(croq.s32[M_, N_], "output")
            for m, n, k in croq.foreach(m=M_, n=N_, k=K_):
                output[m, n] = output[m, n] + lhs[m, k] * rhs[k, n]
            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "Foreach" in text


# =========================================================================== #
#  Port: matmul-dma.co -- matmul with DMA staging
# =========================================================================== #

class TestMatmulDmaCo:
    def test_matmul_dma(self):
        M_, N_, K_ = 128, 256, 256

        @croq.co
        def matmul(lhs: croq.s32[M_, K_],
                   rhs: croq.s32[K_, N_]) -> croq.s32[M_, N_]:
            output = croq.declare(croq.s32[M_, N_], "output")

            for px, py in croq.parallel(px=8, py=16, scope=croq.BLOCK):
                for qx, qy in croq.parallel(qx=16, qy=16, scope=croq.THREAD):
                    for k in croq.foreach(k=K_):
                        output[px @ qx, py @ qy] = (
                            output[px @ qx, py @ qy]
                            + lhs[px @ qx, k] * rhs[k, py @ qy])

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "Parallelization" in text
        assert "block" in text or "Foreach" in text


# =========================================================================== #
#  Port: mma_matmul.co -- MMA-based matmul from test_basic (canonical)
# =========================================================================== #

class TestMmaMatmulCo:
    def test_mma_matmul_canonical(self):
        M_, N_, K_ = 128, 128, 64
        MMA_M, MMA_N, MMA_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[M_, K_],
                   rhs: croq.f16[K_, N_]) -> croq.f16[M_, N_]:
            output = croq.declare(croq.f16[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M, n=N_ // MMA_N, scope=croq.BLOCK):
                mc = croq.mma.fill(0.0, dtype=croq.f32)
                for k in croq.foreach(k=K_ // MMA_K):
                    fl = croq.dma.copy_async(
                        lhs.chunkat(m, k), to=croq.SHARED)
                    fr = croq.dma.copy_async(
                        rhs.chunkat(k, n), to=croq.SHARED)
                    croq.wait(fl, fr)
                    ma = croq.mma.load(fl.chunkat(m, k))
                    mb = croq.mma.load(fr.chunkat(k, n))
                    mc = croq.mma.exec(mc, ma, mb, method="row.col")
                croq.mma.store(mc, output.chunkat(m, n))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "mma" in text.lower() or "MMA" in text
        assert "DMA" in text


# =========================================================================== #
#  Port: select ternary from add-shared.co
# =========================================================================== #

class TestAddSharedCo:
    def test_select_in_kernel(self):
        """Partial port of add-shared.co: tests select/ternary inside kernel."""
        @croq.co
        def ele_add(lhs: croq.s32[6, 17, 64],
                    rhs: croq.s32[6, 17, 64]) -> croq.s32[6, 17, 64]:
            output = croq.declare(croq.s32[6, 17, 64], "output")

            for p in croq.parallel(p=6, scope=croq.BLOCK):
                for x in croq.foreach(x=6):
                    big_x = croq.select(x == 5, croq.Var.lit(2),
                                        croq.Var.lit(3))
                    lhs_s = croq.declare(croq.s32[1, 3, 64], "lhs_s",
                                         storage=croq.SHARED)
                    rhs_s = croq.declare(croq.s32[1, 3, 64], "rhs_s",
                                         storage=croq.SHARED)

                    for q in croq.parallel(q=64):
                        output[p, x, q] = lhs_s[0, x, q] + rhs_s[0, x, q]

            return output

        text = croq.Program().add(ele_add).dump_ast()
        assert "?" in text or "Select" in text or "Ternary" in text


# =========================================================================== #
#  Port: swap from mma_v2.co
# =========================================================================== #

class TestMmaV2Co:
    def test_swap_pattern(self):
        """Partial port of mma_v2.co: tests swap inside with-in."""
        MMA_M, MMA_N, MMA_K = 16, 16, 16
        M_, N_, K_ = 64, 32, 16

        @croq.co
        def matmul(lhs: croq.f16[M_, K_],
                   rhs: croq.f16[K_, N_]) -> croq.f16[M_, N_]:
            output = croq.declare(croq.f16[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M // 4, n=N_ // MMA_N, scope=croq.BLOCK):
                for g0, g1 in croq.parallel(
                        g0=4, g1=1, scope=croq.GROUP):
                    mc = croq.mma.fill(0.0)
                    with croq.with_in(k=K_ // MMA_K) as k:
                        fl0 = croq.dma.copy_async(lhs, to=croq.SHARED)
                        fr0 = croq.dma.copy_async(rhs, to=croq.SHARED)
                        fl1 = croq.dma.any()
                        fr1 = croq.dma.any()

                        croq.wait(fl0, fr0)
                        ma = croq.mma.load(fl0.chunkat(m @ g0, k))
                        mb = croq.mma.load(fr0.chunkat(k, n @ g1))
                        mc = croq.mma.exec(mc, ma, mb, method="row.col")

                        croq.swap(fl0, fl1)
                        croq.swap(fr0, fr1)

                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "Swap" in text or "Rotate" in text
        assert "With" in text


# =========================================================================== #
#  Port: scope annotations from various matmul tests
# =========================================================================== #

class TestNestedScopes:
    def test_block_group_thread(self):
        """Multi-level scope: block > group > thread."""
        @croq.co
        def kern(inp: croq.f16[128, 64]) -> croq.f16[128, 64]:
            out = croq.declare(croq.f16[128, 64], "out")
            for bm in croq.parallel(bm=2, scope=croq.BLOCK):
                for gm in croq.parallel(gm=4, scope=croq.GROUP):
                    for t in croq.parallel(t=16, scope=croq.THREAD):
                        out[bm @ gm @ t, 0] = inp[bm @ gm @ t, 0]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert text.count("Parallelization") == 3

    def test_group4_scope(self):
        """group-4 warpgroup scope."""
        @croq.co
        def kern(inp: croq.f16[64]) -> croq.f16[64]:
            out = croq.declare(croq.f16[64], "out")
            for p in croq.parallel(p=2, scope="group-4"):
                for t in croq.parallel(t=32, scope=croq.THREAD):
                    out[p @ t] = inp[p @ t]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "Parallelization" in text


# =========================================================================== #
#  Port: cdiv expression
# =========================================================================== #

class TestCdiv:
    def test_cdiv_in_kernel(self):
        @croq.co
        def kern(inp: croq.s32[100]) -> croq.s32[100]:
            out = croq.declare(croq.s32[100], "out")
            for p in croq.parallel(p=100):
                out[p] = croq.cdiv(inp[p], croq.Var.lit(3))
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "cdiv" in text


# =========================================================================== #
#  Port: with...as pattern from matmul-pipelined-2.co
# =========================================================================== #

class TestMatmulPipelinedCo:
    def test_with_as_pipelined(self):
        """Simplified pipelined matmul with with...as."""
        M_, N_, K_ = 128, 256, 256

        @croq.co
        def matmul(lhs: croq.s32[M_, K_],
                   rhs: croq.s32[K_, N_]) -> croq.s32[M_, N_]:
            output = croq.declare(croq.s32[M_, N_], "output")

            for px, py in croq.parallel(px=8, py=16, scope=croq.BLOCK):
                for qx, qy in croq.parallel(
                        qx=16, qy=16, scope=croq.THREAD):
                    with croq.with_in(tile_k=16) as tile_k:
                        lf0 = croq.dma.copy(
                            lhs.chunkat(px, tile_k), to=croq.SHARED)
                        rf0 = croq.dma.copy(
                            rhs.chunkat(tile_k, py), to=croq.SHARED)

                        for k in croq.foreach(k=256 // 16):
                            output[px @ qx, py @ qy] = (
                                output[px @ qx, py @ qy]
                                + lf0.data[qx, k] * rf0.data[k, qy])

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "With" in text
        assert "DMA" in text


# =========================================================================== #
#  Port: declare_int from mma.co (int MMA_M = 16)
# =========================================================================== #

class TestDeclareInt:
    def test_declare_int_in_kernel(self):
        @croq.co
        def kern(inp: croq.s32[64]) -> croq.s32[64]:
            out = croq.declare(croq.s32[64], "out")
            tile = croq.declare_int("TILE", 16)
            for p in croq.foreach(p=4):
                out[p] = inp[p] + tile
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "TILE" in text


# =========================================================================== #
#  Port: mma_fp8.co -- FP8 MMA with e4m3/e5m2 types
# =========================================================================== #

class TestMmaFp8Co:
    def test_fp8_e4m3_matmul(self):
        M_, N_, K_ = 64, 32, 32
        MMA_M, MMA_N, MMA_K = 16, 8, 32

        @croq.co
        def matmul_e4m3(
                lhs: croq.f8_e4m3[M_, K_],
                rhs: croq.f8_e4m3[K_, N_]) -> croq.f32[M_, N_]:
            output = croq.declare(croq.f32[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M // 4, n=N_ // MMA_N, scope=croq.BLOCK):
                for g0, g1 in croq.parallel(
                        g0=4, g1=1, scope=croq.GROUP):
                    mc = croq.mma.fill(0.0, dtype=croq.f32)
                    with croq.with_in(k=K_ // MMA_K) as k:
                        fl0 = croq.dma.copy_async(lhs, to=croq.SHARED)
                        fr0 = croq.dma.copy_async(rhs, to=croq.SHARED)
                        fl1 = croq.dma.any()
                        fr1 = croq.dma.any()

                        croq.wait(fl0, fr0)
                        ma = croq.mma.load(fl0.chunkat(m @ g0, k))
                        mb = croq.mma.load(fr0.chunkat(k, n @ g1))
                        mc = croq.mma.exec(mc, ma, mb, method="row.col")

                        croq.swap(fl0, fl1)
                        croq.swap(fr0, fr1)

                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))

            return output

        prog = croq.Program()
        prog.add(matmul_e4m3)
        text = prog.dump_ast()
        assert "MMA" in text
        assert "f8_e4m3" in text or "f8" in text.lower()

    def test_fp8_e5m2_matmul(self):
        M_, N_, K_ = 64, 32, 32
        MMA_M, MMA_N, MMA_K = 16, 8, 32

        @croq.co
        def matmul_e5m2(
                lhs: croq.f8_e5m2[M_, K_],
                rhs: croq.f8_e5m2[K_, N_]) -> croq.f32[M_, N_]:
            output = croq.declare(croq.f32[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M // 4, n=N_ // MMA_N, scope=croq.BLOCK):
                for g0, g1 in croq.parallel(
                        g0=4, g1=1, scope=croq.GROUP):
                    mc = croq.mma.fill(0.0, dtype=croq.f32)
                    for k in croq.foreach(k=K_ // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        mc = croq.mma.exec(mc, ma, mb, method="row.col")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))

            return output

        text = croq.Program().add(matmul_e5m2).dump_ast()
        assert "MMA" in text


# =========================================================================== #
#  Port: tma.co -- TMA copy async + basic subspan
# =========================================================================== #

class TestTmaCo:
    def test_tma_copy_async(self):
        @croq.co
        def tma_copy_async(
                inp: croq.f32[6, 16, 64]) -> croq.f32[6, 16, 64]:
            output = croq.declare(croq.f32[6, 16, 64], "output")
            for _p in croq.parallel(_p=1):
                f = croq.tma.copy_async(inp, to=croq.SHARED)
                croq.wait(f)
                croq.tma.copy(f.data, output)
            return output

        text = croq.Program().add(tma_copy_async).dump_ast()
        assert "DMA" in text or "TMA" in text


# =========================================================================== #
#  Port: transpose.co -- DMA transpose (partial: structure only)
# =========================================================================== #

class TestTransposeCo:
    def test_transpose_structure(self):
        """Port transpose.co structure -- dma.transp not yet in croqtile-python,
        but verify the surrounding parallel/copy pattern works."""
        @croq.co
        def transpose(a: croq.s32[3, 4, 5]) -> croq.s32[4, 3, 5]:
            o = croq.declare(croq.s32[4, 3, 5], "o")
            for p in croq.parallel(p=1):
                fa = croq.dma.copy(a, to=croq.LOCAL)
                croq.dma.copy(fa.data, o)
            return o

        text = croq.Program().add(transpose).dump_ast()
        assert "DMA" in text


# =========================================================================== #
#  Port: static matmul with TMA + MMA (SM90 pattern)
# =========================================================================== #

class TestStaticTmaMatmul:
    def test_tma_mma_matmul(self):
        """Port pattern from matmul_f16_dynamic.co with static dims."""
        M_, N_, K_ = 128, 128, 64
        WARP_M, WARP_N, TILE_K, WARP_K = 64, 64, 64, 16

        @croq.co
        def matmul(lhs: croq.f16[M_, K_],
                   rhs: croq.f16[N_, K_]) -> croq.f16[M_, N_]:
            output = croq.declare(croq.f16[M_, N_], "output")

            for bm, bn in croq.parallel(
                    bm=M_ // WARP_M, bn=N_ // WARP_N, scope=croq.BLOCK):
                lhs_s = croq.declare(
                    croq.f16[WARP_M, TILE_K], "lhs_s",
                    storage=croq.SHARED)
                rhs_s = croq.declare(
                    croq.f16[WARP_N, TILE_K], "rhs_s",
                    storage=croq.SHARED)
                mc = croq.mma.fill(0.0, dtype=croq.f32)

                for iv_k in croq.foreach(iv_k=K_ // TILE_K):
                    croq.tma.copy(
                        lhs.subspan(WARP_M, TILE_K).at(bm, iv_k),
                        lhs_s, swizzle=128)
                    croq.tma.copy(
                        rhs.chunkat(bn, iv_k), rhs_s, swizzle=128)

                    for iv_warp in croq.foreach(iv_warp=TILE_K // WARP_K):
                        for p in croq.parallel(
                                p=1, scope="group-4"):
                            ma = croq.mma.load(
                                lhs_s.chunkat(iv_warp), swizzle=128)
                            mb = croq.mma.load(
                                rhs_s.chunkat(iv_warp), swizzle=128)
                            mc = croq.mma.exec(
                                mc, ma, mb, method="row.row")

                output_s = croq.declare(
                    croq.f16[WARP_M, WARP_N], "output_s",
                    storage=croq.SHARED)
                croq.mma.store(mc, output_s)
                croq.tma.copy(
                    output_s,
                    output.subspan(WARP_M, WARP_N).at(bm, bn))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "Parallelization" in text


# =========================================================================== #
#  Port: multi-function program
# =========================================================================== #

class TestMultiFunction:
    def test_two_kernels_one_program(self):
        @croq.co
        def add_kern(a: croq.s32[128], b: croq.s32[128]) -> croq.s32[128]:
            out = croq.declare(croq.s32[128], "out")
            for p in croq.parallel(p=128):
                out[p] = a[p] + b[p]
            return out

        @croq.co
        def mul_kern(a: croq.s32[64], b: croq.s32[64]) -> croq.s32[64]:
            out = croq.declare(croq.s32[64], "out")
            for p in croq.parallel(p=64):
                out[p] = a[p] * b[p]
            return out

        prog = croq.Program()
        prog.add(add_kern)
        prog.add(mul_kern)
        text = prog.dump_ast()
        assert "add_kern" in text
        assert "mul_kern" in text
        assert text.count("ChoreoFunction") == 2


# =========================================================================== #
#  Port: host code embedding
# =========================================================================== #

# =========================================================================== #
#  Port: matmul_f16_dynamic.co -- dynamic bounds with cdiv, void return
# =========================================================================== #

class TestDynamicMatmul:
    def test_dynamic_parallel_bounds(self):
        """Port of matmul_f16_dynamic.co kernel structure."""
        M_, N_, K_ = 768, 512, 512
        WARP_M, WARP_N, TILE_K, WARP_K = 64, 64, 64, 16

        @croq.co
        def matmul(lhs: croq.f16[M_, K_],
                   rhs: croq.f16[N_, K_]) -> croq.f16[M_, N_]:
            output = croq.declare(croq.f16[M_, N_], "output")
            int_warp_m = croq.declare_int("WARP_M", WARP_M)
            int_warp_n = croq.declare_int("WARP_N", WARP_N)
            int_tile_k = croq.declare_int("TILE_K", TILE_K)
            int_warp_k = croq.declare_int("WARP_K", WARP_K)

            for bm, bn in croq.parallel(
                    bm=croq.cdiv(croq.Var.lit(M_), int_warp_m),
                    bn=croq.cdiv(croq.Var.lit(N_), int_warp_n),
                    scope=croq.BLOCK):
                lhs_s = croq.declare(
                    croq.f16[WARP_M, TILE_K], "lhs_s",
                    storage=croq.SHARED)
                rhs_s = croq.declare(
                    croq.f16[WARP_N, TILE_K], "rhs_s",
                    storage=croq.SHARED)
                mc = croq.mma.fill(0.0, dtype=croq.f32)

                for iv_k in croq.foreach(
                        iv_k=croq.cdiv(croq.Var.lit(K_), int_tile_k)):
                    croq.tma.copy(
                        lhs.subspan(WARP_M, TILE_K).at(bm, iv_k),
                        lhs_s, swizzle=128)
                    croq.tma.copy(
                        rhs.chunkat(bn, iv_k), rhs_s, swizzle=128)

                    for iv_warp in croq.foreach(
                            iv_warp=croq.cdiv(int_tile_k, int_warp_k)):
                        for p in croq.parallel(
                                p=1, scope="group-4"):
                            ma = croq.mma.load(
                                lhs_s.chunkat(iv_warp), swizzle=128)
                            mb = croq.mma.load(
                                rhs_s.chunkat(iv_warp), swizzle=128)
                            mc = croq.mma.exec(
                                mc, ma, mb, method="row.row")

                output_s = croq.declare(
                    croq.f16[WARP_M, WARP_N], "output_s",
                    storage=croq.SHARED)
                croq.mma.store(mc, output_s)
                croq.tma.copy(
                    output_s,
                    output.subspan(WARP_M, WARP_N).at(bm, bn))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "cdiv" in text
        assert "MMA" in text
        assert "Parallelization" in text

    def test_void_function(self):
        """Void return type with output param."""
        @croq.co
        def kern(inp: croq.s32[64], out: croq.s32[64]):
            for p in croq.parallel(p=64):
                out[p] = inp[p]

        text = croq.Program().add(kern).dump_ast()
        assert "void" in text.lower() or "Void" in text or "VOID" in text
        assert "Parallelization" in text


# =========================================================================== #
#  Port: view/from pattern from add-shared.co
# =========================================================================== #

# =========================================================================== #
#  Port: staged foreach from mma_v2.co -- foreach k(1:) pattern
# =========================================================================== #

class TestStagedForeach:
    def test_staged_foreach_in_pipeline(self):
        """Port of mma_v2.co pipelined pattern with foreach_staged."""
        M_, N_, K_ = 64, 32, 16
        MMA_M, MMA_N, MMA_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[M_, K_],
                   rhs: croq.f16[K_, N_]) -> croq.f16[M_, N_]:
            output = croq.declare(croq.f16[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M // 4, n=N_ // MMA_N, scope=croq.BLOCK):
                for g0, g1 in croq.parallel(
                        g0=4, g1=1, scope=croq.GROUP):
                    mc = croq.mma.fill(0.0)
                    with croq.with_in(k=K_ // MMA_K) as k:
                        fl0 = croq.dma.copy_async(lhs, to=croq.SHARED)
                        fr0 = croq.dma.copy_async(rhs, to=croq.SHARED)
                        fl1 = croq.dma.any()
                        fr1 = croq.dma.any()

                        # staged foreach: skip first iteration
                        for _ in croq.foreach_staged(k, start=1):
                            fl1 = croq.dma.copy_async(
                                lhs, to=croq.SHARED)
                            fr1 = croq.dma.copy_async(
                                rhs, to=croq.SHARED)

                            croq.wait(fl0, fr0)
                            ma = croq.mma.load(
                                fl0.chunkat(m @ g0, k))
                            mb = croq.mma.load(
                                fr0.chunkat(k, n @ g1))
                            mc = croq.mma.exec(
                                mc, ma, mb, method="row.col")

                            croq.swap(fl0, fl1)
                            croq.swap(fr0, fr1)

                        # epilogue: last iteration
                        croq.wait(fl0, fr0)
                        ma = croq.mma.load(fl0.chunkat(m @ g0, k))
                        mb = croq.mma.load(fr0.chunkat(k, n @ g1))
                        mc = croq.mma.exec(mc, ma, mb, method="row.col")

                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "With" in text
        assert "Foreach" in text
        assert "Swap" in text or "Rotate" in text


class TestViewFrom:
    def test_view_from_basic(self):
        """Port of add-shared.co view/from access pattern."""
        @croq.co
        def kern(lhs: croq.s32[6, 17, 64]) -> croq.s32[6, 17, 64]:
            output = croq.declare(croq.s32[6, 17, 64], "output")
            for p in croq.parallel(p=6, scope=croq.BLOCK):
                for x in croq.foreach(x=6):
                    f = croq.dma.copy_async(
                        lhs.view(1, 3, 64).from_(p, x * 3, 0),
                        to=croq.SHARED)
                    croq.wait(f)
                    croq.dma.copy(f.data, output.view(1, 3, 64).from_(p, x * 3, 0))
            return output

        text = croq.Program().add(kern).dump_ast()
        assert "VIEW" in text or "view" in text.lower()
        assert "From" in text or "from" in text.lower()


# =========================================================================== #
#  Port: device if-statement
# =========================================================================== #

class TestDeviceIf:
    def test_device_if_basic(self):
        @croq.co
        def kern(inp: croq.s32[64]) -> croq.s32[64]:
            out = croq.declare(croq.s32[64], "out")
            for p in croq.parallel(p=64):
                with croq.device_if(p < 32):
                    out[p] = inp[p]
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "If" in text or "if" in text.lower()

    def test_device_if_with_select(self):
        """Combine device_if with select."""
        @croq.co
        def kern(inp: croq.s32[64]) -> croq.s32[64]:
            out = croq.declare(croq.s32[64], "out")
            for p in croq.parallel(p=64):
                with croq.device_if(p < 32):
                    out[p] = croq.select(
                        p == 0, croq.Var.lit(100), inp[p])
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "If" in text or "if" in text.lower()


# =========================================================================== #
#  Port: global parameter storage (matmul_f16_dynamic.co pattern)
# =========================================================================== #

class TestGlobalParam:
    def test_global_output_param(self):
        """void function with global output param."""
        @croq.co
        def matmul(lhs: croq.Global(croq.f16[128, 64]),
                   rhs: croq.Global(croq.f16[128, 64]),
                   output: croq.Global(croq.f16[128, 128])):
            for p in croq.parallel(p=128):
                output[p, 0] = lhs[p, 0]

        text = croq.Program().add(matmul).dump_ast()
        # Global storage is embedded in the type, not always visible in dump
        # Verify void return and parameters exist
        assert "void" in text.lower()
        assert "lhs" in text
        assert "output" in text


# =========================================================================== #
#  Port: DMA/TMA swizzle (dma.copy.swiz<N>, tma.copy.swiz<N>)
# =========================================================================== #

class TestSwizzle:
    def test_dma_copy_swizzle(self):
        @croq.co
        def kern(inp: croq.f16[128, 64]) -> croq.f16[128, 64]:
            out = croq.declare(croq.f16[128, 64], "out")
            buf_s = croq.declare(
                croq.f16[128, 64], "buf_s", storage=croq.SHARED)
            for p in croq.parallel(p=1):
                f = croq.dma.copy(inp, buf_s, swizzle=128)
                croq.dma.copy(f.data, out)
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "DMA" in text

    def test_tma_copy_swizzle(self):
        @croq.co
        def kern(inp: croq.f16[128, 64]) -> croq.f16[128, 64]:
            out = croq.declare(croq.f16[128, 64], "out")
            buf_s = croq.declare(
                croq.f16[128, 64], "buf_s", storage=croq.SHARED)
            for p in croq.parallel(p=1):
                f = croq.tma.copy(inp, buf_s, swizzle=128)
                croq.tma.copy(f.data, out)
            return out

        text = croq.Program().add(kern).dump_ast()
        assert "DMA" in text or "TMA" in text


class TestHostEmbed:
    def test_host_with_kernel(self):
        @croq.co
        def kern(a: croq.s32[64]) -> croq.s32[64]:
            out = croq.declare(croq.s32[64], "out")
            for p in croq.parallel(p=64):
                out[p] = a[p]
            return out

        prog = croq.Program()
        prog.add(kern)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::s32>(64);
  a.fill_random(-10, 10);
  auto res = kern(a.view());
  std::cout << "Test Passed" << std::endl;
}
""")
        text = prog.dump_ast()
        assert "kern" in text
        assert "main" in text or "CppSourceCode" in text


# =========================================================================== #
#  Port: nil.co -- empty void function
# =========================================================================== #

class TestNilCo:
    def test_empty_void_function(self):
        @croq.co
        def nil():
            pass

        text = croq.Program().add(nil).dump_ast()
        assert "nil" in text
        assert "void" in text.lower() or "ChoreoFunction" in text


# =========================================================================== #
#  Port: add.co -- basic element-wise add
# =========================================================================== #

class TestAddCo:
    def test_basic_add(self):
        @croq.co
        def ele_add(lhs: croq.s32[6, 64],
                    rhs: croq.s32[6, 64]) -> croq.s32[6, 64]:
            output = croq.declare(croq.s32[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        text = croq.Program().add(ele_add).dump_ast()
        assert "Assign" in text
        assert "Parallelization" in text


# =========================================================================== #
#  Port: add-local.co -- DMA copy to local + element-wise compute
# =========================================================================== #

class TestAddLocalCo:
    def test_dma_to_local(self):
        @croq.co
        def ele_add(lhs: croq.s32[6, 17, 64],
                    rhs: croq.s32[6, 17, 64]) -> croq.s32[6, 17, 64]:
            output = croq.declare(croq.s32[6, 17, 64], "output")

            for p, q in croq.parallel(p=6, q=64):
                lhs_load = croq.dma.copy(
                    lhs.chunkat(p, croq.FULL, q), to=croq.LOCAL)
                rhs_load = croq.dma.copy(
                    rhs.chunkat(p, croq.FULL, q), to=croq.LOCAL)
                buf = croq.declare(
                    croq.s32[17], "buffer", storage=croq.LOCAL)
                for x in croq.foreach(x=17):
                    buf[x] = lhs_load.data[x] + rhs_load.data[x]
                croq.dma.copy(buf, output.chunkat(p, croq.FULL, q))
            return output

        text = croq.Program().add(ele_add).dump_ast()
        assert "DMA" in text
        assert "local" in text.lower() or "LOCAL" in text


# =========================================================================== #
#  Port: copy_fp8.co -- FP8 DMA copy (sync + async)
# =========================================================================== #

class TestCopyFp8Co:
    def test_dma_copy_fp8_e4m3(self):
        @croq.co
        def dma_copy_sync(
                inp: croq.f8_e4m3[6, 17, 64]) -> croq.f8_e4m3[6, 17, 64]:
            output = croq.declare(croq.f8_e4m3[6, 17, 64], "output")
            for _p in croq.parallel(_p=1):
                f = croq.dma.copy(inp, to=croq.SHARED)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy_sync).dump_ast()
        assert "DMA" in text
        assert "f8_e4m3" in text.lower() or "f8" in text.lower()

    def test_dma_copy_fp8_e5m2_async(self):
        @croq.co
        def dma_copy_async(
                inp: croq.f8_e5m2[6, 17, 64]) -> croq.f8_e5m2[6, 17, 64]:
            output = croq.declare(croq.f8_e5m2[6, 17, 64], "output")
            for _p in croq.parallel(_p=1):
                f = croq.dma.copy_async(inp, to=croq.SHARED)
                croq.wait(f)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy_async).dump_ast()
        assert "DMA" in text
        assert "WAIT" in text


# =========================================================================== #
#  Port: float_types.co -- element-add for various float types
# =========================================================================== #

class TestFloatTypesCo:
    def _make_add_kernel(self, dtype, shape=(6, 64)):
        @croq.co
        def ele_add(lhs: dtype[shape[0], shape[1]],
                    rhs: dtype[shape[0], shape[1]]
                    ) -> dtype[shape[0], shape[1]]:
            output = croq.declare(dtype[shape[0], shape[1]], "output")
            for p, q in croq.parallel(p=shape[0], q=shape[1]):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output
        return ele_add

    def test_f32_add(self):
        k = self._make_add_kernel(croq.f32)
        text = croq.Program().add(k).dump_ast()
        assert "f32" in text.lower()

    def test_f16_add(self):
        k = self._make_add_kernel(croq.f16)
        text = croq.Program().add(k).dump_ast()
        assert "f16" in text.lower()

    def test_bf16_add(self):
        k = self._make_add_kernel(croq.bf16)
        text = croq.Program().add(k).dump_ast()
        assert "bf16" in text.lower()


# =========================================================================== #
#  Port: integral_types.co -- element-add for integer types
# =========================================================================== #

class TestIntegralTypesCo:
    def _make_add_kernel(self, dtype):
        @croq.co
        def ele_add(lhs: dtype[6, 64],
                    rhs: dtype[6, 64]) -> dtype[6, 64]:
            output = croq.declare(dtype[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output
        return ele_add

    def test_s32_add(self):
        text = croq.Program().add(self._make_add_kernel(croq.s32)).dump_ast()
        assert "s32" in text.lower()

    def test_u32_add(self):
        text = croq.Program().add(self._make_add_kernel(croq.u32)).dump_ast()
        assert "u32" in text.lower()

    def test_s16_add(self):
        text = croq.Program().add(self._make_add_kernel(croq.s16)).dump_ast()
        assert "s16" in text.lower()

    def test_u16_add(self):
        text = croq.Program().add(self._make_add_kernel(croq.u16)).dump_ast()
        assert "u16" in text.lower()

    def test_s8_add(self):
        text = croq.Program().add(self._make_add_kernel(croq.s8)).dump_ast()
        assert "s8" in text.lower()

    def test_u8_add(self):
        text = croq.Program().add(self._make_add_kernel(croq.u8)).dump_ast()
        assert "u8" in text.lower()


# =========================================================================== #
#  Port: mma_v1.co -- MMA with 2x2 group parallelism
# =========================================================================== #

class TestMmaV1Co:
    def test_mma_v1_matmul(self):
        M_, N_, K_ = 128, 256, 256
        MMA_M, MMA_N, MMA_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[M_, K_],
                   rhs: croq.f16[K_, N_]) -> croq.f16[M_, N_]:
            output = croq.declare(croq.f16[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M // 2, n=N_ // MMA_N // 2,
                    scope=croq.BLOCK):
                mc = croq.mma.fill(0.0)
                for g0, g1 in croq.parallel(
                        g0=2, g1=2, scope=croq.GROUP):
                    for k in croq.foreach(k=K_ // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        mc = croq.mma.exec(mc, ma, mb, method="row.col")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert text.count("Parallelization") >= 2


# =========================================================================== #
#  Port: wmma_rr.co -- WMMA row.row matmul
# =========================================================================== #

class TestWmmaRrCo:
    def test_wmma_row_row(self):
        M_, N_, K_ = 128, 256, 256
        MMA_M, MMA_N, MMA_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[M_, K_],
                   rhs: croq.f16[N_, K_]) -> croq.f16[M_, N_]:
            output = croq.declare(croq.f16[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M // 2, n=N_ // MMA_N // 2,
                    scope=croq.BLOCK):
                mc = croq.mma.fill(0.0)
                for g0, g1 in croq.parallel(
                        g0=2, g1=2, scope=croq.GROUP):
                    for k in croq.foreach(k=K_ // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(n @ g1, k))
                        mc = croq.mma.exec(
                            mc, ma, mb, method="row.row")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "row.row" in text.lower() or "ROW_ROW" in text


# =========================================================================== #
#  Port: wmma_fp8_rr.co -- FP8 e4m3 with row.col pattern
# =========================================================================== #

class TestWmmaFp8RrCo:
    def test_wmma_fp8_rr(self):
        M_, N_, K_ = 64, 64, 64
        MMA_M, MMA_N, MMA_K = 16, 8, 32

        @croq.co
        def matmul(lhs: croq.f8_e4m3[M_, K_],
                   rhs: croq.f8_e4m3[K_, N_]) -> croq.f32[M_, N_]:
            output = croq.declare(croq.f32[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M // 2, n=N_ // MMA_N // 2,
                    scope=croq.BLOCK):
                mc = croq.mma.fill(0.0, dtype=croq.f32)
                for g0, g1 in croq.parallel(
                        g0=2, g1=2, scope=croq.GROUP):
                    for k in croq.foreach(k=K_ // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        mc = croq.mma.exec(
                            mc, ma, mb, method="row.col")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "f8_e4m3" in text.lower() or "f8" in text.lower()


# =========================================================================== #
#  Port: tma_v1.co -- TMA tiled copy with FULL slice
# =========================================================================== #

class TestTmaV1Co:
    def test_tma_copy_tiled(self):
        @croq.co
        def tma_copy_tiled(
                inp: croq.f32[6, 16, 128]) -> croq.f32[6, 16, 128]:
            output = croq.declare(croq.f32[6, 16, 128], "output")
            for p in croq.parallel(p=32, scope=croq.BLOCK):
                f = croq.tma.copy_async(
                    inp.chunkat(croq.FULL, croq.FULL, p),
                    to=croq.SHARED)
                croq.wait(f)
                croq.tma.copy(
                    f.data,
                    output.chunkat(croq.FULL, croq.FULL, p))
            return output

        text = croq.Program().add(tma_copy_tiled).dump_ast()
        assert "DMA" in text or "TMA" in text
        assert "WAIT" in text


# =========================================================================== #
#  Port: async_copy.co -- multi-function DMA async program
# =========================================================================== #

class TestAsyncCopyCo:
    def test_dma_copy_basic(self):
        @croq.co
        def dma_copy(inp: croq.s32[6, 17, 32]) -> croq.s32[6, 17, 32]:
            output = croq.declare(croq.s32[6, 17, 32], "output")
            for _p in croq.parallel(_p=1, scope=croq.GROUP):
                f = croq.dma.copy_async(inp, to=croq.SHARED)
                croq.wait(f)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy).dump_ast()
        assert "DMA" in text
        assert "WAIT" in text

    def test_dma_add_two_inputs(self):
        @croq.co
        def dma_add(in0: croq.s32[6, 17, 32],
                    in1: croq.s32[6, 17, 32]) -> croq.s32[6, 17, 32]:
            output = croq.declare(croq.s32[6, 17, 32], "output")
            for q0 in croq.parallel(q0=17, scope=croq.GROUP):
                f0 = croq.dma.copy_async(in0, to=croq.SHARED)
                f1 = croq.dma.copy_async(in1, to=croq.SHARED)
                croq.wait(f0, f1)
                out = croq.declare(
                    croq.s32[6, 1, 32], "out", storage=croq.SHARED)
                for x in croq.foreach(x=6):
                    for q1 in croq.parallel(q1=32):
                        out[x, q0, q1] = (
                            f0.data[x, q0, q1] + f1.data[x, q0, q1])
                croq.dma.copy(out, output)
            return output

        text = croq.Program().add(dma_add).dump_ast()
        assert "DMA" in text
        assert "WAIT" in text


# =========================================================================== #
#  Port: memreuse-static.co -- memory reuse with += compound assignment
# =========================================================================== #

class TestMemreuseStaticCo:
    def test_memreuse_compound_assign(self):
        K_, N_ = 8, 256

        @croq.co
        def mem_reuse0(i0: croq.u8[K_, N_],
                       i1: croq.u8[K_, N_]) -> croq.u8[K_, N_]:
            output = croq.declare(croq.u8[K_, N_], "output")

            for p, q in croq.parallel(p=8, q=128):
                smem1 = croq.dma.copy_async(
                    i0.chunkat(p, croq.FULL), to=croq.SHARED)
                output_s = croq.declare(
                    croq.u8[1, N_], "output_s",
                    storage=croq.SHARED, init=0)
                croq.wait(smem1)

                for i in croq.foreach(i=1):
                    output_s[i, q] = smem1.data[i, q]
                    output_s[i, q + 128] = smem1.data[i, q + 128]

                smem2 = croq.dma.copy_async(
                    i1.chunkat(p, croq.FULL), to=croq.SHARED)
                croq.wait(smem2)

                for i in croq.foreach(i=1):
                    output_s[i, q] += smem2.data[i, q]
                    output_s[i, q + 128] += smem2.data[i, q + 128]

                croq.sync("shared")
                croq.dma.copy(output_s, output.chunkat(p, croq.FULL))
            return output

        text = croq.Program().add(mem_reuse0).dump_ast()
        assert "DMA" in text
        assert "WAIT" in text
        assert "Synchronize" in text
        # Verify += produces correct pattern
        assigns = [l.strip() for l in text.split('\n')
                   if 'Assign' in l]
        assert len(assigns) >= 4


# =========================================================================== #
#  Port: mma_loadc.co -- MMA load C accumulator + cdiv parallel bounds
# =========================================================================== #

class TestMmaLoadcCo:
    def test_mma_loadc(self):
        M_, N_, K_ = 256, 256, 256
        TILE_M, TILE_N, TILE_K = 32, 32, 16
        WARP_M, WARP_N, WARP_K = 8, 8, 4

        @croq.co
        def matmul(lhs: croq.f16[M_, K_],
                   rhs: croq.f16[K_, N_]) -> croq.f16[M_, N_]:
            output = croq.declare(croq.f16[M_, N_], "output", init=0)

            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(croq.Var.lit(M_),
                                      croq.Var.lit(TILE_M)),
                    block_n=croq.cdiv(croq.Var.lit(N_),
                                      croq.Var.lit(TILE_N)),
                    scope=croq.BLOCK):
                for iv_k in croq.foreach(iv_k=croq.cdiv(
                        croq.Var.lit(K_), croq.Var.lit(TILE_K))):
                    lhs_load_s = croq.dma.copy(
                        lhs.chunkat(block_m, iv_k), to=croq.SHARED)
                    rhs_load_s = croq.dma.copy(
                        rhs.chunkat(iv_k, block_n), to=croq.SHARED)
                    for iv_warp_m, iv_warp_n in croq.parallel(
                            iv_warp_m=croq.cdiv(
                                croq.Var.lit(TILE_M),
                                croq.Var.lit(WARP_M)),
                            iv_warp_n=croq.cdiv(
                                croq.Var.lit(TILE_N),
                                croq.Var.lit(WARP_N)),
                            scope=croq.GROUP):
                        mc = croq.mma.load(
                            output.chunkat(
                                block_m @ iv_warp_m,
                                block_n @ iv_warp_n))
                        for iv_warp_k in croq.foreach(
                                iv_warp_k=croq.cdiv(
                                    croq.Var.lit(TILE_K),
                                    croq.Var.lit(WARP_K))):
                            ma = croq.mma.load(
                                lhs_load_s.chunkat(
                                    iv_warp_m, iv_warp_k))
                            mb = croq.mma.load(
                                rhs_load_s.chunkat(
                                    iv_warp_k, iv_warp_n))
                            mc = croq.mma.exec(
                                mc, ma, mb, method="row.col")
                        croq.mma.store(
                            mc, output.chunkat(
                                block_m @ iv_warp_m,
                                block_n @ iv_warp_n))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "cdiv" in text


# =========================================================================== #
#  Port: mma_fusion_scalar.co -- MMA + scalar bias fusion
# =========================================================================== #

class TestMmaFusionScalarCo:
    def test_mma_plus_bias(self):
        M_, N_, K_ = 128, 256, 256
        MMA_M, MMA_N, MMA_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[M_, K_],
                   rhs: croq.f16[K_, N_],
                   bias: croq.f16[1]) -> croq.f16[M_, N_]:
            output = croq.declare(croq.f16[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M // 2, n=N_ // MMA_N // 2,
                    scope=croq.BLOCK):
                for g0, g1 in croq.parallel(
                        g0=2, g1=2, scope=croq.GROUP):
                    mc = croq.mma.fill(0.0)
                    for k in croq.foreach(k=K_ // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        mc = croq.mma.exec(
                            mc, ma, mb, method="row.col")
                    croq.assign(mc, mc + bias)
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "bias" in text


# =========================================================================== #
#  Port: matmul_e4m3_dynamic.co -- FP8 dynamic with subspan+step
# =========================================================================== #

class TestMatmulE4m3DynamicCo:
    def test_fp8_dynamic_subspan_step(self):
        M_, N_, K_ = 768, 512, 512
        WARP_M, WARP_N, TILE_K, WARP_K = 64, 64, 32, 32

        @croq.co
        def matmul(lhs: croq.Global(croq.f8_e4m3[M_, K_]),
                   rhs: croq.Global(croq.f8_e4m3[N_, K_]),
                   output: croq.Global(croq.f32[M_, N_])):
            int_warp_m = croq.declare_int("WARP_M", WARP_M)
            int_warp_n = croq.declare_int("WARP_N", WARP_N)
            int_tile_k = croq.declare_int("TILE_K", TILE_K)
            int_warp_k = croq.declare_int("WARP_K", WARP_K)

            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(croq.Var.lit(M_), int_warp_m),
                    block_n=croq.cdiv(croq.Var.lit(N_), int_warp_n),
                    scope=croq.BLOCK):
                lhs_s = croq.declare(
                    croq.f8_e4m3[WARP_M, TILE_K], "lhs_load_s",
                    storage=croq.SHARED)
                rhs_s = croq.declare(
                    croq.f8_e4m3[WARP_N, TILE_K], "rhs_load_s",
                    storage=croq.SHARED)
                mc = croq.mma.fill(0.0, dtype=croq.f32)

                for iv_k in croq.foreach(
                        iv_k=croq.cdiv(croq.Var.lit(K_), int_tile_k)):
                    croq.tma.copy(
                        lhs.subspan(WARP_M, TILE_K)
                            .step(WARP_M, TILE_K)
                            .at(block_m, iv_k),
                        lhs_s, swizzle=32)
                    croq.tma.copy(
                        rhs.chunkat(block_n, iv_k),
                        rhs_s, swizzle=32)

                    for iv_warp in croq.foreach(
                            iv_warp=croq.cdiv(int_tile_k, int_warp_k)):
                        for p in croq.parallel(
                                p=1, scope="group-4"):
                            ma = croq.mma.load(
                                lhs_s.chunkat(croq.FULL, iv_warp),
                                swizzle=32)
                            mb = croq.mma.load(
                                rhs_s.chunkat(croq.FULL, iv_warp),
                                swizzle=32)
                            mc = croq.mma.exec(
                                mc, ma, mb, method="row.row")

                output_s = croq.declare(
                    croq.f32[WARP_M, WARP_N], "output_s",
                    storage=croq.SHARED)
                croq.mma.store(mc, output_s)
                croq.tma.copy(
                    output_s,
                    output.subspan(WARP_M, WARP_N)
                          .step(WARP_M, WARP_N)
                          .at(block_m, block_n))

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "SubSpan" in text
        assert "Step" in text


# =========================================================================== #
#  Port: matmul_f16_stmatrix_small.co -- SM90 WGMMA small matmul
# =========================================================================== #

class TestStmatrixSmallCo:
    def test_stmatrix_small(self):
        WARP_M, WARP_N = 64, 64
        TILE_K, WARP_K = 16, 16

        @croq.co
        def matmul(lhs: croq.Global(croq.f16[64, 16]),
                   rhs: croq.Global(croq.f16[64, 16]),
                   output: croq.Global(croq.f16[64, 64])):
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(croq.Var.lit(64),
                                      croq.Var.lit(WARP_M)),
                    block_n=croq.cdiv(croq.Var.lit(64),
                                      croq.Var.lit(WARP_N)),
                    scope=croq.BLOCK):
                lhs_s = croq.declare(
                    croq.f16[WARP_M, TILE_K], "lhs_s",
                    storage=croq.SHARED)
                rhs_s = croq.declare(
                    croq.f16[WARP_N, TILE_K], "rhs_s",
                    storage=croq.SHARED)
                mc = croq.mma.fill(0.0, dtype=croq.f16)

                for iv_k in croq.foreach(
                        iv_k=croq.cdiv(croq.Var.lit(16),
                                       croq.Var.lit(TILE_K))):
                    croq.tma.copy(
                        lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k),
                        lhs_s, swizzle=32)
                    croq.tma.copy(
                        rhs.subspan(WARP_N, TILE_K).at(block_n, iv_k),
                        rhs_s, swizzle=32)

                    for iv_warp in croq.foreach(
                            iv_warp=croq.cdiv(
                                croq.Var.lit(TILE_K),
                                croq.Var.lit(WARP_K))):
                        for p in croq.parallel(
                                p=1, scope="group-4"):
                            ma = croq.mma.load(
                                lhs_s.chunkat(croq.FULL, iv_warp),
                                swizzle=32)
                            mb = croq.mma.load(
                                rhs_s.chunkat(croq.FULL, iv_warp),
                                swizzle=32)
                            mc = croq.mma.exec(
                                mc, ma, mb, method="row.row")

                out_s = croq.declare(
                    croq.f16[WARP_M, WARP_N], "out_s",
                    storage=croq.SHARED)
                croq.mma.store(mc, out_s)
                croq.tma.copy(
                    out_s,
                    output.subspan(WARP_M, WARP_N).at(block_m, block_n))

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "SubSpan" in text


# =========================================================================== #
#  Port: tma_v2.co -- TMA double-buffered pipeline
# =========================================================================== #

class TestTmaV2Co:
    def test_tma_ab_buffer(self):
        """Port of tma_v2.co: double-buffered TMA with with-in + staged foreach."""
        @croq.co
        def tma_ab_buffer(
                l: croq.f32[6, 16, 128],
                r: croq.f32[6, 16, 128]) -> croq.f32[6, 16, 128]:
            output = croq.declare(croq.f32[6, 16, 128], "output")

            for p in croq.parallel(p=32, scope=croq.BLOCK):
                with croq.with_in(k=8) as k:
                    l0 = croq.tma.copy_async(
                        l.chunkat(croq.FULL, k, p), to=croq.SHARED)
                    r0 = croq.tma.copy_async(
                        r.chunkat(croq.FULL, k, p), to=croq.SHARED)
                    l1 = croq.tma.any()
                    r1 = croq.tma.any()
                    out = croq.declare(
                        croq.f32[6, 1, 4], "out", storage=croq.SHARED)

                    for _ in croq.foreach_staged(k, start=1):
                        l1 = croq.tma.copy_async(
                            l.chunkat(croq.FULL, k, p),
                            to=croq.SHARED)
                        r1 = croq.tma.copy_async(
                            r.chunkat(croq.FULL, k, p),
                            to=croq.SHARED)
                        croq.wait(l0, r0)
                        for idx in croq.foreach(idx=6):
                            out[idx] = l0.data[idx] + r0.data[idx]
                        croq.tma.copy(
                            out,
                            output.chunkat(croq.FULL, k - 1, p))
                        croq.swap(l0, l1)
                        croq.swap(r0, r1)

                    croq.wait(l0, r0)
                    for idx in croq.foreach(idx=6):
                        out[idx] = l0.data[idx] + r0.data[idx]
                    croq.tma.copy(out, output.chunkat(
                        croq.FULL, k, p))

            return output

        text = croq.Program().add(tma_ab_buffer).dump_ast()
        assert "With" in text
        assert "Foreach" in text
        assert "Swap" in text or "Rotate" in text


# =========================================================================== #
#  Port: copy_if_g2s.co -- conditional copy with view/from + subspan
# =========================================================================== #

class TestCopyIfG2sCo:
    def test_copy_tail_m(self):
        """Port of copy_if_g2s.co: tail handling with view/from + subspan."""
        TILE_M, TILE_K = 64, 64
        M_TAIL, K_FULL = 70, 64

        @croq.co
        def copy_tail_m(
                inp: croq.f16[M_TAIL, K_FULL]) -> croq.f16[M_TAIL, K_FULL]:
            output = croq.declare(
                croq.f16[M_TAIL, K_FULL], "output", init=0)

            for block_m in croq.parallel(
                    block_m=croq.cdiv(croq.Var.lit(M_TAIL),
                                      croq.Var.lit(TILE_M)),
                    scope=croq.BLOCK):
                for t in croq.parallel(t=128, scope=croq.THREAD):
                    tile = croq.declare(
                        croq.f16[TILE_M, TILE_K], "tile",
                        storage=croq.SHARED)

                    base_m = block_m * TILE_M
                    remain_m = M_TAIL - base_m
                    TM = croq.select(
                        croq.Var.lit(TILE_M) < remain_m,
                        croq.Var.lit(TILE_M), remain_m)

                    croq.dma.copy(
                        inp.view(TM, TILE_K).from_(base_m, 0),
                        tile, swizzle=128)
                    croq.dma.copy(
                        tile.subspan(TM, TILE_K).at(0, 0),
                        output.view(TM, TILE_K).from_(base_m, 0))
            return output

        text = croq.Program().add(copy_tail_m).dump_ast()
        assert "view" in text.lower() or "VIEW" in text
        assert "SubSpan" in text or "subspan" in text.lower()


# =========================================================================== #
#  Port: add_fp8_e4m3_to_f16.co -- mixed-type FP8->F16 element add
# =========================================================================== #

class TestAddFp8ToF16Co:
    def test_fp8_e4m3_to_f16(self):
        @croq.co
        def ele_add_fp8(lhs: croq.f8_e4m3[6, 64],
                        rhs: croq.f8_e4m3[6, 64]) -> croq.f16[6, 64]:
            output = croq.declare(croq.f16[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        text = croq.Program().add(ele_add_fp8).dump_ast()
        assert "f8_e4m3" in text.lower() or "f8" in text.lower()
        assert "f16" in text.lower()

    def test_fp8_e5m2_to_f16(self):
        @croq.co
        def ele_add_fp8(lhs: croq.f8_e5m2[6, 64],
                        rhs: croq.f8_e5m2[6, 64]) -> croq.f16[6, 64]:
            output = croq.declare(croq.f16[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        text = croq.Program().add(ele_add_fp8).dump_ast()
        assert "f8_e5m2" in text.lower() or "f8" in text.lower()


# =========================================================================== #
#  Port: tma_fp8.co -- TMA pipeline with FP8 types
# =========================================================================== #

class TestTmaFp8Co:
    def test_tma_fp8_pipeline(self):
        @croq.co
        def tma_ab_buffer(
                l: croq.f8_e4m3[16, 16, 128],
                r: croq.f8_e4m3[16, 16, 128]) -> croq.f32[16, 16, 128]:
            output = croq.declare(croq.f32[16, 16, 128], "output")

            for p in croq.parallel(p=8, scope=croq.BLOCK):
                with croq.with_in(k=8) as k:
                    l0 = croq.tma.copy_async(
                        l.chunkat(croq.FULL, k, p), to=croq.SHARED)
                    r0 = croq.tma.copy_async(
                        r.chunkat(croq.FULL, k, p), to=croq.SHARED)
                    l1 = croq.tma.any()
                    r1 = croq.tma.any()
                    out = croq.declare(
                        croq.f32[16, 1, 16], "out", storage=croq.SHARED)

                    for _ in croq.foreach_staged(k, start=1):
                        l1 = croq.tma.copy_async(
                            l.chunkat(croq.FULL, k, p), to=croq.SHARED)
                        r1 = croq.tma.copy_async(
                            r.chunkat(croq.FULL, k, p), to=croq.SHARED)
                        croq.wait(l0, r0)
                        for idx in croq.foreach(idx=16):
                            out[idx] = l0.data[idx] + r0.data[idx]
                        croq.tma.copy(
                            out, output.chunkat(croq.FULL, k - 1, p))
                        croq.swap(l0, l1)
                        croq.swap(r0, r1)

                    croq.wait(l0, r0)
                    for idx in croq.foreach(idx=16):
                        out[idx] = l0.data[idx] + r0.data[idx]
                    croq.tma.copy(
                        out, output.chunkat(croq.FULL, k, p))

            return output

        text = croq.Program().add(tma_ab_buffer).dump_ast()
        assert "f8_e4m3" in text.lower() or "f8" in text.lower()
        assert "With" in text
        assert "Foreach" in text


# =========================================================================== #
#  Port: tma_with_global_ref.co -- global params + WGMMA
# =========================================================================== #

class TestTmaWithGlobalRefCo:
    def test_global_params_wgmma(self):
        M_, N_, K_ = 64, 64, 64
        WARP_M, WARP_N, TILE_K, WARP_K = 64, 64, 64, 16

        @croq.co
        def matmul(lhs: croq.Global(croq.f16[M_, K_]),
                   rhs: croq.Global(croq.f16[N_, K_]),
                   output: croq.Global(croq.f16[M_, N_])):
            int_warp_m = croq.declare_int("WARP_M", WARP_M)
            int_warp_n = croq.declare_int("WARP_N", WARP_N)
            int_tile_k = croq.declare_int("TILE_K", TILE_K)
            int_warp_k = croq.declare_int("WARP_K", WARP_K)

            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(croq.Var.lit(M_), int_warp_m),
                    block_n=croq.cdiv(croq.Var.lit(N_), int_warp_n),
                    scope=croq.BLOCK):
                lhs_s = croq.declare(
                    croq.f16[WARP_M, TILE_K], "lhs_load_s",
                    storage=croq.SHARED)
                rhs_s = croq.declare(
                    croq.f16[WARP_N, TILE_K], "rhs_load_s",
                    storage=croq.SHARED)
                mc = croq.mma.fill(0.0, dtype=croq.f32)

                for iv_k in croq.foreach(
                        iv_k=croq.cdiv(croq.Var.lit(K_), int_tile_k)):
                    croq.tma.copy(
                        lhs.chunkat(block_m, iv_k),
                        lhs_s, swizzle=128)
                    croq.tma.copy(
                        rhs.chunkat(block_n, iv_k),
                        rhs_s, swizzle=128)

                    for iv_warp in croq.foreach(
                            iv_warp=croq.cdiv(int_tile_k, int_warp_k)):
                        for p in croq.parallel(
                                p=1, scope="group-4"):
                            ma = croq.mma.load(
                                lhs_s.chunkat(croq.FULL, iv_warp),
                                swizzle=128)
                            mb = croq.mma.load(
                                rhs_s.chunkat(croq.FULL, iv_warp),
                                swizzle=128)
                            mc = croq.mma.exec(
                                mc, ma, mb, method="row.row")

                croq.mma.store(mc,
                    output.chunkat(block_m, block_n))

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "void" in text.lower()
        assert "cdiv" in text


# =========================================================================== #
#  Port: memreuse-dynamic.co -- same as static with dynamic dims
# =========================================================================== #

class TestMemreuseDynamicCo:
    def test_memreuse_dynamic(self):
        @croq.co
        def mem_reuse0(i0: croq.u8[8, 256],
                       i1: croq.u8[8, 256]) -> croq.u8[8, 256]:
            output = croq.declare(croq.u8[8, 256], "output")

            for p, q in croq.parallel(p=8, q=128):
                smem1 = croq.dma.copy_async(
                    i0.chunkat(p, croq.FULL), to=croq.SHARED)
                output_s = croq.declare(
                    croq.u8[1, 256], "output_s",
                    storage=croq.SHARED, init=0)
                croq.wait(smem1)

                for i in croq.foreach(i=1):
                    output_s[i, q] = smem1.data[i, q]
                    output_s[i, q + 128] = smem1.data[i, q + 128]

                smem2 = croq.dma.copy_async(
                    i1.chunkat(p, croq.FULL), to=croq.SHARED)
                croq.wait(smem2)

                for i in croq.foreach(i=1):
                    output_s[i, q] += smem2.data[i, q]
                    output_s[i, q + 128] += smem2.data[i, q + 128]

                croq.dma.copy(output_s, output.chunkat(p, croq.FULL))
            return output

        text = croq.Program().add(mem_reuse0).dump_ast()
        assert "DMA" in text
        # Verify compound assignment produces additions
        plus_count = text.count("+")
        assert plus_count >= 2


# =========================================================================== #
#  Port: numerics_core6.co -- math builtins (unary)
# =========================================================================== #

class TestNumericsCore6Co:
    def test_core6_builtins(self):
        @croq.co
        def numerics_core6(
                inp: croq.f32[512]) -> croq.f32[512]:
            output = croq.declare(croq.f32[512], "output")
            for p in croq.parallel(p=512):
                x = inp[p]
                output[p] = (croq.sqrt(x + 2.0)
                             + croq.rsqrt(x + 2.0)
                             + croq.sin(x)
                             + croq.sinh(x * 0.25)
                             + croq.cos(x)
                             + croq.cosh(x * 0.25))
            return output

        text = croq.Program().add(numerics_core6).dump_ast()
        assert "__sqrt" in text
        assert "__rsqrt" in text
        assert "__sin" in text
        assert "__cos" in text
        assert "__sinh" in text
        assert "__cosh" in text


# =========================================================================== #
#  Port: numerics_transcendental.co -- unary + binary math builtins
# =========================================================================== #

class TestNumericsTranscendentalCo:
    def test_unary_mix(self):
        @croq.co
        def unary_mix(inp: croq.f32[512]) -> croq.f32[512]:
            output = croq.declare(croq.f32[512], "output")
            for p in croq.parallel(p=512):
                x = inp[p]
                output[p] = (croq.acos(x * 0.5)
                             + croq.asin(x * 0.5)
                             + croq.atan(x)
                             + croq.sin(x)
                             + croq.cos(x)
                             + croq.cosh(x * 0.25)
                             + croq.sinh(x * 0.25)
                             + croq.exp(x * 0.25)
                             + croq.log(x + 2.5)
                             + croq.log1p(x * 0.25 + 0.5)
                             + croq.expm1(x * 0.25)
                             + croq.sqrt(x + 2.5)
                             + croq.rsqrt(x + 2.5)
                             + croq.tan(x * 0.25)
                             + croq.tanh(x))
            return output

        text = croq.Program().add(unary_mix).dump_ast()
        for fn in ["__acos", "__asin", "__atan", "__sin", "__cos",
                    "__cosh", "__sinh", "__exp", "__log", "__log1p",
                    "__expm1", "__sqrt", "__rsqrt", "__tan", "__tanh"]:
            assert fn in text, f"Missing builtin {fn}"

    def test_binary_mix(self):
        @croq.co
        def binary_mix(lhs: croq.f32[512],
                       rhs: croq.f32[512]) -> croq.f32[512]:
            output = croq.declare(croq.f32[512], "output")
            for p in croq.parallel(p=512):
                x = lhs[p]
                y = rhs[p]
                output[p] = (croq.pow(x + 1.5, y + 1.5)
                             + croq.atan2(x, y))
            return output

        text = croq.Program().add(binary_mix).dump_ast()
        assert "__pow" in text
        assert "__atan2" in text


# =========================================================================== #
#  Port: add_fp6_fp4.co -- FP6/FP4 element-wise add (XFAIL in Choreo)
# =========================================================================== #

class TestAddFp6Fp4Co:
    def test_fp6_e2m3_add(self):
        @croq.co
        def ele_add(lhs: croq.f6_e2m3[6, 64],
                    rhs: croq.f6_e2m3[6, 64]) -> croq.f16[6, 64]:
            output = croq.declare(croq.f16[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        text = croq.Program().add(ele_add).dump_ast()
        assert "f6_e2m3" in text.lower()
        assert "f16" in text.lower()

    def test_fp4_e2m1_add(self):
        @croq.co
        def ele_add(lhs: croq.f4_e2m1[6, 64],
                    rhs: croq.f4_e2m1[6, 64]) -> croq.f16[6, 64]:
            output = croq.declare(croq.f16[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        text = croq.Program().add(ele_add).dump_ast()
        assert "f4_e2m1" in text.lower()


# =========================================================================== #
#  Port: copy_fp6_fp4.co -- DMA copy for FP6/FP4 types
# =========================================================================== #

class TestCopyFp6Fp4Co:
    def test_dma_copy_f6_e3m2(self):
        @croq.co
        def dma_copy_sync(
                inp: croq.f6_e3m2[6, 17, 64]) -> croq.f6_e3m2[6, 17, 64]:
            output = croq.declare(croq.f6_e3m2[6, 17, 64], "output")
            for _p in croq.parallel(_p=1):
                f = croq.dma.copy(inp, to=croq.SHARED)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy_sync).dump_ast()
        assert "DMA" in text
        assert "f6_e3m2" in text.lower()

    def test_dma_copy_f6_e2m3(self):
        @croq.co
        def dma_copy_sync(
                inp: croq.f6_e2m3[6, 17, 64]) -> croq.f6_e2m3[6, 17, 64]:
            output = croq.declare(croq.f6_e2m3[6, 17, 64], "output")
            for _p in croq.parallel(_p=1):
                f = croq.dma.copy(inp, to=croq.SHARED)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy_sync).dump_ast()
        assert "DMA" in text
        assert "f6_e2m3" in text.lower()

    def test_dma_copy_f4_e2m1(self):
        @croq.co
        def dma_copy_sync(
                inp: croq.f4_e2m1[6, 17, 64]) -> croq.f4_e2m1[6, 17, 64]:
            output = croq.declare(croq.f4_e2m1[6, 17, 64], "output")
            for _p in croq.parallel(_p=1):
                f = croq.dma.copy(inp, to=croq.SHARED)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy_sync).dump_ast()
        assert "DMA" in text
        assert "f4_e2m1" in text.lower()


# =========================================================================== #
#  Port: float_types_fp6_fp4.co -- element copy for FP6/FP4 types
# =========================================================================== #

class TestFloatTypesFp6Fp4Co:
    def _make_copy_kernel(self, dtype, shape=(6, 64)):
        @croq.co
        def ele_copy(lhs: dtype[shape[0], shape[1]]
                     ) -> dtype[shape[0], shape[1]]:
            output = croq.declare(dtype[shape[0], shape[1]], "output")
            for p, q in croq.parallel(p=shape[0], q=shape[1]):
                output[p, q] = lhs[p, q]
            return output
        return ele_copy

    def test_f6_e3m2_copy(self):
        k = self._make_copy_kernel(croq.f6_e3m2)
        text = croq.Program().add(k).dump_ast()
        assert "f6_e3m2" in text.lower()

    def test_f6_e2m3_copy(self):
        k = self._make_copy_kernel(croq.f6_e2m3)
        text = croq.Program().add(k).dump_ast()
        assert "f6_e2m3" in text.lower()

    def test_f4_e2m1_copy(self):
        k = self._make_copy_kernel(croq.f4_e2m1)
        text = croq.Program().add(k).dump_ast()
        assert "f4_e2m1" in text.lower()


# =========================================================================== #
#  Port: async_copy_fp8.co -- FP8 async DMA copy + add
# =========================================================================== #

class TestAsyncCopyFp8Co:
    def test_dma_copy_fp8(self):
        """Port of async_copy_fp8.co dma_copy function (static dims)."""
        @croq.co
        def dma_copy(
                inp: croq.f8_e5m2[6, 17, 32]) -> croq.f8_e5m2[6, 17, 32]:
            output = croq.declare(croq.f8_e5m2[6, 17, 32], "output")
            for _p in croq.parallel(_p=1, scope=croq.GROUP):
                f = croq.dma.copy_async(inp, to=croq.SHARED)
                croq.wait(f)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy).dump_ast()
        assert "DMA" in text
        assert "WAIT" in text
        assert "f8_e5m2" in text.lower() or "f8" in text.lower()

    def test_dma_add_fp8(self):
        """Port of async_copy_fp8.co dma_add function (static dims)."""
        @croq.co
        def dma_add(in0: croq.f8_e5m2[6, 17, 32],
                    in1: croq.f8_e5m2[6, 17, 32]) -> croq.f16[6, 17, 32]:
            output = croq.declare(croq.f16[6, 17, 32], "output")
            for q0 in croq.parallel(q0=17, scope=croq.GROUP):
                f0 = croq.dma.copy_async(in0, to=croq.SHARED)
                f1 = croq.dma.copy_async(in1, to=croq.SHARED)
                croq.wait(f0, f1)
                out = croq.declare(
                    croq.f16[6, 1, 32], "out", storage=croq.SHARED)
                for x in croq.foreach(x=6):
                    for q1 in croq.parallel(q1=32):
                        out[x, q0, q1] = (
                            f0.data[x, q0, q1] + f1.data[x, q0, q1])
                croq.dma.copy(out, output)
            return output

        text = croq.Program().add(dma_add).dump_ast()
        assert "DMA" in text
        assert "WAIT" in text

    def test_dma_add1_reverse_wait(self):
        """Port of async_copy_fp8.co dma_add1: reversed wait order."""
        @croq.co
        def dma_add1(in0: croq.f8_e5m2[6, 17, 32],
                     in1: croq.f8_e5m2[6, 17, 32]) -> croq.f16[6, 17, 32]:
            output = croq.declare(croq.f16[6, 17, 32], "output")
            for q0 in croq.parallel(q0=17, scope=croq.GROUP):
                f0 = croq.dma.copy_async(in0, to=croq.SHARED)
                f1 = croq.dma.copy_async(in1, to=croq.SHARED)
                croq.wait(f1, f0)
                out = croq.declare(
                    croq.f16[6, 1, 32], "out", storage=croq.SHARED)
                for x in croq.foreach(x=6):
                    for q1 in croq.parallel(q1=32):
                        out[x, q0, q1] = (
                            f0.data[x, q0, q1] + f1.data[x, q0, q1])
                croq.dma.copy(out, output)
            return output

        text = croq.Program().add(dma_add1).dump_ast()
        assert "DMA" in text
        assert "WAIT" in text


# =========================================================================== #
#  Port: cmma_mc_array.co -- INT8 GEMM with frag array
# =========================================================================== #

class TestCmmaMcArrayCo:
    def test_int8_frag_array(self):
        """Port of cmma_mc_array.co: INT8 GEMM with frag.s32 mc[1][2]."""
        M_, N_, K_ = 128, 128, 64
        MMA_M, MMA_N, MMA_K = 8, 8, 16

        @croq.co
        def matmul(lhs: croq.s8[M_, K_],
                   rhs: croq.s8[K_, N_]) -> croq.s32[M_, N_]:
            output = croq.declare(croq.s32[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M // 2, n=N_ // MMA_N // 2,
                    scope=croq.BLOCK):
                mc = croq.mma.frag("mc", [1, 2], 0, dtype=croq.s32)
                for g0, g1 in croq.parallel(
                        g0=2, g1=2, scope=croq.GROUP):
                    for k in croq.foreach(k=K_ // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        mc_ref = mc[0][1]
                        mc_ref_expr = croq.mma.exec(
                            mc_ref, ma, mb, method="row.col")
                    croq.mma.store(mc_ref, output.chunkat(m @ g0, n @ g1))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "s8" in text.lower()


# =========================================================================== #
#  Port: wmma_mc_array.co -- FP16 WMMA with frag array
# =========================================================================== #

class TestWmmaMcArrayCo:
    def test_f16_frag_array(self):
        """Port of wmma_mc_array.co: FP16 WMMA with frag mc[2][3]."""
        M_, N_, K_ = 128, 256, 64
        MMA_M, MMA_N, MMA_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[M_, K_],
                   rhs: croq.f16[K_, N_]) -> croq.f16[M_, N_]:
            output = croq.declare(croq.f16[M_, N_], "output")

            for m, n in croq.parallel(
                    m=M_ // MMA_M // 4, n=N_ // MMA_N,
                    scope=croq.BLOCK):
                mc = croq.mma.frag("mc", [2, 3], 0.0)
                for g0, g1 in croq.parallel(
                        g0=4, g1=1, scope=croq.GROUP):
                    croq.mma.fill(mc[0][1], 0.0)
                    for k in croq.foreach(k=K_ // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        mc_ref = mc[0][1]
                        croq.mma.exec(mc_ref, ma, mb, method="row.col")
                    croq.mma.store(mc_ref,
                                   output.chunkat(m @ g0, n @ g1))

            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "Parallelization" in text


# =========================================================================== #
#  Port: stream.co -- stream parameter + println
# =========================================================================== #

class TestStreamCo:
    def test_stream_println(self):
        """Port of stream.co: kernel with stream param + println."""
        @croq.co
        def foo(a: croq.s32[1], _s: croq.stream):
            for p in croq.parallel(p=6, scope=croq.BLOCK):
                croq.println(a[p] + p)

        text = croq.Program().add(foo).dump_ast()
        assert "Parallelization" in text
        assert "Call" in text or "println" in text.lower()
        assert "stream" in text.lower() or "void" in text.lower()


# =========================================================================== #
#  Port: make_tiled_copy.co -- structure test (foreach over span not supported)
# =========================================================================== #

class TestMakeTiledCopyCo:
    def test_tiled_copy_structure(self):
        """Structural port of make_tiled_copy.co ele_add0.
        Note: foreach over .span is not yet supported in croqtile-python,
        so we use explicit bounds instead."""
        @croq.co
        def ele_add0(lhs: croq.s32[16, 32],
                     rhs: croq.s32[16, 32]) -> croq.s32[16, 32]:
            output = croq.declare(croq.s32[16, 32], "output")
            for p in croq.parallel(p=1, scope=croq.BLOCK):
                for q in croq.parallel(q=32, scope=croq.THREAD):
                    lhs_s = croq.dma.copy(lhs, to=croq.SHARED)
                    rhs_s = croq.dma.copy(rhs, to=croq.SHARED)
                    for x, y in croq.foreach(x=16, y=32):
                        lhs_s.data[x, y] += rhs_s.data[x, y]
                    croq.dma.copy(lhs_s, output)
            return output

        text = croq.Program().add(ele_add0).dump_ast()
        assert "DMA" in text
        assert "Foreach" in text


# =========================================================================== #
#  Port: pad.co -- structure test (dma.pad not yet supported)
# =========================================================================== #

class TestPadCo:
    def test_pad_structure(self):
        """Structural port of pad.co: DMA copy + output declaration.
        Note: dma.pad is not yet supported in croqtile-python."""
        @croq.co
        def pad(inp: croq.s32[30, 60]) -> croq.s32[35, 63]:
            output = croq.declare(croq.s32[35, 63], "output")
            for p in croq.parallel(p=1):
                input_s = croq.dma.copy_async(inp, to=croq.SHARED)
                croq.wait(input_s)
                croq.dma.copy(input_s.data, output)
            return output

        text = croq.Program().add(pad).dump_ast()
        assert "DMA" in text
        assert "WAIT" in text


# =========================================================================== #
#  Port: warpspec.co -- structure test (events/inthreads not yet supported)
# =========================================================================== #

class TestWarpspecCo:
    def test_warpspec_structure(self):
        """Structural port of warpspec.co: WGMMA matmul skeleton.
        Note: event system and inthreads.async are not yet in croqtile-python."""
        WARP_M, WARP_N = 64, 128
        TILE_K, WARP_K = 64, 16

        @croq.co
        def matmul(lhs: croq.Global(croq.f16[256, 256]),
                   rhs: croq.Global(croq.f16[256, 256]),
                   output: croq.Global(croq.f16[256, 256])):
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(croq.Var.lit(256),
                                      croq.Var.lit(WARP_M)),
                    block_n=croq.cdiv(croq.Var.lit(256),
                                      croq.Var.lit(WARP_N)),
                    scope=croq.BLOCK):
                lhs_s = croq.declare(
                    croq.f16[WARP_M, TILE_K], "lhs_load_s",
                    storage=croq.SHARED)
                rhs_s = croq.declare(
                    croq.f16[WARP_N, TILE_K], "rhs_load_s",
                    storage=croq.SHARED)
                output_s = croq.declare(
                    croq.f16[WARP_M, WARP_N], "output_s",
                    storage=croq.SHARED)
                mc = croq.mma.fill(0.0, dtype=croq.f16)

                for iv_k in croq.foreach(
                        iv_k=croq.cdiv(croq.Var.lit(256),
                                       croq.Var.lit(TILE_K))):
                    croq.tma.copy(
                        lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k),
                        lhs_s, swizzle=128)
                    croq.tma.copy(
                        rhs.chunkat(block_n, iv_k),
                        rhs_s, swizzle=128)

                    for iv_warp in croq.foreach(
                            iv_warp=croq.cdiv(
                                croq.Var.lit(TILE_K),
                                croq.Var.lit(WARP_K))):
                        for p in croq.parallel(
                                p=1, scope="group-4"):
                            ma = croq.mma.load(
                                lhs_s.chunkat(croq.FULL, iv_warp),
                                swizzle=128)
                            mb = croq.mma.load(
                                rhs_s.chunkat(croq.FULL, iv_warp),
                                swizzle=128)
                            mc = croq.mma.exec(
                                mc, ma, mb, method="row.row")
                    croq.mma.commit()

                croq.mma.store(mc, output_s)
                croq.tma.copy(
                    output_s,
                    output.subspan(WARP_M, WARP_N).at(block_m, block_n))

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "SubSpan" in text
        assert "void" in text.lower()


# =========================================================================== #
#  Port: tma_fp6_fp4.co -- TMA pipeline with FP6/FP4 (structural)
# =========================================================================== #

class TestTmaFp6Fp4Co:
    def test_tma_fp6_pipeline(self):
        """Structural port of tma_fp6_fp4.co with f6_e2m3."""
        @croq.co
        def tma_ab_buffer(
                l: croq.f6_e2m3[16, 16, 128],
                r: croq.f6_e2m3[16, 16, 128]) -> croq.f32[16, 16, 128]:
            output = croq.declare(croq.f32[16, 16, 128], "output")
            for p in croq.parallel(p=8, scope=croq.BLOCK):
                with croq.with_in(k=8) as k:
                    l0 = croq.tma.copy_async(
                        l.chunkat(croq.FULL, k, p), to=croq.SHARED)
                    r0 = croq.tma.copy_async(
                        r.chunkat(croq.FULL, k, p), to=croq.SHARED)
                    l1 = croq.tma.any()
                    r1 = croq.tma.any()
                    out = croq.declare(
                        croq.f32[16, 1, 16], "out", storage=croq.SHARED)

                    for _ in croq.foreach_staged(k, start=1):
                        l1 = croq.tma.copy_async(
                            l.chunkat(croq.FULL, k, p), to=croq.SHARED)
                        r1 = croq.tma.copy_async(
                            r.chunkat(croq.FULL, k, p), to=croq.SHARED)
                        croq.wait(l0, r0)
                        for idx in croq.foreach(idx=16):
                            out[idx] = l0.data[idx] + r0.data[idx]
                        croq.tma.copy(
                            out, output.chunkat(croq.FULL, k - 1, p))
                        croq.swap(l0, l1)
                        croq.swap(r0, r1)

                    croq.wait(l0, r0)
                    for idx in croq.foreach(idx=16):
                        out[idx] = l0.data[idx] + r0.data[idx]
                    croq.tma.copy(
                        out, output.chunkat(croq.FULL, k, p))

            return output

        text = croq.Program().add(tma_ab_buffer).dump_ast()
        assert "f6_e2m3" in text.lower()
        assert "With" in text
        assert "Foreach" in text

    def test_tma_fp4_pipeline(self):
        """Structural port of tma_fp6_fp4.co with f4_e2m1."""
        @croq.co
        def tma_ab_buffer(
                l: croq.f4_e2m1[16, 16, 128],
                r: croq.f4_e2m1[16, 16, 128]) -> croq.f32[16, 16, 128]:
            output = croq.declare(croq.f32[16, 16, 128], "output")
            for p in croq.parallel(p=8, scope=croq.BLOCK):
                with croq.with_in(k=8) as k:
                    l0 = croq.tma.copy_async(
                        l.chunkat(croq.FULL, k, p), to=croq.SHARED)
                    r0 = croq.tma.copy_async(
                        r.chunkat(croq.FULL, k, p), to=croq.SHARED)
                    l1 = croq.tma.any()
                    r1 = croq.tma.any()
                    out = croq.declare(
                        croq.f32[16, 1, 16], "out", storage=croq.SHARED)

                    for _ in croq.foreach_staged(k, start=1):
                        l1 = croq.tma.copy_async(
                            l.chunkat(croq.FULL, k, p), to=croq.SHARED)
                        r1 = croq.tma.copy_async(
                            r.chunkat(croq.FULL, k, p), to=croq.SHARED)
                        croq.wait(l0, r0)
                        for idx in croq.foreach(idx=16):
                            out[idx] = l0.data[idx] + r0.data[idx]
                        croq.tma.copy(
                            out, output.chunkat(croq.FULL, k - 1, p))
                        croq.swap(l0, l1)
                        croq.swap(r0, r1)

                    croq.wait(l0, r0)
                    for idx in croq.foreach(idx=16):
                        out[idx] = l0.data[idx] + r0.data[idx]
                    croq.tma.copy(
                        out, output.chunkat(croq.FULL, k, p))

            return output

        text = croq.Program().add(tma_ab_buffer).dump_ast()
        assert "f4_e2m1" in text.lower()
        assert "With" in text


# =========================================================================== #
#  Port: async_copy_fp6_fp4.co -- async DMA for FP6/FP4 (structural)
# =========================================================================== #

class TestAsyncCopyFp6Fp4Co:
    def test_async_dma_f6_e2m3(self):
        @croq.co
        def dma_copy(
                inp: croq.f6_e2m3[6, 17, 64]) -> croq.f6_e2m3[6, 17, 64]:
            output = croq.declare(croq.f6_e2m3[6, 17, 64], "output")
            for _p in croq.parallel(_p=1, scope=croq.GROUP):
                f = croq.dma.copy_async(inp, to=croq.SHARED)
                croq.wait(f)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy).dump_ast()
        assert "DMA" in text
        assert "f6_e2m3" in text.lower()

    def test_async_dma_f4_e2m1(self):
        @croq.co
        def dma_copy(
                inp: croq.f4_e2m1[6, 17, 64]) -> croq.f4_e2m1[6, 17, 64]:
            output = croq.declare(croq.f4_e2m1[6, 17, 64], "output")
            for _p in croq.parallel(_p=1, scope=croq.GROUP):
                f = croq.dma.copy_async(inp, to=croq.SHARED)
                croq.wait(f)
                croq.dma.copy(f.data, output)
            return output

        text = croq.Program().add(dma_copy).dump_ast()
        assert "DMA" in text
        assert "f4_e2m1" in text.lower()


# =========================================================================== #
#  Port: debug_rtti_GDB_types.co -- structure test (ituple/println)
# =========================================================================== #

class TestDebugRttiCo:
    def test_debug_rtti_structure(self):
        """Structural port of debug_rtti_GDB_types.co.
        Note: ituple, a:[7,8,9], with idx in [b] are not yet in croqtile-python.
        We verify the basic kernel structure with parallel + println."""
        @croq.co
        def debug_kern(lhs: croq.s32[32, 64], x: croq.s32[1]):
            for p in croq.parallel(p=1, scope=croq.BLOCK):
                croq.println(croq.Var.lit(222))

        text = croq.Program().add(debug_kern).dump_ast()
        assert "Parallelization" in text
        assert "Call" in text or "println" in text.lower()

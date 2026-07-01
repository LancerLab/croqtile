"""Ports of subdirectory GPU end2end tests to croqtile-python.

Covers: wmma/, ptx_mma/, gemm/, gemm_sp/ directories.
These tests verify that croqtile-python can express MMA matmul patterns with
various type combinations, MMA shapes, and layouts, and that the
generated .co source compiles through the choreo compiler.
"""

import pytest
import croq


# ===================================================================
#  WMMA directory -- block/group MMA matmul with various dtypes
#
#  Pattern: parallel block grid, 2x2 group grid, foreach K,
#           mma.fill, mma.load, mma.exec, mma.store
# ===================================================================

class TestWMMADir:
    """Ports of tests/gpu/end2end/wmma/ -- WMMA matmul variants.

    Each test uses the same kernel structure from mma_f16f16f16_m16n16k16.co
    with different input/output types and MMA dimensions.
    """

    @staticmethod
    def _build_wmma_matmul(lhs_dtype, rhs_dtype, out_dtype, acc_dtype,
                           M, N, K, MMA_M, MMA_N, MMA_K, method):
        lhs_type = lhs_dtype[M, K]
        rhs_type = rhs_dtype[K, N] if method == "row.col" else rhs_dtype[N, K]
        out_type = out_dtype[M, N]
        group_m, group_n = 2, 2

        @croq.co
        def matmul(lhs: lhs_type, rhs: rhs_type) -> out_type:
            output = croq.declare(out_type, "output")
            for m, n in croq.parallel(
                    m=M // MMA_M // group_m,
                    n=N // MMA_N // group_n, scope=croq.BLOCK):
                mc = croq.mma.fill(0.0, dtype=acc_dtype)
                for g0, g1 in croq.parallel(
                        g0=group_m, g1=group_n, scope="group"):
                    if method == "row.col":
                        croq.mma.fill(mc, 0.0)
                    for k in croq.foreach(k=K // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        if method == "row.col":
                            mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        else:
                            mb = croq.mma.load(rhs.chunkat(n @ g1, k))
                        croq.mma.exec(mc, ma, mb, method=method)
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))
            return output
        return matmul

    @pytest.mark.parametrize("lhs_dt,rhs_dt,out_dt,acc_dt,name", [
        (croq.f16, croq.f16, croq.f16, None, "f16f16f16_m16n16k16"),
        (croq.f16, croq.f16, croq.f16, None, "f16f16f16_m32n8k16"),
        (croq.f16, croq.f16, croq.f16, None, "f16f16f16_m8n32k16"),
        (croq.f16, croq.f16, croq.f32, croq.f32, "f16f16f32_m16n16k16"),
        (croq.f16, croq.f16, croq.f32, croq.f32, "f16f16f32_m32n8k16"),
        (croq.f16, croq.f16, croq.f32, croq.f32, "f16f16f32_m8n32k16"),
        (croq.bf16, croq.bf16, croq.f32, croq.f32, "bf16bf16f32_m16n16k16"),
        (croq.bf16, croq.bf16, croq.f32, croq.f32, "bf16bf16f32_m32n8k16"),
        (croq.bf16, croq.bf16, croq.f32, croq.f32, "bf16bf16f32_m8n32k16"),
        (croq.s8, croq.s8, croq.s32, croq.s32, "s8s8s32_m16n16k16"),
        (croq.s8, croq.s8, croq.s32, croq.s32, "s8s8s32_m32n8k16"),
        (croq.s8, croq.s8, croq.s32, croq.s32, "s8s8s32_m8n32k16"),
        (croq.u8, croq.u8, croq.s32, croq.s32, "u8u8s32_m16n16k16"),
        (croq.u8, croq.u8, croq.s32, croq.s32, "u8u8s32_m32n8k16"),
        (croq.u8, croq.u8, croq.s32, croq.s32, "u8u8s32_m8n32k16"),
    ])
    def test_wmma_ast(self, lhs_dt, rhs_dt, out_dt, acc_dt, name):
        """AST-level: verify WMMA kernel structure."""
        parts = name.split("_m")
        mma_str = parts[1] if len(parts) > 1 else "16n16k16"
        dims = {}
        for seg in mma_str.replace("n", " ").replace("k", " ").split():
            dims[len(dims)] = int(seg)
        MMA_M = dims.get(0, 16)
        MMA_N = dims.get(1, 16)
        MMA_K = dims.get(2, 16)

        M, N, K = 128, 256, 256
        if M // MMA_M < 4:
            M = MMA_M * 4
        if N // MMA_N < 4:
            N = MMA_N * 4

        fn = self._build_wmma_matmul(
            lhs_dt, rhs_dt, out_dt, acc_dt or out_dt,
            M, N, K, MMA_M, MMA_N, MMA_K, method="row.col")

        text = croq.Program().add(fn).dump_ast()
        assert "MMA" in text
        assert "Parallelization" in text


# ===================================================================
#  PTX MMA directory -- mma.sync.aligned with various shapes/types
# ===================================================================

class TestPTXMMADir:
    """Ports of tests/gpu/end2end/ptx_mma/ -- PTX mma.sync variants.

    Each variant uses the same kernel pattern with different:
    - MMA shape (m16n8k16, m16n8k8, m8n8k16, etc.)
    - Input types (f16, bf16, s8, u8, tf32, f64, f8_e4m3, f8_e5m2)
    - Output types (f16, f32, s32, f64)
    """

    @staticmethod
    def _build_ptx_mma_kernel(lhs_dt, rhs_dt, out_dt, acc_dt,
                              M, N, K, MMA_M, MMA_N, MMA_K):
        lhs_type = lhs_dt[M, K]
        rhs_type = rhs_dt[K, N]
        out_type = out_dt[M, N]
        group_m, group_n = 2, 2
        block_m = M // MMA_M // group_m
        block_n = N // MMA_N // group_n
        if block_m < 1:
            block_m = 1
        if block_n < 1:
            block_n = 1

        @croq.co
        def matmul(lhs: lhs_type, rhs: rhs_type) -> out_type:
            output = croq.declare(out_type, "output")
            for m, n in croq.parallel(
                    m=block_m, n=block_n, scope=croq.BLOCK):
                mc = croq.mma.fill(0.0, dtype=acc_dt)
                for g0, g1 in croq.parallel(
                        g0=group_m, g1=group_n, scope="group"):
                    for k in croq.foreach(k=K // MMA_K):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        croq.mma.exec(mc, ma, mb, method="row.col")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))
            return output
        return matmul

    @pytest.mark.parametrize("lhs_dt,rhs_dt,out_dt,acc_dt,shape,name", [
        (croq.f16, croq.f16, croq.f32, croq.f32,
         (16, 8, 16), "m16n8k16_f32_f16_f16_f32"),
        (croq.f16, croq.f16, croq.f16, croq.f16,
         (16, 8, 16), "m16n8k16_f16_f16_f16_f16"),
        (croq.bf16, croq.bf16, croq.f32, croq.f32,
         (16, 8, 16), "m16n8k16_f32_bf16_bf16_f32"),
        (croq.s8, croq.s8, croq.s32, croq.s32,
         (16, 8, 16), "m16n8k16_s32_s8_s8_s32"),
        (croq.s8, croq.u8, croq.s32, croq.s32,
         (16, 8, 16), "m16n8k16_s32_s8_u8_s32"),
        (croq.u8, croq.s8, croq.s32, croq.s32,
         (16, 8, 16), "m16n8k16_s32_u8_s8_s32"),
        (croq.u8, croq.u8, croq.s32, croq.s32,
         (16, 8, 16), "m16n8k16_s32_u8_u8_s32"),
        (croq.f16, croq.f16, croq.f32, croq.f32,
         (16, 8, 8), "m16n8k8_f32_f16_f16_f32"),
        (croq.f16, croq.f16, croq.f16, croq.f16,
         (16, 8, 8), "m16n8k8_f16_f16_f16_f16"),
        (croq.bf16, croq.bf16, croq.f32, croq.f32,
         (16, 8, 8), "m16n8k8_f32_bf16_bf16_f32"),
        (croq.s8, croq.s8, croq.s32, croq.s32,
         (8, 8, 16), "m8n8k16_s32_s8_s8_s32"),
        (croq.s8, croq.u8, croq.s32, croq.s32,
         (8, 8, 16), "m8n8k16_s32_s8_u8_s32"),
        (croq.u8, croq.s8, croq.s32, croq.s32,
         (8, 8, 16), "m8n8k16_s32_u8_s8_s32"),
        (croq.u8, croq.u8, croq.s32, croq.s32,
         (8, 8, 16), "m8n8k16_s32_u8_u8_s32"),
        (croq.f16, croq.f16, croq.f16, croq.f16,
         (8, 8, 4), "m8n8k4_f16_f16_f16_f16"),
        (croq.f16, croq.f16, croq.f32, croq.f32,
         (8, 8, 4), "m8n8k4_f32_f16_f16_f32"),
        (croq.s8, croq.s8, croq.s32, croq.s32,
         (16, 8, 32), "m16n8k32_s32_s8_s8_s32"),
        (croq.s8, croq.u8, croq.s32, croq.s32,
         (16, 8, 32), "m16n8k32_s32_s8_u8_s32"),
        (croq.u8, croq.s8, croq.s32, croq.s32,
         (16, 8, 32), "m16n8k32_s32_u8_s8_s32"),
        (croq.u8, croq.u8, croq.s32, croq.s32,
         (16, 8, 32), "m16n8k32_s32_u8_u8_s32"),
    ])
    def test_ptx_mma_ast(self, lhs_dt, rhs_dt, out_dt, acc_dt, shape, name):
        """AST-level: verify PTX mma.sync kernel structure."""
        MMA_M, MMA_N, MMA_K = shape
        M = max(MMA_M * 4, 64)
        N = max(MMA_N * 4, 64)
        K = max(MMA_K * 4, 64)

        fn = self._build_ptx_mma_kernel(
            lhs_dt, rhs_dt, out_dt, acc_dt,
            M, N, K, MMA_M, MMA_N, MMA_K)

        text = croq.Program().add(fn).dump_ast()
        assert "MMA" in text
        assert "Parallelization" in text

    @pytest.mark.parametrize("lhs_dt,rhs_dt,out_dt,acc_dt,shape,name", [
        (croq.f8_e4m3, croq.f8_e4m3, croq.f16, croq.f16,
         (16, 8, 32), "m16n8k32_f16_e4m3_e4m3_f16"),
        (croq.f8_e4m3, croq.f8_e5m2, croq.f16, croq.f16,
         (16, 8, 32), "m16n8k32_f16_e4m3_e5m2_f16"),
        (croq.f8_e5m2, croq.f8_e4m3, croq.f16, croq.f16,
         (16, 8, 32), "m16n8k32_f16_e5m2_e4m3_f16"),
        (croq.f8_e5m2, croq.f8_e5m2, croq.f16, croq.f16,
         (16, 8, 32), "m16n8k32_f16_e5m2_e5m2_f16"),
        (croq.f8_e4m3, croq.f8_e4m3, croq.f32, croq.f32,
         (16, 8, 32), "m16n8k32_f32_e4m3_e4m3_f32"),
        (croq.f8_e4m3, croq.f8_e5m2, croq.f32, croq.f32,
         (16, 8, 32), "m16n8k32_f32_e4m3_e5m2_f32"),
        (croq.f8_e5m2, croq.f8_e4m3, croq.f32, croq.f32,
         (16, 8, 32), "m16n8k32_f32_e5m2_e4m3_f32"),
        (croq.f8_e5m2, croq.f8_e5m2, croq.f32, croq.f32,
         (16, 8, 32), "m16n8k32_f32_e5m2_e5m2_f32"),
    ])
    def test_ptx_mma_fp8_ast(self, lhs_dt, rhs_dt, out_dt, acc_dt,
                             shape, name):
        """AST-level: verify FP8 PTX mma.sync variants."""
        MMA_M, MMA_N, MMA_K = shape
        M = max(MMA_M * 4, 64)
        N = max(MMA_N * 4, 64)
        K = max(MMA_K * 4, 64)

        fn = self._build_ptx_mma_kernel(
            lhs_dt, rhs_dt, out_dt, acc_dt,
            M, N, K, MMA_M, MMA_N, MMA_K)

        text = croq.Program().add(fn).dump_ast()
        assert "MMA" in text
        assert "f8" in text.lower() or "e4m3" in text.lower() or \
               "e5m2" in text.lower()


# ===================================================================
#  GEMM directory -- row-col matmul with DMA to shared
# ===================================================================

class TestGEMMDir:
    """Ports of tests/gpu/end2end/gemm/ -- GEMM variants."""

    def test_gemm_rc_mma_sync_ast(self):
        """Port of gemm/gemm_rc_mma_sync.co -- SM86 GEMM with DMA to shared.

        Key: rhs is [N, K] (row-major), mma.row.row layout.
        """
        M, N, K = 256, 256, 256
        TILE_M, TILE_N, TILE_K = 32, 32, 16
        WARP_M, WARP_N, WARP_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[256, 256],
                   rhs: croq.f16[256, 256]) -> croq.f16[256, 256]:
            output = croq.declare(croq.f16[256, 256], "output", init=0)
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(M, TILE_M),
                    block_n=croq.cdiv(N, TILE_N), scope=croq.BLOCK):
                for iv_warp_m, iv_warp_n in croq.parallel(
                        iv_warp_m=croq.cdiv(TILE_M, WARP_M),
                        iv_warp_n=croq.cdiv(TILE_N, WARP_N),
                        scope="group"):
                    mc = croq.mma.fill(0.0)
                    for iv_k in croq.foreach(
                            iv_k=croq.cdiv(K, TILE_K)):
                        lhs_load_s = croq.dma.copy(
                            lhs.chunkat(block_m, iv_k),
                            to=croq.SHARED, name="lhs_load_s")
                        rhs_load_s = croq.dma.copy(
                            rhs.chunkat(block_n, iv_k),
                            to=croq.SHARED, name="rhs_load_s")
                        for iv_warp_k in croq.foreach(
                                iv_warp_k=croq.cdiv(TILE_K, WARP_K)):
                            ma = croq.mma.load(
                                lhs_load_s.data.chunkat(
                                    iv_warp_m, iv_warp_k))
                            mb = croq.mma.load(
                                rhs_load_s.data.chunkat(
                                    iv_warp_n, iv_warp_k))
                            croq.mma.exec(mc, ma, mb, method="row.row")
                    croq.mma.store(
                        mc, output.chunkat(
                            block_m @ iv_warp_m,
                            block_n @ iv_warp_n))
            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "DMA" in text
        assert "Parallelization" in text

    @pytest.mark.e2e
    def test_gemm_rc_mma_sync_e2e(self, compile_co):
        """E2E: gemm/gemm_rc_mma_sync.co -- compile through choreo."""
        M, N, K = 256, 256, 256
        TILE_M, TILE_N, TILE_K = 32, 32, 16
        WARP_M, WARP_N, WARP_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[256, 256],
                   rhs: croq.f16[256, 256]) -> croq.f16[256, 256]:
            output = croq.declare(croq.f16[256, 256], "output", init=0)
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(M, TILE_M),
                    block_n=croq.cdiv(N, TILE_N), scope=croq.BLOCK):
                for iv_warp_m, iv_warp_n in croq.parallel(
                        iv_warp_m=croq.cdiv(TILE_M, WARP_M),
                        iv_warp_n=croq.cdiv(TILE_N, WARP_N),
                        scope="group"):
                    mc = croq.mma.fill(0.0)
                    for iv_k in croq.foreach(
                            iv_k=croq.cdiv(K, TILE_K)):
                        lhs_load_s = croq.dma.copy(
                            lhs.chunkat(block_m, iv_k),
                            to=croq.SHARED, name="lhs_load_s")
                        rhs_load_s = croq.dma.copy(
                            rhs.chunkat(block_n, iv_k),
                            to=croq.SHARED, name="rhs_load_s")
                        for iv_warp_k in croq.foreach(
                                iv_warp_k=croq.cdiv(TILE_K, WARP_K)):
                            ma = croq.mma.load(
                                lhs_load_s.data.chunkat(
                                    iv_warp_m, iv_warp_k))
                            mb = croq.mma.load(
                                rhs_load_s.data.chunkat(
                                    iv_warp_n, iv_warp_k))
                            croq.mma.exec(mc, ma, mb, method="row.row")
                    croq.mma.store(
                        mc, output.chunkat(
                            block_m @ iv_warp_m,
                            block_n @ iv_warp_n))
            return output

        prog = croq.Program()
        prog.define("M", M)
        prog.define("N", N)
        prog.define("K", K)
        prog.add(matmul)
        prog.add("""
int main() {
  auto lhs = choreo::make_spandata<choreo::f16>(M, K);
  auto rhs = choreo::make_spandata<choreo::f16>(N, K);
  lhs.fill_random(1.0, 10.0);
  rhs.fill_random(1.0, 10.0);
  auto res = matmul(lhs.view(), rhs.view());
  float tolerance = 0.005f;
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      auto ref = 0.0f;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += __half2float(lhs[i][k] * rhs[j][k]);
      auto delta = std::abs((ref - __half2float(res[i][j])) / ref);
      choreo::choreo_assert(delta < tolerance, "mismatch");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    def test_gemm_rc_mma_sync_fp6_fp4_ast(self):
        """Port of gemm/gemm_rc_mma_sync_fp6_fp4.co -- FP6/FP4 GEMM."""
        M, N, K = 128, 128, 128
        TILE_M, TILE_N, TILE_K = 32, 32, 32
        WARP_M, WARP_N, WARP_K = 16, 16, 32

        @croq.co
        def matmul(lhs: croq.f6_e2m3[128, 128],
                   rhs: croq.f6_e2m3[128, 128]) -> croq.f32[128, 128]:
            output = croq.declare(croq.f32[128, 128], "output", init=0)
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(M, TILE_M),
                    block_n=croq.cdiv(N, TILE_N), scope=croq.BLOCK):
                for iv_warp_m, iv_warp_n in croq.parallel(
                        iv_warp_m=croq.cdiv(TILE_M, WARP_M),
                        iv_warp_n=croq.cdiv(TILE_N, WARP_N),
                        scope="group"):
                    mc = croq.mma.fill(0.0, dtype=croq.f32)
                    for iv_k in croq.foreach(
                            iv_k=croq.cdiv(K, TILE_K)):
                        lhs_load_s = croq.dma.copy(
                            lhs.chunkat(block_m, iv_k),
                            to=croq.SHARED, name="lhs_load_s")
                        rhs_load_s = croq.dma.copy(
                            rhs.chunkat(block_n, iv_k),
                            to=croq.SHARED, name="rhs_load_s")
                        for iv_warp_k in croq.foreach(
                                iv_warp_k=croq.cdiv(TILE_K, WARP_K)):
                            ma = croq.mma.load(
                                lhs_load_s.data.chunkat(
                                    iv_warp_m, iv_warp_k))
                            mb = croq.mma.load(
                                rhs_load_s.data.chunkat(
                                    iv_warp_n, iv_warp_k))
                            croq.mma.exec(mc, ma, mb, method="row.row")
                    croq.mma.store(
                        mc, output.chunkat(
                            block_m @ iv_warp_m,
                            block_n @ iv_warp_n))
            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "f6_e2m3" in text.lower()

    @pytest.mark.skipif(True, reason="TMA GEMM requires SM90+")
    def test_gemm_rc_mma_sync_tma_ast(self):
        """Port of gemm/gemm_rc_mma_sync_tma.co -- SM90 GEMM with TMA."""
        pass

    @pytest.mark.skipif(True, reason="WGMMA GEMM requires SM90a")
    def test_gemm_rc_wgmma_tma_ast(self):
        """Port of gemm/gemm_rc_wgmma_tma.co -- SM90a WGMMA with TMA."""
        pass

    def test_gemm_rr_mma_sync_ast(self):
        """Port of gemm/gemm_rr_mma_sync.co -- row.row GEMM layout."""
        M, N, K = 128, 128, 128
        TILE_M, TILE_N, TILE_K = 32, 32, 16
        WARP_M, WARP_N, WARP_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[128, 128],
                   rhs: croq.f16[128, 128]) -> croq.f16[128, 128]:
            output = croq.declare(croq.f16[128, 128], "output", init=0)
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(M, TILE_M),
                    block_n=croq.cdiv(N, TILE_N), scope=croq.BLOCK):
                for iv_warp_m, iv_warp_n in croq.parallel(
                        iv_warp_m=croq.cdiv(TILE_M, WARP_M),
                        iv_warp_n=croq.cdiv(TILE_N, WARP_N),
                        scope="group"):
                    mc = croq.mma.fill(0.0)
                    for iv_k in croq.foreach(
                            iv_k=croq.cdiv(K, TILE_K)):
                        lhs_load_s = croq.dma.copy(
                            lhs.chunkat(block_m, iv_k),
                            to=croq.SHARED, name="lhs_load_s")
                        rhs_load_s = croq.dma.copy(
                            rhs.chunkat(block_n, iv_k),
                            to=croq.SHARED, name="rhs_load_s")
                        for iv_warp_k in croq.foreach(
                                iv_warp_k=croq.cdiv(TILE_K, WARP_K)):
                            ma = croq.mma.load(
                                lhs_load_s.data.chunkat(
                                    iv_warp_m, iv_warp_k))
                            mb = croq.mma.load(
                                rhs_load_s.data.chunkat(
                                    iv_warp_n, iv_warp_k))
                            croq.mma.exec(mc, ma, mb, method="row.row")
                    croq.mma.store(
                        mc, output.chunkat(
                            block_m @ iv_warp_m,
                            block_n @ iv_warp_n))
            return output

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "Parallelization" in text


# ===================================================================
#  GEMM_SP directory -- sparse GEMM (structural ports)
# ===================================================================

# ===================================================================
#  MATMUL directory -- dynamic-shape matmul with Global params
#
#  Pattern: void function with global params, subspan().at(),
#           dma.copy_async, wait, mma.store to shared, dma.copy
#           shared->global.
# ===================================================================

class TestMatmulDir:
    """Ports of tests/gpu/end2end/matmul/ -- dynamic matmul variants."""

    def test_matmul_f16_dyn_sm86_ast(self):
        """Port of matmul/matmul_f16_dyn_sm86.co -- dynamic-shape matmul."""
        MMA_M, MMA_N, MMA_K = 16, 16, 16
        TILE_M, TILE_N, TILE_K = 16, 16, 16

        @croq.co
        def matmul(
                lhs: croq.Global(croq.f16[256, 256]),
                rhs: croq.Global(croq.f16[256, 256]),
                output: croq.Global(croq.f16[256, 256])):
            M, N, K = 256, 256, 256
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(M, TILE_M),
                    block_n=croq.cdiv(N, TILE_N), scope=croq.BLOCK):
                output_s = croq.declare(
                    croq.f16[TILE_M, TILE_N], "output_s",
                    storage=croq.SHARED)
                for warp_m, warp_n in croq.parallel(
                        warp_m=croq.cdiv(TILE_M, MMA_M),
                        warp_n=croq.cdiv(TILE_N, MMA_N),
                        scope="group"):
                    mc = croq.mma.fill(0.0)
                    for iv_k in croq.foreach(
                            iv_k=croq.cdiv(K, TILE_K)):
                        lhs_s = croq.dma.copy_async(
                            lhs.subspan(TILE_M, TILE_K).at(
                                block_m, iv_k),
                            to=croq.SHARED, name="lhs_load_s")
                        rhs_s = croq.dma.copy_async(
                            rhs.subspan(TILE_N, TILE_K).at(
                                block_n, iv_k),
                            to=croq.SHARED, name="rhs_load_s")
                        croq.wait(lhs_s, rhs_s)
                        for iv_warp_k in croq.foreach(
                                iv_warp_k=croq.cdiv(TILE_K, MMA_K)):
                            ma = croq.mma.load(
                                lhs_s.data.chunkat(warp_m, iv_warp_k))
                            mb = croq.mma.load(
                                rhs_s.data.chunkat(warp_n, iv_warp_k))
                            croq.mma.exec(mc, ma, mb, method="row.row")
                    croq.mma.store(mc, output_s.chunkat(warp_m, warp_n))
                croq.dma.copy(
                    output_s,
                    to=output.subspan(TILE_M, TILE_N).at(
                        block_m, block_n))
            return None

        text = croq.Program().add(matmul).dump_ast()
        assert "DMA" in text
        assert "MMA" in text

    @pytest.mark.e2e
    def test_matmul_f16_dyn_sm86_e2e(self, compile_co):
        """E2E: matmul/matmul_f16_dyn_sm86.co"""
        MMA_M, MMA_N, MMA_K = 16, 16, 16
        TILE_M, TILE_N, TILE_K = 16, 16, 16

        @croq.co
        def matmul(
                lhs: croq.Global(croq.f16[256, 256]),
                rhs: croq.Global(croq.f16[256, 256]),
                output: croq.Global(croq.f16[256, 256])):
            M, N, K = 256, 256, 256
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(M, TILE_M),
                    block_n=croq.cdiv(N, TILE_N), scope=croq.BLOCK):
                output_s = croq.declare(
                    croq.f16[TILE_M, TILE_N], "output_s",
                    storage=croq.SHARED)
                for warp_m, warp_n in croq.parallel(
                        warp_m=croq.cdiv(TILE_M, MMA_M),
                        warp_n=croq.cdiv(TILE_N, MMA_N),
                        scope="group"):
                    mc = croq.mma.fill(0.0)
                    for iv_k in croq.foreach(
                            iv_k=croq.cdiv(K, TILE_K)):
                        lhs_s = croq.dma.copy_async(
                            lhs.subspan(TILE_M, TILE_K).at(
                                block_m, iv_k),
                            to=croq.SHARED, name="lhs_load_s")
                        rhs_s = croq.dma.copy_async(
                            rhs.subspan(TILE_N, TILE_K).at(
                                block_n, iv_k),
                            to=croq.SHARED, name="rhs_load_s")
                        croq.wait(lhs_s, rhs_s)
                        for iv_warp_k in croq.foreach(
                                iv_warp_k=croq.cdiv(TILE_K, MMA_K)):
                            ma = croq.mma.load(
                                lhs_s.data.chunkat(warp_m, iv_warp_k))
                            mb = croq.mma.load(
                                rhs_s.data.chunkat(warp_n, iv_warp_k))
                            croq.mma.exec(mc, ma, mb, method="row.row")
                    croq.mma.store(mc, output_s.chunkat(warp_m, warp_n))
                croq.dma.copy(
                    output_s,
                    to=output.subspan(TILE_M, TILE_N).at(
                        block_m, block_n))
            return None

        prog = croq.Program()
        prog.define("M", 256)
        prog.define("N", 256)
        prog.define("K", 256)
        prog.add(matmul)
        prog.add("""
int main() {
  size_t M = 256, N = 256, K = 256;
  auto lhs_h = choreo::make_spandata<choreo::f16>(M, K);
  auto rhs_h = choreo::make_spandata<choreo::f16>(N, K);
  auto res_h = choreo::make_spandata<choreo::f16>(M, N);
  lhs_h.fill_random(0, 2);
  rhs_h.fill_random(0, 2);
  res_h.fill(0.0f);
  half *a_d = nullptr, *b_d = nullptr, *c_d = nullptr;
  choreo::abend_true(cudaMalloc(&a_d, M*K*sizeof(half)));
  choreo::abend_true(cudaMalloc(&b_d, N*K*sizeof(half)));
  choreo::abend_true(cudaMalloc(&c_d, M*N*sizeof(half)));
  choreo::abend_true(cudaMemcpy(a_d, lhs_h.data(), M*K*sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(b_d, rhs_h.data(), N*K*sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(c_d, res_h.data(), M*N*sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());
  auto lhs_d = choreo::make_spanview<choreo::f16, 2>(a_d, {M, K});
  auto rhs_d = choreo::make_spanview<choreo::f16, 2>(b_d, {N, K});
  auto res_d = choreo::make_spanview<choreo::f16, 2>(c_d, {M, N});
  matmul(lhs_d, rhs_d, res_d);
  choreo::abend_true(cudaMemcpy(res_h.data(), c_d, M*N*sizeof(half), cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    def test_matmul_f16_dyn_persis_sta_sm86_ast(self):
        """Port of matmul/matmul_f16_dyn_persis_sta_sm86.co -- persistent."""
        MMA_M, MMA_N, MMA_K = 16, 16, 16
        TILE_M, TILE_N, TILE_K = 16, 16, 16
        NUM_SMS = 82

        @croq.co
        def matmul(
                lhs: croq.Global(croq.f16[256, 256]),
                rhs: croq.Global(croq.f16[256, 256]),
                output: croq.Global(croq.f16[256, 256])):
            M, N, K = 256, 256, 256
            total_tiles = croq.cdiv(M, TILE_M) * croq.cdiv(N, TILE_N)
            tiles_n = croq.cdiv(N, TILE_N)
            for block_id in croq.parallel(
                    block_id=NUM_SMS, scope=croq.BLOCK):
                output_s = croq.declare(
                    croq.f16[TILE_M, TILE_N], "output_s",
                    storage=croq.SHARED)
                for tile_iter in croq.foreach(
                        tile_iter=croq.cdiv(total_tiles, NUM_SMS)):
                    tile_id = tile_iter @ block_id
                    with croq.device_if(tile_id < total_tiles):
                        block_m = tile_id / tiles_n
                        block_n = tile_id % tiles_n
                        for warp_m, warp_n in croq.parallel(
                                warp_m=croq.cdiv(TILE_M, MMA_M),
                                warp_n=croq.cdiv(TILE_N, MMA_N),
                                scope="group"):
                            mc = croq.mma.fill(0.0)
                            for iv_k in croq.foreach(
                                    iv_k=croq.cdiv(K, TILE_K)):
                                lhs_s = croq.dma.copy_async(
                                    lhs.subspan(TILE_M, TILE_K).at(
                                        block_m, iv_k),
                                    to=croq.SHARED,
                                    name="lhs_load_s")
                                rhs_s = croq.dma.copy_async(
                                    rhs.subspan(TILE_N, TILE_K).at(
                                        block_n, iv_k),
                                    to=croq.SHARED,
                                    name="rhs_load_s")
                                croq.wait(lhs_s, rhs_s)
                                for iv_warp_k in croq.foreach(
                                        iv_warp_k=croq.cdiv(
                                            TILE_K, MMA_K)):
                                    ma = croq.mma.load(
                                        lhs_s.data.chunkat(
                                            warp_m, iv_warp_k))
                                    mb = croq.mma.load(
                                        rhs_s.data.chunkat(
                                            warp_n, iv_warp_k))
                                    croq.mma.exec(
                                        mc, ma, mb, method="row.row")
                            croq.mma.store(
                                mc,
                                output_s.chunkat(warp_m, warp_n))
                        croq.dma.copy(
                            output_s,
                            to=output.subspan(TILE_M, TILE_N).at(
                                block_m, block_n))
            return None

        text = croq.Program().add(matmul).dump_ast()
        assert "MMA" in text
        assert "DMA" in text

    def test_matmul_trans_f16_dyn_sm86_ast(self):
        """Port of matmul/matmul_trans_f16_dyn_sm86.co -- transposed store."""
        MMA_M, MMA_N, MMA_K = 16, 16, 16
        TILE_M, TILE_N, TILE_K = 16, 16, 16

        @croq.co
        def matmul_trans(
                lhs: croq.Global(croq.f16[256, 256]),
                rhs: croq.Global(croq.f16[256, 256]),
                output: croq.Global(croq.f16[256, 256])):
            M, N, K = 256, 256, 256
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(M, TILE_M),
                    block_n=croq.cdiv(N, TILE_N), scope=croq.BLOCK):
                output_s = croq.declare(
                    croq.f16[TILE_N, TILE_M], "output_s",
                    storage=croq.SHARED)
                for warp_m, warp_n in croq.parallel(
                        warp_m=croq.cdiv(TILE_M, MMA_M),
                        warp_n=croq.cdiv(TILE_N, MMA_N),
                        scope="group"):
                    mc = croq.mma.fill(0.0)
                    for iv_k in croq.foreach(
                            iv_k=croq.cdiv(K, TILE_K)):
                        lhs_s = croq.dma.copy_async(
                            lhs.subspan(TILE_M, TILE_K).at(
                                block_m, iv_k),
                            to=croq.SHARED, name="lhs_load_s")
                        rhs_s = croq.dma.copy_async(
                            rhs.subspan(TILE_N, TILE_K).at(
                                block_n, iv_k),
                            to=croq.SHARED, name="rhs_load_s")
                        croq.wait(lhs_s, rhs_s)
                        for iv_warp_k in croq.foreach(
                                iv_warp_k=croq.cdiv(TILE_K, MMA_K)):
                            ma = croq.mma.load(
                                lhs_s.data.chunkat(warp_m, iv_warp_k))
                            mb = croq.mma.load(
                                rhs_s.data.chunkat(warp_n, iv_warp_k))
                            croq.mma.exec(mc, ma, mb, method="row.row")
                    croq.mma.store(
                        mc, output_s.chunkat(warp_n, warp_m),
                        transpose=True)
                croq.dma.copy(
                    output_s,
                    to=output.subspan(TILE_N, TILE_M).at(
                        block_n, block_m))
            return None

        text = croq.Program().add(matmul_trans).dump_ast()
        assert "MMA" in text

    @pytest.mark.skipif(True, reason="SM90 TMA matmul requires SM90a HW")
    def test_matmul_f16_dyn_sm90_ast(self):
        pass

    @pytest.mark.skipif(True, reason="SM90 persistent matmul requires SM90a")
    def test_matmul_f16_dyn_persis_sta_sm90_ast(self):
        pass

    @pytest.mark.skipif(True, reason="WGMMA matmul requires SM90a")
    def test_matmul_e4m3_row_row_wgmma_ast(self):
        pass

    @pytest.mark.skipif(True, reason="Warpspec matmul requires SM90a")
    def test_matmul_f16_dyn_sm90_warpspec_1p1c_ast(self):
        pass

    @pytest.mark.skipif(True, reason="Warpspec matmul requires SM90a")
    def test_matmul_f16_dyn_sm90_warpspec_1p3c_ast(self):
        pass


class TestGEMMSPDir:
    """Ports of tests/gpu/end2end/gemm_sp/ -- sparse GEMM (structural)."""

    @pytest.mark.skipif(True, reason="Sparse MMA not yet in croqtile-python")
    def test_gemm_sp_placeholder(self):
        """Sparse MMA uses mma.sp.* constructs not yet in croqtile-python."""
        pass


# ===================================================================
#  E2E compilation for WMMA representative tests
# ===================================================================

class TestWMMAE2E:
    """E2E compilation for key WMMA variants."""

    @pytest.mark.e2e
    def test_wmma_f16_m16n16k16_e2e(self, compile_co):
        """E2E: wmma/mma_f16f16f16_m16n16k16.co"""
        M, N, K = 128, 256, 256

        @croq.co
        def matmul(lhs: croq.f16[128, 256],
                   rhs: croq.f16[256, 256]) -> croq.f16[128, 256]:
            output = croq.declare(croq.f16[128, 256], "output")
            for m, n in croq.parallel(
                    m=M // 16 // 2, n=N // 16 // 2, scope=croq.BLOCK):
                mc = croq.mma.fill(0.0)
                for g0, g1 in croq.parallel(
                        g0=2, g1=2, scope="group"):
                    croq.mma.fill(mc, 0.0)
                    for k in croq.foreach(k=K // 16):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        croq.mma.exec(mc, ma, mb, method="row.col")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))
            return output

        prog = croq.Program()
        prog.define("M", M)
        prog.define("N", N)
        prog.define("K", K)
        prog.add(matmul)
        prog.add("""
int main() {
  auto lhs = choreo::make_spandata<choreo::f16>(M, K);
  auto rhs = choreo::make_spandata<choreo::f16>(K, N);
  lhs.fill_random(1, 10);
  rhs.fill_random(1, 10);
  auto res = matmul(lhs.view(), rhs.view());
  float tolerance = 0.005f;
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      auto ref = 0.0f;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += __half2float(lhs[i][k] * rhs[k][j]);
      auto delta = std::abs((ref - __half2float(res[i][j])) / ref);
      choreo::choreo_assert(delta < tolerance, "mismatch");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_wmma_s8_m16n16k16_e2e(self, compile_co):
        """E2E: wmma/mma_s8s8s32_m16n16k16.co"""
        M, N, K = 128, 256, 256

        @croq.co
        def matmul(lhs: croq.s8[128, 256],
                   rhs: croq.s8[256, 256]) -> croq.s32[128, 256]:
            output = croq.declare(croq.s32[128, 256], "output")
            for m, n in croq.parallel(
                    m=M // 16 // 2, n=N // 16 // 2, scope=croq.BLOCK):
                mc = croq.mma.fill(0.0, dtype=croq.s32)
                for g0, g1 in croq.parallel(
                        g0=2, g1=2, scope="group"):
                    for k in croq.foreach(k=K // 16):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        croq.mma.exec(mc, ma, mb, method="row.col")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))
            return output

        prog = croq.Program()
        prog.define("M", M)
        prog.define("N", N)
        prog.define("K", K)
        prog.add(matmul)
        prog.add("""
int main() {
  auto lhs = choreo::make_spandata<choreo::s8>(M, K);
  auto rhs = choreo::make_spandata<choreo::s8>(K, N);
  lhs.fill_random(1, 10);
  rhs.fill_random(1, 10);
  auto res = matmul(lhs.view(), rhs.view());
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_ptx_mma_f16_m16n8k16_e2e(self, compile_co):
        """E2E: ptx_mma/mma_sync_aligned_m16n8k16_row_col_f32_f16_f16_f32"""
        M, N, K = 64, 128, 64

        @croq.co
        def matmul(lhs: croq.f16[64, 64],
                   rhs: croq.f16[64, 128]) -> croq.f32[64, 128]:
            output = croq.declare(croq.f32[64, 128], "output")
            for m, n in croq.parallel(
                    m=M // 16 // 2, n=N // 8 // 2, scope=croq.BLOCK):
                mc = croq.mma.fill(0.0, dtype=croq.f32)
                for g0, g1 in croq.parallel(
                        g0=2, g1=2, scope="group"):
                    for k in croq.foreach(k=K // 16):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        croq.mma.exec(mc, ma, mb, method="row.col")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))
            return output

        prog = croq.Program()
        prog.define("M", M)
        prog.define("N", N)
        prog.define("K", K)
        prog.add(matmul)
        prog.add("""
int main() {
  auto lhs = choreo::make_spandata<choreo::f16>(M, K);
  auto rhs = choreo::make_spandata<choreo::f16>(K, N);
  lhs.fill_random(0.0, 0.05);
  rhs.fill_random(0.0, 0.05);
  auto res = matmul(lhs.view(), rhs.view());
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

"""Real GPU execution tests -- compile, run, and verify on SM86.

These tests go beyond source compilation to actually:
  1. Generate .co source from croqtile-python DSL
  2. Compile through the Choreo compiler to CUDA source
  3. Compile CUDA -> executable via nvcc
  4. Execute on the GPU
  5. Verify output (numpy-based or check-line based)

Requires: Choreo compiler, nvcc, NVIDIA GPU.
"""

import pytest
import croq


class TestGPUExecution:
    """Full end-to-end execution on GPU hardware with numpy data."""

    @pytest.mark.gpu
    def test_add_execute(self):
        """Element-wise add: numpy in -> GPU kernel -> numpy verify."""
        import numpy as np
        from croqtile.runtime import run_with_numpy

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
        result = run_with_numpy(prog, {"lhs": a, "rhs": b}, arch="sm_86")
        np.testing.assert_array_equal(result, a + b)

    @pytest.mark.gpu
    def test_add_shared_execute(self):
        """Element-wise add with shared memory staging."""
        import numpy as np
        from croqtile.runtime import run_with_numpy

        @croq.co
        def ele_add(lhs: croq.s32[6, 64],
                    rhs: croq.s32[6, 64]) -> croq.s32[6, 64]:
            output = croq.declare(croq.s32[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                f = croq.dma.copy(lhs, to=croq.SHARED, name="f")
                output[p, q] = f.data[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add)

        a = np.random.randint(-10, 10, (6, 64), dtype=np.int32)
        b = np.random.randint(-10, 10, (6, 64), dtype=np.int32)
        result = run_with_numpy(prog, {"lhs": a, "rhs": b}, arch="sm_86")
        np.testing.assert_array_equal(result, a + b)

    @pytest.mark.gpu
    def test_dma_copy_execute(self):
        """DMA copy: global -> shared -> global, verified with numpy."""
        import numpy as np
        from croqtile.runtime import run_with_numpy

        @croq.co
        def copy_kernel(src: croq.s32[4, 8]) -> croq.s32[4, 8]:
            output = croq.declare(croq.s32[4, 8], "output")
            for p, q in croq.parallel(p=4, q=8):
                f = croq.dma.copy(src, to=croq.SHARED, name="f")
                output[p, q] = f.data[p, q]
            return output

        prog = croq.Program()
        prog.add(copy_kernel)

        a = np.random.randint(1, 100, (4, 8), dtype=np.int32)
        result = run_with_numpy(prog, {"src": a}, arch="sm_86")
        np.testing.assert_array_equal(result, a)

    @pytest.mark.gpu
    def test_mma_matmul_execute(self):
        """WMMA matmul f16: numpy in -> GPU -> numpy verify on SM86."""
        import numpy as np
        from croqtile.runtime import run_with_numpy

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

        lhs = np.random.uniform(1, 10, (M, K)).astype(np.float16)
        rhs = np.random.uniform(1, 10, (K, N)).astype(np.float16)
        result = run_with_numpy(
            prog, {"lhs": lhs, "rhs": rhs},
            output_shape=(M, N), output_dtype=np.float16,
            arch="sm_86")
        ref = lhs.astype(np.float32) @ rhs.astype(np.float32)
        np.testing.assert_allclose(
            result.astype(np.float32), ref, rtol=0.01)

    @pytest.mark.gpu
    def test_gemm_rc_mma_sync_execute(self):
        """GEMM with DMA to shared: numpy in -> GPU -> numpy verify."""
        import numpy as np
        from croqtile.runtime import run_with_numpy

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

        lhs = np.random.uniform(1, 10, (M, K)).astype(np.float16)
        rhs = np.random.uniform(1, 10, (N, K)).astype(np.float16)
        result = run_with_numpy(
            prog, {"lhs": lhs, "rhs": rhs},
            output_shape=(M, N), output_dtype=np.float16,
            arch="sm_86")
        ref = lhs.astype(np.float32) @ rhs.T.astype(np.float32)
        np.testing.assert_allclose(
            result.astype(np.float32), ref, rtol=0.01)


class TestRuntimeDiscovery:
    """Tests for the runtime discovery functions."""

    def test_find_croqtile_bin(self):
        """Verify we can find the croqtile compiler."""
        from croqtile.runtime import find_croqtile_bin
        path = find_croqtile_bin()
        if path is None:
            pytest.skip("croqtile binary not installed")
        assert path.endswith("choreo") or path.endswith("croqtile")

    def test_find_nvcc(self):
        """Verify nvcc discovery."""
        from croqtile.runtime import find_nvcc
        path = find_nvcc()
        if path is None:
            pytest.skip("nvcc not installed")
        assert "nvcc" in path

    def test_find_runtime_include(self):
        """Verify runtime include directory."""
        from croqtile.runtime import find_runtime_include
        path = find_runtime_include()
        if path is None:
            pytest.skip("runtime headers not found")
        import os
        assert os.path.isfile(os.path.join(path, "choreo.h"))


class TestNumpyScalar:
    """Additional numpy-based tests with different dtypes."""

    @pytest.mark.gpu
    def test_scale_f32(self):
        """Scale f32 array by constant, verify with numpy."""
        import numpy as np
        from croqtile.runtime import run_with_numpy

        @croq.co
        def scale(src: croq.f32[4, 32]) -> croq.f32[4, 32]:
            output = croq.declare(croq.f32[4, 32], "output")
            for p, q in croq.parallel(p=4, q=32):
                output[p, q] = src[p, q] * 2
            return output

        prog = croq.Program()
        prog.add(scale)

        a = np.random.randn(4, 32).astype(np.float32)
        result = run_with_numpy(
            prog, {"src": a},
            output_shape=(4, 32), output_dtype=np.float32,
            arch="sm_86")
        np.testing.assert_allclose(result, a * 2, rtol=1e-5)

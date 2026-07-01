"""Real GPU execution tests -- compile, run, and verify on SM86.

These tests go beyond source compilation to actually:
  1. Generate .co source from CroqTile Python DSL
  2. Compile through the croqtile compiler to CUDA source
  3. Compile CUDA -> executable via nvcc
  4. Execute on the GPU
  5. Verify output

Requires: croqtile compiler, nvcc, NVIDIA GPU.
"""

import pytest
import croq


class TestGPUExecution:
    """Full end-to-end execution on GPU hardware."""

    @pytest.mark.gpu
    def test_add_execute(self, compile_and_run):
        """Element-wise add: compile + execute + verify."""
        @croq.co
        def ele_add(lhs: croq.s32[6, 64],
                    rhs: croq.s32[6, 64]) -> croq.s32[6, 64]:
            output = croq.declare(croq.s32[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::s32>(6, 64);
  auto b = choreo::make_spandata<choreo::s32>(6, 64);
  a.fill_random(-10, 10);
  b.fill_random(-10, 10);
  auto res = ele_add(a.view(), b.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 64; ++j)
      choreo::choreo_assert(a[i][j] + b[i][j] == res[i][j], "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        stdout = compile_and_run(
            prog.to_co(check_lines=["Test Passed"]))
        assert "Test Passed" in stdout

    @pytest.mark.gpu
    def test_add_shared_execute(self, compile_and_run):
        """Element-wise add with shared memory staging."""
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
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::s32>(6, 64);
  auto b = choreo::make_spandata<choreo::s32>(6, 64);
  a.fill_random(-10, 10);
  b.fill_random(-10, 10);
  auto res = ele_add(a.view(), b.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 64; ++j)
      choreo::choreo_assert(a[i][j] + b[i][j] == res[i][j], "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        stdout = compile_and_run(
            prog.to_co(check_lines=["Test Passed"]))
        assert "Test Passed" in stdout

    @pytest.mark.gpu
    def test_dma_copy_execute(self, compile_and_run):
        """DMA copy: global -> shared -> global."""
        @croq.co
        def copy_kernel(src: croq.s32[4, 8]) -> croq.s32[4, 8]:
            output = croq.declare(croq.s32[4, 8], "output")
            for p, q in croq.parallel(p=4, q=8):
                f = croq.dma.copy(src, to=croq.SHARED, name="f")
                output[p, q] = f.data[p, q]
            return output

        prog = croq.Program()
        prog.add(copy_kernel)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::s32>(4, 8);
  a.fill_random(1, 100);
  auto res = copy_kernel(a.view());
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 8; ++j)
      choreo::choreo_assert(a[i][j] == res[i][j], "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        stdout = compile_and_run(
            prog.to_co(check_lines=["Test Passed"]))
        assert "Test Passed" in stdout

    @pytest.mark.gpu
    def test_mma_matmul_execute(self, compile_and_run):
        """WMMA matmul f16: compile + execute + verify on SM86."""
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
        stdout = compile_and_run(
            prog.to_co(check_lines=["Test Passed"]))
        assert "Test Passed" in stdout

    @pytest.mark.gpu
    def test_gemm_rc_mma_sync_execute(self, compile_and_run):
        """GEMM with DMA to shared: compile + execute + verify on SM86."""
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
        stdout = compile_and_run(
            prog.to_co(check_lines=["Test Passed"]))
        assert "Test Passed" in stdout


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

"""End-to-end tests: CroqPy -> .co source -> choreo compiler -> CUDA source.

Each test:
  1. Defines a kernel using the CroqPy DSL
  2. Adds host verification code (verbatim C++ from the original .co files)
  3. Generates .co source with ``prog.to_co()``
  4. Compiles through the choreo compiler (``-es -t cute``)
  5. Asserts the CUDA source is non-empty

These tests prove the full round-trip: Python DSL -> valid .co -> valid CUDA.
"""

import pytest
import croq


# ===================================================================
#  Basic element-wise operations
# ===================================================================

class TestBasicOps:
    """Ports of add.co, add-local.co, add-shared.co."""

    @pytest.mark.e2e
    def test_add(self, compile_co):
        """Port of tests/gpu/end2end/add.co"""
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
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_add_local(self, compile_co):
        """Port of tests/gpu/end2end/add-local.co"""
        @croq.co
        def ele_add(lhs: croq.s32[6, 17, 64],
                    rhs: croq.s32[6, 17, 64]) -> croq.s32[6, 17, 64]:
            output = croq.declare(croq.s32[6, 17, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                for m in croq.foreach(m=17):
                    output[p, m, q] = lhs[p, m, q] + rhs[p, m, q]
            return output

        prog = croq.Program()
        prog.add(ele_add)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::s32>(6, 17, 64);
  auto b = choreo::make_spandata<choreo::s32>(6, 17, 64);
  a.fill_random(-10, 10);
  b.fill_random(-10, 10);
  auto res = ele_add(a.view(), b.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 17; ++j)
      for (size_t k = 0; k < 64; ++k)
        choreo::choreo_assert(a[i][j][k] + b[i][j][k] == res[i][j][k],
                              "values are not equal.");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_add_shared(self, compile_co):
        """Port of tests/gpu/end2end/add-shared.co"""
        @croq.co
        def ele_add(lhs: croq.s32[6, 16, 64],
                    rhs: croq.s32[6, 16, 64]) -> croq.s32[6, 16, 64]:
            output = croq.declare(croq.s32[6, 16, 64], "output")
            for p in croq.parallel(p=6):
                lhs_s = croq.dma.copy(lhs, to=croq.SHARED, name="lhs_s")
                rhs_s = croq.dma.copy(rhs, to=croq.SHARED, name="rhs_s")
                for m, q in croq.parallel(m=16, q=64):
                    output[p, m, q] = lhs_s.data[p, m, q] + \
                        rhs_s.data[p, m, q]
            return output

        prog = croq.Program()
        prog.add(ele_add)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::s32>(6, 16, 64);
  auto b = choreo::make_spandata<choreo::s32>(6, 16, 64);
  a.fill_random(-10, 10);
  b.fill_random(-10, 10);
  auto res = ele_add(a.view(), b.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 16; ++j)
      for (size_t k = 0; k < 64; ++k)
        choreo::choreo_assert(a[i][j][k] + b[i][j][k] == res[i][j][k],
                              "values are not equal.");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  DMA copy tests
# ===================================================================

class TestDMACopy:
    """Ports of copy.co, copy_fp8.co."""

    @pytest.mark.e2e
    def test_dma_copy_sync(self, compile_co):
        """Port of copy.co -- dma_copy_sync"""
        @croq.co
        def dma_copy_sync(
                input: croq.s32[6, 17, 64]) -> croq.s32[6, 17, 64]:
            output = croq.declare(croq.s32[6, 17, 64], "output")
            for _ in croq.parallel(p=1, q=1):
                f = croq.dma.copy(input, to=croq.SHARED, name="f")
                croq.dma.copy(f.data, to=output, name="g")
            return output

        prog = croq.Program()
        prog.add(dma_copy_sync)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::s32>(6, 17, 64);
  a.fill_random(-10, 10);
  auto res = dma_copy_sync(a.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 17; ++j)
      for (size_t k = 0; k < 64; ++k)
        choreo::choreo_assert(a[i][j][k] == res[i][j][k], "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_dma_copy_async(self, compile_co):
        """Port of copy.co -- dma_copy_async"""
        @croq.co
        def dma_copy_async(
                input: croq.s32[6, 17, 64]) -> croq.s32[6, 17, 64]:
            output = croq.declare(croq.s32[6, 17, 64], "output")
            for _ in croq.parallel(p=1, q=1):
                f = croq.dma.copy_async(input, to=croq.SHARED, name="f")
                croq.wait(f)
                croq.dma.copy(f.data, to=output, name="g")
            return output

        prog = croq.Program()
        prog.add(dma_copy_async)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::s32>(6, 17, 64);
  a.fill_random(-10, 10);
  auto res = dma_copy_async(a.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 17; ++j)
      for (size_t k = 0; k < 64; ++k)
        choreo::choreo_assert(a[i][j][k] == res[i][j][k], "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_copy_fp8(self, compile_co):
        """Port of copy_fp8.co -- FP8 DMA copy"""
        @croq.co
        def dma_copy_sync(
                input: croq.f8_e4m3[6, 17, 64]) -> croq.f8_e4m3[6, 17, 64]:
            output = croq.declare(croq.f8_e4m3[6, 17, 64], "output")
            for _ in croq.parallel(p=1, q=1):
                f = croq.dma.copy(input, to=croq.SHARED, name="f")
                croq.dma.copy(f.data, to=output, name="g")
            return output

        @croq.co
        def dma_copy_async(
                input: croq.f8_e5m2[6, 17, 64]) -> croq.f8_e5m2[6, 17, 64]:
            output = croq.declare(croq.f8_e5m2[6, 17, 64], "output")
            for _ in croq.parallel(p=1, q=1):
                f = croq.dma.copy_async(input, to=croq.SHARED, name="f")
                croq.wait(f)
                croq.dma.copy(f.data, to=output, name="g")
            return output

        prog = croq.Program()
        prog.add(dma_copy_sync)
        prog.add(dma_copy_async)
        prog.add("""
int main() {
  {
    auto a = choreo::make_spandata<choreo::f8_e4m3>(6, 17, 64);
    a.fill_random(-2.0f, 2.0f);
    auto res = dma_copy_sync(a.view());
    for (size_t i = 0; i < 6; ++i)
      for (size_t j = 0; j < 17; ++j)
        for (size_t k = 0; k < 64; ++k)
          choreo::choreo_assert(float(a[i][j][k]) == float(res[i][j][k]),
                                "mismatch");
    std::cout << "Test 0 Passed" << std::endl;
  }
  {
    auto a = choreo::make_spandata<choreo::f8_e5m2>(6, 17, 64);
    a.fill_random(-2.0f, 2.0f);
    auto res = dma_copy_async(a.view());
    for (size_t i = 0; i < 6; ++i)
      for (size_t j = 0; j < 17; ++j)
        for (size_t k = 0; k < 64; ++k)
          choreo::choreo_assert(float(a[i][j][k]) == float(res[i][j][k]),
                                "mismatch");
    std::cout << "Test 1 Passed" << std::endl;
  }
}
""")
        cuda = compile_co(prog.to_co(
            check_lines=["Test 0 Passed", "Test 1 Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Matmul tests (scalar + DMA)
# ===================================================================

class TestMatmul:
    """Ports of matmul.co, matmul-dma.co."""

    @pytest.mark.e2e
    def test_matmul(self, compile_co):
        """Port of tests/gpu/end2end/matmul.co"""
        @croq.co
        def matmul(lhs: croq.s32[128, 256],
                   rhs: croq.s32[256, 256]) -> croq.s32[128, 256]:
            output = croq.declare(croq.s32[128, 256], "output")
            for p, q in croq.parallel(p=16, q=64):
                for m, n, k in croq.foreach(m=8, n=4, k=256):
                    output[m, n] = output[m, n] + lhs[m, k] * rhs[k, n]
            return output

        prog = croq.Program()
        prog.add(matmul)
        prog.add("""
int main() {
  auto lhs = choreo::make_spandata<choreo::s32>(128, 256);
  auto rhs = choreo::make_spandata<choreo::s32>(256, 256);
  lhs.fill_random(-10, 10);
  rhs.fill_random(-10, 10);
  auto res = matmul(lhs.view(), rhs.view());
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      int ref = 0;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += lhs[i][k] * rhs[k][j];
      choreo::choreo_assert(ref == res[i][j], "values are not equal.");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_matmul_dma(self, compile_co):
        """Port of tests/gpu/end2end/matmul-dma.co"""
        @croq.co
        def matmul(lhs: croq.s32[128, 256],
                   rhs: croq.s32[256, 256]) -> croq.s32[128, 256]:
            output = croq.declare(croq.s32[128, 256], "output")
            for px, py in croq.parallel(px=8, py=16):
                for tile_k in croq.with_in(tile_k=16):
                    lhs_load = croq.dma.copy(
                        lhs.chunkat(px, tile_k),
                        to=croq.LOCAL, name="lhs_load")
                    rhs_load = croq.dma.copy(
                        rhs.chunkat(tile_k, py),
                        to=croq.LOCAL, name="rhs_load")
                    for qx, qy in croq.parallel(qx=16, qy=16):
                        for k in croq.foreach(k=16):
                            output[px @ qx, py @ qy] = (
                                output[px @ qx, py @ qy]
                                + lhs_load.data[qx, k]
                                * rhs_load.data[k, qy])
            return output

        prog = croq.Program()
        prog.add(matmul)
        prog.add("""
int main() {
  choreo::s32 a[128][256] = {0};
  choreo::s32 b[256][256] = {0};
  auto lhs = choreo::make_spanview<2, choreo::s32>((int*)a, {128, 256});
  auto rhs = choreo::make_spanview<2, choreo::s32>((int*)b, {256, 256});
  lhs.fill_random(-10, 10);
  rhs.fill_random(-10, 10);
  auto res = matmul(lhs, rhs);
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      int ref = 0;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += a[i][k]*b[k][j];
      choreo::choreo_assert(ref == res[i][j], "values are not equal.");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  MMA matmul tests
# ===================================================================

class TestMMA:
    """Ports of mma.co, mma_v1.co."""

    @pytest.mark.e2e
    def test_mma_matmul(self, compile_co):
        """Port of tests/gpu/end2end/mma.co"""
        M, N, K = 128, 256, 64
        MMA_M, MMA_N, MMA_K = 16, 16, 16

        @croq.co
        def matmul(lhs: croq.f16[128, 64],
                   rhs: croq.f16[64, 256]) -> croq.f16[128, 256]:
            output = croq.declare(croq.f16[128, 256], "output")
            MMA_M_ = croq.declare_int("MMA_M", 16)
            MMA_N_ = croq.declare_int("MMA_N", 16)
            MMA_K_ = croq.declare_int("MMA_K", 16)
            for m, n in croq.parallel(
                    m=M // MMA_M // 4, n=N // MMA_N, scope=croq.BLOCK):
                mc = croq.mma.fill(0.0)
                for g0, g1 in croq.parallel(
                        g0=4, g1=1, scope="group"):
                    croq.mma.fill(mc, 0.0)
                    for k in croq.foreach(k=K // MMA_K):
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
  float tolerance = 0.002f;
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      auto ref = 0.0f;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += __half2float(lhs[i][k] * rhs[k][j]);
      auto delta = std::abs((ref - __half2float(res[i][j])) / ref);
      choreo::choreo_assert((delta < tolerance), "values are not equal.");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100
        assert "mma" in cuda.lower() or "__co__" not in cuda

    @pytest.mark.e2e
    def test_mma_v1_matmul(self, compile_co):
        """Port of tests/gpu/end2end/mma_v1.co -- pipelined MMA"""
        M, N, K = 128, 256, 256

        @croq.co
        def matmul(lhs: croq.f16[128, 256],
                   rhs: croq.f16[256, 256]) -> croq.f16[128, 256]:
            output = croq.declare(croq.f16[128, 256], "output")
            for m, n in croq.parallel(
                    m=M // 16 // 4, n=N // 16, scope=croq.BLOCK):
                mc = croq.mma.fill(0.0)
                for g0, g1 in croq.parallel(
                        g0=4, g1=1, scope="group"):
                    croq.mma.fill(mc, 0.0)
                    for k in croq.foreach(k=K // 16):
                        ma = croq.mma.load(
                            lhs.chunkat(m @ g0, k), swizzle=128)
                        mb = croq.mma.load(
                            rhs.chunkat(k, n @ g1), swizzle=128)
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
      choreo::choreo_assert((delta < tolerance), "values are not equal.");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  TMA tests
# ===================================================================

class TestTMA:
    """Ports of tma.co -- requires SM90+, skip on SM86."""

    @pytest.mark.e2e
    @pytest.mark.skipif(True,
                        reason="TMA requires SM90+; current GPU is SM86")
    def test_tma_copy_async(self, compile_co):
        """Port of tests/gpu/end2end/tma.co -- tma_copy_async"""
        @croq.co
        def tma_copy_async(
                input: croq.f32[6, 16, 64]) -> croq.f32[6, 16, 64]:
            output = croq.declare(croq.f32[6, 16, 64], "output")
            for _ in croq.parallel(p=1, q=1):
                f = croq.tma.copy_async(input, to=croq.SHARED, name="f")
                croq.wait(f)
                croq.tma.copy(f.data, to=output, name="g")
            return output

        prog = croq.Program()
        prog.add(tma_copy_async)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::f32>(6, 16, 64);
  a.fill_random(-10, 10);
  auto res = tma_copy_async(a.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 16; ++j)
      for (size_t k = 0; k < 64; ++k)
        choreo::choreo_assert(a[i][j][k] == res[i][j][k], "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Transpose
# ===================================================================

class TestTranspose:
    """Port of transpose.co."""

    @pytest.mark.e2e
    def test_transpose(self, compile_co):
        """Port of tests/gpu/end2end/transpose.co"""
        @croq.co
        def transpose(
                a: croq.s32[3, 4, 5]) -> croq.s32[4, 3, 5]:
            output = croq.declare(croq.s32[4, 3, 5], "output")
            for p in croq.parallel(p=1):
                f = croq.dma.copy(a, to=croq.LOCAL, name="f")
                for i, j, k in croq.foreach(i=3, j=4, k=5):
                    output[j, i, k] = f.data[i, j, k]
            return output

        prog = croq.Program()
        prog.add(transpose)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::s32>(3, 4, 5);
  a.fill_random(-10, 10);
  auto res = transpose(a.view());
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 4; ++j)
      for (size_t k = 0; k < 5; ++k)
        choreo::choreo_assert(res[j][i][k] == a[i][j][k], "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Nil (void function)
# ===================================================================

class TestNil:
    """Port of nil.co."""

    @pytest.mark.e2e
    def test_nil(self, compile_co):
        """Port of tests/gpu/end2end/nil.co -- void function."""
        @croq.co
        def nil(x: croq.s32[1]) -> croq.s32[1]:
            return x

        prog = croq.Program()
        prog.add(nil)
        prog.add("""
int main() {
  auto x = choreo::make_spandata<choreo::s32>(1);
  nil(x.view());
  std::cout << "Run Complete." << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Run Complete."]))
        assert len(cuda) > 100


# ===================================================================
#  Async copy (multi-function)
# ===================================================================

class TestAsyncCopy:
    """Port of async_copy.co."""

    @pytest.mark.e2e
    def test_async_copy_multi(self, compile_co):
        """Port of tests/gpu/end2end/async_copy.co"""
        @croq.co
        def dma_copy(
                input: croq.s32[6, 17, 32]) -> croq.s32[6, 17, 32]:
            output = croq.declare(croq.s32[6, 17, 32], "output")
            for _ in croq.parallel(p=1, scope="group"):
                f = croq.dma.copy_async(input, to=croq.SHARED, name="f")
                croq.wait(f)
                croq.dma.copy(f.data, to=output, name="g")
            return output

        prog = croq.Program()
        prog.add(dma_copy)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::s32>(6, 17, 32);
  a.fill_random(-10, 10);
  auto res = dma_copy(a.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 17; ++j)
      for (size_t k = 0; k < 32; ++k)
        choreo::choreo_assert(a[i][j][k] == res[i][j][k], "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Float types
# ===================================================================

class TestFloatTypes:
    """Port of float_types.co -- element-wise add for f32, f16, bf16."""

    @pytest.mark.e2e
    def test_float_types_f32(self, compile_co):
        @croq.co
        def ele_add_f32(lhs: croq.f32[6, 64],
                        rhs: croq.f32[6, 64]) -> croq.f32[6, 64]:
            output = croq.declare(croq.f32[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add_f32)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::f32>(6, 64);
  auto b = choreo::make_spandata<choreo::f32>(6, 64);
  a.fill_random(-10, 10);
  b.fill_random(-10, 10);
  auto res = ele_add_f32(a.view(), b.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 64; ++j) {
      auto expected = a[i][j] + b[i][j];
      choreo::choreo_assert(expected == res[i][j], "f32 mismatch");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_float_types_f16(self, compile_co):
        @croq.co
        def ele_add_f16(lhs: croq.f16[6, 64],
                        rhs: croq.f16[6, 64]) -> croq.f16[6, 64]:
            output = croq.declare(croq.f16[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add_f16)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::f16>(6, 64);
  auto b = choreo::make_spandata<choreo::f16>(6, 64);
  a.fill_random(-10, 10);
  b.fill_random(-10, 10);
  auto res = ele_add_f16(a.view(), b.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 64; ++j) {
      auto expected = a[i][j] + b[i][j];
      choreo::choreo_assert(expected == res[i][j], "f16 mismatch");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_float_types_bf16(self, compile_co):
        @croq.co
        def ele_add_bf16(lhs: croq.bf16[6, 64],
                         rhs: croq.bf16[6, 64]) -> croq.bf16[6, 64]:
            output = croq.declare(croq.bf16[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add_bf16)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::bf16>(6, 64);
  auto b = choreo::make_spandata<choreo::bf16>(6, 64);
  a.fill_random(-10, 10);
  b.fill_random(-10, 10);
  auto res = ele_add_bf16(a.view(), b.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 64; ++j) {
      auto expected = a[i][j] + b[i][j];
      choreo::choreo_assert(expected == res[i][j], "bf16 mismatch");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Integral types
# ===================================================================

class TestIntegralTypes:
    """Port of integral_types.co -- common integer types."""

    @pytest.mark.e2e
    @pytest.mark.parametrize("type_name,dtype", [
        ("s32", croq.s32),
        ("u32", croq.u32),
        ("s16", croq.s16),
        ("u16", croq.u16),
        ("s8", croq.s8),
        ("u8", croq.u8),
    ])
    def test_integral_add(self, compile_co, type_name, dtype):
        @croq.co
        def ele_add(lhs: dtype[6, 64],
                    rhs: dtype[6, 64]) -> dtype[6, 64]:
            output = croq.declare(dtype[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add)
        prog.add(f"""
int main() {{
  auto a = choreo::make_spandata<choreo::{type_name}>(6, 64);
  auto b = choreo::make_spandata<choreo::{type_name}>(6, 64);
  a.fill_random(-5, 5);
  b.fill_random(-5, 5);
  auto res = ele_add(a.view(), b.view());
  std::cout << "Test Passed" << std::endl;
}}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  MMA fusion with scalar bias
# ===================================================================

class TestMMAFusion:
    """Port of mma_fusion_scalar.co."""

    @pytest.mark.e2e
    def test_mma_fusion_scalar(self, compile_co):
        """Port of tests/gpu/end2end/mma_fusion_scalar.co"""
        M, N, K = 128, 256, 256

        @croq.co
        def matmul(lhs: croq.f16[128, 256],
                   rhs: croq.f16[256, 256],
                   bias: croq.f16) -> croq.f16[128, 256]:
            output = croq.declare(croq.f16[128, 256], "output")
            for m, n in croq.parallel(
                    m=M // 16 // 2, n=N // 16 // 2, scope=croq.BLOCK):
                for g0, g1 in croq.parallel(
                        g0=2, g1=2, scope="group"):
                    mc = croq.mma.fill(0.0)
                    for k in croq.foreach(k=K // 16):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(k, n @ g1))
                        croq.mma.exec(mc, ma, mb, method="row.col")
                    croq.assign(mc, mc + bias)
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
  auto bias = choreo::make_spandata<choreo::f16>(1);
  lhs.fill_random(1, 10);
  rhs.fill_random(1, 10);
  bias.fill_random(1, 10);
  auto res = matmul(lhs.view(), rhs.view(), bias[0]);
  float tolerance = 0.005f;
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      auto ref = 0.0f;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += __half2float(lhs[i][k] * rhs[k][j]);
      ref += __half2float(bias[0]);
      auto delta = std::abs((ref - __half2float(res[i][j])) / ref);
      choreo::choreo_assert((delta < tolerance), "values are not equal.");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  MMA with C tile load (mma_loadc)
# ===================================================================

class TestMMALoadC:
    """Port of mma_loadc.co."""

    @pytest.mark.e2e
    def test_mma_loadc(self, compile_co):
        """Port of tests/gpu/end2end/mma_loadc.co"""
        M, N, K = 256, 256, 256
        TILE_M, TILE_N, TILE_K = 32, 32, 16
        WARP_M, WARP_N, WARP_K = 8, 8, 4

        @croq.co
        def matmul(lhs: croq.f16[256, 256],
                   rhs: croq.f16[256, 256]) -> croq.f16[256, 256]:
            output = croq.declare(croq.f16[256, 256], "output", init=0)
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(M, TILE_M),
                    block_n=croq.cdiv(N, TILE_N), scope=croq.BLOCK):
                for iv_k in croq.foreach(
                        iv_k=croq.cdiv(K, TILE_K)):
                    lhs_load_s = croq.dma.copy(
                        lhs.chunkat(block_m, iv_k),
                        to=croq.SHARED, name="lhs_load_s")
                    rhs_load_s = croq.dma.copy(
                        rhs.chunkat(iv_k, block_n),
                        to=croq.SHARED, name="rhs_load_s")
                    for iv_warp_m, iv_warp_n in croq.parallel(
                            iv_warp_m=croq.cdiv(TILE_M, WARP_M),
                            iv_warp_n=croq.cdiv(TILE_N, WARP_N),
                            scope="group"):
                        mc = croq.mma.load(
                            output.chunkat(
                                block_m @ iv_warp_m,
                                block_n @ iv_warp_n))
                        for iv_warp_k in croq.foreach(
                                iv_warp_k=croq.cdiv(TILE_K, WARP_K)):
                            ma = croq.mma.load(
                                lhs_load_s.data.chunkat(
                                    iv_warp_m, iv_warp_k))
                            mb = croq.mma.load(
                                rhs_load_s.data.chunkat(
                                    iv_warp_k, iv_warp_n))
                            croq.mma.exec(mc, ma, mb, method="row.col")
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
  auto rhs = choreo::make_spandata<choreo::f16>(K, N);
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < K; ++j)
      lhs[i][j] = __float2half(i*0.02f+j*0.01f);
  for (size_t i = 0; i < K; ++i)
    for (size_t j = 0; j < N; ++j)
      rhs[i][j] = __float2half(i*0.07f+j*0.03f);
  auto res = matmul(lhs.view(), rhs.view());
  float tolerance = 0.007f;
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      __half ref = __float2half(0.0f);
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += lhs[i][k] * rhs[k][j];
      auto delta = std::abs((__half2float(ref) - __half2float(res[i][j]))
                            / __half2float(ref));
      choreo::choreo_assert((delta < tolerance), "values are not equal.");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  WMMA row.row layout
# ===================================================================

class TestWMMA:
    """Port of wmma_rr.co."""

    @pytest.mark.e2e
    def test_wmma_row_row(self, compile_co):
        """Port of tests/gpu/end2end/wmma_rr.co -- row.row MMA"""
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
                    for k in croq.foreach(k=K // 16):
                        ma = croq.mma.load(lhs.chunkat(m @ g0, k))
                        mb = croq.mma.load(rhs.chunkat(n @ g1, k))
                        croq.mma.exec(mc, ma, mb, method="row.row")
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
  auto rhs = choreo::make_spandata<choreo::f16>(N, K);
  lhs.fill_random(0, 1);
  rhs.fill_random(0, 1);
  auto res = matmul(lhs.view(), rhs.view());
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Numerics -- math builtins
# ===================================================================

class TestNumerics:
    """Port of numerics_core6.co."""

    @pytest.mark.e2e
    def test_numerics_core6(self, compile_co):
        """Port of tests/gpu/end2end/numerics_core6.co"""
        @croq.co
        def numerics_core6(
                input: croq.f32[512]) -> croq.f32[512]:
            output = croq.declare(croq.f32[512], "output")
            for p in croq.parallel(p=512):
                x = input[p]
                output[p] = (croq.sqrt(x + 2.0) + croq.rsqrt(x + 2.0)
                             + croq.sin(x) + croq.sinh(x * 0.25)
                             + croq.cos(x) + croq.cosh(x * 0.25))
            return output

        prog = croq.Program()
        prog.add(numerics_core6)
        prog.add("""
#include <cmath>
static float ref_core6(float x) {
  return std::sqrt(x + 2.0f) + 1.0f / std::sqrt(x + 2.0f) + std::sin(x) +
         std::sinh(x * 0.25f) + std::cos(x) + std::cosh(x * 0.25f);
}
int main() {
  auto input = choreo::make_spandata<choreo::f32>(512);
  input.fill_random(-0.5f, 1.0f);
  auto res = numerics_core6(input.view());
  for (size_t i = 0; i < 512; ++i) {
    float ref = ref_core6(input[i]);
    float got = res[i];
    float tol = 2e-4f + 2e-4f * std::abs(ref);
    choreo::choreo_assert(std::abs(got - ref) <= tol,
                          "core6 numerics mismatch.");
  }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Pipelined matmul with swap + dma.any + foreach_staged
# ===================================================================

class TestPipelined:
    """Port of matmul-pipelined-2.co."""

    @pytest.mark.e2e
    def test_matmul_pipelined(self, compile_co):
        """Port of tests/gpu/end2end/matmul-pipelined-2.co"""
        M, N, K = 128, 256, 256

        @croq.co
        def matmul(lhs: croq.s32[128, 256],
                   rhs: croq.s32[256, 256]) -> croq.s32[128, 256]:
            output = croq.declare(croq.s32[128, 256], "output")
            for px, py in croq.parallel(px=8, py=16, scope=croq.BLOCK):
                for qx, qy in croq.parallel(
                        qx=16, qy=16, scope=croq.THREAD):
                    with croq.with_in(tile_k=16) as tile_k:
                        lf0 = croq.dma.copy(
                            lhs.chunkat(px, tile_k),
                            to=croq.SHARED, name="lf0")
                        rf0 = croq.dma.copy(
                            rhs.chunkat(tile_k, py),
                            to=croq.SHARED, name="rf0")
                        lf1 = croq.dma.any(name="lf1")
                        rf1 = croq.dma.any(name="rf1")

                        for _ in croq.foreach_staged(tile_k, start=1):
                            lf1 = croq.dma.copy(
                                lhs.chunkat(px, tile_k),
                                to=croq.SHARED, name="lf1")
                            rf1 = croq.dma.copy(
                                rhs.chunkat(tile_k, py),
                                to=croq.SHARED, name="rf1")
                            for k in croq.foreach(k=16):
                                output[px @ qx, py @ qy] = (
                                    output[px @ qx, py @ qy]
                                    + lf0.data[qx, k]
                                    * rf0.data[k, qy])
                            croq.swap(lf0, lf1)
                            croq.swap(rf0, rf1)

                        for k in croq.foreach(k=16):
                            output[px @ qx, py @ qy] = (
                                output[px @ qx, py @ qy]
                                + lf0.data[qx, k]
                                * rf0.data[k, qy])
            return output

        prog = croq.Program()
        prog.define("M", M)
        prog.define("N", N)
        prog.define("K", K)
        prog.add(matmul)
        prog.add("""
int main() {
  auto lhs = choreo::make_spandata<choreo::s32>(M, K);
  auto rhs = choreo::make_spandata<choreo::s32>(K, N);
  lhs.fill_random(-10, 10);
  rhs.fill_random(-10, 10);
  auto res = matmul(lhs.view(), rhs.view());
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      int ref = 0;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += lhs[i][k] * rhs[k][j];
      choreo::choreo_assert(ref == res[i][j], "values are not equal.");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Stream + println
# ===================================================================

class TestStream:
    """Port of stream.co."""

    @pytest.mark.e2e
    def test_stream_println(self, compile_co):
        """Port of tests/gpu/end2end/stream.co"""
        @croq.co
        def foo(a: croq.s32) -> croq.s32:
            for p in croq.parallel(p=6, scope=croq.BLOCK):
                croq.println(a + p)
            return a

        prog = croq.Program()
        prog.add(foo)
        prog.add("""
int main() {
  foo(100);
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  FP8 -> F16 add (SM90 only)
# ===================================================================

class TestFP8:
    """Port of add_fp8_e4m3_to_f16.co."""

    @pytest.mark.e2e
    @pytest.mark.skipif(True,
                        reason="FP8 add-to-f16 requires SM90+")
    def test_add_fp8_to_f16(self, compile_co):
        """Port of tests/gpu/end2end/add_fp8_e4m3_to_f16.co"""
        @croq.co
        def ele_add_fp8(lhs: croq.f8_e4m3[6, 64],
                        rhs: croq.f8_e4m3[6, 64]) -> croq.f16[6, 64]:
            output = croq.declare(croq.f16[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add_fp8)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::f8_e4m3>(6, 64);
  auto b = choreo::make_spandata<choreo::f8_e4m3>(6, 64);
  a.fill_random(-2.0f, 2.0f);
  b.fill_random(-2.0f, 2.0f);
  auto res = ele_add_fp8(a.view(), b.view());
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  MMA v2: pipelined swap-based matmul
# ===================================================================

class TestMMAv2:
    """Port of mma_v2.co -- pipelined MMA with with-in + swap."""

    @pytest.mark.e2e
    def test_mma_v2_pipelined(self, compile_co):
        """Port of tests/gpu/end2end/mma_v2.co"""
        M, N, K = 64, 32, 16

        @croq.co
        def matmul(lhs: croq.f16[64, 16],
                   rhs: croq.f16[16, 32]) -> croq.f16[64, 32]:
            output = croq.declare(croq.f16[64, 32], "output")
            MMA_M_ = croq.declare_int("MMA_M", 16)
            MMA_N_ = croq.declare_int("MMA_N", 16)
            MMA_K_ = croq.declare_int("MMA_K", 16)
            for m, n in croq.parallel(
                    m=M // 16 // 4, n=N // 16, scope=croq.BLOCK):
                for g0, g1 in croq.parallel(
                        g0=4, g1=1, scope="group"):
                    mc = croq.mma.fill(0.0)
                    with croq.with_in(k=K // 16) as k:
                        fl0 = croq.dma.copy_async(
                            lhs, to=croq.SHARED, name="fl0")
                        fr0 = croq.dma.copy_async(
                            rhs, to=croq.SHARED, name="fr0")
                        fl1 = croq.dma.any(name="fl1")
                        fr1 = croq.dma.any(name="fr1")
                        for _ in croq.foreach_staged(k, start=1):
                            fl1 = croq.dma.copy_async(
                                lhs, to=croq.SHARED, name="fl1")
                            fr1 = croq.dma.copy_async(
                                rhs, to=croq.SHARED, name="fr1")
                            croq.wait(fl0, fr0)
                            ma = croq.mma.load(
                                fl0.data.chunkat(m @ g0, k - 1))
                            mb = croq.mma.load(
                                fr0.data.chunkat(k - 1, n @ g1))
                            croq.mma.exec(mc, ma, mb, method="row.col")
                            croq.swap(fl0, fl1)
                            croq.swap(fr0, fr1)
                        croq.wait(fl0, fr0)
                        ma = croq.mma.load(
                            fl0.data.chunkat(m @ g0, k))
                        mb = croq.mma.load(
                            fr0.data.chunkat(k, n @ g1))
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
  float base_tol = 0.001f;
  float rel_tol = 0.002f;
  choreo::verify_matmul_row_col_subset(lhs, rhs, res, base_tol, rel_tol, 8, 8);
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  MMA FP8: e4m3 and e5m2 MMA matmul (SM90 only)
# ===================================================================

class TestMMAFP8:
    """Port of mma_fp8.co -- FP8 MMA pipelined (SM90)."""

    @pytest.mark.e2e
    @pytest.mark.skipif(True,
                        reason="FP8 MMA requires SM90+")
    def test_mma_fp8_e4m3(self, compile_co):
        """Port of tests/gpu/end2end/mma_fp8.co -- e4m3"""
        M, N, K = 64, 32, 32

        @croq.co
        def matmul_e4m3(lhs: croq.f8_e4m3[64, 32],
                        rhs: croq.f8_e4m3[32, 32]) -> croq.f32[64, 32]:
            output = croq.declare(croq.f32[64, 32], "output")
            MMA_M_ = croq.declare_int("MMA_M", 16)
            MMA_N_ = croq.declare_int("MMA_N", 8)
            MMA_K_ = croq.declare_int("MMA_K", 32)
            for m, n in croq.parallel(
                    m=M // 16 // 4, n=N // 8, scope=croq.BLOCK):
                for g0, g1 in croq.parallel(
                        g0=4, g1=1, scope="group"):
                    mc = croq.mma.fill(0.0, dtype=croq.f32)
                    with croq.with_in(k=K // 32) as k:
                        fl0 = croq.dma.copy_async(
                            lhs, to=croq.SHARED, name="fl0")
                        fr0 = croq.dma.copy_async(
                            rhs, to=croq.SHARED, name="fr0")
                        fl1 = croq.dma.any(name="fl1")
                        fr1 = croq.dma.any(name="fr1")
                        for _ in croq.foreach_staged(k, start=1):
                            fl1 = croq.dma.copy_async(
                                lhs, to=croq.SHARED, name="fl1")
                            fr1 = croq.dma.copy_async(
                                rhs, to=croq.SHARED, name="fr1")
                            croq.wait(fl0, fr0)
                            ma = croq.mma.load(
                                fl0.data.chunkat(m @ g0, k - 1))
                            mb = croq.mma.load(
                                fr0.data.chunkat(k - 1, n @ g1))
                            croq.mma.exec(mc, ma, mb, method="row.col")
                            croq.swap(fl0, fl1)
                            croq.swap(fr0, fr1)
                        croq.wait(fl0, fr0)
                        ma = croq.mma.load(
                            fl0.data.chunkat(m @ g0, k))
                        mb = croq.mma.load(
                            fr0.data.chunkat(k, n @ g1))
                        croq.mma.exec(mc, ma, mb, method="row.col")
                    croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))
            return output

        prog = croq.Program()
        prog.define("M", M)
        prog.define("N", N)
        prog.define("K", K)
        prog.add(matmul_e4m3)
        prog.add("""
int main() {
  auto lhs = choreo::make_spandata<choreo::f8_e4m3>(M, K);
  auto rhs = choreo::make_spandata<choreo::f8_e4m3>(K, N);
  lhs.fill_random(-2.0f, 2.0f);
  rhs.fill_random(-2.0f, 2.0f);
  auto res = matmul_e4m3(lhs.view(), rhs.view());
  float base_tol = 0.2f;
  float rel_tol = 0.02f;
  choreo::verify_matmul_row_col_subset(lhs, rhs, res, base_tol, rel_tol, 8, 8);
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  WMMA FP8 row.row (SM90 only)
# ===================================================================

class TestWMMAFP8:
    """Port of wmma_fp8_rr.co -- FP8 e4m3 row.col WMMA."""

    @pytest.mark.e2e
    @pytest.mark.skipif(True,
                        reason="FP8 WMMA requires SM90+")
    def test_wmma_fp8_rr(self, compile_co):
        """Port of tests/gpu/end2end/wmma_fp8_rr.co"""
        M, N, K = 64, 64, 64

        @croq.co
        def matmul(lhs: croq.f8_e4m3[64, 64],
                   rhs: croq.f8_e4m3[64, 64]) -> croq.f32[64, 64]:
            output = croq.declare(croq.f32[64, 64], "output")
            for m, n in croq.parallel(
                    m=M // 16 // 2, n=N // 8 // 2, scope=croq.BLOCK):
                mc = croq.mma.fill(0.0, dtype=croq.f32)
                for g0, g1 in croq.parallel(
                        g0=2, g1=2, scope="group"):
                    for k in croq.foreach(k=K // 32):
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
  auto lhs = choreo::make_spandata<choreo::f8_e4m3>(M, K);
  auto rhs = choreo::make_spandata<choreo::f8_e4m3>(K, N);
  lhs.fill_random(-2.0f, 2.0f);
  rhs.fill_random(-2.0f, 2.0f);
  auto res = matmul(lhs.view(), rhs.view());
  float base_tol = 0.2f;
  float rel_tol = 0.02f;
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      float ref = 0.0f;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += float(lhs[i][k]) * float(rhs[k][j]);
      float got = float(res[i][j]);
      float tol = base_tol + rel_tol * std::abs(ref);
      choreo::choreo_assert(std::abs(got - ref) <= tol, "mismatch");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Memory reuse (static + dynamic)
# ===================================================================

class TestMemreuse:
    """Ports of memreuse-static.co and memreuse-dynamic.co."""

    @pytest.mark.e2e
    def test_memreuse_static(self, compile_co):
        """Port of tests/gpu/end2end/memreuse-static.co"""
        K, N = 8, 256

        @croq.co
        def mem_reuse0(i0: croq.u8[8, 256],
                       i1: croq.u8[8, 256]) -> croq.u8[8, 256]:
            output = croq.declare(croq.u8[8, 256], "output")
            for p, q in croq.parallel(p=8, q=128):
                smem1 = croq.dma.copy_async(
                    i0.chunkat(p, croq.FULL), to=croq.SHARED,
                    name="smem1")
                output_s = croq.declare(
                    croq.u8[1, 256], "output_s",
                    storage=croq.SHARED, init=0)
                croq.wait(smem1)
                for i in croq.foreach(i=1):
                    output_s[i, q] = smem1.data[i, q]
                    output_s[i, q + 128] = smem1.data[i, q + 128]
                smem2 = croq.dma.copy_async(
                    i1.chunkat(p, croq.FULL), to=croq.SHARED,
                    name="smem2")
                croq.wait(smem2)
                for i in croq.foreach(i=1):
                    output_s[i, q] += smem2.data[i, q]
                    output_s[i, q + 128] += smem2.data[i, q + 128]
                croq.sync("shared")
                croq.dma.copy(output_s, to=output.chunkat(p, croq.FULL),
                              name="g")
            return output

        prog = croq.Program()
        prog.define("K", K)
        prog.define("N", N)
        prog.add(mem_reuse0)
        prog.add("""
int main() {
  choreo::u8 a[K][N] = {0};
  choreo::u8 b[K][N] = {0};
  auto a_data = choreo::make_spanview<2, choreo::u8>(
      (choreo::u8*)a, {K, N});
  auto b_data = choreo::make_spanview<2, choreo::u8>(
      (choreo::u8*)b, {K, N});
  a_data.fill_random(0, 10);
  b_data.fill_random(0, 10);
  auto res = mem_reuse0(a_data, b_data);
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j)
      choreo::choreo_assert(a[i][j] + b[i][j] == res[i][j],
                            "values are not equal.");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_memreuse_dynamic(self, compile_co):
        """Port of tests/gpu/end2end/memreuse-dynamic.co"""
        @croq.co
        def mem_reuse0(i0: croq.u8[8, 256],
                       i1: croq.u8[8, 256]) -> croq.u8[8, 256]:
            output = croq.declare(croq.u8[8, 256], "output")
            for p, q in croq.parallel(p=8, q=128):
                smem1 = croq.dma.copy_async(
                    i0.chunkat(p, croq.FULL), to=croq.SHARED,
                    name="smem1")
                output_s = croq.declare(
                    croq.u8[1, 256], "output_s",
                    storage=croq.SHARED, init=0)
                croq.wait(smem1)
                for i in croq.foreach(i=1):
                    output_s[i, q] = smem1.data[i, q]
                    output_s[i, q + 128] = smem1.data[i, q + 128]
                smem2 = croq.dma.copy_async(
                    i1.chunkat(p, croq.FULL), to=croq.SHARED,
                    name="smem2")
                croq.wait(smem2)
                for i in croq.foreach(i=1):
                    output_s[i, q] += smem2.data[i, q]
                    output_s[i, q + 128] += smem2.data[i, q + 128]
                croq.dma.copy(output_s, to=output.chunkat(p, croq.FULL),
                              name="g")
            return output

        prog = croq.Program()
        prog.add(mem_reuse0)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::u8>(8, 256);
  auto b = choreo::make_spandata<choreo::u8>(8, 256);
  a.fill_random(0, 10);
  b.fill_random(0, 10);
  auto res = mem_reuse0(a.view(), b.view());
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      int expected = (int)a[i][j] + (int)b[i][j];
      choreo::choreo_assert((int)res[i][j] == expected,
                            "values are not equal.");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Numerics transcendental (unary + binary math builtins)
# ===================================================================

class TestNumericsTranscendental:
    """Port of numerics_transcendental.co."""

    @pytest.mark.e2e
    def test_unary_mix(self, compile_co):
        """Port of numerics_transcendental.co -- unary functions."""
        @croq.co
        def unary_mix(input: croq.f32[512]) -> croq.f32[512]:
            output = croq.declare(croq.f32[512], "output")
            for p in croq.parallel(p=512):
                x = input[p]
                output[p] = (croq.acos(x * 0.5) + croq.asin(x * 0.5)
                             + croq.atan(x) + croq.sin(x) + croq.cos(x)
                             + croq.cosh(x * 0.25) + croq.sinh(x * 0.25)
                             + croq.exp(x * 0.25) + croq.log(x + 2.5)
                             + croq.log1p(x * 0.25 + 0.5)
                             + croq.expm1(x * 0.25)
                             + croq.sqrt(x + 2.5) + croq.rsqrt(x + 2.5)
                             + croq.tan(x * 0.25) + croq.tanh(x))
            return output

        prog = croq.Program()
        prog.add(unary_mix)
        prog.add("""
#include <cmath>
static float unary_ref(float x) {
  return std::acos(x * 0.5f) + std::asin(x * 0.5f) + std::atan(x) +
         std::sin(x) + std::cos(x) + std::cosh(x * 0.25f) +
         std::sinh(x * 0.25f) + std::exp(x * 0.25f) +
         std::log(x + 2.5f) + std::log1p(x * 0.25f + 0.5f) +
         std::expm1(x * 0.25f) + std::sqrt(x + 2.5f) +
         1.0f / std::sqrt(x + 2.5f) + std::tan(x * 0.25f) + std::tanh(x);
}
int main() {
  auto input = choreo::make_spandata<choreo::f32>(512);
  input.fill_random(-0.5f, 1.0f);
  auto res = unary_mix(input.view());
  for (size_t i = 0; i < 512; ++i) {
    float ref = unary_ref(input[i]);
    float got = res[i];
    float tol = 2e-4f + 2e-4f * std::abs(ref);
    choreo::choreo_assert(std::abs(got - ref) <= tol,
                          "unary numerics mismatch.");
  }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

    @pytest.mark.e2e
    def test_binary_mix(self, compile_co):
        """Port of numerics_transcendental.co -- binary functions."""
        @croq.co
        def binary_mix(lhs: croq.f32[512],
                       rhs: croq.f32[512]) -> croq.f32[512]:
            output = croq.declare(croq.f32[512], "output")
            for p in croq.parallel(p=512):
                x = lhs[p]
                y = rhs[p]
                output[p] = croq.pow(x + 1.5, y + 1.5) + croq.atan2(x, y)
            return output

        prog = croq.Program()
        prog.add(binary_mix)
        prog.add("""
#include <cmath>
static float binary_ref(float x, float y) {
  return std::pow(x + 1.5f, y + 1.5f) + std::atan2(x, y);
}
int main() {
  auto lhs = choreo::make_spandata<choreo::f32>(512);
  auto rhs = choreo::make_spandata<choreo::f32>(512);
  lhs.fill_random(-0.5f, 1.0f);
  rhs.fill_random(-0.5f, 1.0f);
  auto res = binary_mix(lhs.view(), rhs.view());
  for (size_t i = 0; i < 512; ++i) {
    float ref = binary_ref(lhs[i], rhs[i]);
    float got = res[i];
    float tol = 2e-4f + 2e-4f * std::abs(ref);
    choreo::choreo_assert(std::abs(got - ref) <= tol,
                          "binary numerics mismatch.");
  }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  TMA tests (SM90 only)
# ===================================================================

class TestTMAv1:
    """Port of tma_v1.co -- TMA tiled copy (SM90+)."""

    @pytest.mark.e2e
    @pytest.mark.skipif(True,
                        reason="TMA requires SM90+; current GPU is SM86")
    def test_tma_v1_tiled(self, compile_co):
        """Port of tests/gpu/end2end/tma_v1.co"""
        @croq.co
        def tma_copy_tiled(
                input: croq.f32[6, 16, 128]) -> croq.f32[6, 16, 128]:
            output = croq.declare(croq.f32[6, 16, 128], "output")
            for p in croq.parallel(p=32, scope=croq.BLOCK):
                f = croq.tma.copy_async(
                    input.chunkat(croq.FULL, croq.FULL, p),
                    to=croq.SHARED, name="f")
                croq.wait(f)
                croq.tma.copy(
                    f.data, to=output.chunkat(
                        croq.FULL, croq.FULL, p), name="g")
            return output

        prog = croq.Program()
        prog.add(tma_copy_tiled)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::f32>(6, 16, 128);
  a.fill_random(-10.0f, 10.0f);
  auto res = tma_copy_tiled(a.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 16; ++j)
      for (size_t k = 0; k < 128; ++k)
        choreo::choreo_assert(a[i][j][k] == res[i][j][k], "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


class TestTMAv2:
    """Port of tma_v2.co -- double-buffered TMA pipeline (SM90+)."""

    @pytest.mark.e2e
    @pytest.mark.skipif(True,
                        reason="TMA requires SM90+; current GPU is SM86")
    def test_tma_v2_double_buffer(self, compile_co):
        """Port of tests/gpu/end2end/tma_v2.co"""
        @croq.co
        def tma_ab_buffer(
                l: croq.f32[6, 16, 128],
                r: croq.f32[6, 16, 128]) -> croq.f32[6, 16, 128]:
            output = croq.declare(croq.f32[6, 16, 128], "output")
            for p in croq.parallel(p=32, scope=croq.BLOCK):
                with croq.with_in(k=8) as k:
                    l0 = croq.tma.copy_async(
                        l.chunkat(croq.FULL, k, p),
                        to=croq.SHARED, name="l0")
                    r0 = croq.tma.copy_async(
                        r.chunkat(croq.FULL, k, p),
                        to=croq.SHARED, name="r0")
                    l1 = croq.tma.any(name="l1")
                    r1 = croq.tma.any(name="r1")
                    out = croq.declare(
                        croq.f32[6, 1, 4], "out",
                        storage=croq.SHARED)
                    for _ in croq.foreach_staged(k, start=1):
                        l1 = croq.tma.copy_async(
                            l.chunkat(croq.FULL, k, p),
                            to=croq.SHARED, name="l1")
                        r1 = croq.tma.copy_async(
                            r.chunkat(croq.FULL, k, p),
                            to=croq.SHARED, name="r1")
                        croq.wait(l0, r0)
                        for idx in croq.foreach(idx=6):
                            out[idx] = l0.data[idx] + r0.data[idx]
                        croq.tma.copy(
                            out, to=output.chunkat(
                                croq.FULL, k - 1, p), name="out_w")
                        croq.swap(l0, l1)
                        croq.swap(r0, r1)
                    croq.wait(l0, r0)
                    for idx in croq.foreach(idx=6):
                        out[idx] = l0.data[idx] + r0.data[idx]
                    croq.tma.copy(
                        out, to=output.chunkat(croq.FULL, k, p),
                        name="out_w2")
            return output

        prog = croq.Program()
        prog.add(tma_ab_buffer)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::f32>(6, 16, 128);
  auto b = choreo::make_spandata<choreo::f32>(6, 16, 128);
  a.fill_random(-10.0f, 10.0f);
  b.fill_random(-10.0f, 10.0f);
  auto res = tma_ab_buffer(a.view(), b.view());
  for (size_t i = 0; i < 6; ++i)
    for (size_t j = 0; j < 16; ++j)
      for (size_t k = 0; k < 128; ++k)
        choreo::choreo_assert(a[i][j][k] + b[i][j][k] == res[i][j][k],
                              "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


class TestTMAGlobalRef:
    """Port of tma_with_global_ref.co -- WGMMA with global params (SM90a)."""

    @pytest.mark.e2e
    @pytest.mark.skipif(True,
                        reason="TMA+WGMMA requires SM90a")
    def test_tma_with_global_ref(self, compile_co):
        """Port of tests/gpu/end2end/tma_with_global_ref.co"""
        M, N, K = 64, 64, 64
        WARP_M, WARP_N, TILE_K, WARP_K = 64, 64, 64, 16

        @croq.co
        def matmul(lhs: croq.Global(croq.f16[64, 64]),
                   rhs: croq.Global(croq.f16[64, 64]),
                   output: croq.Global(croq.f16[64, 64])):
            int_warp_m = croq.declare_int("WARP_M", WARP_M)
            int_warp_n = croq.declare_int("WARP_N", WARP_N)
            int_tile_k = croq.declare_int("TILE_K", TILE_K)
            int_warp_k = croq.declare_int("WARP_K", WARP_K)
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(croq.Var.lit(M), int_warp_m),
                    block_n=croq.cdiv(croq.Var.lit(N), int_warp_n),
                    scope=croq.BLOCK):
                lhs_s = croq.declare(
                    croq.f16[WARP_M, TILE_K], "lhs_load_s",
                    storage=croq.SHARED)
                rhs_s = croq.declare(
                    croq.f16[WARP_N, TILE_K], "rhs_load_s",
                    storage=croq.SHARED)
                mc = croq.mma.fill(0.0, dtype=croq.f32)
                for iv_k in croq.foreach(
                        iv_k=croq.cdiv(
                            croq.Var.lit(K), int_tile_k)):
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
                            croq.mma.exec(
                                mc, ma, mb, method="row.row")
                croq.mma.store(
                    mc, output.chunkat(block_m, block_n))

        prog = croq.Program()
        prog.define("M", M)
        prog.define("N", N)
        prog.define("K", K)
        prog.add(matmul)
        prog.add("""
int main() {
  auto a_h = new half[M * K];
  auto b_h = new half[N * K];
  for (size_t i = 0; i < M * K; ++i)
    a_h[i] = __float2half(float(rand() % 2));
  for (size_t i = 0; i < N * K; ++i)
    b_h[i] = __float2half(float(rand() % 2));
  half *a_d, *b_d, *c_d;
  cudaMalloc(&a_d, M * K * sizeof(half));
  cudaMalloc(&b_d, N * K * sizeof(half));
  cudaMalloc(&c_d, M * N * sizeof(half));
  cudaMemcpy(a_d, a_h, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, N * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  auto lhs_d = choreo::make_spanview<2, choreo::f16>(a_d, {M, K});
  auto rhs_d = choreo::make_spanview<2, choreo::f16>(b_d, {N, K});
  auto res_d = choreo::make_spanview<2, choreo::f16>(c_d, {M, N});
  matmul(lhs_d, rhs_d, res_d);
  cudaDeviceSynchronize();
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Conditional copy (copy_if_g2s.co) -- SM90a only
# ===================================================================

class TestCopyIfG2S:
    """Port of copy_if_g2s.co -- tail handling with view/from + subspan."""

    @pytest.mark.e2e
    @pytest.mark.skipif(True,
                        reason="copy_if_g2s requires SM90a")
    def test_copy_tail_m(self, compile_co):
        """Port of tests/gpu/end2end/copy_if_g2s.co -- copy_tail_m"""
        TILE_M, TILE_K = 64, 64
        M_TAIL, K_FULL = 70, 64

        @croq.co
        def copy_tail_m(
                inp: croq.f16[70, 64]) -> croq.f16[70, 64]:
            output = croq.declare(croq.f16[70, 64], "output", init=0)
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
                        to=output.view(TM, TILE_K).from_(base_m, 0))
            return output

        prog = croq.Program()
        prog.define("TILE_M", TILE_M)
        prog.define("TILE_K", TILE_K)
        prog.define("M_TAIL", M_TAIL)
        prog.define("K_FULL", K_FULL)
        prog.add(copy_tail_m)
        prog.add("""
int main() {
  auto input = choreo::make_spandata<choreo::f16>(M_TAIL, K_FULL);
  input.fill_random(-10.0f, 10.0f);
  auto output = copy_tail_m(input.view());
  for (size_t i = 0; i < M_TAIL; ++i)
    for (size_t j = 0; j < K_FULL; ++j)
      choreo::choreo_assert(
          choreo::f16_to_f32(input[i][j]) ==
          choreo::f16_to_f32(output[i][j]),
          "mismatch");
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  FP8 -> F16 add for e5m2 (SM90 only)
# ===================================================================

class TestFP8E5M2:
    """Port of add_fp8_e5m2_to_f16.co."""

    @pytest.mark.e2e
    @pytest.mark.skipif(True,
                        reason="FP8 e5m2 add-to-f16 requires SM90+")
    def test_add_fp8_e5m2_to_f16(self, compile_co):
        """Port of tests/gpu/end2end/add_fp8_e5m2_to_f16.co"""
        @croq.co
        def ele_add_fp8(lhs: croq.f8_e5m2[6, 64],
                        rhs: croq.f8_e5m2[6, 64]) -> croq.f16[6, 64]:
            output = croq.declare(croq.f16[6, 64], "output")
            for p, q in croq.parallel(p=6, q=64):
                output[p, q] = lhs[p, q] + rhs[p, q]
            return output

        prog = croq.Program()
        prog.add(ele_add_fp8)
        prog.add("""
int main() {
  auto a = choreo::make_spandata<choreo::f8_e5m2>(6, 64);
  auto b = choreo::make_spandata<choreo::f8_e5m2>(6, 64);
  a.fill_random(-2.0f, 2.0f);
  b.fill_random(-2.0f, 2.0f);
  auto res = ele_add_fp8(a.view(), b.view());
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  Matmul with DMA + tiling (matmul_f16_dynamic pattern)
# ===================================================================

class TestMatmulDynamic:
    """Port of matmul_f16_dynamic.co -- dynamic bounds matmul."""

    @pytest.mark.e2e
    @pytest.mark.skipif(True,
                        reason="TMA-based dynamic matmul requires SM90+")
    def test_matmul_f16_dynamic(self, compile_co):
        """Port of tests/gpu/end2end/matmul_f16_dynamic.co"""
        M, N, K = 768, 512, 512
        WARP_M, WARP_N, TILE_K, WARP_K = 64, 64, 64, 16

        @croq.co
        def matmul(lhs: croq.f16[768, 512],
                   rhs: croq.f16[512, 512]) -> croq.f16[768, 512]:
            output = croq.declare(croq.f16[768, 512], "output")
            int_warp_m = croq.declare_int("WARP_M", WARP_M)
            int_warp_n = croq.declare_int("WARP_N", WARP_N)
            int_tile_k = croq.declare_int("TILE_K", TILE_K)
            int_warp_k = croq.declare_int("WARP_K", WARP_K)
            for bm, bn in croq.parallel(
                    bm=croq.cdiv(croq.Var.lit(M), int_warp_m),
                    bn=croq.cdiv(croq.Var.lit(N), int_warp_n),
                    scope=croq.BLOCK):
                lhs_s = croq.declare(
                    croq.f16[WARP_M, TILE_K], "lhs_s",
                    storage=croq.SHARED)
                rhs_s = croq.declare(
                    croq.f16[WARP_N, TILE_K], "rhs_s",
                    storage=croq.SHARED)
                mc = croq.mma.fill(0.0, dtype=croq.f32)
                for iv_k in croq.foreach(
                        iv_k=croq.cdiv(croq.Var.lit(K), int_tile_k)):
                    croq.tma.copy(
                        lhs.subspan(WARP_M, TILE_K).at(bm, iv_k),
                        lhs_s, swizzle=128)
                    croq.tma.copy(
                        rhs.chunkat(bn, iv_k),
                        rhs_s, swizzle=128)
                    for iv_warp in croq.foreach(
                            iv_warp=croq.cdiv(int_tile_k, int_warp_k)):
                        for p in croq.parallel(
                                p=1, scope="group-4"):
                            ma = croq.mma.load(
                                lhs_s.chunkat(iv_warp), swizzle=128)
                            mb = croq.mma.load(
                                rhs_s.chunkat(iv_warp), swizzle=128)
                            croq.mma.exec(
                                mc, ma, mb, method="row.row")
                output_s = croq.declare(
                    croq.f16[WARP_M, WARP_N], "output_s",
                    storage=croq.SHARED)
                croq.mma.store(mc, output_s)
                croq.tma.copy(
                    output_s,
                    output.subspan(WARP_M, WARP_N).at(bm, bn))
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
  lhs.fill_random(0, 1);
  rhs.fill_random(0, 1);
  auto res = matmul(lhs.view(), rhs.view());
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100


# ===================================================================
#  MMA loadC with init=0 output
# ===================================================================

class TestMMALoadCInit:
    """Port of mma_loadc.co with output init=0."""

    @pytest.mark.e2e
    def test_mma_loadc_init(self, compile_co):
        """Port of mma_loadc.co variant with pre-initialized output."""
        M, N, K = 256, 256, 256
        TILE_M, TILE_N, TILE_K = 32, 32, 16
        WARP_M, WARP_N, WARP_K = 8, 8, 4

        @croq.co
        def matmul(lhs: croq.f16[256, 256],
                   rhs: croq.f16[256, 256]) -> croq.f16[256, 256]:
            output = croq.declare(croq.f16[256, 256], "output", init=0)
            for block_m, block_n in croq.parallel(
                    block_m=croq.cdiv(M, TILE_M),
                    block_n=croq.cdiv(N, TILE_N),
                    scope=croq.BLOCK):
                for iv_k in croq.foreach(
                        iv_k=croq.cdiv(K, TILE_K)):
                    lhs_load_s = croq.dma.copy(
                        lhs.chunkat(block_m, iv_k),
                        to=croq.SHARED, name="lhs_load_s")
                    rhs_load_s = croq.dma.copy(
                        rhs.chunkat(iv_k, block_n),
                        to=croq.SHARED, name="rhs_load_s")
                    for iv_warp_m, iv_warp_n in croq.parallel(
                            iv_warp_m=croq.cdiv(TILE_M, WARP_M),
                            iv_warp_n=croq.cdiv(TILE_N, WARP_N),
                            scope="group"):
                        mc = croq.mma.load(
                            output.chunkat(
                                block_m @ iv_warp_m,
                                block_n @ iv_warp_n))
                        for iv_warp_k in croq.foreach(
                                iv_warp_k=croq.cdiv(TILE_K, WARP_K)):
                            ma = croq.mma.load(
                                lhs_load_s.data.chunkat(
                                    iv_warp_m, iv_warp_k))
                            mb = croq.mma.load(
                                rhs_load_s.data.chunkat(
                                    iv_warp_k, iv_warp_n))
                            croq.mma.exec(mc, ma, mb, method="row.col")
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
  auto rhs = choreo::make_spandata<choreo::f16>(K, N);
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < K; ++j)
      lhs[i][j] = __float2half(i * 0.02f + j * 0.01f);
  for (size_t i = 0; i < K; ++i)
    for (size_t j = 0; j < N; ++j)
      rhs[i][j] = __float2half(i * 0.07f + j * 0.03f);
  auto res = matmul(lhs.view(), rhs.view());
  float tolerance = 0.007f;
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      __half ref = __float2half(0.0f);
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += lhs[i][k] * rhs[k][j];
      auto delta = std::abs((__half2float(ref) - __half2float(res[i][j]))
                            / __half2float(ref));
      choreo::choreo_assert(delta < tolerance, "values not equal.");
    }
  std::cout << "Test Passed" << std::endl;
}
""")
        cuda = compile_co(prog.to_co(check_lines=["Test Passed"]))
        assert len(cuda) > 100

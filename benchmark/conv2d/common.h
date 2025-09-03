float* conv2d_cpu(
    const float* input,   // [N, Cin, H, W]
    const float* weight,  // [Cout, Cin, Kh, Kw]
    int N, int Cin, int H, int W,
    int Cout, int Kh, int Kw,
    int stride, int padding, int dilation) {
    int Ho = (H + 2 * padding - dilation * (Kh - 1) - 1) / stride + 1;
    int Wo = (W + 2 * padding - dilation * (Kw - 1) - 1) / stride + 1;

    float* output = new float[N * Cout * Ho * Wo];

    for (int n = 0; n < N; n++)
        for (int co = 0; co < Cout; co++)
            for (int ho = 0; ho < Ho; ho++)
                for (int wo = 0; wo < Wo; wo++) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < Cin; ci++) {
                        for (int kh = 0; kh < Kh; kh++) {
                            for (int kw = 0; kw < Kw; kw++) {
                                int h_in = ho * stride - padding + kh * dilation;
                                int w_in = wo * stride - padding + kw * dilation;
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    float in_val = input[((n * Cin + ci) * H + h_in) * W + w_in];
                                    float w_val  = weight[((co * Cin + ci) * Kh + kh) * Kw + kw];
                                    sum += in_val * w_val;
                                }
                            }
                        }
                    }
                    output[((n * Cout + co) * Ho + ho) * Wo + wo] = sum;
                }

    return output;
}

#if 1

__co_device__ inline float v_dot(float *lhs, float *rhs, int elem_count) {
  int vector_length = sizeof(__vector float) / sizeof(float);
  int vector_count = elem_count / vector_length;
  
  float result = 0;

  if (vector_count > 0) {
    // Use TCLE leaptr for vectorized memory access (float4 = 4*4=16 bytes)
    auto lhs_ptr = tcle::leaptr<__vector float, 1>(lhs, sizeof(__vector float));
    auto rhs_ptr = tcle::leaptr<__vector float, 1>(rhs, sizeof(__vector float));
    __vector float vresult = (__vector float)0;
    for (int i = 0; i < vector_count; ++i) {
      __vector float lhs_vec = lhs_ptr.load<0>();
      __vector float rhs_vec = rhs_ptr.load<0>();
      vresult += lhs_vec * rhs_vec;
    }
    for (int i = 0; i < vector_length; ++i)
      result += vresult[i];
  }

  // Handle remaining elements
  int offset = vector_count * vector_length;
  for (int i = 0; i + offset < elem_count; ++i)
    result += lhs[offset + i] * rhs[offset + i];

  return result;
}
#else
__co_device__ inline float v_dot(float *lhs, float *rhs, int elem_count) {
  // Use TCLE leaptr for vectorized memory access (float4 = 4*4=16 bytes)
  int vector4_length = sizeof(__vector4 float) / sizeof(float);
  int vector4_count = elem_count / vector4_length;

  __vector4 float v4result = (__vector4 float)0;
  for (int i = 0; i < vector4_count; ++i) {
    v4result += *((__vector4 float*)lhs + i) * *((__vector4 float*)rhs + i);
  }
  
  float result = 0;

  if (vector4_count > 0)
    for (int i = 0; i < vector4_length; ++i)
      result += v4result[i];

  int vector_length = sizeof(__vector float) / sizeof(float);
  int vector_count = (elem_count % vector4_length) / vector_length;

  if (vector_count > 0) {
    __vector float vresult = (__vector float)0;
    __vector float * lhs_p = (__vector float * )(lhs + vector4_count * vector4_length);
    __vector float * rhs_p = (__vector float * )(rhs + vector4_count * vector4_length);
    for (int i = 0; i < vector_count; ++i) {
      vresult += *((__vector float*)lhs_p + i) * *((__vector float*)rhs_p + i);
    }
    // TODO: sip exception here
    // done: due to vector alignment!
    for (int i = 0; i < vector_length; ++i)
      result += vresult[i];
  }

  // Handle remaining elements
  int offset = vector4_count * vector4_length + vector_count * vector_length;
  for (int i = 0; i + offset < elem_count; ++i)
    result += lhs[offset + i] * rhs[offset + i];

  return result;
}
#endif

__co_device__ extern "C" void k_matmul(float * lhs, float * rhs, float * out, int m, int n, int k) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      float res = 0;
#ifndef NO_VRED
      res = v_dot(&lhs[i*k], &rhs[j*k], k);
#else
      for (int z = 0; z < k; ++z)
        res += lhs[i*k+z]*rhs[j*k+z];
#endif
      out[i*n + j] += res;
   }
}


void test(std::string name, std::function<spanned_data<float, 4>(spanned_view<float, 4UL>, spanned_view<float, 4UL>, int, int, int)> f, std::vector<int> ids, std::vector<int> kds, int stride, int padding, int dilation) {
  auto input = choreo::make_spandata<choreo::f32>(ids[0], ids[1], ids[2], ids[3]);
  // input.fill(1.0f);
  input.fill_random(-1.0f, 1.0f);
  
  auto kernel = choreo::make_spandata<choreo::f32>(kds[0], kds[1], kds[2], kds[3]);
  // kernel.fill(1.0f);
  kernel.fill_random(-1.0f, 1.0f);

  auto start = std::chrono::high_resolution_clock::now();
  auto res = f(input.view(), kernel.view(), stride, padding, dilation);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

#ifdef __CHECK__
  auto res_cpu = conv2d_cpu(input.data(), kernel.data(), ids[0], ids[1], ids[2], ids[3], kds[0], kds[2], kds[3], stride, padding, dilation);

  auto nearlyEqual = [](float a, float b,
                 float absEps = 1e-2f, float relEps = 1e-1f) {
    float diff = std::fabs(a - b);
    if (diff <= absEps) return true;
    return diff <= relEps * std::max(std::fabs(a), std::fabs(b));
  };

  size_t idx = 0;
  for (int i = 0; i < res.shape()[0]; ++i)
    for (int j = 0; j < res.shape()[1]; ++j)
      for (int k = 0; k < res.shape()[2]; ++k)
        for (int m = 0; m < res.shape()[3]; ++m, ++idx) {
          auto expect = res_cpu[idx];
          auto actual = res[i][j][k][m];
          if (!nearlyEqual(actual, expect)) {
            std::cerr << "[" << i << "," << j << "," << k << "," << m << "]:\n";
            std::cerr << "\t" << "expect: " << expect << "\n";
            std::cerr << "\t" << "actual: " << actual << "\n";
            choreo::choreo_assert(false, "error");
          }
        }
  delete[] res_cpu;
  std::cout << name << " is PASS.\n";
#endif
  std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
}
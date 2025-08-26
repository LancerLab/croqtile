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

void test(std::string name, std::function<spanned_data<float, 4>(spanned_view<float, 4UL>, spanned_view<float, 4UL>, int, int, int)> f, std::vector<int> ids, std::vector<int> kds, int stride, int padding, int dilation) {
  auto input = choreo::make_spandata<choreo::f32>(ids[0], ids[1], ids[2], ids[3]);
  // input.fill(1.0f);
  input.fill_random(-1.0f, 1.0f);

  auto kernel = choreo::make_spandata<choreo::f32>(kds[0], kds[1], kds[2], kds[3]);
  // kernel.fill(1.0f);
  kernel.fill_random(-1.0f, 1.0f);

  auto res = f(input.view(), kernel.view(), stride, padding, dilation);
  auto res_cpu = conv2d_cpu(input.data(), kernel.data(), ids[0], ids[1], ids[2], ids[3], kds[0], kds[2], kds[3], stride, padding, dilation);

  auto nearlyEqual = [](float a, float b,
                 float absEps = 1e-3f, float relEps = 1e-2f) {
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
}
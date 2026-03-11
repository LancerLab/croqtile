#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TopkSoftmaxResult {
  std::vector<float> weights;
  std::vector<int32_t> indices;
};

struct TestCase {
  std::string name;
  int num_tokens;
  int num_experts;
  int topk;
  bool renormalize;
  float moe_softcapping;
  std::vector<float> gating_output;
  std::vector<float> correction_bias;
  std::vector<float> expected_weights;
  std::vector<int32_t> expected_indices;
};

TopkSoftmaxResult topkSoftmaxCpu(const std::vector<float>& gating_output,
                                 int num_tokens, int num_experts, int topk,
                                 bool renormalize, float moe_softcapping,
                                 const std::vector<float>* correction_bias) {
  if (num_tokens <= 0 || num_experts <= 0) {
    throw std::invalid_argument("num_tokens and num_experts must be positive");
  }
  if (topk < 1 || topk > num_experts) {
    throw std::invalid_argument("topk must be in [1, num_experts]");
  }
  if (static_cast<int>(gating_output.size()) != num_tokens * num_experts) {
    throw std::invalid_argument("gating_output size does not match shape");
  }
  if (correction_bias != nullptr &&
      static_cast<int>(correction_bias->size()) != num_experts) {
    throw std::invalid_argument("correction_bias must be [num_experts]");
  }

  TopkSoftmaxResult result;
  result.weights.resize(static_cast<size_t>(num_tokens) * topk);
  result.indices.resize(static_cast<size_t>(num_tokens) * topk);

  std::vector<float> logits(static_cast<size_t>(num_experts));
  std::vector<std::pair<float, int32_t>> ranked(static_cast<size_t>(num_experts));

  for (int token = 0; token < num_tokens; ++token) {
    float row_max = 0.0f;
    for (int expert = 0; expert < num_experts; ++expert) {
      float value = gating_output[static_cast<size_t>(token) * num_experts + expert];
      if (moe_softcapping != 0.0f) {
        value = std::tanh(value / moe_softcapping) * moe_softcapping;
      }
      if (correction_bias != nullptr) {
        value += (*correction_bias)[expert];
      }
      logits[expert] = value;
      if (expert == 0 || value > row_max) row_max = value;
    }

    float row_sum = 0.0f;
    for (int expert = 0; expert < num_experts; ++expert) {
      float prob = std::exp(logits[expert] - row_max);
      logits[expert] = prob;
      row_sum += prob;
    }

    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    for (int expert = 0; expert < num_experts; ++expert) {
      logits[expert] *= inv_sum;
      ranked[expert] = {logits[expert], static_cast<int32_t>(expert)};
    }

    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& lhs, const auto& rhs) {
                       return lhs.first > rhs.first;
                     });

    float selected_sum = 0.0f;
    for (int i = 0; i < topk; ++i) {
      result.weights[static_cast<size_t>(token) * topk + i] = ranked[i].first;
      result.indices[static_cast<size_t>(token) * topk + i] = ranked[i].second;
      selected_sum += ranked[i].first;
    }

    if (renormalize && selected_sum > 0.0f) {
      const float inv_selected_sum = 1.0f / selected_sum;
      for (int i = 0; i < topk; ++i) {
        result.weights[static_cast<size_t>(token) * topk + i] *= inv_selected_sum;
      }
    }
  }

  return result;
}

bool nearlyEqual(float lhs, float rhs, float atol = 1e-6f, float rtol = 1e-5f) {
  const float diff = std::fabs(lhs - rhs);
  const float scale = std::max(std::fabs(lhs), std::fabs(rhs));
  return diff <= atol + rtol * scale;
}

void verifyCase(const TestCase& test_case) {
  const std::vector<float>* bias = test_case.correction_bias.empty()
                                       ? nullptr
                                       : &test_case.correction_bias;
  TopkSoftmaxResult got =
      topkSoftmaxCpu(test_case.gating_output, test_case.num_tokens,
                     test_case.num_experts, test_case.topk,
                     test_case.renormalize, test_case.moe_softcapping, bias);

  if (got.weights.size() != test_case.expected_weights.size() ||
      got.indices.size() != test_case.expected_indices.size()) {
    throw std::runtime_error("reference output shape mismatch");
  }

  for (size_t i = 0; i < got.indices.size(); ++i) {
    if (got.indices[i] != test_case.expected_indices[i]) {
      std::cerr << test_case.name << ": index mismatch at flat position " << i
                << ", expected " << test_case.expected_indices[i] << ", got "
                << got.indices[i] << "\n";
      throw std::runtime_error("index comparison failed");
    }
  }

  for (size_t i = 0; i < got.weights.size(); ++i) {
    if (!nearlyEqual(got.weights[i], test_case.expected_weights[i])) {
      std::cerr << std::fixed << std::setprecision(10)
                << test_case.name << ": weight mismatch at flat position " << i
                << ", expected " << test_case.expected_weights[i] << ", got "
                << got.weights[i] << "\n";
      throw std::runtime_error("weight comparison failed");
    }
  }

  std::cout << "Verified " << test_case.name << "\n";
}

std::vector<TestCase> buildTestCases() {
  return {
      {
          "basic",
          2,
          4,
          2,
          false,
          0.0f,
          {
              1.0f, 0.5f, -0.25f, 0.5f,
              2.0f, 2.0f, 1.0f, -3.0f,
          },
          {},
          {
              0.4000694454f, 0.2426543832f,
              0.4211204946f, 0.4211204946f,
          },
          {
              0, 1,
              0, 1,
          },
      },
      {
          "renorm_bias",
          2,
          5,
          3,
          true,
          0.0f,
          {
              0.25f, -0.5f, 1.5f, 1.5f, -1.0f,
              3.0f, -2.0f, 0.0f, 0.25f, 0.25f,
          },
          {
              0.0f, 0.5f, -0.25f, 0.25f, 0.0f,
          },
          {
              0.5465493798f, 0.3314989507f, 0.1219516546f,
              0.8725905418f, 0.0716265962f, 0.0557828471f,
          },
          {
              3, 2, 0,
              0, 3, 4,
          },
      },
      {
          "softcap",
          2,
          4,
          2,
          true,
          2.5f,
          {
              4.0f, -1.0f, 0.5f, 2.0f,
              -3.5f, -3.5f, 7.0f, 1.0f,
          },
          {
              0.25f, -0.5f, 0.0f, 0.125f,
          },
          {
              0.6833217144f, 0.3166782856f,
              0.8032459021f, 0.1967540681f,
          },
          {
              0, 3,
              2, 3,
          },
      },
  };
}

} // namespace

int main() {
  try {
    for (const auto& test_case : buildTestCases()) {
      verifyCase(test_case);
    }
  } catch (const std::exception& ex) {
    std::cerr << "Verification failed: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "Test Passed\n";
  return EXIT_SUCCESS;
}

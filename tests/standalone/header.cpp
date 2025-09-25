#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>

#include "runtime/choreo.h"

// Test the compilation
void foo(choreo::f32* a, choreo::f16* b) {
  choreo::spanned_view<choreo::f32, 2> spanned_a(a, {1, 2});
  auto spanned_d = choreo::copy_as_spanned<3>(b, {4, 7, 8});
}

template <typename T, size_t N>
void bar(const choreo::spanned_view<T, N>& b) {}

void foobar(float* a) { bar(choreo::make_spanview<2>(a, {1, 2})); }

void fooboo() {
  float a[2][3];
  bar(choreo::make_spanview(a));
}

template <size_t M, size_t N, size_t K, typename T>
choreo::spanned_data<T, 3> pass_and_return(T* l) {
  return choreo::copy_as_spanned<3>(l, {M, N, K});
}

template <int M, int N, int K, typename T>
void generate_elements(T* d) {
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
      for (size_t k = 0; k < K; ++k) {
        size_t index = i * (N * K) + j * K + k;
        d[index] = i + j + k;
      }
}

class HeaderTest : public ::testing::Test {
protected:
  std::string temp_filename;
  int tempfile_desc;

  virtual void SetUp() {
    char temp_template[] = "/tmp/optiontest-XXXXXX";
    tempfile_desc = mkstemp(temp_template);
    if (tempfile_desc == -1) {
      perror("Error creating temporary file");
      exit(EXIT_FAILURE);
    }
    temp_filename = temp_template; // Update filename
  }

  virtual void TearDown() {
    close(tempfile_desc);          // Close file descriptor
    remove(temp_filename.c_str()); // Delete file
  }

  void createFileWithContent(const std::string& filename,
                             const std::string& content) {
    std::ofstream out(filename);
    ASSERT_TRUE(out.is_open());
    out << content;
    out.close();
  }
};

// Test the execution
TEST_F(HeaderTest, SpanViewTest) {
  choreo::f32 in0[100];
  choreo::f32 in1[100];
  choreo::f32 out[100];
  // fill in0 with even numbers
  float value = 0.0f;
  std::generate(in0, in0 + 100, [&value]() {
    auto current = value;
    value += 2.0f;
    return current;
  });
  // fill in1 with 1.0f
  std::fill_n(in1, 100, 1.0f);
  // fill out with odd numbers
  value = 1.0f;
  std::generate(out, out + 100, [&value]() {
    auto current = value;
    value += 2.0f;
    return current;
  });

  auto sv0 = choreo::make_spanview<2>(in0, {10, 10});
  auto sv1 = choreo::make_spanview<3>(in1, {10, 10, 1});
  auto sv2 = choreo::make_spanview<4>(out, {1, 10, 10, 1});

  ASSERT_TRUE(sv0.rank() == 2);
  ASSERT_TRUE(sv1.rank() == 3);
  ASSERT_TRUE(sv2.rank() == 4);
  ASSERT_FALSE(sv0.shape() == sv1.shape());
  ASSERT_TRUE(sv0.element_count() == sv1.element_count());
  ASSERT_TRUE(sv0.element_count() == sv2.element_count());

  std::flush(std::cout);
  auto s = sv0.shape();
  for (size_t i = 0; i < s[0]; ++i)
    for (size_t j = 0; j < s[1]; ++j) {
      std::flush(std::cout);
      ASSERT_TRUE(sv0[i][j] + sv1[i][j][0] == sv2[0][i][j][0]);
    }
  std::flush(std::cout);
}

TEST_F(HeaderTest, SpanDataReturnTest) {
  choreo::s32 in0[128];
  // fill in0 with even numbers
  float value = 0;
  std::generate(in0, in0 + 128, [&value]() {
    auto current = value;
    value += 3;
    return current;
  });

  auto sd = pass_and_return<32, 2, 2>((choreo::s32*)in0);

  ASSERT_TRUE(sd.rank() == 3);
  ASSERT_TRUE(sd.shape()[0] == 32);
  ASSERT_TRUE(sd.shape()[1] == 2);
  ASSERT_TRUE(sd.shape()[2] == 2);

  auto s = sd.shape();
  for (size_t i = 0; i < s[0]; ++i)
    for (size_t j = 0; j < s[1]; ++j)
      for (size_t k = 0; k < s[2]; ++k)
        ASSERT_TRUE(sd[i][j][k] == in0[i * s[1] * s[2] + j * s[2] + k]);
}

TEST_F(HeaderTest, SpanDataCopyTest) {
  auto in = choreo::make_spandata<choreo::s32, 4>({32, 8, 7, 1});

  generate_elements<32, 8, 7>(in.data());

  ASSERT_TRUE(in.rank() == 4);
  ASSERT_TRUE(in.shape()[0] == 32);
  ASSERT_TRUE(in.shape()[1] == 8);
  ASSERT_TRUE(in.shape()[2] == 7);
  ASSERT_TRUE(in.shape()[3] == 1);

  auto s = in.shape();
  for (size_t i = 0; i < s[0]; ++i)
    for (size_t j = 0; j < s[1]; ++j)
      for (size_t k = 0; k < s[2]; ++k)
        for (size_t l = 0; l < s[3]; ++l)
          ASSERT_TRUE((size_t)in[i][j][k][l] == i + j + k + l);
}

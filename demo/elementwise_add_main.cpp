// clang-format off
// topsfc main.cc -I/opt/tops/include -L/opt/tops/lib -ltopsrt -o a.out
// ./a.out foo.bin
// clang-format on

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "tops/tops_ext.h"      // topsruntime
#include "tops/tops_runtime.h"  // topsruntime

#define CHECK(call)                                                      \
  {                                                                      \
    const topsError_t error = call;                                      \
    if (error != topsSuccess) {                                          \
      std::cerr << "Failed to call topsruntime api: " << __FILE__ << ":" \
                << __LINE__ << ", error=" << error << std::endl;         \
      return 1;                                                          \
    }                                                                    \
  }

template <typename T>
std::vector<uint8_t> ToU8V(const std::vector<T> &v) {
  auto u8_data = reinterpret_cast<const uint8_t *>(v.data());
  return std::vector<uint8_t>(u8_data, u8_data + sizeof(T) * v.size());
}

std::vector<int32_t> ToI32V(const std::vector<uint8_t> &v) {
  auto i32_data = reinterpret_cast<const int32_t *>(v.data());
  return std::vector<int32_t>(i32_data, i32_data + v.size() / sizeof(int32_t));
}

// UTILS to get cardinal
size_t calcCardinal(const std::vector<int64_t>& vec) {
    size_t result = 1;
    for (const auto& elem : vec) {
        result *= static_cast<size_t>(elem);
    }
    return result;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Specify the binary file.\n";
    return 1;
  }

  std::vector<char> binary;
  // Read bin file and store to a vector
  std::ifstream ifs(argv[1], std::ios::binary);
  std::copy(std::istreambuf_iterator<char>(ifs),
            std::istreambuf_iterator<char>(), std::back_inserter(binary));
  ifs.close();

  // Create executable
  topsExecutable_t executable = nullptr;
  CHECK(topsCreateExecutable(&executable, binary.data(), binary.size()));
  topsStream_t stream = nullptr;
  CHECK(topsStreamCreate(&stream));

  // Init input & Prepare output buffer
  std::vector<int64_t> flatten_dims = {6, 17, 128};
  std::vector<size_t> flatten_ranks{1, 1};
  size_t size_in_count = calcCardinal(flatten_dims);
  size_t size_in_bytes = 4*size_in_count;

  std::vector<int32_t> input_data0 = std::vector<int32_t>(size_in_count, 1);
  std::vector<uint8_t> host_mem0 = ToU8V(input_data0);
  void *device_mem0 = nullptr;
  CHECK(topsMalloc(&device_mem0, size_in_bytes));
  CHECK(topsMemcpy(device_mem0, reinterpret_cast<void *>(host_mem0.data()),
                   size_in_bytes, topsMemcpyHostToDevice));

  std::vector<int32_t> input_data1 = std::vector<int32_t>(size_in_count, 2);
  std::vector<uint8_t> host_mem1 = ToU8V(input_data1);
  void *device_mem1 = nullptr;
  CHECK(topsMalloc(&device_mem1, size_in_bytes));
  CHECK(topsMemcpy(device_mem1, reinterpret_cast<void *>(host_mem1.data()),
                   size_in_bytes, topsMemcpyHostToDevice));

  void *device_mem2 = nullptr;
  CHECK(topsMalloc(&device_mem2, size_in_bytes));

  // Launch
  void *device_inputs[] = {device_mem0, device_mem1};
  void *device_outputs[] = {device_mem2};
  CHECK(topsLaunchExecutableV2(
      executable, nullptr, device_inputs,
      sizeof(device_inputs) / sizeof(void *), &flatten_dims[0],
      &flatten_ranks[0], device_outputs,
      sizeof(device_outputs) / sizeof(void *), stream));
  CHECK(topsStreamSynchronize(stream));

  // Copy output data from device to host
  std::vector<uint8_t> host_mem2(size_in_bytes, 0);
  CHECK(topsMemcpy(reinterpret_cast<void *>(host_mem2.data()), device_mem2,
                   size_in_bytes, topsMemcpyDeviceToHost));

  // Free
  topsFree(device_mem0);
  topsFree(device_mem1);
  topsFree(device_mem2);
  topsStreamDestroy(stream);
  topsDestroyExecutable(executable);

  auto output_data = ToI32V(host_mem2);
  for (int i = 0; i< flatten_dims[0]; ++i) {
    for (int j = 0; j< flatten_dims[1]; ++j) {
      for (int k = 0; k< flatten_dims[2]; ++k) {
        int index = k + j*flatten_dims[2] + i*flatten_dims[1];
        assert(input_data0[index]+input_data1[index] == output_data[index] && "unequal results");
        std::cout << "calc at [" << i << ", " << j << ", " << k << "] : " << input_data0[index] << " + " << input_data1[index]
                  << " = " << output_data[index] << std::endl;
      }
    }
  }
  std::cout << "test passed" << std::endl;
  return 0;
}

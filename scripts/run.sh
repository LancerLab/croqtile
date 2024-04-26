#!/usr/bin/env bash

set -x
set -e

# This the the choreo generated bash script to compile factor code

# step 1: write the kernel source code into a temp file
kernel_src=/tmp/1714104412747419274_1___choreo_ele_add_micro_kernel.cpp
cat <<EOF > ${kernel_src}

// kernel program
extern "C" void kernel(int * a, int * b, int * c, int n) {
  for (int i = 0; i < n; ++i)
    c[i] = a[i] + b[i];
}

EOF

# step 2: write the factor source code into a temp file
factor_src=/tmp/1714104412747422279_1___choreo_ele_add_factor.cpp
cat <<EOF > ${factor_src}

#include <vector>

#include "gcu/factor/factor.h"

using namespace factor;
void __choreo_ele_add() {
  include_("/tmp/1714104412747419274_1___choreo_ele_add_micro_kernel.cpp");
  auto lhs_type = DRAMType(IntType(32), {6, 17, 128});
  auto rhs_type = DRAMType(IntType(32), {6, 17, 128});
  auto output_type = DRAMType(IntType(32), {6, 17, 128});

  // choreo-factor dataflow function
  D(main_)({lhs_type, rhs_type}, [&](auto args) {
    auto output = alloc_(lhs_type);
    Dim3 grid_dim(1);
    Dim3 block_dim(6);
    Value stream = alloc_stream_();
    create_stream_(stream);
    auto ts = launch_kernel_("__choreo_ele_add", grid_dim, block_dim, stream, {args[0], args[1]}, {output});
    destroy_stream_(stream);
    dealloc_stream_(stream);
    return std::vector<Value>{output};
  }); // end of choreo-factor dataflow program

  
  D(func_)("__choreo_ele_add", {lhs_type, rhs_type}, {output_type}, [&](auto args, auto results) {
    auto thread_id = thread_id_();
    for_(0, 17, 1, [&](auto x) {
      for_(0, 8, 1, [&](auto y) {
        auto lhs_load_buffer = alloc_(L1Type(IntType(32),{1, 1, 16}));
        auto lhs_load = alloc_dma_(SDMAType());
        async_load_(lhs_load, args[0], lhs_load_buffer, {thread_id,x,y});
        auto rhs_load_buffer = alloc_(L1Type(IntType(32),{1, 1, 16}));
        auto rhs_load = alloc_dma_(SDMAType());
        async_load_(rhs_load, args[1], rhs_load_buffer, {thread_id,x,y});
        wait_dma_(lhs_load);
        wait_dma_(rhs_load);
        auto l1_out = alloc_(L1Type(IntType(32),{1, 1, 16}));
        call_("kernel", {lhs_load_buffer.addr_(),rhs_load_buffer.addr_(),l1_out.addr_(),16});
        auto out_store = alloc_dma_(SDMAType());
        async_store_(out_store, l1_out, results[0], {thread_id,x,y});
        wait_dma_(out_store);
      }); // end of choreo-foreach block
    }); // end of choreo-foreach block
  }); // end of choreo-factor kernel function
}

MODULE_REGISTER("module__choreo_ele_add", __choreo_ele_add);
EOF

# step 3: compile factor code into a binary
factor_bin=\"/tmp/1714104412747424387_1___choreo_ele_add_factor.fb\"

# step 4: generate the host source
host_src=/tmp/1714104412747426234_1___choreo_ele_add_host.cpp
cat <<EOF > ${host_src}
#include "choreo.h"



// device program

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

// dependant on the topsruntime
#include "tops/tops_ext.h"
#include "tops/tops_runtime.h"

// choreo header
#include "choreo.h"

using namespace choreo;

namespace {

// Nasty data copy. Need optimization together with factor
template <typename T, int Rank>
static inline std::vector<uint8_t> ToFactorData(const spanned<T, Rank> &v) {
  auto u8_data = reinterpret_cast<const uint8_t *>(v.data);
    return std::vector<uint8_t>(u8_data, v.bytes());
}

template <int N, typename T>
static inline spanned<N, T> ToSpanned(const std::vector<uint8_t> &v) {
  return make_spanned<N, T, uint8_t>(v.data());
}

// must be true
#define CHECK(a) choreo_assert(a, ##a, __FILE__, __LINE__)

} // end anonymous namespace
choreo::spanned<choreo::s32, 3> ele_add(const choreo::spanned<choreo::s32, 3> & sp0, const choreo::spanned<choreo::s32, 3> & sp1);
#include "choreo.h"



// device program
choreo::spanned<choreo::s32, 3> ele_add(const choreo::spanned<choreo::s32, 3> & sp2, const choreo::spanned<choreo::s32, 3> & sp3) {
std::vector<char> binary;
  // Read bin file and store to a vector
  std::ifstream ifs("/tmp/1714104412747424387_1___choreo_ele_add_factor.fb", std::ios::binary);
  std::copy(std::istreambuf_iterator<char>(ifs),
            std::istreambuf_iterator<char>(), std::back_inserter(binary));
  ifs.close();

  // Create executable
  topsExecutable_t executable = nullptr;
  CHECK(topsCreateExecutable(&executable, binary.data(), binary.size()));
  topsStream_t stream = nullptr;
  CHECK(topsStreamCreate(&stream));

  void *in_mem0 = nullptr;
  CHECK(topsMalloc(&in_mem0, 52224));
  CHECK(topsMemcpy(in_mem0, reinterpret_cast<void *>(sp0.data, 52224, topsMemcpyHostToDevice));
  void *in_mem1 = nullptr;
  CHECK(topsMalloc(&in_mem1, 52224));
  CHECK(topsMemcpy(in_mem1, reinterpret_cast<void *>(sp1.data, 52224, topsMemcpyHostToDevice));
  void *in_mem2 = nullptr;
  CHECK(topsMalloc(&in_mem2, 52224));
  CHECK(topsMemcpy(in_mem2, reinterpret_cast<void *>(sp2.data, 52224, topsMemcpyHostToDevice));
  void *in_mem3 = nullptr;
  CHECK(topsMalloc(&in_mem3, 52224));
  CHECK(topsMemcpy(in_mem3, reinterpret_cast<void *>(sp3.data, 52224, topsMemcpyHostToDevice));
  void * device_inputs[] = {in_mem0, in_mem1, in_mem2, in_mem3};

  void * out_mem = nullptr;
  CHECK(topsMalloc(&out_mem, 52224));
  void *device_outputs[] = {out_mem};
  size_t input_dim = 6;
  size_t input_rank = 1;

  CHECK(topsLaunchExecutableV2(
      executable, nullptr, device_inputs,
      sizeof(device_inputs) / sizeof(void *), &input_dim,
      &input_rank, device_outputs,
      sizeof(device_outputs) / sizeof(void *), stream));
  CHECK(topsStreamSynchronize(stream));


  // Copy output data from device to host
  std::vector<uint8_t> host_mem2 = {0, 0, 0, 0}; // TODO: manage the memory by mdspan
  CHECK(topsMemcpy(reinterpret_cast<void *>(host_mem2.data()), out_mem,
                 52224, topsMemcpyDeviceToHost));
// Free up the resources
  topsFree(in_mem0);
  topsFree(in_mem1);
  topsFree(in_mem2);
  topsFree(in_mem3);

  topsStreamDestroy(stream);
  topsDestroyExecutable(executable);

  return 0;
}
EOF

# step 5: compile the host source to target executable
target=__choreo_ele_add
# TODO: sfc ${host_src} -o ${target}
~/choreo/scripts/factor_compile_and_exec.sh ${factor_src} ${factor_bin} ${host_src} ${target}

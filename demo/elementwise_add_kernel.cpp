#include <vector>
#include "gcu/factor/factor.h"

using namespace factor;
void __choreo_ele_add() {
  include_("/tmp/1714104412747419274_1___choreo_ele_add_micro_kernel.cpp");
  // include_("/root/choreo/demo/__choreo_ele_add_micro_kernel.cpp");
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

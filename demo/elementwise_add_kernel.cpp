// clang-format off
// topsfc foo.cc -gcu-arch=gcu200 -resource=4c24s
// clang-format on

#include <vector>
#include "gcu/factor/factor.h"

using namespace factor;

void choreo_ele_add(void) {
  include_("micro_kernel.cpp");
  auto lhs_type = DRAMType(IntType(32), {6, 17, 128});
  auto rhs_type = DRAMType(IntType(32), {6, 17, 128});
  auto output_type = DRAMType(IntType(32), {6, 17, 128});
  auto l2_tile_type = SRAMType(IntType(32), {1, 17, 128});
  auto l1_tile_type = L1Type(IntType(32), {1, 1, 128});
  
  D(main_)({lhs_type, rhs_type}, [&](auto args) {
    auto output = alloc_(output_type);
    Dim3 grid_dim(1);
    Dim3 block_dim(6);
    Value stream = alloc_stream_();
    create_stream_(stream);
    launch_kernel_("ele_add", grid_dim, block_dim, stream,
                   {args[0], args[1]}, {output});
    destroy_stream_(stream);
    dealloc_stream_(stream);
    return std::vector<Value>{output};
  });

  D(func_)("ele_add", {lhs_type, rhs_type}, {output_type}, [&](auto args, auto results) {
    auto thread_id = thread_id_();

    // alloc dma resources
    auto l2_lhs_load = alloc_dma_(CDMAType());
    auto l2_rhs_load = alloc_dma_(CDMAType());
    auto l2_out_store = alloc_dma_(CDMAType());
    auto l1_lhs_load = alloc_dma_(SDMAType());
    auto l1_rhs_load = alloc_dma_(SDMAType());
    auto l1_out_store = alloc_dma_(SDMAType());

    // alloc mem resources
    auto l2_lhs = alloc_(l2_tile_type);
    auto l2_rhs = alloc_(l2_tile_type);
    auto l2_out = alloc_(l2_tile_type);
    auto l1_lhs = alloc_(l1_tile_type);
    auto l1_rhs = alloc_(l1_tile_type);
    auto l1_out = alloc_(l1_tile_type);

    memset_(l2_out_store, l2_out, 0);
    memset_(l1_out_store, l1_out, 0);

    // do cdma
    async_load_(l2_lhs_load, args[0], l2_lhs, {thread_id, 0, 0});
    async_load_(l2_rhs_load, args[1], l2_rhs, {thread_id, 0, 0});
    wait_dma_(l2_lhs_load);
    wait_dma_(l2_rhs_load);

    for_(0, 17, 1, [&](auto x) {
      // do sdma
      async_load_(l1_lhs_load, l2_lhs, l1_lhs, {0, x, 0});
      async_load_(l1_rhs_load, l2_rhs, l1_rhs, {0, x, 0});
      wait_dma_(l1_lhs_load);
      wait_dma_(l1_rhs_load);

      // TIPS: Kernel in Destination-Passing-Style
      call_("add_i32_dps", {l1_lhs.addr_(), l1_rhs.addr_(), l1_out.addr_(), 128});

      // TIPS: Kernel in non Destination-Passing-Style
      // auto output = call_("add_i32_ndps", {l1_lhs({x, y}), l1_rhs({x, y})}, {IntType(32)});
      // l1_out[{x, y}] = output[0];

      // TIPS: You can either use syncthreads or not
      // syncthreads_();

      // TIPS: use Factor overloaded operator+ for vadd
      // auto reg_lhs = loadv_(l1_lhs, {0, 0}, 1, 5);
      // auto reg_rhs = loadv_(l1_rhs, {0, 0}, 1, 5);
      // auto reg_out = reg_lhs + reg_rhs;
      // storev_(reg_out, l1_out, {0, 0}, 1);

      async_store_(l1_out_store, l1_out, l2_out, {0, x, 0});
      wait_dma_(l1_out_store);
    });
    async_store_(l2_out_store, l2_out, results[0], {thread_id, 0, 0});
    wait_dma_(l2_out_store);
  });
}

MODULE_REGISTER("choreo_ele_add", choreo_ele_add);

#include "embedding_cache_replace_kernel.h"
#include <iostream>
namespace flashtensor {
template <
    typename key_type,
    key_type empty_key,
    int set_associativity,
    int warp_size
>
__global__ void EmbeddingCacheReplace_kernel(
    // attribute
    key_type *keys,
    float *vals,
    uint64_t *slot_counter,
    int capacity_in_set,
    int embedding_vec_size,
    uint64_t global_counter,
    // input
    key_type *d_keys,
    float *d_values,
    int len
) {
    // L1 buffer size: 1024*2*32*8B + 1024*2*32*8B + 8K*(8+4)B = 512KB + 512KB + 96KB
    const int PART_SIZE = 1024;
    const int TILE_SIZE = 8 * 1024;
    __valigned__ __local__ __private__ key_type buffer_key[PART_SIZE][set_associativity][warp_size];
    __valigned__ __local__ __private__ uint64_t buffer_counter[PART_SIZE][set_associativity][warp_size];
    __valigned__ __local__ __private__ key_type buffer_query[TILE_SIZE];
    __valigned__ __local__ __private__ uint32_t buffer_src_set[TILE_SIZE];

    int thread_num = gridDim.x * blockDim.x;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // |--A--|--B--|--C--|--D--|  split keys into parts for each thread
    int k_total_len = ceil_(capacity_in_set, thread_num);
    int k_total_start = thread_idx * k_total_len;
    int k_total_end = min_(k_total_start + k_total_len, capacity_in_set);
    k_total_len = k_total_end - k_total_start;

    // |{-a-}{-b-}|  tiling keys
    int k_loop_num = ceil_(k_total_len, PART_SIZE);
    for (int k_loop_idx = 0; k_loop_idx < k_loop_num; k_loop_idx++) {
        int k_len = ceil_(k_total_len, k_loop_num);
        int k_start = k_total_start + k_loop_idx * k_len;
        int k_end = min_(k_start + k_len, k_total_end);
        k_len = k_end - k_start;

        // copy key
        tops::private_dte ctx_key;
        auto s_key(tops_dte_scope(ctx_key));
        int tensor_key_shape[] = {capacity_in_set, set_associativity, warp_size, 8};
        int buffer_key_shape[] = {k_len, set_associativity, warp_size, 8};
        tops::mdspan global_key(tops::Global, (char*)keys, tensor_key_shape);
        tops::mdspan private_key(tops::Private, (__private__ char*)buffer_key, buffer_key_shape);
        tops::slice(ctx_key, private_key, global_key, {k_start, 0, 0, 0});

        // copy counter
        tops::private_dte ctx_counter;
        auto s_counter(tops_dte_scope(ctx_counter));
        int tensor_counter_shape[] = {capacity_in_set, set_associativity, warp_size, 8};
        int buffer_counter_shape[] = {k_len, set_associativity, warp_size, 8};
        tops::mdspan global_slot_counter(tops::Global, (char*)slot_counter, tensor_counter_shape);
        tops::mdspan private_slot_counter(tops::Private, (__private__ char*)buffer_counter, buffer_counter_shape);
        tops::slice(ctx_counter, private_slot_counter, global_slot_counter, {k_start, 0, 0, 0});

        // perpare for deslice value
        const int ev_num = 4;
        int ev_idx = 0;
        int flag_ev_value[ev_num]{};
        tops::event ev_value[ev_num];
        tops::private_dte ctx_value[ev_num];
        tops::dte_scope_ex<tops::private_dte> s_value[ev_num]{
            tops_dte_scope(ctx_value[0]),
            tops_dte_scope(ctx_value[1]),
            tops_dte_scope(ctx_value[2]),
            tops_dte_scope(ctx_value[3])
        };
        int table_value_shape[] = {capacity_in_set, set_associativity, warp_size, embedding_vec_size};
        int new_value_shape[] = {1, 1, 1, embedding_vec_size};
        tops::mdspan global_table_value(tops::Global, vals, table_value_shape);

        // tiling d_keys
        int m_loop_num = ceil_(len, TILE_SIZE);
        for (int m_loop_idx = 0; m_loop_idx < m_loop_num; m_loop_idx++) {
            int m_len = ceil_(len, m_loop_num);
            int m_start = m_loop_idx * m_len;
            int m_end = min_(m_start + m_len, len);
            m_len = m_end - m_start;

            // copy d_keys
            tops::private_dte ctx_query;
            auto s_query(tops_dte_scope(ctx_query));
            tops::mdspan global_query(tops::Global, (char*)d_keys, len*8);
            tops::mdspan private_query(tops::Private, (__private__ char*)buffer_query, m_len*8);
            tops::slice(ctx_query, private_query, global_query, {m_start*8});

            embedding_cache_util::MurmurHash3_32(buffer_query, buffer_src_set, capacity_in_set, m_len);

            // for each d_keys
            for (int query_idx = 0; query_idx < m_len; query_idx++) {
                int src_set = buffer_src_set[query_idx];
                if (!(k_start <= src_set && src_set < k_end)) {
                    continue;
                }
                key_type query = buffer_query[query_idx];
                int src_slab = (uint64_t)query % set_associativity;

                // find key in set
                int flag_need_replace = 1;
                int src_slot;
                #pragma unroll 2
                for (int cnt = 0; cnt < set_associativity; cnt++) {
                    auto warp_key = buffer_key[src_set - k_start][src_slab];
                    src_slot = embedding_cache_util::index<key_type, warp_size>(warp_key, query);
                    if (src_slot >= 0) {
                        flag_need_replace = 0;
                        break;
                    } else if (warp_key[warp_size - 1] == empty_key) {
                        src_slot = embedding_cache_util::index<key_type, warp_size>(warp_key, empty_key);
                        break;
                    }
                    src_slab = (src_slab + 1) % set_associativity;
                }
                // if not found, get LRU slot
                if (src_slot == -1) {
                    auto warp_counter = buffer_counter[src_set - k_start][0];
                    int idx = embedding_cache_util::argmin<uint64_t, set_associativity*warp_size>(warp_counter, warp_size, src_slab);
                    src_slab = idx / warp_size;
                    src_slot = idx % warp_size;
                }
                // do replace key, value, counter
                if (flag_need_replace) {
                    // replace key
                    buffer_key[src_set - k_start][src_slab][src_slot] = query;
                    // replace value
                    float *new_value = d_values + (m_start + query_idx) * embedding_vec_size;
                    tops::mdspan global_new_value(tops::Global, new_value, new_value_shape);
                    if (flag_ev_value[ev_idx]) {
                        ev_value[ev_idx].wait();
                    }
                    ev_value[ev_idx] = tops::deslice_async(ctx_value[ev_idx], global_table_value, global_new_value, {src_set, src_slab, src_slot, 0});
                    flag_ev_value[ev_idx] = 1;
                    ev_idx = (ev_idx + 1) % ev_num;
                }
                buffer_counter[src_set - k_start][src_slab][src_slot] = global_counter;
            }

            for (int ev_idx = 0; ev_idx < ev_num; ev_idx++) {
                if (flag_ev_value[ev_idx]) {
                    ev_value[ev_idx].wait();
                    flag_ev_value[ev_idx] = 0;
                }
            }
        }

        // write key
        tops::deslice(ctx_key, global_key, private_key, {k_start, 0, 0, 0});
        // write counter
        tops::deslice(ctx_counter, global_slot_counter, private_slot_counter, {k_start, 0, 0, 0});
    }
}


// HOST
void EmbeddingCacheReplaceForward(
    // input
    int64_t *keys,
    float *vals,
    uint64_t *slot_counter,
    int capacity_in_set,
    int embedding_vec_size,
    uint64_t global_counter,
    // output
    int64_t *d_keys,
    float *d_values,
    int len,
    // kernel parameter
    int dim_grid,
    int dim_block,
    int num_bytes_in_SharedMem,
    topsStream_t stream
) {
    const int64_t empty_key = std::numeric_limits<int64_t>::max();
    EmbeddingCacheReplace_kernel
        <int64_t, empty_key, 2, 32>
        <<<dim_grid, dim_block, num_bytes_in_SharedMem, stream>>>
        (keys, vals, slot_counter, capacity_in_set, embedding_vec_size, global_counter,
        d_keys, d_values, len);
}


void EmbeddingCacheReplaceForward(
    // input
    uint64_t *keys,
    float *vals,
    uint64_t *slot_counter,
    int capacity_in_set,
    int embedding_vec_size,
    uint64_t global_counter,
    // output
    uint64_t *d_keys,
    float *d_values,
    int len,
    // kernel parameter
    int dim_grid,
    int dim_block,
    int num_bytes_in_SharedMem,
    topsStream_t stream
) {
    const uint64_t empty_key = std::numeric_limits<uint64_t>::max();
    EmbeddingCacheReplace_kernel
        <uint64_t, empty_key, 2, 32>
        <<<dim_grid, dim_block, num_bytes_in_SharedMem, stream>>>
        (keys, vals, slot_counter, capacity_in_set, embedding_vec_size, global_counter,
        d_keys, d_values, len);
}
}


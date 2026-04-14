#ifndef __CHROEO_CUTE_HEADER_LIBRARY_HPP__
#define __CHROEO_CUTE_HEADER_LIBRARY_HPP__

// Note: No manual inclusion. (included by "choreo.h")

#ifdef __CHOREO_TARGET_CUTE__

namespace choreo {

  #ifndef CHOREO_PTX_BARRIER_MAX_SPINS
    #define CHOREO_PTX_BARRIER_MAX_SPINS (1u << 24)
  #endif

__device__ __forceinline__ uint32_t tma_to_shared_u32(const void* p) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
  #else
  (void)p;
  return 0;
  #endif
}

__device__ __forceinline__ void tma_mbarrier_init(uint64_t* bar,
                                                  uint32_t thread_count) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(bar_ptr), "r"(thread_count));
  #else
  (void)bar;
  (void)thread_count;
  #endif
}

__device__ __forceinline__ void tma_mbarrier_expect_tx(uint64_t* bar,
                                                       uint32_t bytes) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
      :
      : "r"(bar_ptr), "r"(bytes));
  #else
  (void)bar;
  (void)bytes;
  #endif
}

__device__ __forceinline__ void
tma_mbarrier_expect_tx_noarrive(uint64_t* bar, uint32_t bytes) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(bar_ptr), "r"(bytes));
  #else
  (void)bar;
  (void)bytes;
  #endif
}

__device__ __forceinline__ void
tma_load_2d_shared_cta_global_mbarrier(void* dst, const void* tma_map,
                                       uint64_t* bar, int32_t coord0,
                                       int32_t coord1) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(tma_map);
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  uint32_t dst_ptr = tma_to_shared_u32(dst);
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::"
               "complete_tx::bytes "
               "[%0], [%1, {%3, %4}], [%2];"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(bar_ptr), "r"(coord0),
                 "r"(coord1)
               : "memory");
  #else
  (void)dst;
  (void)tma_map;
  (void)bar;
  (void)coord0;
  (void)coord1;
  #endif
}

__device__ __forceinline__ void
tma_load_2d_shared_cluster_global_mbarrier(void* dst, const void* tma_map,
                                           uint64_t* bar, int32_t coord0,
                                           int32_t coord1) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(tma_map);
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  uint32_t dst_ptr = tma_to_shared_u32(dst);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
               "complete_tx::bytes "
               "[%0], [%1, {%3, %4}], [%2];"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(bar_ptr), "r"(coord0),
                 "r"(coord1)
               : "memory");
  #else
  (void)dst;
  (void)tma_map;
  (void)bar;
  (void)coord0;
  (void)coord1;
  #endif
}

__device__ __forceinline__ void
tma_load_2d_shared_cluster_global_mbarrier_multicast(
    void* dst, const void* tma_map, uint64_t* bar, int32_t coord0,
    int32_t coord1, uint16_t multicast_mask) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(tma_map);
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  uint32_t dst_ptr = tma_to_shared_u32(dst);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile."
               "mbarrier::complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%3, %4}], [%2], %5;"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(bar_ptr), "r"(coord0),
                 "r"(coord1), "h"(multicast_mask)
               : "memory");
  #else
  (void)dst;
  (void)tma_map;
  (void)bar;
  (void)coord0;
  (void)coord1;
  (void)multicast_mask;
  #endif
}

__device__ __forceinline__ uint32_t tma_cluster_rank() {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t rank;
  asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank));
  return rank;
  #else
  return 0;
  #endif
}

__device__ __forceinline__ uint32_t tma_cluster_dim() {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t dim;
  asm volatile("mov.u32 %0, %cluster_nctarank;\n" : "=r"(dim));
  return dim;
  #else
  return 1;
  #endif
}

__device__ __forceinline__ void tma_cluster_sync() {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");
  #endif
}

__device__ __forceinline__ void
tma_mbarrier_arrive_cluster(uint64_t* bar, uint32_t cta_id,
                            uint32_t count = 1) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_addr = tma_to_shared_u32(bar);
  asm volatile("{\n\t"
               ".reg .b32 remAddr32;\n\t"
               "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
               "mbarrier.arrive.shared::cluster.b64 _, [remAddr32], %2;\n\t"
               "}"
               :
               : "r"(smem_addr), "r"(cta_id), "r"(count));
  #else
  (void)bar;
  (void)cta_id;
  (void)count;
  #endif
}

__device__ __forceinline__ void
tma_mbarrier_wait_parity_cluster(uint64_t* bar, int phase_bit) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  uint32_t spins = 0;
  while (true) {
    uint32_t ready = 0;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], "
        "%2;\n"
        "selp.b32 %0, 1, 0, P1;\n"
        "}\n"
        : "=r"(ready)
        : "r"(bar_ptr), "r"(phase_bit)
        : "memory");
    if (ready) return;
    if (++spins >= CHOREO_PTX_BARRIER_MAX_SPINS) {
      printf("WARN: cluster mbarrier wait exceeded max spins\n");
      return;
    }
  }
  #else
  (void)bar;
  (void)phase_bit;
  #endif
}

__device__ __forceinline__ void tma_mbarrier_arrive(uint64_t* bar,
                                                    uint32_t count = 1) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(bar_ptr), "r"(count)
               : "memory");
  #else
  (void)bar;
  (void)count;
  #endif
}

__device__ __forceinline__ void tma_mbarrier_wait_parity(uint64_t* bar,
                                                         int phase_bit) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t bar_ptr = tma_to_shared_u32(bar);
  uint32_t spins = 0;
  while (true) {
    uint32_t ready = 0;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2;\n"
        "selp.b32 %0, 1, 0, P1;\n"
        "}\n"
        : "=r"(ready)
        : "r"(bar_ptr), "r"(phase_bit)
        : "memory");

    if (ready) return;

    ++spins;
    if (spins >= CHOREO_PTX_BARRIER_MAX_SPINS) {
      printf("[choreo-rt] PTX mbarrier wait timeout: bar=%u, phase=%d, "
             "spins=%u\\n",
             bar_ptr, phase_bit, spins);
      __trap();
    }

    if ((spins & 0x3FFu) == 0u) __nanosleep(64);
  }
  #else
  (void)bar;
  (void)phase_bit;
  #endif
}

// SM90+ (Hopper+) - TMA barrier and token
struct TMAAtom {
  cuda::barrier<cuda::thread_scope_block>* bar;
  cuda::barrier<cuda::thread_scope_block>::arrival_token tok;
  uint64_t* ptx_bar = nullptr;
  int ptx_phase = 0;
  bool use_ptx_mbarrier = false;
  //  TMAAtom(cuda::barrier<cuda::thread_scope_block> *b): bar(b) {}
  __device__ auto& barrier() { return *bar; }
  __device__ auto& token() { return tok; }
  __device__ uint64_t* ptx_barrier() { return ptx_bar; }
  __device__ int ptx_phase_bit() const { return ptx_phase; }
  __device__ void toggle_ptx_phase() { ptx_phase ^= 1; }
  __device__ bool IsPTXMBarrier() const { return use_ptx_mbarrier; }
  __device__ void EnablePTXMBarrier(uint64_t* b) {
    ptx_bar = b;
    ptx_phase = 0;
    use_ptx_mbarrier = true;
  }
};

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
using TMALoadAtom = cute::SM90_TMA_LOAD;
using TMAStoreAtom = cute::SM90_TMA_STORE;

  #endif

using AsyncCopyAtom = cute::AutoCopyAsync;

__device__ __attribute__((always_inline)) static inline void __co_abort__() {
  __trap();
}

// this facilitate the wait-N implementation for sm_80+
// it is must be warp-wise since async copy is warp-wise.
// a ring with 6 elements is 8-bytes. Therefore it may cost up-to 256-bytes
// (32-warps) shared memory
struct future;
template <int N>
struct future_ring {
  static_assert(N < UCHAR_MAX, "ring size is too large.");
  int8_t ring[N];

  uint8_t head = 0; // lastest commit
  uint8_t tail = 0; // oldest commit

  __device__ void commit(future*);
  // note: return the remaining count N to feed cp_async_wait<N>
  __device__ int discard(future*);
  __device__ void init() {
    head = 0;
    tail = 0;
  }
};

using AtomType = void; // erase the type

// choreo device future
struct future {
  AtomType* atom = nullptr;
  void* d = nullptr;  // data: future's user must guarantee it is valid
  void* md = nullptr; // metadata: optional structured sparsity metadata

  bool is_tma = false;

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  future_ring<6>* ring;
  int8_t id;
  #else
  // make host compilation happy
  future_ring<6>* ring;
  int8_t id;
  #endif
  __device__ void set_ring(future_ring<6>* r) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    ring = r + __CHOREO_GROUP_ID__;
  #else
  // make host compilation happy
  #endif
  }
  // for runtime check purpose
  //
  // ST_NONE -> ST_INITED -> ST_TRIGGERED -> ST_WAITED
  //                              ^              |
  //                              +--------------+
  enum Status {
    ST_NONE = 0,
    ST_INITED = 1,
    ST_TRIGGERED = 2,
    ST_WAITED = 3,
  };

  #ifdef __CHOREO_DMA_DIAGNOSIS__
  Status s = ST_NONE;
  const char* name = nullptr;
  // source code locations
  unsigned line = 0;
  unsigned column = 0;

  __device__ future(const char* n, unsigned l, unsigned c, void* data = nullptr,
                    void* mdata = nullptr)
      : d(data), md(mdata ? mdata : data), s(ST_NONE), name(n), line(l),
        column(c) {}
  #else
  __device__ future(const char* n, unsigned l, unsigned c, void* data = nullptr,
                    void* mdata = nullptr)
      : d(data), md(mdata ? mdata : data) {
    (void)n;
    (void)l;
    (void)c;
  }

  __device__ future(void* data = nullptr, void* mdata = nullptr)
      : d(data), md(mdata ? mdata : data) {}
  #endif //__CHOREO_DMA_DIAGNOSIS__

  // context is retrieved to invoke data operations
  __device__ auto get_atom() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_NONE) s = ST_INITED;
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is not initialized.\n",
             line, column);
      __co_abort__();
    }
  #endif // __CHOREO_DMA_DIAGNOSIS__
    return atom;
  }

  // when async, an event is obtained for later waiting
  __device__ void set_atom(AtomType* a) {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_TRIGGERED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "is triggered on an in-flight event.\n",
             line, column);
      __co_abort__();
    } else if (s != ST_NONE) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "has been initialized before setting atom.\n",
             line, column);
      __co_abort__();
    }
    s = ST_INITED;
  #endif // __CHOREO_DMA_DIAGNOSIS__

    atom = a;
  }

  // when sync, no wait is required. simply change the status
  __device__ void set_nowait() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s != ST_INITED && s != ST_WAITED) {
      if (s == ST_NONE)
        printf("[choreo-rt] WARN: future (defined at line %u:%u) maybe in none "
               "state when setting nowait.\n",
               line, column);
      else if (s == ST_TRIGGERED)
        printf("[choreo-rt] WARN: future (defined at line %u:%u) maybe in "
               "triggered state when setting nowait.\n",
               line, column);
      else
        printf("[choreo-rt] WARN: future (defined at line %u:%u) maybe in an "
               "invalid state when setting nowait.\n",
               line, column);
    }
    s = ST_WAITED;
  #endif // __CHOREO_DMA_DIAGNOSIS__
  }

  __device__ void set_data(void* data) { d = data; }
  __device__ void set_atom_data(AtomType* a, void* data) {
    set_atom(a);
    set_data(data);
  }

  __device__ void wait_impl() {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (is_tma) {
      auto* tma_atom = ((TMAAtom*)atom);
      if (tma_atom->IsPTXMBarrier()) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        tma_mbarrier_wait_parity(tma_atom->ptx_barrier(),
                                 tma_atom->ptx_phase_bit());
    #else
        __co_abort__();
    #endif
      } else {
        auto& barrier = tma_atom->barrier();
        auto& token = tma_atom->token();
        barrier.wait(std::move(token));
      }
      return;
    }
    // TODO: make shared memory be allocated by host
    __shared__ int discard_count;
    discard_count = 0;
    // cautious: must be warp based
    if (__CHOREO_GROUP_SINGLE__) {
      assert(ring && "ring is invalid.");
      discard_count = ring->discard(this);
    }
    // sync it to warp threads
    discard_count = __shfl_sync(0xffffffff, discard_count, 0);
    switch (discard_count) {
    case -1: break;
    case 0: cute::cp_async_wait<0>(); break;
    case 1: cute::cp_async_wait<1>(); break;
    case 2: cute::cp_async_wait<2>(); break;
    case 3: cute::cp_async_wait<3>(); break;
    case 4: cute::cp_async_wait<4>(); break;
    case 5: cute::cp_async_wait<5>(); break;
    default:
    #ifdef __CHOREO_DMA_DIAGNOSIS__
      printf("[choreo-rt] Unable to wait the %d futures (current defined at "
             "line %u:%u).\n",
             discard_count, line, column);
    #else
      printf("[choreo-rt] Unable to wait the %d futures.\n", discard_count);
    #endif // DIAGNOSIS
      __co_abort__();
      break;
    }
    __syncwarp();

  #else
  // cuda host compilation
  #endif
  }

  __device__ void trigger() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has been triggered without atom set.\n",
             line, column);
      __co_abort__();
    }
    s = ST_TRIGGERED;
  #endif // __CHOREO_DMA_DIAGNOSIS__

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // cautious: must be warp based
    if (__CHOREO_GROUP_SINGLE__) {
      assert(ring && "ring is invalid.");
      ring->commit(this);
    }
  #else
  // cuda host compilation
  #endif
  }

  __device__ void wait() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_TRIGGERED) {
      s = ST_WAITED;
    } else if (s == ST_WAITED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has been waited multiple times.\n",
             line, column);
      __co_abort__();
    } else if (s == ST_INITED) {
      printf("[choreo-rt] WARN: maybe wait the future (defined at line %u:%u) "
             "before trigger it.\n",
             line, column);
    } else
      assert(s == ST_NONE); // waiting on not triggered future is acceptable
  #endif                    // __CHOREO_DMA_DIAGNOSIS__
    wait_impl();
  }

  __device__ void* data() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (!d) {
      printf("[choreo-rt] internal error: future (defined at line %u:%u) is "
             "not associated with a data.\n",
             line, column);
      __co_abort__();
    }
    if (s == ST_TRIGGERED) {
      // TODO: requires krt %s support to print future name
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) is "
             "not waited before using.\n",
             line, column);
      __co_abort__();
    }
  #endif // __CHOREO_DMA_DIAGNOSIS__
    return d;
  }

  __device__ void* mdata() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (!md) {
      printf("[choreo-rt] internal error: future (defined at line %u:%u) is "
             "not associated with a metadata.\n",
             line, column);
      __co_abort__();
    }
    if (s == ST_TRIGGERED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) is "
             "not waited before using.\n",
             line, column);
      __co_abort__();
    }
  #endif // __CHOREO_DMA_DIAGNOSIS__
    return md ? md : d;
  }

  __device__ void destroy() {}

  __device__ ~future() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_TRIGGERED)
      printf("[choreo-rt] WARN: the future (defined at line %u:%u) may never "
             "have been waited.\n",
             line, column);
    if (s >= ST_INITED) destroy();
  #else
    destroy();
  #endif // __CHOREO_DMA_DIAGNOSIS__
  }
  __device__ future(const future& f) = delete;
  __device__ future(future&& f) = delete;
  __device__ future& operator=(const future& f) = delete;
};

template <int N>
inline __device__ void future_ring<N>::commit(future* f) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // the uniqueness of id is guaranteed by choreo
  ring[head] = f->id;
  head = (head + 1) % N;

    #ifdef __CHOREO_DEBUG_FUTURE_RING__
  printf("committed feature: %d, [%d, %d)\n", f->id, tail, head);
    #endif

  #else
  // cuda host compilation
  #endif // CUDA_ARCH
}

template <int N>
inline __device__ int future_ring<N>::discard(future* f) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

    #ifdef __CHOREO_DEBUG_FUTURE_RING__
  printf("discarding feature: %d, [%d, %d)\n", f->id, tail, head);
    #endif

  uint8_t p = tail;
  while (p != head) {
    if (ring[p] == f->id) {
      int size = (head - p - 1 + N) % N;
      tail = (p + 1) % N;
      return size;
    } else
      p = (p + 1) % N;
  }

  // the ring is now empty
  if (tail == head) p = (p + 1) % N;

  while (p != tail) {
    // has been discarded already
    if (ring[p] == f->id)
      return -1;
    else
      p = (p + 1) % N;
  }

    #ifdef __CHOREO_DMA_DIAGNOSIS__
  printf("[choreo-rt] Internal error: future %d (defined at line %u:%u) "
         "is not committed.\n",
         f->id, f->line, f->column);
    #else
  printf("[choreo-rt] Internal error: future %d is not committed.\n", f->id);
    #endif

      //  __co_abort__();
  #else
  // cuda host compilation
  #endif // CUDA_ARCH
  return -1;
}

__device__ static inline void swap(future& a, future& b) {
  auto atom = a.atom;
  auto d = a.d;

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  future_ring<6>* ring = a.ring;
  int8_t id = a.id;
  #else
  #endif

  #ifdef __CHOREO_DMA_DIAGNOSIS__
  auto name = a.name;
  auto s = a.s;
  auto l = a.line;
  auto c = a.column;
  #endif

  a.atom = b.atom;
  a.d = b.d;

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  a.ring = b.ring;
  a.id = b.id;
  #else
  #endif

  #ifdef __CHOREO_DMA_DIAGNOSIS__
  a.name = b.name;
  a.s = b.s;
  a.line = b.line;
  a.column = b.column;
  #endif

  b.atom = atom;
  b.d = d;

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  b.ring = ring;
  b.id = id;
  #else
  #endif

  #ifdef __CHOREO_DMA_DIAGNOSIS__
  b.name = name;
  b.s = s;
  b.line = l;
  b.column = c;
  #endif
}

// ------------------- C++17 utilities -------------------
template <class...>
using void_t = void;

template <class T, class = void>
struct has_value : std::false_type {};
template <class T>
struct has_value<T, void_t<decltype(T::value)>> : std::true_type {};

// std::conjunction for C++14/17
template <class...>
struct conj : std::true_type {};
template <class B1>
struct conj<B1> : B1 {};
template <class B1, class... Bn>
struct conj<B1, Bn...> : std::conditional<B1::value, conj<Bn...>, B1>::type {};

// ------------------- compile-time stride checks -------------------
// One-stride divisibility check (only meaningful if stride is static
// cute::C<...>)
template <int Nelems, class StrideT>
struct stride_ok_ct
    : std::bool_constant<has_value<StrideT>::value &&
                         ((int(StrideT::value) % Nelems == 0) ||
                          (Nelems % int(StrideT::value) == 0))> {};

// Fold over stride tuple at indices I...
template <int Nelems, class Strides, std::size_t... I>
struct all_strides_ok_ct_impl {
  using type = conj<stride_ok_ct<
      Nelems,
      typename std::remove_cv<typename std::remove_reference<
          decltype(std::get<I>(std::declval<Strides>()))>::type>::type>...>;
  static constexpr bool value = type::value;
};

// Are ALL strides compile-time constants? (no divisibility yet)
template <class Strides, std::size_t... I>
struct all_strides_are_static_impl {
  using type =
      conj<has_value<typename std::remove_cv<typename std::remove_reference<
          decltype(std::get<I>(std::declval<Strides>()))>::type>::type>...>;
  static constexpr bool value = type::value;
};

// Trait: for a given Src tensor, can we prove Bits-wide vector is OK at compile
// time?
template <int Bits, class Src>
struct layout_vec_ok_ct {
  using E = typename Src::value_type;
  static constexpr int Nelems = Bits / (8 * int(sizeof(E)));
  using Strides = decltype(std::declval<Src>().layout().stride());
  static constexpr std::size_t R =
      decltype(size(std::declval<Strides>()))::value;

  static constexpr bool value =
      all_strides_ok_ct_impl<Nelems, Strides,
                             std::make_index_sequence<R>{}>::value;
};

// Trait: are ALL strides of Src compile-time constants?
template <class Src>
struct all_strides_are_static {
  using Strides = decltype(std::declval<Src>().layout().stride());
  static constexpr std::size_t R =
      decltype(size(std::declval<Strides>()))::value;

  static constexpr bool value =
      all_strides_are_static_impl<Strides,
                                  std::make_index_sequence<R>{}>::value;
};

// ------------------- runtime pointer alignment (bytes) -------------------
template <int Bits, class Ptr>
CUTE_HOST_DEVICE bool aligned_at_least(Ptr p) {
  constexpr std::uintptr_t A = Bits / 8;
  return (reinterpret_cast<std::uintptr_t>(p) % A) == 0;
}

template <size_t Bits, class Ptr>
CUTE_HOST_DEVICE std::uintptr_t aligned_up_ptr(Ptr p) {
  constexpr std::uintptr_t Align = Bits / 8;
  auto ptr = reinterpret_cast<std::uintptr_t>(p);
  return (ptr + Align - 1) & ~(Align - 1);
}

// ------------------- universal copy (any rank, C++17) -------------------
template <class Src, class Dst>
CUTE_HOST_DEVICE void opt_copy(const Src& src, Dst& dst) {
  // Fallback: 32-bit (scalar element width) - always safe for any
  // shape/stride/alignment
  copy(cute::AutoVectorizingCopyWithAssumedAlignment<32>{}, src, dst);
}

template <bool Swizzle, class Element, int ThrRows, int ThrCols, int ValRows,
          int ValCols, int CopyBits = 128, class SrcTensor, class DstTensor,
          class Pred>
__device__ static inline void copy_if_g2s(const SrcTensor& src, DstTensor& dst,
                                          Pred pred) {
  if constexpr (CopyBits == 128) {
    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,
                        Element>{},
        cute::make_layout(
            cute::make_shape(cute::Int<ThrRows>{}, cute::Int<ThrCols>{}),
            cute::make_stride(cute::Int<ThrCols>{}, cute::Int<1>{})),
        cute::make_layout(
            cute::make_shape(cute::Int<ValRows>{}, cute::Int<ValCols>{})));
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto src_thr = thr_copy.partition_S(src);
    auto dst_thr = [&]() {
      if constexpr (Swizzle) {
        auto dst_pi = cute::as_position_independent_swizzle_tensor(dst);
        return thr_copy.partition_D(dst_pi);
      } else {
        return thr_copy.partition_D(dst);
      }
    }();
    auto coord_thr =
        thr_copy.partition_S(cute::make_identity_tensor(cute::shape(src)));
    auto pred_thr = cute::make_tensor<bool>(cute::shape(src_thr));
    CUTE_UNROLL
    for (int i = 0; i < cute::size(pred_thr); ++i) {
      pred_thr(i) = pred(coord_thr(i));
    }
    cute::copy_if(tiled_copy, pred_thr, src_thr, dst_thr);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
  } else {
    // Synchronous fallback for non-aligned strides: element-wise
    // copy with zero-fill for predicated-out elements.
    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<Element>, Element>{},
        cute::make_layout(
            cute::make_shape(cute::Int<ThrRows>{}, cute::Int<ThrCols>{}),
            cute::make_stride(cute::Int<ThrCols>{}, cute::Int<1>{})),
        cute::make_layout(
            cute::make_shape(cute::Int<ValRows>{}, cute::Int<ValCols>{})));
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto src_thr = thr_copy.partition_S(src);
    auto dst_thr = [&]() {
      if constexpr (Swizzle) {
        auto dst_pi = cute::as_position_independent_swizzle_tensor(dst);
        return thr_copy.partition_D(dst_pi);
      } else {
        return thr_copy.partition_D(dst);
      }
    }();
    auto coord_thr =
        thr_copy.partition_S(cute::make_identity_tensor(cute::shape(src)));
    CUTE_UNROLL
    for (int i = 0; i < cute::size(dst_thr); ++i) {
      bool valid = pred(coord_thr(i));
      dst_thr(i) = valid ? src_thr(i) : Element(0);
    }
  }
}

template <class Element, int ThrRows, int ThrCols, int ValRows, int ValCols,
          class SrcTensor, class DstTensor, class Pred>
__device__ static inline void copy_if_s2g(const SrcTensor& src, DstTensor& dst,
                                          Pred pred) {
  auto tiled_copy = cute::make_tiled_copy(
      cute::Copy_Atom<cute::UniversalCopy<Element>, Element>{},
      cute::make_layout(
          cute::make_shape(cute::Int<ThrRows>{}, cute::Int<ThrCols>{}),
          cute::make_stride(cute::Int<ThrCols>{}, cute::Int<1>{})),
      cute::make_layout(
          cute::make_shape(cute::Int<ValRows>{}, cute::Int<ValCols>{})));
  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  auto src_thr = thr_copy.partition_S(src);
  auto dst_thr = thr_copy.partition_D(dst);
  auto coord_thr =
      thr_copy.partition_S(cute::make_identity_tensor(cute::shape(src)));
  auto pred_thr = cute::make_tensor<bool>(cute::shape(src_thr));
  CUTE_UNROLL
  for (int i = 0; i < cute::size(pred_thr); ++i) {
    pred_thr(i) = pred(coord_thr(i));
  }
  cute::copy_if(tiled_copy, pred_thr, src_thr, dst_thr);
}

// TODO: move to choreo_mma_wrapper.h
// ------------------- inline mma PTX -------------------

template <class MMA>
struct MMA_Policy {
  static constexpr bool supported = false;
};

// Accumulation type casting helper
template <class T, class F>
struct AccumTCast {
  static constexpr bool supported = false;
};

template <>
struct AccumTCast<f16, f32> {
  static constexpr bool supported = true;
  __device__ static inline f16 cast(f32 val) { return f32_to_f16(val); }
};

template <>
struct AccumTCast<f32, f16> {
  static constexpr bool supported = true;
  __device__ static inline f32 cast(f16 val) { return f16_to_f32(val); }
};

template <class T, class F>
__device__ static inline T cast_if(F val) {
  if constexpr (AccumTCast<F, T>::supported) {
    return AccumTCast<F, T>::cast(val);
  } else {
    static_assert(std::is_same<T, F>::value,
                  "unsupported accumulation type casting");
    return val;
  }
}

template <>
struct AccumTCast<bf16, f32> {
  static constexpr bool supported = true;
  __device__ static inline bf16 cast(f32 val) { return f32_to_bf16(val); }
};

template <>
struct AccumTCast<f32, bf16> {
  static constexpr bool supported = true;
  __device__ static inline f32 cast(bf16 val) { return bf16_to_f32(val); }
};

  #ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
template <>
struct AccumTCast<f8_e4m3, f32> {
  static constexpr bool supported = true;
  __device__ static inline f8_e4m3 cast(f32 val) { return f8_e4m3(val); }
};

template <>
struct AccumTCast<f32, f8_e4m3> {
  static constexpr bool supported = true;
  __device__ static inline f32 cast(f8_e4m3 val) { return float(val); }
};

template <>
struct AccumTCast<f8_e5m2, f32> {
  static constexpr bool supported = true;
  __device__ static inline f8_e5m2 cast(f32 val) { return f8_e5m2(val); }
};

template <>
struct AccumTCast<f32, f8_e5m2> {
  static constexpr bool supported = true;
  __device__ static inline f32 cast(f8_e5m2 val) { return float(val); }
};
  #endif

// --------------- load A policies ---------------
struct Policy_A_M8N8K4 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, f16>::value) {
      int row;
      if (lane < 16)
        row = lane & 3;
      else
        row = (lane & 3) + 4;
      auto A_u32 = cute::recast<uint32_t>(A);
      uint32_t a0 = A_u32(row, 0);
      uint32_t a1 = A_u32(row, 1);
      return cutlass::Array<uint32_t, 2>{a0, a1};
    } else if constexpr (std::is_same<value_type, double>::value) {
      int row = lane >> 2;
      int col = lane & 3;
      double a0 = A(row, col);
      return cutlass::Array<double, 1>{a0};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA");
    }
  }
};

// for s8, u8
struct Policy_A_M8N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row = gid;
    int col = tid_in_group * 4;
    uint32_t a0 = 0;
      // TODO: if use recast, res error
  #pragma unroll
    for (int i = 3; i >= 0; i--) a0 = (a0 << 8) | bitcast_u32(A(row, col + i));
    return cutlass::Array<uint32_t, 1>{a0};
  }
};

// for s4, u4
struct Policy_A_M8N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_A_M8N8K128 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for tf32
struct Policy_A_M16N8K4 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col = tid_in_group;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, tf32>::value ||
                  std::is_same<value_type, float>::value) {
      auto A_u32 = cute::recast<uint32_t>(A);
      uint32_t a0 = A_u32(row0, col);
      uint32_t a1 = A_u32(row1, col);
      return cutlass::Array<uint32_t, 2>{a0, a1};
    } else if constexpr (std::is_same<value_type, double>::value) {
      double a0 = A(row0, col);
      double a1 = A(row1, col);
      return cutlass::Array<double, 2>{a0, a1};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_A_M16N8K8 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group;
    int col1 = tid_in_group + 4;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, f16>::value ||
                  std::is_same<value_type, bf16>::value) {
      auto A_u32 = cute::recast<uint32_t>(A);
      uint32_t a0 = A_u32(row0, col0);
      uint32_t a1 = A_u32(row1, col0);
      return cutlass::Array<uint32_t, 2>{a0, a1};
    } else if constexpr (std::is_same<value_type, tf32>::value ||
                         std::is_same<value_type, float>::value) {
      auto A_u32 = cute::recast<uint32_t>(A);
      uint32_t a0 = A_u32(row0, col0);
      uint32_t a1 = A_u32(row1, col0);
      uint32_t a2 = A_u32(row0, col1);
      uint32_t a3 = A_u32(row1, col1);
      return cutlass::Array<uint32_t, 4>{a0, a1, a2, a3};
    } else if constexpr (std::is_same<value_type, double>::value) {
      double a0 = A(row0, col0);
      double a1 = A(row1, col0);
      double a2 = A(row0, col1);
      double a3 = A(row1, col1);
      return cutlass::Array<double, 4>{a0, a1, a2, a3};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_A_M16N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid, row1 = gid + 8;

    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value) {
      int col0 = tid_in_group, col1 = tid_in_group + 4, col2 = tid_in_group + 8,
          col3 = tid_in_group + 12;
      double a0 = A(row0, col0);
      double a1 = A(row1, col0);
      double a2 = A(row0, col1);
      double a3 = A(row1, col1);
      double a4 = A(row0, col2);
      double a5 = A(row1, col2);
      double a6 = A(row0, col3);
      double a7 = A(row1, col3);
      return cutlass::Array<double, 8>{a0, a1, a2, a3, a4, a5, a6, a7};
    } else if constexpr (std::is_same<value_type, f16>::value ||
                         std::is_same<value_type, bf16>::value) {
      auto A_u32 = cute::recast<uint32_t>(A);
      int col0 = tid_in_group, col1 = tid_in_group + 4;
      uint32_t a0 = A_u32(row0, col0);
      uint32_t a1 = A_u32(row1, col0);
      uint32_t a2 = A_u32(row0, col1);
      uint32_t a3 = A_u32(row1, col1);
      return cutlass::Array<uint32_t, 4>{a0, a1, a2, a3};
    } else if constexpr (std::is_same<value_type, uint8_t>::value ||
                         std::is_same<value_type, int8_t>::value ||
                         std::is_same<value_type, f8_e4m3>::value ||
                         std::is_same<value_type, f8_e5m2>::value) {
      int col = tid_in_group * 4;
      uint32_t a0 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a0 = (a0 << 8) | bitcast_u32(A(row0, col + i));
      uint32_t a1 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a1 = (a1 << 8) | bitcast_u32(A(row1, col + i));
      return cutlass::Array<uint32_t, 2>{a0, a1};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_A_M16N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, f8_e4m3>::value ||
                  std::is_same<value_type, f8_e5m2>::value) {
      int col0 = tid_in_group;
      int col1 = tid_in_group + 4;
      auto A_u32 = cute::recast<uint32_t>(A);
      uint32_t a0 = A_u32(row0, col0);
      uint32_t a1 = A_u32(row1, col0);
      uint32_t a2 = A_u32(row0, col1);
      uint32_t a3 = A_u32(row1, col1);
      return cutlass::Array<uint32_t, 4>{a0, a1, a2, a3};
    } else if constexpr (std::is_same<value_type, uint8_t>::value ||
                         std::is_same<value_type, int8_t>::value) {
      int col0 = tid_in_group * 4;
      int col1 = tid_in_group * 4 + 16;
      uint32_t a0 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a0 = (a0 << 8) | bitcast_u32(A(row0, col0 + i));
      uint32_t a1 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a1 = (a1 << 8) | bitcast_u32(A(row1, col0 + i));
      uint32_t a2 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a2 = (a2 << 8) | bitcast_u32(A(row0, col1 + i));
      uint32_t a3 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        a3 = (a3 << 8) | bitcast_u32(A(row1, col1 + i));
      return cutlass::Array<uint32_t, 4>{a0, a1, a2, a3};
    }
  }
};

// Sparse A load policy for m16n8k16 (compressed K/2 = 8 elements per row)
// Each thread loads 2 half values from the compressed sparse matrix
struct Policy_A_Sparse_M16N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;   // 0..7
    int thread_id = lane & 0x3; // 0..3

    using value_type = typename Tensor::value_type;
    static_assert(std::is_same<value_type, f16>::value ||
                      std::is_same<value_type, bf16>::value,
                  "Sparse m16n8k16 only supports f16/bf16");

    // For 2:4 sparse m16n8k16: A is [16, 8] (compressed from [16, 16])
    // Each thread loads 2 values packed into one uint32_t
    int col_base = thread_id * 2; // 0, 2, 4, 6
    uint32_t a0 = (bitcast_u32(A(group_id, col_base + 1)) << 16) |
                  bitcast_u32(A(group_id, col_base));
    uint32_t a1 = (bitcast_u32(A(group_id + 8, col_base + 1)) << 16) |
                  bitcast_u32(A(group_id + 8, col_base));
    return cutlass::Array<uint32_t, 2>{a0, a1};
  }
};

// Sparse A load policy for m16n8k32 (compressed K/2 = 16 elements per row)
// Each thread loads 4 half values from the compressed sparse matrix
struct Policy_A_Sparse_M16N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;   // 0..7
    int thread_id = lane & 0x3; // 0..3

    using value_type = typename Tensor::value_type;
    static_assert(std::is_same<value_type, f16>::value ||
                      std::is_same<value_type, bf16>::value,
                  "Sparse m16n8k32 only supports f16/bf16");

    // For 2:4 sparse m16n8k32: A is [16, 16] (compressed from [16, 32])
    // Fragment layout requires 8 half values -> 4 uint32_t registers
    // Following the sptc-demo pattern for manual lane loading
    value_type vals[8];
  #pragma unroll
    for (int ai = 0; ai < 8; ++ai) {
      int row = (ai < 2 || (ai >= 4 && ai < 6)) ? group_id : (group_id + 8);
      int col_base = (ai < 4) ? (thread_id * 4) : (thread_id * 4 + 16);
      int chunk = col_base / 4;
      int val_idx = ai & 0x1;
      // A is [16, 16] compressed; chunk * 2 + val_idx gives column
      vals[ai] = A(row, chunk * 2 + val_idx);
    }
    uint32_t p0 = (bitcast_u32(vals[1]) << 16) | bitcast_u32(vals[0]);
    uint32_t p1 = (bitcast_u32(vals[3]) << 16) | bitcast_u32(vals[2]);
    uint32_t p2 = (bitcast_u32(vals[5]) << 16) | bitcast_u32(vals[4]);
    uint32_t p3 = (bitcast_u32(vals[7]) << 16) | bitcast_u32(vals[6]);
    return cutlass::Array<uint32_t, 4>{p0, p1, p2, p3};
  }
};

// Sparse A load policy for m16n8k64 (compressed K/2 = 32 elements per row)
// Each thread loads 16 fp8 values -> 4 uint32_t registers
struct Policy_A_Sparse_M16N8K64 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;   // 0..7
    int thread_id = lane & 0x3; // 0..3

    using value_type = typename Tensor::value_type;
    static_assert(std::is_same<value_type, f8_e4m3>::value ||
                      std::is_same<value_type, f8_e5m2>::value,
                  "Sparse m16n8k64 only supports fp8 types");

    // For 2:4 sparse m16n8k64: A is [16, 32] (compressed from [16, 64])
    // Recast to uint32 (4 fp8 per reg) and follow dense K32 layout
    auto A_u32 = cute::recast<uint32_t>(A);
    int col0 = thread_id;     // 0..3
    int col1 = thread_id + 4; // 4..7
    uint32_t a0 = A_u32(group_id, col0);
    uint32_t a1 = A_u32(group_id + 8, col0);
    uint32_t a2 = A_u32(group_id, col1);
    uint32_t a3 = A_u32(group_id + 8, col1);
    return cutlass::Array<uint32_t, 4>{a0, a1, a2, a3};
  }
};

// Sparse metadata load policy for m16n8k32
struct Policy_E_Sparse_M16N8K32 {
  template <class Tensor>
  __device__ static uint32_t load(Tensor const& E) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2; // 0..7
    int thread_id = lane & 3; // 0..3
    if (thread_id == 0 || thread_id == 1) {
      uint32_t e0 = E(group_id, 0);
      uint32_t e1 = E(group_id + 8, 0);
      uint32_t lo0 = e0 & 0xFFFFu;
      uint32_t hi0 = (e0 >> 16) & 0xFFFFu;
      uint32_t lo1 = e1 & 0xFFFFu;
      uint32_t hi1 = (e1 >> 16) & 0xFFFFu;
      if (thread_id == 0) return (lo1 << 16) | lo0;
      return (hi1 << 16) | hi0;
    }
    return 0;
  }
};

// Sparse metadata load policy for m16n8k16
struct Policy_E_Sparse_M16N8K16 {
  template <class Tensor>
  __device__ static uint32_t load(Tensor const& E) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2; // 0..7
    uint32_t lo = E(group_id, 0) & 0xFFFFu;
    uint32_t hi = E(group_id + 8, 0) & 0xFFFFu;
    return (hi << 16) | lo;
  }
};

// Sparse metadata load policy for m16n8k64 (FP8 path)
// Map the two 32-bit metadata words for each row pair across the 4 lanes.
struct Policy_E_Sparse_M16N8K64 {
  template <class Tensor>
  __device__ static uint32_t load(Tensor const& E) {
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;   // 0..7
    int thread_id = lane & 0x3; // 0..3

    uint32_t row0_lo = E(group_id, 0);
    uint32_t row0_hi = E(group_id, 1);
    uint32_t row1_lo = E(group_id + 8, 0);
    uint32_t row1_hi = E(group_id + 8, 1);

    if (thread_id == 0) return row0_lo;
    if (thread_id == 1) return row1_lo;
    if (thread_id == 2) return row0_hi;
    return row1_hi;
  }
};

// for s4, u4 and e2m1
struct Policy_A_M16N8K64 {

  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_A_M16N8K128 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_A_M16N8K256 {
  template <class Tensor>
  __device__ static auto load(Tensor const& A) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// --------------- load B policies ---------------
struct Policy_B_M8N8K4 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, f16>::value) {
      int col;
      if (lane < 16)
        col = lane & 3;
      else
        col = (lane & 3) + 4;
      uint32_t b0 = (bitcast_u32(B(1, col)) << 16) | bitcast_u32(B(0, col));
      uint32_t b1 = (bitcast_u32(B(3, col)) << 16) | bitcast_u32(B(2, col));
      return cutlass::Array<uint32_t, 2>{b0, b1};
    } else if constexpr (std::is_same<value_type, double>::value) {
      int row = lane & 3;
      int col = lane >> 2;
      double b0 = B(row, col);
      return cutlass::Array<double, 1>{b0};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA");
    }
  }
};

// for s8, u8
struct Policy_B_M8N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row = tid_in_group * 4;
    int col = gid;
    uint32_t b0 = 0;
  #pragma unroll
    for (int i = 3; i >= 0; i--) b0 = (b0 << 8) | bitcast_u32(B(row + i, col));
    return cutlass::Array<uint32_t, 1>{b0};
  }
};

// for s4, u4
struct Policy_B_M8N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_B_M8N8K128 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

struct Policy_B_M16N8K4 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row = tid_in_group;
    int col = gid;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, tf32>::value ||
                  std::is_same<value_type, float>::value) {
      uint32_t b0 = bitcast_u32(B(row, col));
      return cutlass::Array<uint32_t, 1>{b0};
    } else if constexpr (std::is_same<value_type, double>::value) {
      double b0_d = B(row, col);
      return cutlass::Array<double, 1>{b0_d};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_B_M16N8K8 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = tid_in_group;
    int row1 = tid_in_group + 4;
    int col = gid;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, f16>::value ||
                  std::is_same<value_type, bf16>::value) {
      int row = tid_in_group * 2;
      uint32_t b0 =
          (bitcast_u32(B(row + 1, col)) << 16) | bitcast_u32(B(row, col));
      return cutlass::Array<uint32_t, 1>{b0};
    } else if constexpr (std::is_same<value_type, tf32>::value ||
                         std::is_same<value_type, float>::value) {
      auto B_u32 = cute::recast<uint32_t>(B);
      uint32_t b0 = B_u32(row0, col);
      uint32_t b1 = B_u32(row1, col);
      return cutlass::Array<uint32_t, 2>{b0, b1};
    } else if constexpr (std::is_same<value_type, double>::value) {
      double b0 = B(row0, col);
      double b1 = B(row1, col);
      return cutlass::Array<double, 2>{b0, b1};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_B_M16N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;

    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value) {
      int row = tid_in_group;
      int col = gid;
      double b0 = B(row, col);
      double b1 = B(row + 4, col);
      double b2 = B(row + 8, col);
      double b3 = B(row + 12, col);
      return cutlass::Array<double, 4>{b0, b1, b2, b3};
    } else if constexpr (std::is_same<value_type, f16>::value ||
                         std::is_same<value_type, bf16>::value) {
      int row0 = tid_in_group * 2;
      int row1 = tid_in_group * 2 + 8;
      int col = gid;
      uint32_t b0 =
          (bitcast_u32(B(row0 + 1, col)) << 16) | bitcast_u32(B(row0, col));
      uint32_t b1 =
          (bitcast_u32(B(row1 + 1, col)) << 16) | bitcast_u32(B(row1, col));
      return cutlass::Array<uint32_t, 2>{b0, b1};
    } else if constexpr (std::is_same<value_type, uint8_t>::value ||
                         std::is_same<value_type, int8_t>::value ||
                         std::is_same<value_type, f8_e4m3>::value ||
                         std::is_same<value_type, f8_e5m2>::value) {
      int row = tid_in_group * 4;
      int col = gid;
      uint32_t b0 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        b0 = (b0 << 8) | bitcast_u32(B(row + i, col));
      return cutlass::Array<uint32_t, 1>{b0};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

// Sparse B policy for m16n8k16 with K-major [N, K] layout
struct Policy_B_Sparse_M16N8K16 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;

    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value) {
      int row = tid_in_group;
      int col = gid;
      double b0 = B(col, row);
      double b1 = B(col, row + 4);
      double b2 = B(col, row + 8);
      double b3 = B(col, row + 12);
      return cutlass::Array<double, 4>{b0, b1, b2, b3};
    } else if constexpr (std::is_same<value_type, f16>::value ||
                         std::is_same<value_type, bf16>::value) {
      int row0 = tid_in_group * 2;
      int row1 = tid_in_group * 2 + 8;
      int col = gid;
      uint32_t b0 =
          (bitcast_u32(B(col, row0 + 1)) << 16) | bitcast_u32(B(col, row0));
      uint32_t b1 =
          (bitcast_u32(B(col, row1 + 1)) << 16) | bitcast_u32(B(col, row1));
      return cutlass::Array<uint32_t, 2>{b0, b1};
    } else if constexpr (std::is_same<value_type, uint8_t>::value ||
                         std::is_same<value_type, int8_t>::value ||
                         std::is_same<value_type, f8_e4m3>::value ||
                         std::is_same<value_type, f8_e5m2>::value) {
      int row = tid_in_group * 4;
      int col = gid;
      uint32_t b0 = 0;
  #pragma unroll
      for (int i = 3; i >= 0; i--)
        b0 = (b0 << 8) | bitcast_u32(B(col, row + i));
      return cutlass::Array<uint32_t, 1>{b0};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

// for s8, u8, e4m3, e5m2, e3m2, e2m3, e2m1, and also f16/bf16 for sparse MMA
struct Policy_B_M16N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane % 4;

    int row0 = tid_in_group * 4;
    int row1 = tid_in_group * 4 + 16;
    int col = gid;
    if constexpr (std::is_same<typename Tensor::value_type, f16>::value ||
                  std::is_same<typename Tensor::value_type, bf16>::value) {
      // For f16/bf16 m16n8k32: B is [32, 8], need 4 registers
      int row2 = tid_in_group * 4 + 8;
      uint32_t b0 =
          (bitcast_u32(B(row0 + 1, col)) << 16) | bitcast_u32(B(row0, col));
      uint32_t b1 =
          (bitcast_u32(B(row0 + 3, col)) << 16) | bitcast_u32(B(row0 + 2, col));
      uint32_t b2 =
          (bitcast_u32(B(row2 + 1, col)) << 16) | bitcast_u32(B(row2, col));
      uint32_t b3 =
          (bitcast_u32(B(row2 + 3, col)) << 16) | bitcast_u32(B(row2 + 2, col));
      return cutlass::Array<uint32_t, 4>{b0, b1, b2, b3};
    } else if constexpr (std::is_same<typename Tensor::value_type, s8>::value ||
                         std::is_same<typename Tensor::value_type, u8>::value ||
                         std::is_same<typename Tensor::value_type,
                                      f8_e4m3>::value ||
                         std::is_same<typename Tensor::value_type,
                                      f8_e5m2>::value) {
      uint32_t b0 = (bitcast_u32(B(row0 + 3, col)) << 24) |
                    (bitcast_u32(B(row0 + 2, col)) << 16) |
                    (bitcast_u32(B(row0 + 1, col)) << 8) |
                    bitcast_u32(B(row0, col));
      uint32_t b1 = (bitcast_u32(B(row1 + 3, col)) << 24) |
                    (bitcast_u32(B(row1 + 2, col)) << 16) |
                    (bitcast_u32(B(row1 + 1, col)) << 8) |
                    bitcast_u32(B(row1, col));

      return cutlass::Array<uint32_t, 2>{b0, b1};
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

// Sparse row.row path expects B in K-major (shape [N, K]).
// This policy swaps B indices for f16/bf16 to match that layout.
struct Policy_B_Sparse_M16N8K32 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane % 4;

    int row0 = tid_in_group * 2;
    int col = gid;
    if constexpr (std::is_same<typename Tensor::value_type, f16>::value ||
                  std::is_same<typename Tensor::value_type, bf16>::value) {
      // Swap indices: B is [N, K]
      uint32_t b0 =
          (bitcast_u32(B(col, row0 + 1)) << 16) | bitcast_u32(B(col, row0));
      uint32_t b1 =
          (bitcast_u32(B(col, row0 + 9)) << 16) | bitcast_u32(B(col, row0 + 8));
      uint32_t b2 = (bitcast_u32(B(col, row0 + 17)) << 16) |
                    bitcast_u32(B(col, row0 + 16));
      uint32_t b3 = (bitcast_u32(B(col, row0 + 25)) << 16) |
                    bitcast_u32(B(col, row0 + 24));
      return cutlass::Array<uint32_t, 4>{b0, b1, b2, b3};
    } else {
      return Policy_B_M16N8K32::load(B);
    }
  }
};

// Sparse B policy for m16n8k64 with K-major [N, K] layout (fp8 path)
struct Policy_B_Sparse_M16N8K64 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;

    using value_type = typename Tensor::value_type;
    static_assert(std::is_same<value_type, f8_e4m3>::value ||
                      std::is_same<value_type, f8_e5m2>::value,
                  "Sparse m16n8k64 only supports fp8 types");

    // Swap indices: B is [N, K]
    int col = gid;
    int row0 = tid_in_group * 4;
    int row1 = tid_in_group * 4 + 16;
    int row2 = tid_in_group * 4 + 32;
    int row3 = tid_in_group * 4 + 48;
    uint32_t b0 = (bitcast_u32(B(col, row0 + 3)) << 24) |
                  (bitcast_u32(B(col, row0 + 2)) << 16) |
                  (bitcast_u32(B(col, row0 + 1)) << 8) |
                  bitcast_u32(B(col, row0));
    uint32_t b1 = (bitcast_u32(B(col, row1 + 3)) << 24) |
                  (bitcast_u32(B(col, row1 + 2)) << 16) |
                  (bitcast_u32(B(col, row1 + 1)) << 8) |
                  bitcast_u32(B(col, row1));
    uint32_t b2 = (bitcast_u32(B(col, row2 + 3)) << 24) |
                  (bitcast_u32(B(col, row2 + 2)) << 16) |
                  (bitcast_u32(B(col, row2 + 1)) << 8) |
                  bitcast_u32(B(col, row2));
    uint32_t b3 = (bitcast_u32(B(col, row3 + 3)) << 24) |
                  (bitcast_u32(B(col, row3 + 2)) << 16) |
                  (bitcast_u32(B(col, row3 + 1)) << 8) |
                  bitcast_u32(B(col, row3));
    return cutlass::Array<uint32_t, 4>{b0, b1, b2, b3};
  }
};

// for s4, u4 and e2m1
struct Policy_B_M16N8K64 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_B_M16N8K128 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// for b1
struct Policy_B_M16N8K256 {
  template <class Tensor>
  __device__ static auto load(Tensor const& B) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    // TODO
  }
};

// ------------------- store/load D fragment -------------------
struct Policy_D_M8N8 {
  template <class Tensor, class AccumT>
  __device__ static void store(Tensor& D, AccumT const* d) {
    int lane = threadIdx.x & 31;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value ||
                  std::is_same<value_type, s32>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int gid = lane >> 2;
      int tid_in_group = lane & 3;
      int row = gid;
      int col0 = tid_in_group * 2;
      int col1 = col0 + 1;
      auto D_casted = cute::recast<AccumT>(D);
      D_casted(row, col0) = d[0];
      D_casted(row, col1) = d[1];
    } else if constexpr (std::is_same<AccumT, float>::value) {
      int row = (lane & 1);
      if (lane >= 16) row += 4;
      int col = lane & 2;
      D(row, col) = cast_if<value_type>(d[0]);
      D(row, col + 1) = cast_if<value_type>(d[1]);
      D(row + 2, col) = cast_if<value_type>(d[2]);
      D(row + 2, col + 1) = cast_if<value_type>(d[3]);
      D(row, col + 4) = cast_if<value_type>(d[4]);
      D(row, col + 4 + 1) = cast_if<value_type>(d[5]);
      D(row + 2, col + 4) = cast_if<value_type>(d[6]);
      D(row + 2, col + 4 + 1) = cast_if<value_type>(d[7]);
    } else if constexpr (std::is_same<AccumT, f16>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int row = (lane & 3);
      if (lane >= 16) row = row + 4;
      D(row, 0) = d[0];
      D(row, 1) = d[1];
      D(row, 2) = d[2];
      D(row, 3) = d[3];
      D(row, 4) = d[4];
      D(row, 5) = d[5];
      D(row, 6) = d[6];
      D(row, 7) = d[7];
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }

  template <class Tensor, class AccumT>
  __device__ static void store_trans(Tensor& D, AccumT const* d) {
    int lane = threadIdx.x & 31;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value ||
                  std::is_same<value_type, s32>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int gid = lane >> 2;
      int tid_in_group = lane & 3;
      int row = gid;
      int col0 = tid_in_group * 2;
      int col1 = col0 + 1;
      auto D_casted = cute::recast<AccumT>(D);
      D_casted(col0, row) = d[0];
      D_casted(col1, row) = d[1];
    } else if constexpr (std::is_same<AccumT, float>::value) {
      int row = (lane & 1);
      if (lane >= 16) row += 4;
      int col = lane & 2;
      D(col, row) = cast_if<value_type>(d[0]);
      D(col + 1, row) = cast_if<value_type>(d[1]);
      D(col, row + 2) = cast_if<value_type>(d[2]);
      D(col + 1, row + 2) = cast_if<value_type>(d[3]);
      D(col + 4, row) = cast_if<value_type>(d[4]);
      D(col + 4 + 1, row) = cast_if<value_type>(d[5]);
      D(col + 4, row + 2) = cast_if<value_type>(d[6]);
      D(col + 4 + 1, row + 2) = cast_if<value_type>(d[7]);
    } else if constexpr (std::is_same<AccumT, f16>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int row = (lane & 3);
      if (lane >= 16) row = row + 4;
      D(0, row) = d[0];
      D(1, row) = d[1];
      D(2, row) = d[2];
      D(3, row) = d[3];
      D(4, row) = d[4];
      D(5, row) = d[5];
      D(6, row) = d[6];
      D(7, row) = d[7];
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }

  template <class Tensor, class AccumT>
  __device__ static void load(Tensor const& D, AccumT* d) {
    int lane = threadIdx.x & 31;
    using value_type = typename Tensor::value_type;
    if constexpr (std::is_same<value_type, double>::value ||
                  std::is_same<value_type, s32>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int gid = lane >> 2;
      int tid_in_group = lane & 3;
      int row = gid;
      int col0 = tid_in_group * 2;
      int col1 = col0 + 1;
      auto D_casted = cute::recast<AccumT>(D);
      d[0] = D_casted(row, col0);
      d[1] = D_casted(row, col1);
    } else if constexpr (std::is_same<value_type, float>::value) {
      int row = (lane & 1);
      if (lane >= 16) row += 4;
      int col = lane & 2;
      d[0] = cast_if<AccumT>(D(row, col));
      d[1] = cast_if<AccumT>(D(row, col + 1));
      d[2] = cast_if<AccumT>(D(row + 2, col));
      d[3] = cast_if<AccumT>(D(row + 2, col + 1));
      d[4] = cast_if<AccumT>(D(row, col + 4));
      d[5] = cast_if<AccumT>(D(row, col + 4 + 1));
      d[6] = cast_if<AccumT>(D(row + 2, col + 4));
      d[7] = cast_if<AccumT>(D(row + 2, col + 4 + 1));
    } else if constexpr (std::is_same<value_type, f16>::value) {
      static_assert(std::is_same<AccumT, value_type>::value,
                    "AccumT must be same as value_type");
      int row = (lane & 3);
      if (lane >= 16) row = row + 4;
      d[0] = D(row, 0);
      d[1] = D(row, 1);
      d[2] = D(row, 2);
      d[3] = D(row, 3);
      d[4] = D(row, 4);
      d[5] = D(row, 5);
      d[6] = D(row, 6);
      d[7] = D(row, 7);
    } else {
      static_assert(sizeof(Tensor) != sizeof(Tensor),
                    "unsupported data type in this MMA policy");
    }
  }
};

struct Policy_D_M16N8 {
  template <class Tensor, class AccumT>
  __device__ static void load(Tensor const& D, AccumT* d) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    d[0] = cast_if<AccumT>(D(row0, col0));
    d[1] = cast_if<AccumT>(D(row0, col1));
    d[2] = cast_if<AccumT>(D(row1, col0));
    d[3] = cast_if<AccumT>(D(row1, col1));
  }

  template <class Tensor, class AccumT>
  __device__ static void store(Tensor& D, AccumT const* d) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    using value_type = typename Tensor::value_type;
    D(row0, col0) = cast_if<value_type>(d[0]);
    D(row0, col1) = cast_if<value_type>(d[1]);
    D(row1, col0) = cast_if<value_type>(d[2]);
    D(row1, col1) = cast_if<value_type>(d[3]);
  }

  template <class Tensor, class AccumT>
  __device__ static void store_mask_row(Tensor& D, AccumT const* d,
                                        int row_guard) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
      D(row0, col0) = cast_if<value_type>(d[0]);
      D(row0, col1) = cast_if<value_type>(d[1]);
    }
    if (row1 < row_guard) {
      D(row1, col0) = cast_if<value_type>(d[2]);
      D(row1, col1) = cast_if<value_type>(d[3]);
    }
  }

  template <class Tensor, class AccumT>
  __device__ static void store_mask_col(Tensor& D, AccumT const* d,
                                        int col_guard) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    using value_type = typename Tensor::value_type;
    if (col0 < col_guard) {
      D(row0, col0) = cast_if<value_type>(d[0]);
      D(row1, col0) = cast_if<value_type>(d[2]);
    }
    if (col1 < col_guard) {
      D(row0, col1) = cast_if<value_type>(d[1]);
      D(row1, col1) = cast_if<value_type>(d[3]);
    }
  }

  template <class Tensor, class AccumT>
  __device__ static void store_mask_row_col(Tensor& D, AccumT const* d,
                                            int row_guard, int col_guard) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
      if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[0]);
      if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[1]);
    }
    if (row1 < row_guard) {
      if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[2]);
      if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[3]);
    }
  }

  template <class Tensor, class AccumT>
  __device__ static void store_trans(Tensor& D, AccumT const* d) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tid_in_group = lane & 3;
    int row0 = gid;
    int row1 = gid + 8;
    int col0 = tid_in_group * 2;
    int col1 = tid_in_group * 2 + 1;
    using value_type = typename Tensor::value_type;
    D(col0, row0) = cast_if<value_type>(d[0]);
    D(col1, row0) = cast_if<value_type>(d[1]);
    D(col0, row1) = cast_if<value_type>(d[2]);
    D(col1, row1) = cast_if<value_type>(d[3]);
  }
};

// --------------- TMA primitives (SM90+) ---------------
// TMA SWIZZLE pattern enum for CUtensorMapSwizzle
enum class TMA_Swizzle {
  NONE = 0, // No swizzle
  B32 = 1,  // 32B swizzle
  B64 = 2,  // 64B swizzle
  B128 = 3  // 128B swizzle
};

// Helper function to convert TMA_Swizzle to CUtensorMapSwizzle string
// representation
inline constexpr const char* cuda_stringify(TMA_Swizzle swizzle) {
  switch (swizzle) {
  case TMA_Swizzle::NONE:
    return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE";
  case TMA_Swizzle::B32: return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B";
  case TMA_Swizzle::B64: return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B";
  case TMA_Swizzle::B128:
    return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B";
  default: return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE";
  }
}

// --------------- WGMMA primitives (SM90+) ---------------
// refer to:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-leading-dimension-byte-offset
// 9.7.15.5.1.2.2. Matrix Descriptor Format
// SWIZZLE pattern enum
enum class WGMMA_Swizzle : uint64_t {
  NS = 0,  // No swizzle
  B32 = 3, // 32B swizzle
  B64 = 2, // 64B swizzle
  B128 = 1 // 128B swizzle
};

// Major order enum
enum class WGMMA_MajorOrder {
  K_MAJOR, // K dimension is major (leading)
  MN_MAJOR // M and N dimensions are major (leading)
};

// mma shape enum
enum class WGMMA_MMAShape {
  M64N64K16,
};

// helper functions to get mma property at compile time
template <WGMMA_MMAShape Shape>
__device__ constexpr int get_mma_m() {
  if constexpr (Shape == WGMMA_MMAShape::M64N64K16) return 64;
  return 0;
}

template <WGMMA_MMAShape Shape>
__device__ constexpr int get_mma_n() {
  if constexpr (Shape == WGMMA_MMAShape::M64N64K16) return 64;
  return 0;
}

template <WGMMA_MMAShape Shape>
__device__ constexpr int get_mma_k() {
  if constexpr (Shape == WGMMA_MMAShape::M64N64K16) return 16;
  return 0;
}

template <WGMMA_MajorOrder MajorOrder>
__device__ constexpr int get_trans_a() {
  if constexpr (MajorOrder == WGMMA_MajorOrder::K_MAJOR) return 0;
  if constexpr (MajorOrder == WGMMA_MajorOrder::MN_MAJOR) return 1;
  return 0;
}

template <WGMMA_MajorOrder MajorOrder>
__device__ constexpr int get_trans_b() {
  if constexpr (MajorOrder == WGMMA_MajorOrder::K_MAJOR) return 0;
  if constexpr (MajorOrder == WGMMA_MajorOrder::MN_MAJOR) return 1;
  return 0;
}

// Helper function to encode matrix descriptor
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
  return (((x) & 0x3FFFF) >> 0x4);
}

// Unified shared memory descriptor encoding template
// Automatically determines stride and leading dimension based on major order
// and swizzle
template <WGMMA_MajorOrder MajorOrder, WGMMA_Swizzle Swizzle, typename T>
__device__ static inline uint64_t wgmma_make_smem_desc(T* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0x0000000000000000;
  desc |= matrix_descriptor_encode(addr);

  // Determine stride and leading dimension based on major order and swizzle
  uint64_t LBO = 0;
  uint64_t SBO = 0;

  if constexpr (MajorOrder == WGMMA_MajorOrder::K_MAJOR) {
    // K-major layout: stride varies by swizzle pattern
    switch (Swizzle) {
    case WGMMA_Swizzle::NS:
      LBO = 256;
      SBO = 128;
      break;
    case WGMMA_Swizzle::B32:
      LBO = 16;
      SBO = 256;
      break;
    case WGMMA_Swizzle::B64:
      LBO = 16;
      SBO = 512;
      break;
    case WGMMA_Swizzle::B128:
      LBO = 16;
      SBO = 1024;
      break;
    }
  } else { // MN_MAJOR
    // For swizzled MN-major GMMA layouts, the leading-byte offset remains one
    // 128-bit line (16B for f16/bf16). Only the stride-byte offset scales with
    // the swizzle width; this matches CuTe's canonical GMMA descriptor
    // construction.
    switch (Swizzle) {
    case WGMMA_Swizzle::NS:
      LBO = 16;
      SBO = 128;
      break;
    case WGMMA_Swizzle::B32:
      LBO = 16;
      SBO = 256;
      break;
    case WGMMA_Swizzle::B64:
      LBO = 16;
      SBO = 512;
      break;
    case WGMMA_Swizzle::B128:
      LBO = 16;
      SBO = 1024;
      break;
    }
  }

  desc |= matrix_descriptor_encode(LBO) << 16;
  desc |= matrix_descriptor_encode(SBO) << 32;
  desc |= static_cast<uint64_t>(Swizzle) << 62;

  return desc;
}

// WGMMA fence/sync primitives
__device__ static inline void warpgroup_arrive() {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  #endif
}

__device__ static inline void warpgroup_commit_batch() {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  #endif
}

template <int PD>
__device__ static inline void warpgroup_wait() {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  static_assert(PD >= 0 && PD <= 7, "WGMMA wait: N must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(PD) : "memory");
  #endif
}

// Unified WGMMA template with automatic descriptor selection
// Template parameters:
//   - InputT: input data type (__half or __nv_bfloat16)
//   - OutputT: output data type (float or same as InputT)
//   - MajorOrderA: major order for matrix A (K_MAJOR or MN_MAJOR)
//   - SwizzleA: swizzle pattern for matrix A
//   - MajorOrderB: major order for matrix B (K_MAJOR or MN_MAJOR)
//   - SwizzleB: swizzle pattern for matrix B
template <typename InputT, typename OutputT,
          WGMMA_MajorOrder MajorOrderA = WGMMA_MajorOrder::K_MAJOR,
          WGMMA_Swizzle SwizzleA = WGMMA_Swizzle::NS,
          WGMMA_MajorOrder MajorOrderB = WGMMA_MajorOrder::K_MAJOR,
          WGMMA_Swizzle SwizzleB = WGMMA_Swizzle::NS>
__device__ static __forceinline__ void wgmma_m64n64k16(OutputT d[4][8],
                                                       InputT* sA, InputT* sB) {
  static_assert(
      std::is_same_v<InputT, __half> || std::is_same_v<InputT, __nv_bfloat16> ||
          std::is_same_v<InputT, f8_e4m3> || std::is_same_v<InputT, f8_e5m2>,
      "wgmma_m64n64k16_unified requires __half, __nv_bfloat16 or fp8 input "
      "type");
  static_assert(
      std::is_same_v<OutputT, float> || std::is_same_v<OutputT, InputT>,
      "wgmma_m64n64k16_unified requires float or same as InputT output type");

  uint64_t desc_a = wgmma_make_smem_desc<MajorOrderA, SwizzleA>(&sA[0]);
  uint64_t desc_b = wgmma_make_smem_desc<MajorOrderB, SwizzleB>(&sB[0]);
  constexpr uint64_t trans_a = get_trans_a<MajorOrderA>();
  constexpr uint64_t trans_b = get_trans_b<MajorOrderB>();

  // Determine PTX instruction based on input and output types
  if constexpr (std::is_same_v<InputT, __half> &&
                std::is_same_v<OutputT, __half>) {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+h"(*(uint16_t*)&d[0][0]), "+h"(*(uint16_t*)&d[0][1]),
                   "+h"(*(uint16_t*)&d[0][2]), "+h"(*(uint16_t*)&d[0][3]),
                   "+h"(*(uint16_t*)&d[0][4]), "+h"(*(uint16_t*)&d[0][5]),
                   "+h"(*(uint16_t*)&d[0][6]), "+h"(*(uint16_t*)&d[0][7]),
                   "+h"(*(uint16_t*)&d[1][0]), "+h"(*(uint16_t*)&d[1][1]),
                   "+h"(*(uint16_t*)&d[1][2]), "+h"(*(uint16_t*)&d[1][3]),
                   "+h"(*(uint16_t*)&d[1][4]), "+h"(*(uint16_t*)&d[1][5]),
                   "+h"(*(uint16_t*)&d[1][6]), "+h"(*(uint16_t*)&d[1][7]),
                   "+h"(*(uint16_t*)&d[2][0]), "+h"(*(uint16_t*)&d[2][1]),
                   "+h"(*(uint16_t*)&d[2][2]), "+h"(*(uint16_t*)&d[2][3]),
                   "+h"(*(uint16_t*)&d[2][4]), "+h"(*(uint16_t*)&d[2][5]),
                   "+h"(*(uint16_t*)&d[2][6]), "+h"(*(uint16_t*)&d[2][7]),
                   "+h"(*(uint16_t*)&d[3][0]), "+h"(*(uint16_t*)&d[3][1]),
                   "+h"(*(uint16_t*)&d[3][2]), "+h"(*(uint16_t*)&d[3][3]),
                   "+h"(*(uint16_t*)&d[3][4]), "+h"(*(uint16_t*)&d[3][5]),
                   "+h"(*(uint16_t*)&d[3][6]), "+h"(*(uint16_t*)&d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(1), "n"(1), "n"(1),
                   "n"(trans_a), "n"(trans_b));
  #endif
  } else if constexpr (std::is_same_v<InputT, __half> &&
                       std::is_same_v<OutputT, float>) {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                   "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                   "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                   "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                   "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                   "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                   "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                   "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(1), "n"(1), "n"(1),
                   "n"(trans_a), "n"(trans_b));
  #endif
  } else if constexpr (std::is_same_v<InputT, __nv_bfloat16> &&
                       std::is_same_v<OutputT, __nv_bfloat16>) {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.bf16.bf16.bf16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+h"(*(uint16_t*)&d[0][0]), "+h"(*(uint16_t*)&d[0][1]),
                   "+h"(*(uint16_t*)&d[0][2]), "+h"(*(uint16_t*)&d[0][3]),
                   "+h"(*(uint16_t*)&d[0][4]), "+h"(*(uint16_t*)&d[0][5]),
                   "+h"(*(uint16_t*)&d[0][6]), "+h"(*(uint16_t*)&d[0][7]),
                   "+h"(*(uint16_t*)&d[1][0]), "+h"(*(uint16_t*)&d[1][1]),
                   "+h"(*(uint16_t*)&d[1][2]), "+h"(*(uint16_t*)&d[1][3]),
                   "+h"(*(uint16_t*)&d[1][4]), "+h"(*(uint16_t*)&d[1][5]),
                   "+h"(*(uint16_t*)&d[1][6]), "+h"(*(uint16_t*)&d[1][7]),
                   "+h"(*(uint16_t*)&d[2][0]), "+h"(*(uint16_t*)&d[2][1]),
                   "+h"(*(uint16_t*)&d[2][2]), "+h"(*(uint16_t*)&d[2][3]),
                   "+h"(*(uint16_t*)&d[2][4]), "+h"(*(uint16_t*)&d[2][5]),
                   "+h"(*(uint16_t*)&d[2][6]), "+h"(*(uint16_t*)&d[2][7]),
                   "+h"(*(uint16_t*)&d[3][0]), "+h"(*(uint16_t*)&d[3][1]),
                   "+h"(*(uint16_t*)&d[3][2]), "+h"(*(uint16_t*)&d[3][3]),
                   "+h"(*(uint16_t*)&d[3][4]), "+h"(*(uint16_t*)&d[3][5]),
                   "+h"(*(uint16_t*)&d[3][6]), "+h"(*(uint16_t*)&d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(1), "n"(1), "n"(1),
                   "n"(trans_a), "n"(trans_b));
  #endif
  } else if constexpr (std::is_same_v<InputT, __nv_bfloat16> &&
                       std::is_same_v<OutputT, float>) {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                   "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                   "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                   "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                   "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                   "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                   "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                   "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(1), "n"(1), "n"(1),
                   "n"(trans_a), "n"(trans_b));
  #endif
  } else if constexpr ((std::is_same_v<InputT, f8_e4m3> ||
                        std::is_same_v<InputT, f8_e5m2>) &&
                       std::is_same_v<OutputT, float>) {
  #if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f8.f8 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                   "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                   "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                   "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                   "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                   "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                   "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                   "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(1), "n"(1), "n"(1),
                   "n"(trans_a), "n"(trans_b));
  #endif
  }
}

// wgmma store d
// reference:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shape
struct Policy_WGMMA_D_M64K16 {
  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row(Tensor& D, AccumT* d, int row_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_col(Tensor& D, AccumT* d, int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      if (col0 < col_guard) {
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      }
      if (col1 < col_guard) {
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row_col(Tensor& D, AccumT* d, int row_guard,
                                            int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[c * 4]);
        if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(row0, col0) = cast_if<value_type>(d[c * 4]);
      D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_trans(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(col0, row0) = cast_if<value_type>(d[c * 4]);
      D(col1, row0) = cast_if<value_type>(d[c * 4 + 1]);
      D(col0, row1) = cast_if<value_type>(d[c * 4 + 2]);
      D(col1, row1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }
};

// Store policy for 64x64x8 WGMMA (accumulator layout matches K=16)
struct Policy_WGMMA_D_M64K8 {
  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row(Tensor& D, AccumT* d, int row_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_col(Tensor& D, AccumT* d, int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      if (col0 < col_guard) {
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      }
      if (col1 < col_guard) {
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row_col(Tensor& D, AccumT* d, int row_guard,
                                            int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[c * 4]);
        if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;             // 0-31: lane within warp
    int warp = tid / 32;             // 0-3: which warp in warp group
    int row0 = warp * 16 + lane / 4; // first row
    int row1 = row0 + 8;             // second row
    int col_num = N / 8;             // number of column pairs
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(row0, col0) = cast_if<value_type>(d[c * 4]);
      D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_trans(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(col0, row0) = cast_if<value_type>(d[c * 4]);
      D(col1, row0) = cast_if<value_type>(d[c * 4 + 1]);
      D(col0, row1) = cast_if<value_type>(d[c * 4 + 2]);
      D(col1, row1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }
};

// Store policy for 64x64x32 WGMMA (accumulator layout matches K=16)
struct Policy_WGMMA_D_M64K32 {
  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_col(Tensor& D, AccumT* d, int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      if (col0 < col_guard) {
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      }
      if (col1 < col_guard) {
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row(Tensor& D, AccumT* d, int row_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8; // number of column pairs
    using value_type = typename Tensor::value_type;

    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }

    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row_col(Tensor& D, AccumT* d, int row_guard,
                                            int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[c * 4]);
        if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(row0, col0) = cast_if<value_type>(d[c * 4]);
      D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_trans(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(col0, row0) = cast_if<value_type>(d[c * 4]);
      D(col1, row0) = cast_if<value_type>(d[c * 4 + 1]);
      D(col0, row1) = cast_if<value_type>(d[c * 4 + 2]);
      D(col1, row1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }
};

// Store policy for 64x64x64 WGMMA (accumulator layout matches K=32 paths)
struct Policy_WGMMA_D_M64K64 {
  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row(Tensor& D, AccumT* d, int row_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_col(Tensor& D, AccumT* d, int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      if (col0 < col_guard) {
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      }
      if (col1 < col_guard) {
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row_col(Tensor& D, AccumT* d, int row_guard,
                                            int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[c * 4]);
        if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(row0, col0) = cast_if<value_type>(d[c * 4]);
      D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_trans(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(col0, row0) = cast_if<value_type>(d[c * 4]);
      D(col1, row0) = cast_if<value_type>(d[c * 4 + 1]);
      D(col0, row1) = cast_if<value_type>(d[c * 4 + 2]);
      D(col1, row1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }
};

// Store policy for 64x64x256 WGMMA (binary accumulator layout matches K=16)
struct Policy_WGMMA_D_M64K256 {
  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row(Tensor& D, AccumT* d, int row_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_col(Tensor& D, AccumT* d, int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      if (col0 < col_guard) {
        D(row0, col0) = cast_if<value_type>(d[c * 4]);
        D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      }
      if (col1 < col_guard) {
        D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
        D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_mask_row_col(Tensor& D, AccumT* d, int row_guard,
                                            int col_guard) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
    if (row0 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row0, col0) = cast_if<value_type>(d[c * 4]);
        if (col1 < col_guard) D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      }
    }
    if (row1 < row_guard) {
  #pragma unroll
      for (int c = 0; c < col_num; c++) {
        int col0 = c * 8 + (tid % 4) * 2;
        int col1 = col0 + 1;
        if (col0 < col_guard) D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
        if (col1 < col_guard) D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
      }
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(row0, col0) = cast_if<value_type>(d[c * 4]);
      D(row0, col1) = cast_if<value_type>(d[c * 4 + 1]);
      D(row1, col0) = cast_if<value_type>(d[c * 4 + 2]);
      D(row1, col1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }

  template <class Tensor, typename AccumT, int N>
  __device__ static void store_trans(Tensor& D, AccumT* d) {
    int tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_num = N / 8;
    using value_type = typename Tensor::value_type;
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int col0 = c * 8 + (tid % 4) * 2;
      int col1 = col0 + 1;
      D(col0, row0) = cast_if<value_type>(d[c * 4]);
      D(col1, row0) = cast_if<value_type>(d[c * 4 + 1]);
      D(col0, row1) = cast_if<value_type>(d[c * 4 + 2]);
      D(col1, row1) = cast_if<value_type>(d[c * 4 + 3]);
    }
  }
};

// unified MMA load/store fragment interfaces
// load A fragment
template <class MMA, class Tensor>
__device__ static inline auto load_fragment_a(Tensor const& A) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  return MMA_Policy<MMA>::typeA::load(A);
}

// load B fragment
template <class MMA, class Tensor>
__device__ static inline auto load_fragment_b(Tensor const& B) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  return MMA_Policy<MMA>::typeB::load(B);
}

// load E fragment (metadata)
template <class MMA, class Tensor>
__device__ static inline auto load_fragment_e(Tensor const& E) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  return MMA_Policy<MMA>::typeE::load(E);
}

// load d fragment with pointer
template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static inline void load_fragment_d(Tensor const& D,
                                              AccumT* const d) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "load d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "load d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template load<Tensor, AccumT, N>(D, d);
  else
    MMA_Policy<MMA>::typeD::template load<Tensor, AccumT>(D, d);
}

// store d fragment with pointer
template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static inline void store_fragment_d(Tensor& D, AccumT* const d) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "store d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "store d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template store<Tensor, AccumT, N>(D, d);
  else
    MMA_Policy<MMA>::typeD::template store<Tensor, AccumT>(D, d);
}

template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static inline void
store_fragment_d_mask_row(Tensor& D, AccumT* const d, int row_guard) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "store d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "store d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template store_mask_row<Tensor, AccumT, N>(
        D, d, row_guard);
  else
    MMA_Policy<MMA>::typeD::template store_mask_row<Tensor, AccumT>(D, d,
                                                                    row_guard);
}

template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static inline void
store_fragment_d_mask_col(Tensor& D, AccumT* const d, int col_guard) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "store d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "store d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template store_mask_col<Tensor, AccumT, N>(
        D, d, col_guard);
  else
    MMA_Policy<MMA>::typeD::template store_mask_col<Tensor, AccumT>(D, d,
                                                                    col_guard);
}

template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static inline void
store_fragment_d_mask_row_col(Tensor& D, AccumT* const d, int row_guard,
                              int col_guard) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "store d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "store d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template store_mask_row_col<Tensor, AccumT, N>(
        D, d, row_guard, col_guard);
  else
    MMA_Policy<MMA>::typeD::template store_mask_row_col<Tensor, AccumT>(
        D, d, row_guard, col_guard);
}

template <class MMA, int N = 0, class Tensor, class AccumT>
__device__ static inline void store_fragment_d_trans(Tensor& D,
                                                     AccumT* const d) {
  static_assert(MMA_Policy<MMA>::supported, "No policy for this MMA");
  static_assert(std::is_same<AccumT, float>::value ||
                    std::is_same<AccumT, double>::value ||
                    std::is_same<AccumT, f16>::value ||
                    std::is_same<AccumT, s32>::value,
                "store d only supports float/double/f16/s32 accumulator type");
  static_assert(AccumTCast<typename Tensor::value_type, AccumT>::supported ||
                    std::is_same<typename Tensor::value_type, AccumT>::value,
                "store d unsupported type cast");
  if constexpr (N > 0)
    MMA_Policy<MMA>::typeD::template store_trans<Tensor, AccumT, N>(D, d);
  else
    MMA_Policy<MMA>::typeD::template store_trans<Tensor, AccumT>(D, d);
}

template <class MMA, int N, class Tensor>
__device__ static inline void
store_fragment_d_stmatrix_f32_bf16(Tensor& D, float* const d) {
  static_assert(std::is_same<typename Tensor::value_type, bf16>::value,
                "stmatrix f32->bf16 store requires bf16 output tensor");
  static_assert(
      (N % 8) == 0,
      "stmatrix f32->bf16 store currently requires N to be divisible by 8");

  int tid = threadIdx.x % 128;
  int lane = tid % 32;
  int warp = tid / 32;

  bf16* d_base_ptr = &D(0, 0);
  uint32_t d_base_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(d_base_ptr));

  constexpr int w_iters = N / 16;
  constexpr int tail_cols = N % 16;
  constexpr int full_cols = w_iters * 16;
  constexpr uint32_t kStepBytes = static_cast<uint32_t>(16 * sizeof(bf16));

  uint32_t lane_offset =
      static_cast<uint32_t>((lane % 8) * N + (lane / 16) * N * 8 + (lane & 8));
  uint32_t addr = d_base_addr +
                  static_cast<uint32_t>((warp * 16 * N) * sizeof(bf16)) +
                  lane_offset * sizeof(bf16);

  #pragma unroll
  for (int w = 0; w < w_iters; ++w) {
    int base = w * 8;
    bf16 v0 = utils::from_f32<bf16>(d[base + 0]);
    bf16 v1 = utils::from_f32<bf16>(d[base + 1]);
    bf16 v2 = utils::from_f32<bf16>(d[base + 2]);
    bf16 v3 = utils::from_f32<bf16>(d[base + 3]);
    bf16 v4 = utils::from_f32<bf16>(d[base + 4]);
    bf16 v5 = utils::from_f32<bf16>(d[base + 5]);
    bf16 v6 = utils::from_f32<bf16>(d[base + 6]);
    bf16 v7 = utils::from_f32<bf16>(d[base + 7]);

    uint32_t r0 = static_cast<uint32_t>(bitcast_uint(v0)) |
                  (static_cast<uint32_t>(bitcast_uint(v1)) << 16);
    uint32_t r1 = static_cast<uint32_t>(bitcast_uint(v4)) |
                  (static_cast<uint32_t>(bitcast_uint(v5)) << 16);
    uint32_t r2 = static_cast<uint32_t>(bitcast_uint(v2)) |
                  (static_cast<uint32_t>(bitcast_uint(v3)) << 16);
    uint32_t r3 = static_cast<uint32_t>(bitcast_uint(v6)) |
                  (static_cast<uint32_t>(bitcast_uint(v7)) << 16);

    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], "
                 "{%1, %2, %3, %4};\n"
                 :
                 : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    addr += kStepBytes;
  }

  if constexpr (tail_cols != 0) {
    static_assert(
        tail_cols == 8,
        "stmatrix f32->bf16 tail path only supports one 8-column remainder");
    int row0 = warp * 16 + lane / 4;
    int row1 = row0 + 8;
    int col0 = full_cols + (tid % 4) * 2;
    int col1 = col0 + 1;
    int base = w_iters * 8;

    D(row0, col0) = utils::from_f32<bf16>(d[base + 0]);
    D(row0, col1) = utils::from_f32<bf16>(d[base + 1]);
    D(row1, col0) = utils::from_f32<bf16>(d[base + 2]);
    D(row1, col1) = utils::from_f32<bf16>(d[base + 3]);
  }
}

template <class MMA, int N, class Tensor>
__device__ static inline void
store_fragment_d_stmatrix_trans_f32_bf16(Tensor& D, float* const d) {
  static_assert(
      std::is_same<typename Tensor::value_type, bf16>::value,
      "transposed stmatrix f32->bf16 store requires bf16 output tensor");
  static_assert((N % 8) == 0, "transposed stmatrix f32->bf16 store currently "
                              "requires N to be divisible by 8");

  int tid = threadIdx.x % 128;
  int lane = tid % 32;
  int warp = tid / 32;

  bf16* d_base_ptr = &D(0, 0);
  uint32_t d_base_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(d_base_ptr));

  const uint32_t ld = static_cast<uint32_t>(&D(1, 0) - &D(0, 0));
  constexpr int w_iters = N / 16;
  const uint32_t kStepBytes = static_cast<uint32_t>(16 * ld * sizeof(bf16));

  uint32_t lane_offset = static_cast<uint32_t>(warp * 16) +
                         static_cast<uint32_t>(lane % 8) * ld +
                         static_cast<uint32_t>(lane / 16) * ld * 8 +
                         static_cast<uint32_t>(lane & 8);
  uint32_t addr = d_base_addr + lane_offset * sizeof(bf16);

  #pragma unroll
  for (int w = 0; w < w_iters; ++w) {
    int base = w * 8;
    bf16 v0 = utils::from_f32<bf16>(d[base + 0]);
    bf16 v1 = utils::from_f32<bf16>(d[base + 1]);
    bf16 v2 = utils::from_f32<bf16>(d[base + 2]);
    bf16 v3 = utils::from_f32<bf16>(d[base + 3]);
    bf16 v4 = utils::from_f32<bf16>(d[base + 4]);
    bf16 v5 = utils::from_f32<bf16>(d[base + 5]);
    bf16 v6 = utils::from_f32<bf16>(d[base + 6]);
    bf16 v7 = utils::from_f32<bf16>(d[base + 7]);

    uint32_t r0 = static_cast<uint32_t>(bitcast_uint(v0)) |
                  (static_cast<uint32_t>(bitcast_uint(v1)) << 16);
    uint32_t r1 = static_cast<uint32_t>(bitcast_uint(v2)) |
                  (static_cast<uint32_t>(bitcast_uint(v3)) << 16);
    uint32_t r2 = static_cast<uint32_t>(bitcast_uint(v4)) |
                  (static_cast<uint32_t>(bitcast_uint(v5)) << 16);
    uint32_t r3 = static_cast<uint32_t>(bitcast_uint(v6)) |
                  (static_cast<uint32_t>(bitcast_uint(v7)) << 16);

    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%0], "
                 "{%1, %2, %3, %4};\n"
                 :
                 : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    addr += kStepBytes;
  }
}

// stmatrix-based store for WGMMA accumulators (--stmatrix flag).
//
// Uses PTX stmatrix.sync.aligned.m8n8.x1.b16 to write 8x8 sub-tiles of
// the WGMMA accumulator to shared memory in a bank-conflict-free manner.
//
// Because stmatrix stores in dense row-major stride-8 format (128-byte
// blocks), but the destination tensor D has stride N, we use a per-warp
// temp buffer in shared memory: stmatrix writes to the temp buffer, then
// threads scatter to the correct strided locations in D.
//
// A register shuffle (__shfl_sync) re-maps the WGMMA accumulator register
// layout (row=lane/4, col_pair=lane%4) to the stmatrix-expected layout
// (row=lane%8, col_pair=lane/8) before each stmatrix call.
//
// NOTE: This is a demonstrative implementation.  For maximum performance,
//       the output shared memory layout should be changed to tile-8 format
//       so that stmatrix can write directly without the scatter step, and
//       TMA output should use a matching swizzled tensor map.
template <class MMA, int N, class Tensor, class AccumT>
__device__ static inline void store_fragment_d_stmatrix(Tensor& D,
                                                        AccumT* const d) {
  static_assert(sizeof(AccumT) == 2,
                "stmatrix store requires 16-bit accumulator type, f16 or bf16");
  static_assert((N % 8) == 0,
                "stmatrix store currently requires N to be divisible by 8");

  int tid = threadIdx.x % 128;
  int lane = tid % 32;
  int warp = tid / 32;

  AccumT* d_base_ptr = &D(0, 0);
  uint32_t d_base_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(d_base_ptr));

  constexpr int w_iters = N / 16;
  constexpr uint32_t kStepBytes = static_cast<uint32_t>(16 * sizeof(AccumT));

  uint32_t lane_offset =
      static_cast<uint32_t>((lane % 8) * N + (lane / 16) * N * 8 + (lane & 8));
  uint32_t addr = d_base_addr +
                  static_cast<uint32_t>((warp * 16 * N) * sizeof(AccumT)) +
                  lane_offset * sizeof(AccumT);
  auto d_u32 = reinterpret_cast<uint32_t const*>(d);

  #pragma unroll
  for (int w = 0; w < w_iters; ++w) {
    uint32_t r0 = d_u32[w * 4 + 0];
    uint32_t r1 = d_u32[w * 4 + 2];
    uint32_t r2 = d_u32[w * 4 + 1];
    uint32_t r3 = d_u32[w * 4 + 3];

    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], "
                 "{%1, %2, %3, %4};\n"
                 :
                 : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    addr += kStepBytes;
  }
}

template <class MMA, int N, class Tensor, class AccumT>
__device__ static inline void store_fragment_d_stmatrix_trans(Tensor& D,
                                                              AccumT* const d) {
  static_assert(sizeof(AccumT) == 2,
                "stmatrix store requires 16-bit accumulator type, f16 or bf16");
  static_assert((N % 8) == 0,
                "stmatrix store currently requires N to be divisible by 8");

  int tid = threadIdx.x % 128;
  int lane = tid % 32;
  int warp = tid / 32;

  AccumT* d_base_ptr = &D(0, 0);
  uint32_t d_base_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(d_base_ptr));

  const uint32_t ld = static_cast<uint32_t>(&D(1, 0) - &D(0, 0));
  constexpr int w_iters = N / 16;
  const uint32_t kStepBytes = static_cast<uint32_t>(16 * ld * sizeof(AccumT));

  uint32_t lane_offset = static_cast<uint32_t>(warp * 16) +
                         static_cast<uint32_t>(lane % 8) * ld +
                         static_cast<uint32_t>(lane / 16) * ld * 8 +
                         static_cast<uint32_t>(lane & 8);
  uint32_t addr = d_base_addr + lane_offset * sizeof(AccumT);
  auto d_u32 = reinterpret_cast<uint32_t const*>(d);

  #pragma unroll
  for (int w = 0; w < w_iters; ++w) {
    uint32_t r0 = d_u32[w * 4 + 0];
    uint32_t r1 = d_u32[w * 4 + 1];
    uint32_t r2 = d_u32[w * 4 + 2];
    uint32_t r3 = d_u32[w * 4 + 3];

    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%0], "
                 "{%1, %2, %3, %4};\n"
                 :
                 : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    addr += kStepBytes;
  }
}

// M64 WGMMA accumulator scaling.
//
// d[i] += scale_d[i] * scale_a[row] * scale_b
//
// Thread->row mapping (M64 WGMMA layout, warpgroup of 128 threads):
//   row0 = warp*16 + (lane>>2)       in [0..55]
//   row1 = row0 + 8                  in [8..63]
//
// valid_rows: number of valid rows in scale_a. Threads whose rows
// exceed this are zeroed. When valid_rows >= 64 (the full M64 tile),
// the masking comparisons always pass and the cost is negligible.
// A single implementation avoids duplicate template instantiations
// that would increase register pressure in the calling kernel.
template <typename AccT, typename ScaleT, int N>
__device__ __forceinline__ void
scale_accumulator(AccT* d, AccT* scale_d, ScaleT* scale_a_ptr, int scale_a_ld,
                  int valid_rows, ScaleT scale_b) {
  static_assert(std::is_same_v<ScaleT, f32>,
                "scale_accumulator: ScaleT must be f32");
  static_assert(std::is_same_v<AccT, f16> || std::is_same_v<AccT, float>,
                "scale_accumulator: AccT must be f16 or float");

  int itd = threadIdx.x & 127;
  int lane = itd & 31;
  int warp = itd >> 5;
  int row0 = warp * 16 + (lane >> 2);
  int row1 = row0 + 8;
  constexpr int col_num = N / 8;

  int row0_valid = row0 < valid_rows;
  int row1_valid = row1 < valid_rows;
  auto* sa_ptr0 = scale_a_ptr + (row0_valid ? row0 : 0) * scale_a_ld;
  auto* sa_ptr1 = scale_a_ptr + (row1_valid ? row1 : 0) * scale_a_ld;

  float sa0 = 0.0f, sa1 = 0.0f;
  #if defined(__CUDA_ARCH__)
  if (__isShared(scale_a_ptr)) {
    sa0 = *sa_ptr0;
    sa1 = *sa_ptr1;
  } else if (__isGlobal(scale_a_ptr)) {
    sa0 = __ldg(sa_ptr0);
    sa1 = __ldg(sa_ptr1);
  } else {
    sa0 = *sa_ptr0;
    sa1 = *sa_ptr1;
  }
  #else
  sa0 = *sa_ptr0;
  sa1 = *sa_ptr1;
  #endif
  sa0 *= scale_b * static_cast<float>(row0_valid);
  sa1 *= scale_b * static_cast<float>(row1_valid);

  if constexpr (std::is_same_v<AccT, f16>) {
  #if defined(__USE_CUDA_TYPE__)
    auto* d2 = reinterpret_cast<__half2*>(d);
    auto const* scale_d2 = reinterpret_cast<__half2 const*>(scale_d);
    __half2 sa0_h2 = __float2half2_rn(sa0);
    __half2 sa1_h2 = __float2half2_rn(sa1);
    #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int base2 = c * 2;
      d2[base2 + 0] = __hfma2(scale_d2[base2 + 0], sa0_h2, d2[base2 + 0]);
      d2[base2 + 1] = __hfma2(scale_d2[base2 + 1], sa1_h2, d2[base2 + 1]);
    }
  #else
    #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int base = c * 4;
      d[base + 0] += utils::from_f32<f16>(to_f32(scale_d[base + 0]) * sa0);
      d[base + 1] += utils::from_f32<f16>(to_f32(scale_d[base + 1]) * sa0);
      d[base + 2] += utils::from_f32<f16>(to_f32(scale_d[base + 2]) * sa1);
      d[base + 3] += utils::from_f32<f16>(to_f32(scale_d[base + 3]) * sa1);
    }
  #endif
  } else if constexpr (std::is_same_v<AccT, float>) {
  #pragma unroll
    for (int c = 0; c < col_num; c++) {
      int base = c * 4;
      d[base + 0] = fmaf(scale_d[base + 0], sa0, d[base + 0]);
      d[base + 1] = fmaf(scale_d[base + 1], sa0, d[base + 1]);
      d[base + 2] = fmaf(scale_d[base + 2], sa1, d[base + 2]);
      d[base + 3] = fmaf(scale_d[base + 3], sa1, d[base + 3]);
    }
  }
}

// --------------- MMA policy specializations ---------------
struct MMA {};
struct CUTE_MMA : MMA {};
struct CUTE_MMA_M8N8K4 : CUTE_MMA {};
struct CUTE_MMA_M8N8K16 : CUTE_MMA {};
struct CUTE_MMA_M16N8K4 : CUTE_MMA {};
struct CUTE_MMA_M16N8K8 : CUTE_MMA {};
struct CUTE_MMA_M16N8K16 : CUTE_MMA {};
struct CUTE_MMA_M16N8K32 : CUTE_MMA {};
struct CUTE_MMA_M16N8K128 : CUTE_MMA {};
// Sparse MMA atom types for 2:4 structured sparsity
struct CUTE_MMA_SPARSE_M16N8K16 : CUTE_MMA {};
struct CUTE_MMA_SPARSE_M16N8K32 : CUTE_MMA {};
struct CUTE_MMA_SPARSE_M16N8K64 : CUTE_MMA {};
struct CUTE_WGMMA : MMA {};
struct CUTE_WGMMA_M64K8 : CUTE_WGMMA {};
struct CUTE_WGMMA_M64K16 : CUTE_WGMMA {};
struct CUTE_WGMMA_M64K32 : CUTE_WGMMA {};
struct CUTE_WGMMA_M64K64 : CUTE_WGMMA {};
struct CUTE_WGMMA_M64k256 : CUTE_WGMMA {};

template <>
struct MMA_Policy<CUTE_MMA_M8N8K4> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M8N8K4;
  using typeB = Policy_B_M8N8K4;
  using typeD = Policy_D_M8N8;
};

template <>
struct MMA_Policy<CUTE_MMA_M8N8K16> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M8N8K16;
  using typeB = Policy_B_M8N8K16;
  using typeD = Policy_D_M8N8;
};

template <>
struct MMA_Policy<CUTE_MMA_M16N8K4> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M16N8K4;
  using typeB = Policy_B_M16N8K4;
  using typeD = Policy_D_M16N8;
};

template <>
struct MMA_Policy<CUTE_MMA_M16N8K8> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M16N8K8;
  using typeB = Policy_B_M16N8K8;
  using typeD = Policy_D_M16N8;
};

template <>
struct MMA_Policy<CUTE_MMA_M16N8K16> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M16N8K16;
  using typeB = Policy_B_M16N8K16;
  using typeD = Policy_D_M16N8;
};

template <>
struct MMA_Policy<CUTE_MMA_M16N8K32> {
  static constexpr bool supported = true;
  using typeA = Policy_A_M16N8K32;
  using typeB = Policy_B_M16N8K32;
  using typeD = Policy_D_M16N8;
};

// Sparse MMA policies for 2:4 structured sparsity
template <>
struct MMA_Policy<CUTE_MMA_SPARSE_M16N8K16> {
  static constexpr bool supported = true;
  using typeA = Policy_A_Sparse_M16N8K16;
  using typeB = Policy_B_Sparse_M16N8K16;
  using typeD = Policy_D_M16N8;
  using typeE = Policy_E_Sparse_M16N8K16;
};

template <>
struct MMA_Policy<CUTE_MMA_SPARSE_M16N8K32> {
  static constexpr bool supported = true;
  using typeA = Policy_A_Sparse_M16N8K32;
  using typeB = Policy_B_Sparse_M16N8K32;
  using typeD = Policy_D_M16N8;
  using typeE = Policy_E_Sparse_M16N8K32;
};

template <>
struct MMA_Policy<CUTE_MMA_SPARSE_M16N8K64> {
  static constexpr bool supported = true;
  using typeA = Policy_A_Sparse_M16N8K64;
  using typeB = Policy_B_Sparse_M16N8K64;
  using typeD = Policy_D_M16N8;
  using typeE = Policy_E_Sparse_M16N8K64;
};

// wgmma policies
template <>
struct MMA_Policy<CUTE_WGMMA_M64K16> {

  static constexpr bool supported = true;
  using typeD = Policy_WGMMA_D_M64K16;
};

template <>
struct MMA_Policy<CUTE_WGMMA_M64K8> {
  static constexpr bool supported = true;
  using typeD = Policy_WGMMA_D_M64K8;
};

template <>
struct MMA_Policy<CUTE_WGMMA_M64K32> {
  static constexpr bool supported = true;
  using typeD = Policy_WGMMA_D_M64K32;
};

template <>
struct MMA_Policy<CUTE_WGMMA_M64K64> {
  static constexpr bool supported = true;
  using typeD = Policy_WGMMA_D_M64K64;
};

template <>
struct MMA_Policy<CUTE_WGMMA_M64k256> {
  static constexpr bool supported = true;
  using typeD = Policy_WGMMA_D_M64K256;
};

// TODO: all 16x8x64 (sub byte)

// TODO: all 16x8x128 (b1)

// TODO: all 16x8x256 (b1)

// --------------- WGMMA policies (SM90+) ---------------
// Note: WGMMA uses PTX inline assembly directly via wgmma_m64n64k16<>
// template. No MMA_Policy specializations are needed for WGMMA as it bypasses
// the cute MMA policy system.

} // namespace choreo

namespace cute {

// --------------- Sparse MMA PTX wrappers for SM80+ ---------------
// These implement mma.sp.sync.aligned instructions for 2:4 structured sparsity

// fp16 m16n8k16 sparse MMA: C = A_sparse * B + C
struct SM80_SPARSE_16x8x16_F32F16F16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(float& d0, float& d1, float& d2, float& d3,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& b0, uint32_t const& b1,
                                   float const& c0, float const& c1,
                                   float const& c2, float const& c3,
                                   uint32_t const& e, int const& spsel = 0) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #if (__CUDACC_VER_MAJOR__ > 12) ||                                         \
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16."
        "f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(e));
    #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(e));
    #endif
  #endif
  }
};

// fp16 m16n8k32 sparse MMA: C = A_sparse * B + C
struct SM80_SPARSE_16x8x32_F32F16F16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #if (__CUDACC_VER_MAJOR__ > 12) ||                                         \
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32."
                 "f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, "
                 "%1, %2, %3}, %12, 0x0;\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(e));
    #else
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, "
                 "%1, %2, %3}, %12, 0x0;\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(e));
    #endif
  #endif
  }
};

// bf16 m16n8k16 sparse MMA: C = A_sparse * B + C
struct SM80_SPARSE_16x8x16_F32BF16BF16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(float& d0, float& d1, float& d2, float& d3,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& b0, uint32_t const& b1,
                                   float const& c0, float const& c1,
                                   float const& c2, float const& c3,
                                   uint32_t const& e, int const& spsel = 0) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #if (__CUDACC_VER_MAJOR__ > 12) ||                                         \
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16."
        "f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(e));
    #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(e));
    #endif
  #endif
  }
};

// bf16 m16n8k32 sparse MMA: C = A_sparse * B + C
struct SM80_SPARSE_16x8x32_F32BF16BF16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #if (__CUDACC_VER_MAJOR__ > 12) ||                                         \
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32."
                 "bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, "
                 "%1, %2, %3}, %12, 0x0;\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(e));
    #else
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, "
                 "%1, %2, %3}, %12, 0x0;\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(e));
    #endif
  #endif
  }
};

  #ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
// fp8 m16n8k64 sparse MMA: C = A_sparse * B + C (SM90+)
struct SM90_SPARSE_16x8x64_F16E4M3E4M3F16_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t& d0, uint32_t& d1,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& a2, uint32_t const& a3,
                                   uint32_t const& b0, uint32_t const& b1,
                                   uint32_t const& b2, uint32_t const& b3,
                                   uint32_t const& c0, uint32_t const& c1,
                                   uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3."
        "f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F16E4M3E5M2F16_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t& d0, uint32_t& d1,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& a2, uint32_t const& a3,
                                   uint32_t const& b0, uint32_t const& b1,
                                   uint32_t const& b2, uint32_t const& b3,
                                   uint32_t const& c0, uint32_t const& c1,
                                   uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e5m2."
        "f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.f16.e4m3.e5m2.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F16E5M2E4M3F16_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t& d0, uint32_t& d1,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& a2, uint32_t const& a3,
                                   uint32_t const& b0, uint32_t const& b1,
                                   uint32_t const& b2, uint32_t const& b3,
                                   uint32_t const& c0, uint32_t const& c1,
                                   uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e5m2.e4m3."
        "f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.f16.e5m2.e4m3.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F16E5M2E5M2F16_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t& d0, uint32_t& d1,
                                   uint32_t const& a0, uint32_t const& a1,
                                   uint32_t const& a2, uint32_t const& a3,
                                   uint32_t const& b0, uint32_t const& b1,
                                   uint32_t const& b2, uint32_t const& b3,
                                   uint32_t const& c0, uint32_t const& c1,
                                   uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e5m2.e5m2."
        "f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.f16.e5m2.e5m2.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, {%10, %11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F32E4M3E4M3F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F32E4M3E5M2F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e4m3.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F32E5M2E4M3F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e5m2.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #endif
    #endif
  }
};

struct SM90_SPARSE_16x8x64_F32E5M2E5M2F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float& d0, float& d1, float& d2, float& d3, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, float const& c0, float const& c1, float const& c2,
      float const& c3, uint32_t const& e, int const& spsel = 0) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    (void)spsel;
      #if (__CUDACC_VER_MAJOR__ > 12) ||                                       \
          (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e5m2.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
      #endif
    #endif
  }
};
  #endif // __CHOREO_TARGET_NATIVE_FP8_SUPPORT__

} // namespace cute

namespace choreo {
namespace nv_cute {

namespace numerics {

template <typename T>
using DecayT = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
CUTE_HOST_DEVICE auto to_math(T value) {
  using U = DecayT<T>;
  if constexpr (std::is_same_v<U, f64>) {
    return static_cast<double>(value);
  } else {
    return static_cast<float>(to_f32(value));
  }
}

template <typename T, typename U>
CUTE_HOST_DEVICE T from_math(U value) {
  if constexpr (std::is_same_v<T, f64>) {
    return static_cast<f64>(value);
  } else if constexpr (std::is_same_v<T, f32>) {
    return static_cast<f32>(value);
  } else {
    return utils::from_f32<T>(static_cast<float>(value));
  }
}

  #define CHOREO_CUTE_UNARY_NUMERIC(NAME, F32, F64)                            \
    template <typename T>                                                      \
    CUTE_HOST_DEVICE auto NAME(T value) {                                      \
      using U = DecayT<T>;                                                     \
      if constexpr (std::is_same_v<U, f64>) {                                  \
        return from_math<U>(F64(to_math(value)));                              \
      } else {                                                                 \
        return from_math<U>(F32(to_math(value)));                              \
      }                                                                        \
    }

  #define CHOREO_CUTE_BINARY_NUMERIC(NAME, F32, F64)                           \
    template <typename A, typename B>                                          \
    CUTE_HOST_DEVICE auto NAME(A lhs, B rhs) {                                 \
      using R = DecayT<A>;                                                     \
      if constexpr (std::is_same_v<DecayT<A>, f64> ||                          \
                    std::is_same_v<DecayT<B>, f64>) {                          \
        return from_math<R>(F64(to_math(lhs), to_math(rhs)));                  \
      } else {                                                                 \
        return from_math<R>(F32(to_math(lhs), to_math(rhs)));                  \
      }                                                                        \
    }

CHOREO_CUTE_UNARY_NUMERIC(acos, ::acosf, ::acos);
CHOREO_CUTE_UNARY_NUMERIC(asin, ::asinf, ::asin);
CHOREO_CUTE_UNARY_NUMERIC(atan, ::atanf, ::atan);
CHOREO_CUTE_UNARY_NUMERIC(ceil, ::ceilf, ::ceil);
CHOREO_CUTE_UNARY_NUMERIC(cos, ::cosf, ::cos);
CHOREO_CUTE_UNARY_NUMERIC(cosh, ::coshf, ::cosh);
CHOREO_CUTE_UNARY_NUMERIC(exp, ::expf, ::exp);
CHOREO_CUTE_UNARY_NUMERIC(expm1, ::expm1f, ::expm1);
CHOREO_CUTE_UNARY_NUMERIC(floor, ::floorf, ::floor);
CHOREO_CUTE_UNARY_NUMERIC(log, ::logf, ::log);
CHOREO_CUTE_UNARY_NUMERIC(log1p, ::log1pf, ::log1p);
CHOREO_CUTE_UNARY_NUMERIC(round, ::roundf, ::round);
CHOREO_CUTE_UNARY_NUMERIC(sin, ::sinf, ::sin);
CHOREO_CUTE_UNARY_NUMERIC(sinh, ::sinhf, ::sinh);
CHOREO_CUTE_UNARY_NUMERIC(sqrt, ::sqrtf, ::sqrt);
CHOREO_CUTE_UNARY_NUMERIC(tan, ::tanf, ::tan);
CHOREO_CUTE_UNARY_NUMERIC(tanh, ::tanhf, ::tanh);

CHOREO_CUTE_BINARY_NUMERIC(atan2, ::atan2f, ::atan2);
CHOREO_CUTE_BINARY_NUMERIC(pow, ::powf, ::pow);

template <typename T>
CUTE_HOST_DEVICE auto rsqrt(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  return from_math<U>(decltype(x)(1) /
                      (std::is_same_v<U, f64> ? ::sqrt(x) : ::sqrtf(x)));
}

template <typename T>
CUTE_HOST_DEVICE auto sigmoid(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  auto one = decltype(x)(1);
  auto y = one / (one + (std::is_same_v<U, f64> ? ::exp(-x) : ::expf(-x)));
  return from_math<U>(y);
}

template <typename T>
CUTE_HOST_DEVICE auto softplus(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  auto y = std::is_same_v<U, f64> ? ::log1p(::exp(x)) : ::log1pf(::expf(x));
  return from_math<U>(y);
}

template <typename T>
CUTE_HOST_DEVICE auto gelu(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  auto k0 = std::is_same_v<U, f64> ? 0.044715 : 0.044715f;
  auto k1 = std::is_same_v<U, f64> ? 0.7978845608028654 : 0.7978845834732056f;
  auto y = x * (std::is_same_v<U, f64> ? 0.5 : 0.5f) *
           ((std::is_same_v<U, f64> ? 1.0 : 1.0f) +
            (std::is_same_v<U, f64> ? ::tanh(k1 * (x + k0 * x * x * x))
                                    : ::tanhf(k1 * (x + k0 * x * x * x))));
  return from_math<U>(y);
}

template <typename T>
CUTE_HOST_DEVICE auto sign(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  return from_math<U>((x > 0) - (x < 0));
}

template <typename T>
CUTE_HOST_DEVICE auto isfinite(T value) {
  using U = DecayT<T>;
  auto x = to_math(value);
  return from_math<U>(::isfinite(x) ? 1 : 0);
}

  #undef CHOREO_CUTE_UNARY_NUMERIC
  #undef CHOREO_CUTE_BINARY_NUMERIC

} // namespace numerics

// warp-level operation implementation
namespace warp_cooperative {

constexpr int _WARP_SIZE = 32;

template <typename FragTy, typename FTy>
__device__ __attribute__((always_inline)) inline void
fragment_elementwise(FragTy& frag, FTy&& f) {
  #pragma unroll
  for (int i = 0; i < frag.num_elements; ++i)
    frag.x[i] = f(frag.x[i]); // each lane updates its own elements
}

template <typename FragTy, typename ETy, typename FTy>
__device__ __attribute__((always_inline)) inline void
fragment_scalar_elementwise(FragTy& frag, const ETy& s, FTy&& f) {
  #pragma unroll
  for (int i = 0; i < frag.num_elements; ++i)
    frag.x[i] = f(frag.x[i], s); // each lane updates its own elements
}

template <typename FragTy, typename ETy, typename FTy>
__device__ __attribute__((always_inline)) inline void
fragment_scalarx2_elementwise(FragTy& frag, const ETy& s0, const ETy& s1,
                              FTy&& f) {
  #pragma unroll
  for (int i = 0; i < frag.num_elements; ++i)
    frag.x[i] = f(frag.x[i], s0, s1); // each lane updates its own elements
}

template <int M, int N, int LD /*leading dim*/, typename ETy, typename FTy>
__device__ __attribute__((always_inline)) inline void
inplace_matrix_uop(ETy __restrict__* m, FTy&& f) {
  int lane;
  asm("mov.u32 %0, %laneid;" : "=r"(lane));

  int MN = M * N;
  #pragma unroll
  for (int idx = lane; idx < MN; idx += _WARP_SIZE) {
    int r = idx / N;
    m[r * LD + (idx - r * N)] = f(m[r * LD + (idx - r * N)]);
  }
}

template <int M, int N, int LD /*leading dim*/, typename ETy, typename FTy>
__device__ __attribute__((always_inline)) inline void
inplace_matrix_vector_bop(ETy __restrict__* m, const ETy __restrict__* v0,
                          FTy&& f) {
  int lane;
  asm("mov.u32 %0, %laneid;" : "=r"(lane));

  int MN = M * N;
  #pragma unroll
  for (int idx = lane; idx < MN; idx += _WARP_SIZE) {
    int r = idx / N;
    int c = idx - r * N;
    m[r * LD + c] = f(m[r * LD + c], v0[r]);
  }
}

template <int M, int N, int LD /*leading dim*/, typename ETy, typename FTy>
__device__ __attribute__((always_inline)) inline void
inplace_matrix_vectorx2_bop(ETy __restrict__* m, const ETy __restrict__* v0,
                            const ETy __restrict__* v1, FTy&& f) {
  int lane;
  asm("mov.u32 %0, %laneid;" : "=r"(lane));

  int MN = M * N;
  #pragma unroll
  for (int idx = lane; idx < MN; idx += _WARP_SIZE) {
    int r = idx / N;
    int c = idx - r * N;
    m[r * LD + c] = f(m[r * LD + c], v0[r], v1[r]);
  }
}

} // end namespace warp_cooperative

} // end namespace nv_cute
} // end namespace choreo

#endif // __CHOREO_TARGET_CUTE__

#endif //__CHROEO_CUTE_HEADER_LIBRARY_HPP__

#ifdef __TOPSCC__
  #include <krt/builtins.h>
  #if __GCU_ARCH__ >= 300 && __GCU_ARCH__ < 500
    #include <tops/topscc_types.h>
  #endif

  #if defined(__CHOREO_TARGET_NATIVE_BF16_SUPPORT__) && __GCU_ARCH__ >= 300
    #include <tops/tops_bf16.h>

namespace choreo {

class bf16 {
  tops::__ef_bfloat16 val_;

public:
  __co_any__ bf16() = default;
  __co_any__ bf16(float f) : val_(f) {}
  __co_any__ bf16(double f) : val_(static_cast<float>(f)) {}
  __co_any__ bf16& operator=(float f) {
    val_ = f;
    return *this;
  }
  __co_any__ bf16& operator=(double f) {
    val_ = static_cast<float>(f);
    return *this;
  }
  __co_any__ operator float() const { return tops::__bfloat162float(val_); }

  __co_any__ bool operator==(double value) {
    auto vf = static_cast<float>(value);
    if (std::isnan(vf)) return std::isnan(float(*this));
    return float(*this) == vf;
  }
  template <typename T>
  __co_any__ bool operator==(T value) {
    if constexpr (std::is_same<T, bf16>::value) {
      auto vf = float(value);
      if (std::isnan(vf)) return std::isnan(float(*this));
      return float(*this) == vf;
    } else {
      auto vf = static_cast<float>(value);
      if (std::isnan(vf)) return std::isnan(float(*this));
      return float(*this) == vf;
    }
  }
  template <typename T>
  __co_any__ bool operator>(T value) {
    if constexpr (std::is_same<T, bf16>::value)
      return float(*this) > float(value);
    else
      return float(*this) > static_cast<float>(value);
  }
  template <typename T>
  __co_any__ bool operator<(T value) {
    if constexpr (std::is_same<T, bf16>::value)
      return float(*this) < float(value);
    else
      return float(*this) < static_cast<float>(value);
  }
};

using bfloat16 = bf16;
using bfp16 = bf16;

__co_any__ inline static bf16 f32_to_bf16(float value) { return bf16(value); }

__co_any__ inline static float bf16_to_f32(bf16 value) {
  return static_cast<float>(value);
}

    #define __CHOREO_BF16_DEFINED__
    #define __CHOREO_BF16_CONVERT_DEFINED__

} // namespace choreo

  #elif defined(__CHOREO_TARGET_NATIVE_BF16_SUPPORT__)
    #undef __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
  #endif // __CHOREO_TARGET_NATIVE_BF16_SUPPORT__ && __GCU_ARCH__

namespace choreo {

using stream_t = topsStream_t;

// --- light-weight choreo device library --- //

__device__ __attribute__((always_inline)) static inline void __co_abort__() {
  #ifdef __CHOREO_USE_TOPS_ABORT__
  tops::abort();
  #endif
}

  #if __GCU_ARCH__ == 400
using choreo_dte_ctx_t = tops_dte_ctx_base_s;
using choreo_event = tops::event;
__device__ __forceinline__ void tops_init_dte(tops_dte_ctx_base_s* ctx) {
  ctx->init_comm();
}
__device__ __forceinline__ void tops_destroy_dte(tops_dte_ctx_base_s* ctx) {
  ctx->destroy_comm();
}
  #elif __GCU_ARCH__ == 500
struct choreo_dte_ctx_t {
  __device__ __forceinline__ void init() {}
  __device__ __forceinline__ void destory() {}
};
__device__ __forceinline__ void tops_init_dte(choreo_dte_ctx_t* ctx) {
  ctx->init();
}
__device__ __forceinline__ void tops_destroy_dte(choreo_dte_ctx_t* ctx) {
  ctx->destory();
}
struct choreo_event {
  choreo_dte_ctx_t* ctx;
};

  #elif __GCU_ARCH__ == 300
// GCU300 uses tops::private_cdte (CDTE) for Global<->Shared DMA and
// tops::private_dte (dynamic SDTE) for transfers involving Private memory.
// TOPSCC_PRIVATE_DTE_AUTO_INIT enables RAII lifecycle management.
using choreo_dte_ctx_t = tops_dte_ctx_base_s;
using choreo_cdte = tops::shared_dte;
using choreo_cdte_priv = tops::private_cdte;
using choreo_sdte = tops::private_dte;
using choreo_event = tops::event;
__device__ __forceinline__ void tops_init_dte(tops_dte_ctx_base_s* ctx) {
  // no-op: TOPSCC_PRIVATE_DTE_AUTO_INIT handles init in DTE constructor
}
__device__ __forceinline__ void tops_destroy_dte(tops_dte_ctx_base_s* ctx) {
  // no-op: TOPSCC_PRIVATE_DTE_AUTO_INIT handles destroy in DTE destructor
}
  #else
using choreo_dte_ctx_t = tops_dte_ctx_t;
using choreo_cdte = tops_dte_ctx_t;
using choreo_cdte_priv = tops_dte_ctx_t;
using choreo_sdte = tops_dte_ctx_t;
using choreo_event = tops::event;
  #endif

// choreo device future
struct future {
  choreo_dte_ctx_t* ctx = nullptr;
  choreo_event e;
  void* d = nullptr;  // data: future's user must guarantee it is valid
  void* md = nullptr; // metadata: optional structured sparsity metadata

  // DTE lifecycle state (functional) -- also used for diagnosis checks when
  // __CHOREO_DMA_DIAGNOSIS__ is defined:
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
  Status s = ST_NONE;
#if defined(__GCU_ARCH__) && __GCU_ARCH__ == 300
  bool explicit_init = false;
#endif

  #ifdef __CHOREO_DMA_DIAGNOSIS__
  // diagnosis-only fields: source location for runtime error messages
  const char* name = nullptr;
  unsigned line = 0;
  unsigned column = 0;
  #endif // __CHOREO_DMA_DIAGNOSIS__

  #ifdef __CHOREO_DMA_DIAGNOSIS__
  __device__ future(choreo_dte_ctx_t& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr, void* mdata = nullptr)
      : ctx(&dte), d(data), md(mdata ? mdata : data), s(ST_NONE), name(n),
        line(l), column(c) {}
    #if __GCU_ARCH__ == 400
  __device__ future(tops::local_dte& dte, const char* n, unsigned l, unsigned c,
                    void* data = nullptr, void* mdata = nullptr)
      : ctx(&dte), d(data), md(mdata ? mdata : data), s(ST_NONE), name(n),
        line(l), column(c) {}
  __device__ future(tops::shared_dte& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr, void* mdata = nullptr)
      : ctx(&dte), d(data), md(mdata ? mdata : data), s(ST_NONE), name(n),
        line(l), column(c) {}
  __device__ future(tops::private_dte& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr, void* mdata = nullptr)
      : ctx(&dte), d(data), md(mdata ? mdata : data), s(ST_NONE), name(n),
        line(l), column(c) {}
  #elif __GCU_ARCH__ == 300
  __device__ future(choreo_sdte& dte, const char* n, unsigned l, unsigned c,
                    void* data = nullptr, void* mdata = nullptr)
      : ctx(reinterpret_cast<choreo_dte_ctx_t*>(&dte)), d(data),
        md(mdata ? mdata : data), s(ST_NONE), explicit_init(false),
        name(n), line(l), column(c) {}
  __device__ future(choreo_cdte& dte, const char* n, unsigned l, unsigned c,
                    void* data = nullptr, void* mdata = nullptr)
      : ctx(reinterpret_cast<choreo_dte_ctx_t*>(&dte)), d(data),
        md(mdata ? mdata : data), s(ST_NONE), explicit_init(true),
        name(n), line(l), column(c) {}
  __device__ future(choreo_cdte_priv& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr, void* mdata = nullptr)
      : ctx(reinterpret_cast<choreo_dte_ctx_t*>(&dte)), d(data),
        md(mdata ? mdata : data), s(ST_NONE), explicit_init(false),
        name(n), line(l), column(c) {}
    #endif // __GCU_ARCH__
  #else    // !__CHOREO_DMA_DIAGNOSIS__
  __device__ future(choreo_dte_ctx_t& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr, void* mdata = nullptr)
      : ctx(&dte), d(data), md(mdata ? mdata : data), s(ST_NONE) {
    (void)n;
    (void)l;
    (void)c;
  }
    #if __GCU_ARCH__ == 400
  __device__ future(tops::local_dte& dte, const char* n, unsigned l, unsigned c,
                    void* data = nullptr, void* mdata = nullptr)
      : ctx(&dte), d(data), md(mdata ? mdata : data), s(ST_NONE) {
    (void)n;
    (void)l;
    (void)c;
  }
  __device__ future(tops::shared_dte& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr, void* mdata = nullptr)
      : ctx(&dte), d(data), md(mdata ? mdata : data), s(ST_NONE) {
    (void)n;
    (void)l;
    (void)c;
  }
  __device__ future(tops::private_dte& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr, void* mdata = nullptr)
      : ctx(&dte), d(data), md(mdata ? mdata : data), s(ST_NONE) {
    (void)n;
    (void)l;
    (void)c;
  }
    #elif __GCU_ARCH__ == 300
  __device__ future(choreo_sdte& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr, void* mdata = nullptr)
      : ctx(reinterpret_cast<choreo_dte_ctx_t*>(&dte)), d(data),
        md(mdata ? mdata : data), s(ST_NONE), explicit_init(false) {
    (void)n;
    (void)l;
    (void)c;
  }
  __device__ future(choreo_cdte& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr, void* mdata = nullptr)
      : ctx(reinterpret_cast<choreo_dte_ctx_t*>(&dte)), d(data),
        md(mdata ? mdata : data), s(ST_NONE), explicit_init(true) {
    (void)n;
    (void)l;
    (void)c;
  }
  __device__ future(choreo_cdte_priv& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr, void* mdata = nullptr)
      : ctx(reinterpret_cast<choreo_dte_ctx_t*>(&dte)), d(data),
        md(mdata ? mdata : data), s(ST_NONE), explicit_init(false) {
    (void)n;
    (void)l;
    (void)c;
  }
    #endif // __GCU_ARCH__
  #endif   // __CHOREO_DMA_DIAGNOSIS__

  // context is retrieved to invoke data operations
  __device__ auto get_ctx() {
    if (s == ST_NONE) {
  #if defined(__GCU_ARCH__) && __GCU_ARCH__ == 300
      if (explicit_init) ctx->init_comm();
  #else
      tops_init_dte(ctx);
  #endif
      s = ST_INITED;
    }
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is not initialized.\n",
             line, column);
      __co_abort__();
    }
  #endif // __CHOREO_DMA_DIAGNOSIS__
    return ctx;
  }

  // when async, an event is obtained for later waiting
  __device__ void set_event(choreo_event& ev) {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_TRIGGERED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "is triggered on an in-flight event.\n",
             line, column);
      __co_abort__();
    } else if (s == ST_NONE) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is not initialized before triggering.\n",
             line, column);
      __co_abort__();
    }
    if (ev.ctx != ctx) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is used inconsistently.\n",
             line, column);
      __co_abort__();
    }
  #endif // __CHOREO_DMA_DIAGNOSIS__
    e = ev;
    s = ST_TRIGGERED;
  }

  // when sync, no wait is required. simply change the status
  __device__ void set_nowait() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is used incorrectly.\n",
             line, column);
      __co_abort__();
    }
  #endif // __CHOREO_DMA_DIAGNOSIS__
    s = ST_WAITED;
  }

  __device__ void set_data(void* data) { d = data; }
  __device__ void set_event_data(choreo_event& ev, void* data) {
    set_event(ev);
    set_data(data);
  }

  __device__ void wait() {
    if (s == ST_TRIGGERED) {
  #if __GCU_ARCH__ <= 400
      tops::wait(e);
  #elif __GCU_ARCH__ == 500
  #endif
      s = ST_WAITED;
    }
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    else if (s == ST_WAITED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has been waited multiple times.\n",
             line, column);
      __co_abort__();
    } else if (s == ST_INITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is used incorrectly.\n",
             line, column);
      __co_abort__();
    } else {
      assert(s == ST_NONE); // waiting on not triggered future is acceptable
    }
  #endif // __CHOREO_DMA_DIAGNOSIS__
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
    return md;
  }

  __device__ ~future() {
  #ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_TRIGGERED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has never been waited.\n",
             line, column);
      __co_abort__();
    }
  #endif // __CHOREO_DMA_DIAGNOSIS__
  #if defined(__GCU_ARCH__) && __GCU_ARCH__ == 300
    if (s >= ST_INITED && explicit_init) ctx->destroy_comm();
  #else
    if (s >= ST_INITED) tops_destroy_dte(ctx);
  #endif
  }
  __device__ future(const future& f) = delete;
  __device__ future(future&& f) = delete;
  __device__ future& operator=(const future& f) = delete;
};

__device__ static inline void swap(future& a, future& b) {
  auto ctx = a.ctx;
  auto e = a.e;
  auto d = a.d;
  auto s = a.s;
  #ifdef __CHOREO_DMA_DIAGNOSIS__
  auto l = a.line;
  auto c = a.column;
  #endif // __CHOREO_DMA_DIAGNOSIS__

  a.ctx = b.ctx;
  a.e = b.e;
  a.d = b.d;
  a.s = b.s;
  #ifdef __CHOREO_DMA_DIAGNOSIS__
  a.line = b.line;
  a.column = b.column;
  #endif // __CHOREO_DMA_DIAGNOSIS__

  b.ctx = ctx;
  b.e = e;
  b.d = d;
  b.s = s;
  #ifdef __CHOREO_DMA_DIAGNOSIS__
  b.line = l;
  b.column = c;
  #endif // __CHOREO_DMA_DIAGNOSIS__
}

} // end namespace choreo

#endif // __TOPSCC__

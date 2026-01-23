#ifdef __TOPSCC__
#include <krt/builtins.h>

namespace choreo {

// --- light-weight choreo device library --- //

__device__ __attribute__((always_inline)) static inline void __co_abort__() {
#if __GCU_ARCH__ >= 300
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

#else
using choreo_dte_ctx_t = tops_dte_ctx_t;
using choreo_event = tops::event;
#endif

// choreo device future
struct future {
  choreo_dte_ctx_t* ctx = nullptr;
  choreo_event e;
  void* d = nullptr;  // data: future's user must guarantee it is valid
  void* md = nullptr; // metadata: optional structured sparsity metadata

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
  Status s = ST_NONE;
  const char* name = nullptr;
  // source code locations
  unsigned line = 0;
  unsigned column = 0;

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
#endif

  // context is retrieved to invoke data operations
  __device__ auto get_ctx() {
    if (s == ST_NONE) {
      tops_init_dte(ctx);
      s = ST_INITED;
    }
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is not initialized.\n",
             line, column);
      __co_abort__();
    }
    return ctx;
  }

  // when async, an event is obtained for later waiting
  __device__ void set_event(choreo_event& ev) {
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

    e = ev;
    s = ST_TRIGGERED;
  }

  // when sync, no wait is required. simply change the status
  __device__ void set_nowait() {
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is used incorrectly.\n",
             line, column);
      __co_abort__();
    }
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
    } else if (s == ST_WAITED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has been waited multiple times.\n",
             line, column);
      __co_abort__();
    } else if (s == ST_INITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is used incorrectly.\n",
             line, column);
      __co_abort__();
    } else
      assert(s == ST_NONE); // waiting on not triggered future is acceptable
  }

#if 0
  __device__ choreo_event& event() {
    if (s == ST_TRIGGERED) {
      printf("[choreo-rt] internal error: future (defined at line %u:%u) is not associated with an event.\n",
             line, column);
      __co_abort__();
    }
    return e;
  }
#endif

  __device__ void* data() {
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
    return d;
  }

  __device__ void* mdata() {
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
    return md;
  }

  __device__ ~future() {
    if (s == ST_TRIGGERED) {
      // TODO: requires krt %s support to print future name
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has never been waited.\n",
             line, column);
      __co_abort__();
    }
    if (s >= ST_INITED) tops_destroy_dte(ctx);
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
  auto l = a.line;
  auto c = a.column;

  a.ctx = b.ctx;
  a.e = b.e;
  a.d = b.d;
  a.s = b.s;
  a.line = b.line;
  a.column = b.column;

  b.ctx = ctx;
  b.e = e;
  b.d = d;
  b.s = s;
  b.line = l;
  b.column = c;
}

} // end namespace choreo

#endif

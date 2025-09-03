__cok__ {
  struct st {};
  __device__ void abc(const std::vector<std::map<st*,std::string>> &a = {std::make_pair<>((st*)a, "aa")}) {}

  __device__ static inline constexpr st const *def(const ID &&a = foo<ID>(1, {}) + 1, float* b = nullptr) {
    return a + (int)b;
  }

  __device__  
  extern int foo(int a,  __private__ __align__ char* c = nullptr, float f = 1.0f, void** v = nullptr) {} ;

  __device__ extern int index ();

  template <typename T>
  __device__ __attribute__((noinline, no_mem_alias_in_vldst_tar))
  void dynamic_scaled_int8_quant_kernel(
    int8_t* out, float* scales, T* in,
    const DYNAMIC_SCALED_INT8_QUANT_OP_PARAS para) {}
}
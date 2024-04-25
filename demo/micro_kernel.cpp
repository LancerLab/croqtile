

extern "C" void add_i32_dps(int* a, int* b, int* c, int n) { 
  for (int i=0; i<n; ++i) {
    c[i] = a[i] + b[i]; 
  }
  return; 
}

extern "C" int add_i32_ndps(int a, int b) { return a + b; }


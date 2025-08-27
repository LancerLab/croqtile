// 3->4
#define TEST1(func, in_dim0, in_dim1, in_dim2, out_dim0, out_dim1, out_dim2,   \
              out_dim3)                                                        \
  auto input_##func =                                                          \
      choreo::make_spandata<choreo::f32>(in_dim0, in_dim1, in_dim2);           \
  input_##func.fill(1.0f);                                                     \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(input_##func.view());                                 \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  assert(res_##func.span(0) == out_dim0);                                      \
  assert(res_##func.span(1) == out_dim1);                                      \
  assert(res_##func.span(2) == out_dim2);                                      \
  assert(res_##func.span(3) == out_dim3);                                      \
  printf("Test %s Passed!\n", #func);                                          \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Execution time: " << duration_##func.count()                   \
            << " microseconds" << std::endl;

// 4->2
#define TEST2(func, in_dim0, in_dim1, in_dim2, in_dim3, out_dim0, out_dim1)    \
  auto input_##func =                                                          \
      choreo::make_spandata<choreo::f32>(in_dim0, in_dim1, in_dim2, in_dim3);  \
  input_##func.fill(1.0f);                                                     \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(input_##func.view());                                 \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  assert(res_##func.span(0) == out_dim0);                                      \
  assert(res_##func.span(1) == out_dim1);                                      \
  printf("Test %s Passed!\n", #func);                                          \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Execution time: " << duration_##func.count()                   \
            << " microseconds" << std::endl;

// 4->3
#define TEST3(func, in_dim0, in_dim1, in_dim2, in_dim3, out_dim0, out_dim1,    \
              out_dim2)                                                        \
  auto input_##func =                                                          \
      choreo::make_spandata<choreo::f32>(in_dim0, in_dim1, in_dim2, in_dim3);  \
  input_##func.fill(1.0f);                                                     \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(input_##func.view());                                 \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  assert(res_##func.span(0) == out_dim0);                                      \
  assert(res_##func.span(1) == out_dim1);                                      \
  assert(res_##func.span(2) == out_dim2);                                      \
  printf("Test %s Passed!\n", #func);                                          \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Execution time: " << duration_##func.count()                   \
            << " microseconds" << std::endl;

// 4->4
#define TEST4(func, in_dim0, in_dim1, in_dim2, in_dim3, out_dim0, out_dim1,    \
              out_dim2, out_dim3)                                              \
  auto input_##func =                                                          \
      choreo::make_spandata<choreo::f32>(in_dim0, in_dim1, in_dim2, in_dim3);  \
  input_##func.fill(1.0f);                                                     \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(input_##func.view());                                 \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  assert(res_##func.span(0) == out_dim0);                                      \
  assert(res_##func.span(1) == out_dim1);                                      \
  assert(res_##func.span(2) == out_dim2);                                      \
  assert(res_##func.span(3) == out_dim3);                                      \
  printf("Test %s Passed!\n", #func);                                          \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Execution time: " << duration_##func.count()                   \
            << " microseconds" << std::endl;

// 3->2
#define TEST5(func, in_dim0, in_dim1, in_dim2, out_dim0, out_dim1)             \
  auto input_##func =                                                          \
      choreo::make_spandata<choreo::f32>(in_dim0, in_dim1, in_dim2);           \
  input_##func.fill(1.0f);                                                     \
  auto start_##func = std::chrono::high_resolution_clock::now();               \
  auto res_##func = func(input_##func.view());                                 \
  auto end_##func = std::chrono::high_resolution_clock::now();                 \
  assert(res_##func.span(0) == out_dim0);                                      \
  assert(res_##func.span(1) == out_dim1);                                      \
  printf("Test %s Passed!\n", #func);                                          \
  auto duration_##func =                                                       \
      std::chrono::duration_cast<std::chrono::microseconds>(end_##func -       \
                                                            start_##func);     \
  std::cout << "Execution time: " << duration_##func.count()                   \
            << " microseconds" << std::endl;

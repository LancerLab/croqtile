
#ifndef __CHOREO_CUDA_H__
#define __CHOREO_CUDA_H__

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

namespace choreo {

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

}  // end namespace choreo

#endif  // __CHOREO_CUDA_H__

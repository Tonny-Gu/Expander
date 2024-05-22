#pragma once

#include <iostream>
#include "field/M31.hpp"

namespace cuda {

using HostF = gkr::M31_field::M31;
using HostBatchF = gkr::M31_field::VectorizedM31;

#define CUDA_CHECK(ret) __cuda_check((ret), #ret, __FILE__, __LINE__)

void __cuda_check(cudaError_t err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::abort();
  }
}

static constexpr int64_t div_ceil(int64_t a, int64_t b) {
  return a / b + (a % b > 0);
}

constexpr int32_t block_size = 32;

} // namespace cuda
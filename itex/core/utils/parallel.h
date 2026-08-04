/* Copyright (c) 2023 Intel Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ITEX_CORE_UTILS_PARALLEL_H_
#define ITEX_CORE_UTILS_PARALLEL_H_

#include "itex/core/utils/parallel_openmp.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace itex {

// Returns the maximum number of threads that may be used in a parallel region
inline int GetNumThreads(const OpKernelContext* ctx) {
  return ctx->eigen_cpu_device().numThreadsInPool();
}

// Returns the current thread number (starting from 0)
// in the current parallel region, or 0 in the sequential region.
inline int GetThreadNum(const OpKernelContext* ctx) {
  return ctx->eigen_cpu_device().currentThreadId();
}

template <typename F>
inline void ParallelFor(const OpKernelContext* ctx, int64_t n, const Eigen::TensorOpCost& cost,
                        const F& f) {
  ctx->eigen_cpu_device().parallelFor(n, cost, f);
}
}  // namespace itex

#endif  // ITEX_CORE_UTILS_PARALLEL_H_

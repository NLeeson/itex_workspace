/* Copyright (c) 2026 Intel Corporation

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

#ifndef ITEX_CORE_UTILS_CPU_ONEDNN_RUNTIME_H_
#define ITEX_CORE_UTILS_CPU_ONEDNN_RUNTIME_H_

#include <dlfcn.h>
#include "itex/core/utils/env_var.h"

namespace itex {

// True when this process opened the OpenMP oneDNN CPU library.
// Compile-time THREADPOOL builds never load OpenMP. Otherwise the CPU
// wrapper publishes the library it actually dlopened.
inline bool LoadedCpuOnednnIsOpenMP() {
#if defined(CC_THREADPOOL_BUILD) || defined(ITEX_CPU_THREADPOOL_BUILD)
  return false;
#else
  using ProbeFn = int (*)();
  auto* probe = reinterpret_cast<ProbeFn>(
      dlsym(RTLD_DEFAULT, "ITEX_CpuOnednnRuntimeIsOpenMP"));
  if (probe != nullptr) {
    return probe() != 0;
  }
  bool enable_omp = true;
  auto status = ReadBoolFromEnvVar("ITEX_OMP_THREADPOOL", true, &enable_omp);
  if (!status.ok()) return true;
  return enable_omp;
#endif
}

}  // namespace itex

#endif  // ITEX_CORE_UTILS_CPU_ONEDNN_RUNTIME_H_

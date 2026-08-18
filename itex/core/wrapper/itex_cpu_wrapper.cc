/* Copyright (c) 2021-2022 Intel Corporation

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

#include "itex/core/wrapper/itex_cpu_wrapper.h"

#include <cpuid.h>
#include <dlfcn.h>
#include <cstdlib>
#include <link.h>
#include <string>

#include "itex/core/devices/device_backend_util.h"
#include "itex/core/utils/cpu_info.h"
#include "itex/core/utils/logging.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

static void* handle;
static int loaded_cpu_onednn_is_openmp = 1;
static void* LoadCpuLibrary() __attribute__((constructor));
static void UnloadCpuLibrary() __attribute__((destructor));

extern "C" __attribute__((visibility("default"))) int
ITEX_CpuOnednnRuntimeIsOpenMP(void) {
  return loaded_cpu_onednn_is_openmp;
}

namespace {

std::string CurrentWrapperPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&LoadCpuLibrary), &info) != 0 &&
      info.dli_fname != nullptr) {
    return info.dli_fname;
  }
  return "unknown";
}

std::string LoadedLibraryPath(void* library_handle) {
  if (library_handle == nullptr) return "unknown";
  link_map* map = nullptr;
  if (dlinfo(library_handle, RTLD_DI_LINKMAP, &map) == 0 && map != nullptr &&
      map->l_name != nullptr && map->l_name[0] != '\0') {
    return map->l_name;
  }
  return "unknown";
}

bool ParseEnvBool(const char* value, bool default_value, const char* name) {
  if (value == nullptr) return default_value;
  std::string normalized(value);
  if (normalized == "1" || normalized == "true" || normalized == "True") {
    return true;
  }
  if (normalized == "0" || normalized == "false" || normalized == "False") {
    return false;
  }
  ITEX_LOG(WARNING) << name << " parse failed for value: " << normalized;
  return default_value;
}

bool DebugWiringEnabled() {
  static bool enabled =
      ParseEnvBool(std::getenv("ITEX_DEBUG_WIRING"), false, "ITEX_DEBUG_WIRING");
  return enabled;
}

void LogCpuWrapperWiring(const std::string& event,
                         const std::string& extra = "") {
  if (!DebugWiringEnabled()) return;
  ITEX_LOG(INFO) << "ITEX_DEBUG_WIRING component=cpu_wrapper event=" << event
                 << " wrapper_path=" << CurrentWrapperPath()
                 << " backend=CPU compiled={USING_NEXTPLUGGABLE_DEVICE="
#ifdef USING_NEXTPLUGGABLE_DEVICE
                 << "1"
#else
                 << "0"
#endif
                 << ", INTEL_GPU_ONLY="
#ifdef INTEL_GPU_ONLY
                 << "1"
#else
                 << "0"
#endif
                 << "} " << extra;
}

}  // namespace

void* LoadCpuLibrary() {
  if (itex_get_backend() == ITEX_BACKEND_DEFAULT) {
    itex_freeze_backend(ITEX_BACKEND_CPU);
  }
  LogCpuWrapperWiring("load_start");
  bool enable_omp =
      ParseEnvBool(std::getenv("ITEX_OMP_THREADPOOL"), true, "ITEX_OMP_THREADPOOL");
#ifdef ITEX_CPU_THREADPOOL_BUILD
  if (enable_omp) {
    ITEX_LOG(WARNING)
        << "ITEX_OMP_THREADPOOL=1 is ignored because this ITEX build was "
        << "configured with --define=build_with_threadpool=true.";
  }
  enable_omp = false;
#endif
  const char* onednn_lib =
      enable_omp ? "libonednn_cpu_so.so" : "libonednn_cpu_eigen_so.so";
  onednn_handle = dlopen(onednn_lib, RTLD_NOW | RTLD_GLOBAL);
  if (!onednn_handle) {
    ITEX_LOG(FATAL) << dlerror();
  }
  loaded_cpu_onednn_is_openmp = enable_omp ? 1 : 0;
  ITEX_LOG(INFO) << "oneDNN CPU runtime: " << (enable_omp ? "OMP" : "THREADPOOL")
                 << " (" << onednn_lib << ")";
  LogCpuWrapperWiring("onednn_loaded",
                      std::string("library=") + onednn_lib +
                          " runtime=" + (enable_omp ? "OMP" : "THREADPOOL") +
                          " library_path=" + LoadedLibraryPath(onednn_handle));

  if (itex::port::CPUIDAVX512()) {
    handle = dlopen("libitex_cpu_internal_avx512.so", RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      const char* error_msg = dlerror();
      ITEX_LOG(WARNING)
          << "AVX512 CPU library is loaded failed, try to load AVX2.";
      ITEX_LOG(WARNING) << error_msg;
    } else {
      ITEX_LOG(INFO)
          << "Intel Extension for Tensorflow* AVX512 CPU backend is loaded.";
      LogCpuWrapperWiring("backend_loaded",
                          "library=libitex_cpu_internal_avx512.so library_path=" +
                              LoadedLibraryPath(handle));
      return handle;
    }
  }
  handle = dlopen("libitex_cpu_internal_avx2.so", RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char* error_msg = dlerror();
    ITEX_LOG(FATAL) << "Could not load dynamic library: " << error_msg;
  }
  ITEX_LOG(INFO)
      << "Intel Extension for Tensorflow* AVX2 CPU backend is loaded.";
  LogCpuWrapperWiring("backend_loaded",
                      "library=libitex_cpu_internal_avx2.so library_path=" +
                          LoadedLibraryPath(handle));
  return handle;
}

void UnloadCpuLibrary() {
  if (handle) {
    dlclose(handle);
  }
}

void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status) {
  typedef void (*tf_initgraph_internal)(TP_OptimizerRegistrationParams*,
                                        TF_Status*);

  if (handle) {
    auto tf_initgraph = reinterpret_cast<tf_initgraph_internal>(
        dlsym(handle, "TF_InitGraph_Internal"));
    if (tf_initgraph != nullptr) {
      tf_initgraph(params, status);
    } else {
      const char* error_msg = dlerror();
      ITEX_LOG(FATAL) << error_msg;
    }
  } else {
    ITEX_LOG(WARNING) << "Graph module not found.";
  }
}

void TF_InitKernel() {
  typedef void (*tf_initkernel_internal)();

  if (handle) {
    auto tf_initkernel = reinterpret_cast<tf_initkernel_internal>(
        dlsym(handle, "TF_InitKernel_Internal"));
    if (*tf_initkernel != nullptr) {
      tf_initkernel();
    } else {
      const char* error_msg = dlerror();
      ITEX_LOG(FATAL) << error_msg;
    }
  }
}

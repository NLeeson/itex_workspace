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

#ifndef ITEX_CORE_UTILS_TF_EIGEN_THREADPOOL_DEVICE_H_
#define ITEX_CORE_UTILS_TF_EIGEN_THREADPOOL_DEVICE_H_

#if defined(INTEL_CPU_ONLY) && !defined(ITEX_BUILD_JAX)

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "itex/core/utils/env_var.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/threadpool_interface.h"
#include "tensorflow/c/kernels.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// Keep this header independent of TensorFlow's private C++ headers. These
// declarations mirror the pure-C bridge implemented by
// tf_eigen_threadpool_bridge.cc.
extern "C" {
void* ITEX_GetTensorFlowThreadPool(void* opaque_context);
int ITEX_TensorFlowThreadPoolNumThreads(void* opaque_threadpool);
int ITEX_TensorFlowThreadPoolCurrentThreadId(void* opaque_threadpool);
void ITEX_TensorFlowThreadPoolScheduleWithHint(
    void* opaque_threadpool, void (*task)(void*), void* argument, int start,
    int end);
}

namespace itex {
namespace tf_eigen_threadpool_device_internal {

inline bool DebugWiringEnabled() {
  static bool enabled = []() {
    bool value = false;
    auto status = ReadBoolFromEnvVar("ITEX_DEBUG_WIRING", false, &value);
    if (!status.ok()) {
      ITEX_LOG(WARNING) << "ITEX_DEBUG_WIRING parse failed: " << status;
      return false;
    }
    return value;
  }();
  return enabled;
}

inline void RunTask(void* opaque_task) {
  std::unique_ptr<std::function<void()>> task(
      static_cast<std::function<void()>*>(opaque_task));
  (*task)();
}

// Eigen-facing adapter over TensorFlow's existing intra-op pool. This object
// owns no workers and never participates in TensorFlow pool destruction.
class TensorFlowThreadPoolAdapter final : public Eigen::ThreadPoolInterface {
 public:
  explicit TensorFlowThreadPoolAdapter(void* tensorflow_threadpool)
      : tensorflow_threadpool_(tensorflow_threadpool) {
    ITEX_CHECK(tensorflow_threadpool_ != nullptr)
        << "TensorFlow Eigen CPU threadpool is unavailable";
    ITEX_CHECK_GT(NumThreads(), 0)
        << "TensorFlow Eigen CPU threadpool has no workers";
  }

  void Schedule(std::function<void()> fn) override {
    ScheduleWithHint(std::move(fn), 0, NumThreads());
  }

  void ScheduleWithHint(std::function<void()> fn, int start,
                        int limit) override {
    const int num_threads = NumThreads();
    const int safe_start = std::max(0, std::min(start, num_threads - 1));
    const int safe_limit =
        std::max(safe_start + 1, std::min(limit, num_threads));
    auto* task = new std::function<void()>(std::move(fn));
    ::ITEX_TensorFlowThreadPoolScheduleWithHint(
        tensorflow_threadpool_, &RunTask, task, safe_start, safe_limit);
  }

  // TensorFlow owns this pool and its lifecycle. Cancellation through this
  // borrowed view must not affect TensorFlow or unrelated operations.
  void Cancel() override {}

  int NumThreads() const override {
    return ::ITEX_TensorFlowThreadPoolNumThreads(tensorflow_threadpool_);
  }

  int CurrentThreadId() const override {
    return ::ITEX_TensorFlowThreadPoolCurrentThreadId(tensorflow_threadpool_);
  }

  void* tensorflow_threadpool() const { return tensorflow_threadpool_; }

 private:
  void* tensorflow_threadpool_ = nullptr;
};

struct CachedEigenDevice {
  explicit CachedEigenDevice(void* tensorflow_threadpool)
      : adapter(tensorflow_threadpool),
        device(&adapter, adapter.NumThreads()) {}

  TensorFlowThreadPoolAdapter adapter;
  Eigen::ThreadPoolDevice device;
};

inline const Eigen::ThreadPoolDevice& GetTensorFlowEigenCpuDevice(
    TF_OpKernelContext* context) {
  void* tensorflow_threadpool =
      ::ITEX_GetTensorFlowThreadPool(static_cast<void*>(context));
  ITEX_CHECK(tensorflow_threadpool != nullptr)
      << "TensorFlow Eigen CPU threadpool is unavailable";

  static std::mutex mu;
  static std::vector<std::unique_ptr<CachedEigenDevice>> devices;

  std::lock_guard<std::mutex> lock(mu);
  for (const auto& cached : devices) {
    if (cached->adapter.tensorflow_threadpool() == tensorflow_threadpool) {
      return cached->device;
    }
  }

  devices.emplace_back(new CachedEigenDevice(tensorflow_threadpool));
  CachedEigenDevice* cached = devices.back().get();
  if (DebugWiringEnabled()) {
    ITEX_LOG(INFO)
        << "ITEX_DEBUG_WIRING component=itex_eigen_threadpool "
           "event=adapter_created"
        << " pool=" << tensorflow_threadpool
        << " ownership=borrowed_from_tensorflow"
        << " tensorflow_threads=" << cached->adapter.NumThreads()
        << " eigen_device=" << static_cast<void*>(&cached->device)
        << " creates_workers=0"
        << " scheduling=ScheduleWithHint"
        << " cancellation=noop_borrowed_view"
        << " cache_key=tensorflow_pool"
        << " consumer=eigen_cpu_device";
  }
  return cached->device;
}

}  // namespace tf_eigen_threadpool_device_internal

inline const Eigen::ThreadPoolDevice& GetTensorFlowEigenCpuDevice(
    TF_OpKernelContext* context) {
  return tf_eigen_threadpool_device_internal::GetTensorFlowEigenCpuDevice(
      context);
}

}  // namespace itex

#endif  // INTEL_CPU_ONLY && !ITEX_BUILD_JAX

#endif  // ITEX_CORE_UTILS_TF_EIGEN_THREADPOOL_DEVICE_H_

/* Copyright (c) 2023 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ITEX_CORE_UTILS_ONEDNN_MKL_THREADPOOL_H_
#define ITEX_CORE_UTILS_ONEDNN_MKL_THREADPOOL_H_

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dnnl.hpp"             // NOLINT(build/include_subdir)
#include "dnnl_threadpool.hpp"  // NOLINT(build/include_subdir)
#include "itex/core/utils/blocking_counter.h"
#include "itex/core/utils/cpu_info.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/tf_eigen_threadpool_bridge.h"
#include "itex/core/utils/threadpool.h"

namespace itex {

using dnnl::threadpool_interop::threadpool_iface;

// Divide 'n' units of work equally among 'teams' threads. If 'n' is not
// divisible by 'teams' and has a remainder 'r', the first 'r' teams have one
// unit of work more than the rest. Returns the range of work that belongs to
// the team 'tid'.
template <typename T, typename U>
inline void balance211(T n, U team, U tid, T* n_start, T* n_end) {
  if (team <= 1 || n == 0) {
    *n_start = 0;
    *n_end = n;
    return;
  }
  T min_per_team = n / team;
  T remainder = n - min_per_team * team;
  *n_start = tid * min_per_team + std::min(tid, remainder);
  *n_end = *n_start + min_per_team + (tid < remainder);
}

inline void run_jobs(bool balance, int i, int n, int njobs,
                     const std::function<void(int, int)>& fn) {
  if (balance) {
    int start, end;
    balance211(n, njobs, i, &start, &end);
    for (int j = start; j < end; j++) fn(j, n);
  } else {
    fn(i, n);
  }
}

namespace mkl_threadpool_internal {

inline bool DebugWiringEnabled() {
  static bool enabled = []() {
    bool value = false;
    auto status =
        itex::ReadBoolFromEnvVar("ITEX_DEBUG_WIRING", false, &value);
    if (!status.ok()) {
      ITEX_LOG(WARNING) << "ITEX_DEBUG_WIRING parse failed: " << status;
      return false;
    }
    return value;
  }();
  return enabled;
}

inline const char* EnvOrUnset(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? "<unset>" : value;
}

inline void LogAdapterCreated(void* tensorflow_threadpool,
                              int requested_threads, int effective_threads) {
  if (!DebugWiringEnabled()) return;
  ITEX_LOG(INFO)
      << "ITEX_DEBUG_WIRING component=onednn_threadpool "
         "event=adapter_created"
      << " pool=" << tensorflow_threadpool
      << " ownership=borrowed_from_tensorflow"
      << " tensorflow_threads="
      << ::ITEX_TensorFlowThreadPoolNumThreads(tensorflow_threadpool)
      << " requested_threads=" << requested_threads
      << " effective_threads=" << effective_threads
      << " scheduling=ScheduleWithHint"
      << " caller_participates=1"
      << " completion=BlockingCounter"
      << " parallel_for_returns_after_completion=1"
      << " onednn_wait=noop_after_synchronous_parallel_for"
      << " cache_key=tensorflow_pool_plus_effective_threads"
      << " env={TF_NUM_INTRAOP_THREADS="
      << EnvOrUnset("TF_NUM_INTRAOP_THREADS")
      << ",TF_NUM_INTEROP_THREADS=" << EnvOrUnset("TF_NUM_INTEROP_THREADS")
      << ",OMP_NUM_THREADS=" << EnvOrUnset("OMP_NUM_THREADS")
      << ",KMP_BLOCKTIME=" << EnvOrUnset("KMP_BLOCKTIME")
      << ",DNNL_MAX_CPU_ISA=" << EnvOrUnset("DNNL_MAX_CPU_ISA") << "}";
}

struct ScheduledJob {
  bool balance;
  int index;
  int work_items;
  int num_jobs;
  const std::function<void(int, int)>* function;
  BlockingCounter* pending_jobs;
};

inline void RunScheduledJob(void* opaque_job) {
  std::unique_ptr<ScheduledJob> job(
      static_cast<ScheduledJob*>(opaque_job));
  run_jobs(job->balance, job->index, job->work_items, job->num_jobs,
           *job->function);
  job->pending_jobs->DecrementCount();
}

}  // namespace mkl_threadpool_internal

struct MklDnnThreadPool : public threadpool_iface {
  MklDnnThreadPool() = default;

  explicit MklDnnThreadPool(const OpKernelContext* ctx, int num_threads = -1)
      : MklDnnThreadPool(
            ::ITEX_GetTensorFlowThreadPool(
                const_cast<OpKernelContext*>(ctx)->Get()),
            num_threads) {}

  explicit MklDnnThreadPool(void* tensorflow_threadpool,
                            int num_threads = -1)
      : tensorflow_threadpool_(tensorflow_threadpool),
        num_threads_(NormalizeThreadCount(tensorflow_threadpool,
                                          num_threads)) {
    ITEX_CHECK(tensorflow_threadpool_ != nullptr)
        << "TensorFlow Eigen CPU threadpool is unavailable";
    mkl_threadpool_internal::LogAdapterCreated(
        tensorflow_threadpool_, num_threads, num_threads_);
  }

  int get_num_threads() const override { return num_threads_; }

  bool get_in_parallel() const override {
    return tensorflow_threadpool_ != nullptr &&
           ::ITEX_TensorFlowThreadPoolCurrentThreadId(
               tensorflow_threadpool_) != -1;
  }

  uint64_t get_flags() const override { return 0; }

  void parallel_for(int n, const std::function<void(int, int)>& fn) override {
    if (n == 0) return;
    if (n == 1 || tensorflow_threadpool_ == nullptr) {
      fn(0, 1);
      return;
    }

    const int nthr = get_num_threads();
    const int njobs = std::min(n, nthr);
    const bool balance = nthr < n;
    LogFirstParallelFor(n, nthr, njobs, balance);
    if (njobs <= 1) {
      run_jobs(balance, 0, n, njobs, fn);
      return;
    }

    // The caller executes one job while the remaining jobs run on the
    // TensorFlow-owned intra-op pool. TensorFlow retains pool ownership and
    // lifecycle management.
    const int njobs_to_schedule = njobs - 1;
    BlockingCounter pending_jobs(njobs_to_schedule);
    for (int i = 0; i < njobs_to_schedule; ++i) {
      auto* job = new mkl_threadpool_internal::ScheduledJob{
          balance, i, n, njobs, &fn, &pending_jobs};
      ::ITEX_TensorFlowThreadPoolScheduleWithHint(
          tensorflow_threadpool_,
          &mkl_threadpool_internal::RunScheduledJob, job, i, i + 1);
    }

    run_jobs(balance, njobs - 1, n, njobs, fn);
    pending_jobs.Wait();
  }

  void wait() override {}
  ~MklDnnThreadPool() override = default;

  bool Matches(void* tensorflow_threadpool, int num_threads) const {
    return tensorflow_threadpool_ == tensorflow_threadpool &&
           num_threads_ ==
               NormalizeThreadCount(tensorflow_threadpool, num_threads);
  }

 private:
  static int NormalizeThreadCount(void* tensorflow_threadpool,
                                  int num_threads) {
    int max_threads = ::ITEX_TensorFlowThreadPoolNumThreads(
        tensorflow_threadpool);
    if (max_threads < 1) max_threads = 1;
    if (num_threads == -1) return max_threads;
    if (num_threads < 1) return 1;
    return std::min(max_threads, num_threads);
  }

  void LogFirstParallelFor(int work_items, int threads, int jobs,
                           bool balance) const {
    if (!mkl_threadpool_internal::DebugWiringEnabled()) return;
    bool expected = false;
    if (!parallel_for_logged_.compare_exchange_strong(expected, true)) return;
    ITEX_LOG(INFO)
        << "ITEX_DEBUG_WIRING component=onednn_threadpool "
           "event=first_parallel_for"
        << " pool=" << tensorflow_threadpool_
        << " caller_thread_id="
        << ::ITEX_TensorFlowThreadPoolCurrentThreadId(tensorflow_threadpool_)
        << " caller_already_in_pool=" << (get_in_parallel() ? 1 : 0)
        << " work_items=" << work_items << " threads=" << threads
        << " jobs=" << jobs << " scheduled_jobs=" << std::max(0, jobs - 1)
        << " caller_jobs=" << (jobs > 0 ? 1 : 0)
        << " balanced_partition=" << (balance ? 1 : 0)
        << " completion_wait=blocking_counter";
  }

  void* tensorflow_threadpool_ = nullptr;  // Borrowed from TensorFlow.
  int num_threads_ = 1;
  mutable std::atomic<bool> parallel_for_logged_{false};
};

inline MklDnnThreadPool* GetMklDnnThreadPool(const OpKernelContext* ctx,
                                             int num_threads = -1) {
  void* tensorflow_threadpool = ::ITEX_GetTensorFlowThreadPool(
      const_cast<OpKernelContext*>(ctx)->Get());
  ITEX_CHECK(tensorflow_threadpool != nullptr)
      << "TensorFlow Eigen CPU threadpool is unavailable";

  static std::mutex mu;
  static std::vector<std::unique_ptr<MklDnnThreadPool>> threadpools;

  std::lock_guard<std::mutex> lock(mu);
  for (const auto& threadpool : threadpools) {
    if (threadpool->Matches(tensorflow_threadpool, num_threads)) {
      return threadpool.get();
    }
  }

  threadpools.emplace_back(
      new MklDnnThreadPool(tensorflow_threadpool, num_threads));
  return threadpools.back().get();
}

}  // namespace itex

#endif  // ITEX_CORE_UTILS_ONEDNN_MKL_THREADPOOL_H_

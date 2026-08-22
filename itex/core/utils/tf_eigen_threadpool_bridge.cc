#ifndef ITEX_BUILD_JAX

// This bridge always accesses TensorFlow's CPU Eigen device, even when it is
// compiled as part of the XPU build graph. The XPU transition adds
// EIGEN_USE_GPU globally; upstream Eigen interprets that macro as CUDA unless
// HIP is selected, which makes the Tensor header include cuda_runtime.h.
// Remove GPU mode locally before any TensorFlow/Eigen header is included.
#ifdef EIGEN_USE_GPU
#undef EIGEN_USE_GPU
#endif

// TensorFlow compiles DeviceBase's Eigen CPU device support with this macro
// enabled. It must be defined before including TensorFlow/Eigen headers so
// Eigen::ThreadPoolDevice is fully defined rather than only forward-declared.
#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#include "itex/core/utils/tf_eigen_threadpool_bridge.h"

#include "tensorflow/core/framework/op_kernel.h"

namespace {

Eigen::ThreadPoolInterface* GetTensorFlowThreadPool(void* opaque_context) {
  if (opaque_context == nullptr) {
    return nullptr;
  }

  // opaque_context is TF_OpKernelContext* from the TensorFlow C API, which is
  // defined as: struct TF_OpKernelContext { ::tensorflow::OpKernelContext* context; };
  auto* tf_context =
      *reinterpret_cast<::tensorflow::OpKernelContext**>(opaque_context);
  if (tf_context == nullptr) {
    return nullptr;
  }

  auto* tf_device = tf_context->device();
  if (tf_device != nullptr) {
    const auto* cpu_worker_threads = tf_device->tensorflow_cpu_worker_threads();
    if (cpu_worker_threads != nullptr && cpu_worker_threads->workers != nullptr) {
      return cpu_worker_threads->workers->AsEigenThreadPool();
    }

    const auto* eigen_device = tf_device->eigen_cpu_device();
    if (eigen_device != nullptr && eigen_device->getPool() != nullptr) {
      return eigen_device->getPool();
    }
  }

  return nullptr;
}

}  // namespace

extern "C" void* ITEX_GetTensorFlowThreadPool(void* opaque_context) {
  return static_cast<void*>(GetTensorFlowThreadPool(opaque_context));
}

extern "C" int ITEX_TensorFlowThreadPoolNumThreads(
    void* opaque_threadpool) {
  auto* threadpool =
      static_cast<Eigen::ThreadPoolInterface*>(opaque_threadpool);
  return threadpool == nullptr ? 0 : threadpool->NumThreads();
}

extern "C" int ITEX_TensorFlowThreadPoolCurrentThreadId(
    void* opaque_threadpool) {
  auto* threadpool =
      static_cast<Eigen::ThreadPoolInterface*>(opaque_threadpool);
  return threadpool == nullptr ? -1 : threadpool->CurrentThreadId();
}

extern "C" void ITEX_TensorFlowThreadPoolScheduleWithHint(
    void* opaque_threadpool, ITEX_TensorFlowThreadPoolTask task,
    void* argument, int start, int end) {
  auto* threadpool =
      static_cast<Eigen::ThreadPoolInterface*>(opaque_threadpool);
  if (threadpool == nullptr || task == nullptr) {
    return;
  }

  threadpool->ScheduleWithHint(
      [task, argument]() { task(argument); }, start, end);
}

#endif  // ITEX_BUILD_JAX

#ifndef ITEX_BUILD_JAX

#include "itex/core/utils/tf_eigen_threadpool_bridge.h"

#include "tensorflow/core/framework/op_kernel.h"

namespace {

::tensorflow::OpKernelContext* GetTensorFlowContext(void* opaque_context) {
  return reinterpret_cast<::tensorflow::OpKernelContext*>(opaque_context);
}

Eigen::ThreadPoolInterface* GetTensorFlowThreadPool(void* opaque_context) {
  auto* tf_context = GetTensorFlowContext(opaque_context);
  if (tf_context == nullptr) {
    return nullptr;
  }

  auto* tf_device = tf_context->device();
  if (tf_device == nullptr) {
    return nullptr;
  }

  const auto* eigen_device = tf_device->eigen_cpu_device();
  return eigen_device == nullptr ? nullptr : eigen_device->getPool();
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

// Compatibility for OpKernelContext::eigen_cpu_device(). New oneDNN scheduling
// uses the opaque functions above so TensorFlow's Eigen types do not enter the
// normal ITEX translation units.
extern "C" const void* ITEX_GetTensorFlowEigenCpuDevice(
    const void* opaque_context) {
  auto* tf_context = GetTensorFlowContext(
      const_cast<void*>(opaque_context));
  if (tf_context == nullptr || tf_context->device() == nullptr) {
    return nullptr;
  }
  return static_cast<const void*>(
      tf_context->device()->eigen_cpu_device());
}

#endif  // ITEX_BUILD_JAX

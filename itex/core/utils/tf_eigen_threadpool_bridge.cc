#ifndef ITEX_BUILD_JAX

#include "tensorflow/core/framework/op_kernel.h"

extern "C" const void* ITEX_GetTensorFlowEigenCpuDevice(
    const void* opaque_context) {
  if (opaque_context == nullptr) {
    return nullptr;
  }

  const auto* tf_context =
      reinterpret_cast<const ::tensorflow::OpKernelContext*>(
          opaque_context);

  const auto* tf_device = tf_context->device();
  if (tf_device == nullptr) {
    return nullptr;
  }

  return static_cast<const void*>(
      tf_device->eigen_cpu_device());
}

#endif  // ITEX_BUILD_JAX

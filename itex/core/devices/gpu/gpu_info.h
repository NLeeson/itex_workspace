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

#ifndef ITEX_CORE_DEVICES_GPU_GPU_INFO_H_
#define ITEX_CORE_DEVICES_GPU_GPU_INFO_H_

#include "third_party/eigen3/Eigen/Core"
#include "third_party/build_option/sycl/runtime/eigen_itex_gpu_runtime.h"

#ifndef INTEL_CPU_ONLY
#ifndef EIGEN_CXX11_TENSOR_TENSOR_DEVICE_GPU_H
#if !defined(EIGEN_USE_GPU)
#define ITEX_GPU_INFO_EIGEN_USE_GPU_WAS_UNDEFINED
#define EIGEN_USE_GPU
#endif
#if !defined(EIGEN_USE_DPCPP)
#define ITEX_GPU_INFO_EIGEN_USE_DPCPP_WAS_UNDEFINED
#define EIGEN_USE_DPCPP
#endif
#include "unsupported/Eigen/CXX11/Tensor"
#ifdef ITEX_GPU_INFO_EIGEN_USE_DPCPP_WAS_UNDEFINED
#undef EIGEN_USE_DPCPP
#undef ITEX_GPU_INFO_EIGEN_USE_DPCPP_WAS_UNDEFINED
#endif
#ifdef ITEX_GPU_INFO_EIGEN_USE_GPU_WAS_UNDEFINED
#undef EIGEN_USE_GPU
#undef ITEX_GPU_INFO_EIGEN_USE_GPU_WAS_UNDEFINED
#endif
#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_GPU_H
#endif  // INTEL_CPU_ONLY

namespace itex {

#ifndef INTEL_CPU_ONLY
struct DeviceInfo {
  explicit DeviceInfo(gpuDeviceProp_t device_prop, sycl::device device,
                      sycl::context context, int max_work_group_size)
      : device_prop_(device_prop),
        device_(device),
        context_(context),
        max_work_group_size_(max_work_group_size) {}

  gpuDeviceProp_t getGPUDeviceProperties() { return device_prop_; }
  sycl::device getDeivce() { return device_; }
  sycl::context getContext() { return context_; }
  int getMaxWorkGroupSize() { return max_work_group_size_; }

 private:
  gpuDeviceProp_t device_prop_;
  sycl::device device_;
  sycl::context context_;
  int max_work_group_size_;
};

DeviceInfo* GetDeviceInfo(ITEX_GPUStream* stream);

#endif

}  // namespace itex
#endif  // ITEX_CORE_DEVICES_GPU_GPU_INFO_H_

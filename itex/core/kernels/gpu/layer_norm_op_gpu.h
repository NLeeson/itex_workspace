/* Copyright (c) 2022-2026 Intel Corporation

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

#ifndef ITEX_CORE_KERNELS_GPU_LAYER_NORM_OP_GPU_H_
#define ITEX_CORE_KERNELS_GPU_LAYER_NORM_OP_GPU_H_

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/kernels/common/layer_norm_op.h"
#include "itex/core/utils/gpu_helper.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/plugin_tensor.h"
#include "itex/core/utils/tensor_shape.h"

namespace itex {

// oneDNN Intel GPU LayerNorm compiles OpenCL-C and needs an OpenCL GPU ICD.
// ITEX XPU is Level Zero only, so the GPU kernel is a native SYCL launch.
template <typename T, typename U>
struct LayerNormFwdSyclKernel {
  LayerNormFwdSyclKernel(const T* src, const U* scale, const U* shift, T* dst,
                         U* mean, U* var, int cols, float epsilon)
      : src_(src),
        scale_(scale),
        shift_(shift),
        dst_(dst),
        mean_(mean),
        var_(var),
        cols_(cols),
        epsilon_(epsilon) {}

  void operator()(sycl::nd_item<1> item) const {
    const int row = static_cast<int>(item.get_group(0));
    const int lid = static_cast<int>(item.get_local_id(0));
    const int lsz = static_cast<int>(item.get_local_range(0));
    const T* row_src = src_ + static_cast<size_t>(row) * static_cast<size_t>(cols_);
    T* row_dst = dst_ + static_cast<size_t>(row) * static_cast<size_t>(cols_);

    float sum = 0.f;
    float sumsq = 0.f;
    for (int c = lid; c < cols_; c += lsz) {
      const float v = static_cast<float>(row_src[c]);
      sum += v;
      sumsq += v * v;
    }

    auto group = item.get_group();
    sum = sycl::reduce_over_group(group, sum, sycl::plus<float>());
    sumsq = sycl::reduce_over_group(group, sumsq, sycl::plus<float>());

    const float inv_cols = 1.f / static_cast<float>(cols_);
    const float mean = sum * inv_cols;
    float var = sumsq * inv_cols - mean * mean;
    if (var < 0.f) var = 0.f;
    const float inv = sycl::rsqrt(var + epsilon_);

    if (lid == 0 && mean_ != nullptr) {
      mean_[row] = static_cast<U>(mean);
      var_[row] = static_cast<U>(var);
    }

    for (int c = lid; c < cols_; c += lsz) {
      const float v = static_cast<float>(row_src[c]);
      const float y = (v - mean) * inv * static_cast<float>(scale_[c]) +
                      static_cast<float>(shift_[c]);
      row_dst[c] = static_cast<T>(y);
    }
  }

 private:
  const T* src_;
  const U* scale_;
  const U* shift_;
  T* dst_;
  U* mean_;
  U* var_;
  int cols_;
  float epsilon_;
};

template <typename T, typename U, bool is_inteltf_ln>
class LayerNormOp<GPUDevice, T, U, is_inteltf_ln> : public OpKernel {
 public:
  explicit LayerNormOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    if (context->HasAttr("is_training")) {
      OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
    }
    if (context->HasAttr("data_format")) {
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format_));
    }
    OP_REQUIRES(
        context, tensor_format_ == "NHWC",
        errors::InvalidArgument("Invalid data format, only support NHWC"));
    is_inplace_ = false;
    if (context->HasAttr("is_inplace")) {
      OP_REQUIRES_OK(context, context->GetAttr("is_inplace", &is_inplace_));
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& src_tensor = context->input(0);
    const Tensor& scale_tensor = context->input(1);
    const Tensor& shift_tensor = context->input(2);

    const TensorShape src_tf_shape = src_tensor.shape();
    const int ndims = src_tf_shape.dims();

    OP_REQUIRES(context, ndims >= 2 && ndims <= 4,
                errors::InvalidArgument("input must be 2, 3 or 4-dimensional",
                                        src_tensor.shape().DebugString()));
    OP_REQUIRES(context, scale_tensor.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale_tensor.shape().DebugString()));
    OP_REQUIRES(context, shift_tensor.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        shift_tensor.shape().DebugString()));

    TensorShape mean_var_shape;
    for (int i = 0; i < ndims - 1; ++i) {
      mean_var_shape.AddDim(src_tf_shape.dim_size(i));
    }

    Tensor* dst_tensor = nullptr;
    Tensor* layer_mean_tensor = nullptr;
    Tensor* layer_variance_tensor = nullptr;

    if (src_tf_shape.num_elements() == 0) {
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {0}, 0, src_tf_shape, &dst_tensor));
      if (!is_inteltf_ln) {
        AllocateTFOutputs(context, mean_var_shape, &layer_mean_tensor,
                          &layer_variance_tensor, true);
      }
      return;
    }

    const int cols = static_cast<int>(src_tf_shape.dim_size(ndims - 1));
    OP_REQUIRES(context, scale_tensor.dim_size(0) == cols,
                errors::InvalidArgument(
                    "scale size must match input last dimension ", cols,
                    " but got ", scale_tensor.shape().DebugString()));
    OP_REQUIRES(context, shift_tensor.dim_size(0) == cols,
                errors::InvalidArgument(
                    "offset size must match input last dimension ", cols,
                    " but got ", shift_tensor.shape().DebugString()));
    const int rows = static_cast<int>(src_tf_shape.num_elements() / cols);

    if (is_inplace_) {
      context->set_output(0, src_tensor);
      dst_tensor = context->mutable_output(0);
    } else {
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {0}, 0, src_tensor.shape(), &dst_tensor));
    }

    if (!is_inteltf_ln) {
      AllocateTFOutputs(context, mean_var_shape, &layer_mean_tensor,
                        &layer_variance_tensor);
    }

    auto* stream = context->GetDeviceStream();
    OP_REQUIRES(context, stream != nullptr,
                errors::Internal("No SYCL stream for LayerNorm"));

    const int max_wg = static_cast<int>(
        stream->get_device()
            .template get_info<sycl::info::device::max_work_group_size>());
    int wg = 256;
    if (wg > max_wg) wg = max_wg;
    if (wg > cols) {
      wg = cols;
      if (wg < 1) wg = 1;
    }

    const T* src = src_tensor.flat<T>().data();
    const U* scale = scale_tensor.flat<U>().data();
    const U* shift = shift_tensor.flat<U>().data();
    T* dst = dst_tensor->flat<T>().data();
    U* mean = (is_training_ && layer_mean_tensor != nullptr)
                  ? layer_mean_tensor->flat<U>().data()
                  : nullptr;
    U* var = (is_training_ && layer_variance_tensor != nullptr)
                 ? layer_variance_tensor->flat<U>().data()
                 : nullptr;

    stream->submit([&](sycl::handler& cgh) {
      LayerNormFwdSyclKernel<T, U> task(src, scale, shift, dst, mean, var, cols,
                                        epsilon_);
      cgh.parallel_for<LayerNormFwdSyclKernel<T, U>>(
          sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows) *
                                           static_cast<size_t>(wg)),
                            sycl::range<1>(static_cast<size_t>(wg))),
          task);
    });
  }

 private:
  bool is_inplace_;
  float epsilon_;
  bool is_training_ = false;
  string tensor_format_ = "NHWC";

  void AllocateTFOutputs(OpKernelContext* context, TensorShape mean_var_shape,
                         Tensor** layer_mean_tensor,
                         Tensor** layer_variance_tensor,
                         bool init_val = false) {
    OP_REQUIRES_OK(context, context->allocate_output(1, mean_var_shape,
                                                     layer_mean_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, mean_var_shape,
                                                     layer_variance_tensor));
    if (init_val) {
      U nan = Eigen::NumTraits<U>::quiet_NaN();
      const int kSize = mean_var_shape.num_elements();
      auto* stream = context->GetDeviceStream();
      DeviceFill<GPUDevice, U>((*layer_mean_tensor)->flat<U>().data(), nan,
                               kSize, stream);
      DeviceFill<GPUDevice, U>((*layer_variance_tensor)->flat<U>().data(), nan,
                               kSize, stream);
    }
  }
};

}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_LAYER_NORM_OP_GPU_H_

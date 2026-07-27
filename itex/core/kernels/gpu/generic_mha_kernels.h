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

#ifndef ITEX_CORE_KERNELS_GPU_GENERIC_MHA_KERNELS_H_
#define ITEX_CORE_KERNELS_GPU_GENERIC_MHA_KERNELS_H_

#include <algorithm>
#include <cmath>

#include "itex/core/kernels/gpu/softmax_op_functor.h"
#include "itex/core/utils/onednn/onednn_util.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"

namespace itex {
namespace generic_mha {

using dnnl::memory;
using GPUDevice = Eigen::GpuDevice;

inline memory::desc PlainDesc(const memory::dims& dims,
                              const memory::dims& strides) {
  return memory::desc(dims, OneDnnType<float>(), strides);
}

inline void ExecuteMatmul(OpKernelContext* ctx, const dnnl::engine& engine,
                          const dnnl::stream& stream, const Tensor& src,
                          const memory::desc& src_md, const Tensor& weights,
                          const memory::desc& weights_md, Tensor* dst,
                          const memory::desc& dst_md) {
  auto src_mem = CreateDnnlMemory(src_md, engine, GetTensorBuffer<float>(&src));
  auto weights_mem =
      CreateDnnlMemory(weights_md, engine, GetTensorBuffer<float>(&weights));
  auto dst_mem = CreateDnnlMemory(dst_md, engine, GetTensorBuffer<float>(dst));
  auto primitive_desc =
      dnnl::matmul::primitive_desc(engine, src_md, weights_md, dst_md);
  dnnl::matmul(primitive_desc)
      .execute(stream, {{DNNL_ARG_SRC, src_mem},
                        {DNNL_ARG_WEIGHTS, weights_mem},
                        {DNNL_ARG_DST, dst_mem}});
}

class ScaleAndMaskKernel;

struct ScaleAndMask {
  ScaleAndMask(float* scores, const float* mask, int64 total, int heads,
               int queries, int keys, int mask_batches, int mask_heads,
               int mask_queries, float scale, bool use_mask)
      : scores(scores),
        mask(mask),
        total(total),
        heads(heads),
        queries(queries),
        keys(keys),
        mask_batches(mask_batches),
        mask_heads(mask_heads),
        mask_queries(mask_queries),
        scale(scale),
        use_mask(use_mask) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64 index = item.get_global_linear_id();
    if (index >= total) return;
    float value = scores[index] * scale;
    if (use_mask) {
      int64 remaining = index;
      const int key = remaining % keys;
      remaining /= keys;
      const int query = remaining % queries;
      remaining /= queries;
      const int head = remaining % heads;
      const int batch = remaining / heads;
      const int mask_batch = mask_batches == 1 ? 0 : batch;
      const int mask_head = mask_heads == 1 ? 0 : head;
      const int mask_query = mask_queries == 1 ? 0 : query;
      const int64 mask_index =
          ((mask_batch * mask_heads + mask_head) * mask_queries + mask_query) *
              keys +
          key;
      value += mask[mask_index];
    }
    scores[index] = value;
  }

  float* scores;
  const float* mask;
  int64 total;
  int heads;
  int queries;
  int keys;
  int mask_batches;
  int mask_heads;
  int mask_queries;
  float scale;
  bool use_mask;
};

class ApplyDropoutKernel;

struct ApplyDropout {
  ApplyDropout(const float* input, const bool* mask, float* output, int64 total,
               float scale, bool use_dropout)
      : input(input),
        mask(mask),
        output(output),
        total(total),
        scale(scale),
        use_dropout(use_dropout) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64 index = item.get_global_linear_id();
    if (index >= total) return;
    output[index] = use_dropout ? (mask[index] ? input[index] * scale : 0.0f)
                                : input[index];
  }

  const float* input;
  const bool* mask;
  float* output;
  int64 total;
  float scale;
  bool use_dropout;
};

class SoftmaxBackwardKernel;

struct SoftmaxBackward {
  SoftmaxBackward(const float* probability, const float* probability_grad,
                  const bool* dropout_mask, float* score_grad, int rows,
                  int columns, float attention_scale, float dropout_scale,
                  bool use_dropout)
      : probability(probability),
        probability_grad(probability_grad),
        dropout_mask(dropout_mask),
        score_grad(score_grad),
        rows(rows),
        columns(columns),
        attention_scale(attention_scale),
        dropout_scale(dropout_scale),
        use_dropout(use_dropout) {}

  void operator()(sycl::nd_item<1> item) const {
    const int row = item.get_group(0);
    if (row >= rows) return;
    const int lane = item.get_local_id(0);
    const int local_size = item.get_local_range(0);
    const int offset = row * columns;
    float partial_sum = 0.0f;
    for (int column = lane; column < columns; column += local_size) {
      const int index = offset + column;
      float grad = probability_grad[index];
      if (use_dropout) {
        grad = dropout_mask[index] ? grad * dropout_scale : 0.0f;
      }
      partial_sum += grad * probability[index];
    }
    const float row_sum = sycl::reduce_over_group(item.get_group(), partial_sum,
                                                  sycl::plus<float>());
    for (int column = lane; column < columns; column += local_size) {
      const int index = offset + column;
      float grad = probability_grad[index];
      if (use_dropout) {
        grad = dropout_mask[index] ? grad * dropout_scale : 0.0f;
      }
      score_grad[index] =
          (grad - row_sum) * probability[index] * attention_scale;
    }
  }

  const float* probability;
  const float* probability_grad;
  const bool* dropout_mask;
  float* score_grad;
  int rows;
  int columns;
  float attention_scale;
  float dropout_scale;
  bool use_dropout;
};

template <typename KernelName, typename Kernel>
void LaunchElementwise(OpKernelContext* ctx, int64 total, const Kernel& task) {
  auto* stream = ctx->eigen_gpu_device().stream();
  const int max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const int group_size = std::min(256, max_group_size);
  const int64 global_size =
      ((total + group_size - 1) / group_size) * group_size;
  stream->submit([&](sycl::handler& cgh) {
    cgh.parallel_for<KernelName>(sycl::nd_range<1>(sycl::range<1>(global_size),
                                                   sycl::range<1>(group_size)),
                                 task);
  });
}

inline void LaunchSoftmaxBackward(OpKernelContext* ctx, const Tensor& softmax,
                                  const Tensor& probability_grad,
                                  const Tensor& dropout_mask,
                                  Tensor* score_grad, int rows, int columns,
                                  float attention_scale, float dropout_prob,
                                  bool use_dropout) {
  auto* stream = ctx->eigen_gpu_device().stream();
  const int max_group_size =
      stream->get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  const int group_size = std::min(256, max_group_size);
  const float dropout_scale = use_dropout ? 1.0f / (1.0f - dropout_prob) : 1.0f;
  SoftmaxBackward task(softmax.flat<float>().data(),
                       probability_grad.flat<float>().data(),
                       use_dropout ? dropout_mask.flat<bool>().data() : nullptr,
                       score_grad->flat<float>().data(), rows, columns,
                       attention_scale, dropout_scale, use_dropout);
  stream->submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SoftmaxBackwardKernel>(
        sycl::nd_range<1>(sycl::range<1>(rows * group_size),
                          sycl::range<1>(group_size)),
        task);
  });
}

inline void Forward(OpKernelContext* ctx, const Tensor& query,
                    const Tensor& key, const Tensor& value,
                    const Tensor& attention_mask, const Tensor& dropout_mask,
                    Tensor* output, Tensor* softmax, Tensor* dropped_softmax,
                    int batch, int heads, int queries, int keys, int head_size,
                    float dropout_prob, bool use_mask, bool use_dropout) {
  const int hidden = heads * head_size;
  const int64 score_count = static_cast<int64>(batch) * heads * queries * keys;
  auto engine = CreateDnnlEngine<GPUDevice>(*ctx);
  auto stream = CreateDnnlStream(*ctx, engine);

  const auto query_md = PlainDesc(
      {batch, heads, queries, head_size},
      {heads * queries * head_size, queries * head_size, head_size, 1});
  const auto key_transposed_md =
      PlainDesc({batch, heads, head_size, keys},
                {heads * keys * head_size, keys * head_size, 1, head_size});
  const auto score_md =
      PlainDesc({batch, heads, queries, keys},
                {heads * queries * keys, queries * keys, keys, 1});
  ExecuteMatmul(ctx, engine, stream, query, query_md, key, key_transposed_md,
                softmax, score_md);

  int mask_batches = 1;
  int mask_heads = 1;
  int mask_queries = 1;
  const float* mask_data = nullptr;
  if (use_mask) {
    mask_batches = attention_mask.dim_size(0);
    mask_heads = attention_mask.dim_size(1);
    mask_queries = attention_mask.dim_size(2);
    mask_data = attention_mask.flat<float>().data();
  }
  ScaleAndMask scale_and_mask(
      softmax->flat<float>().data(), mask_data, score_count, heads, queries,
      keys, mask_batches, mask_heads, mask_queries,
      1.0f / std::sqrt(static_cast<float>(head_size)), use_mask);
  LaunchElementwise<ScaleAndMaskKernel>(ctx, score_count, scale_and_mask);

  SoftmaxFunctor<GPUDevice, float>()(ctx->eigen_gpu_device(),
                                     softmax->flat_inner_dims<float>(), softmax,
                                     false);
  const float dropout_scale = use_dropout ? 1.0f / (1.0f - dropout_prob) : 1.0f;
  ApplyDropout apply_dropout(
      softmax->flat<float>().data(),
      use_dropout ? dropout_mask.flat<bool>().data() : nullptr,
      dropped_softmax->flat<float>().data(), score_count, dropout_scale,
      use_dropout);
  LaunchElementwise<ApplyDropoutKernel>(ctx, score_count, apply_dropout);

  const auto value_md =
      PlainDesc({batch, heads, keys, head_size},
                {heads * keys * head_size, keys * head_size, head_size, 1});
  const auto output_md = PlainDesc({batch, heads, queries, head_size},
                                   {queries * hidden, head_size, hidden, 1});
  ExecuteMatmul(ctx, engine, stream, *dropped_softmax, score_md, value,
                value_md, output, output_md);
}

inline void Backward(OpKernelContext* ctx, const Tensor& query,
                     const Tensor& key, const Tensor& value,
                     const Tensor& dropout_mask, const Tensor& softmax,
                     const Tensor& dropped_softmax, const Tensor& output_grad,
                     Tensor* query_grad, Tensor* key_grad, Tensor* value_grad,
                     int batch, int heads, int queries, int keys, int head_size,
                     float dropout_prob) {
  const int hidden = heads * head_size;
  const bool use_dropout = dropout_prob != 0.0f;
  auto engine = CreateDnnlEngine<GPUDevice>(*ctx);
  auto stream = CreateDnnlStream(*ctx, engine);

  const auto score_md =
      PlainDesc({batch, heads, queries, keys},
                {heads * queries * keys, queries * keys, keys, 1});
  const auto score_transposed_md =
      PlainDesc({batch, heads, keys, queries},
                {heads * queries * keys, queries * keys, 1, keys});
  const auto output_grad_md =
      PlainDesc({batch, heads, queries, head_size},
                {queries * hidden, head_size, hidden, 1});
  const auto key_value_md =
      PlainDesc({batch, heads, keys, head_size},
                {heads * keys * head_size, keys * head_size, head_size, 1});
  ExecuteMatmul(ctx, engine, stream, dropped_softmax, score_transposed_md,
                output_grad, output_grad_md, value_grad, key_value_md);

  Tensor probability_grad;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                          DT_FLOAT, TensorShape({batch, heads, queries, keys}),
                          &probability_grad));
  const auto value_transposed_md =
      PlainDesc({batch, heads, head_size, keys},
                {heads * keys * head_size, keys * head_size, 1, head_size});
  ExecuteMatmul(ctx, engine, stream, output_grad, output_grad_md, value,
                value_transposed_md, &probability_grad, score_md);

  Tensor score_grad;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                          DT_FLOAT, TensorShape({batch, heads, queries, keys}),
                          &score_grad));
  LaunchSoftmaxBackward(ctx, softmax, probability_grad, dropout_mask,
                        &score_grad, batch * heads * queries, keys,
                        1.0f / std::sqrt(static_cast<float>(head_size)),
                        dropout_prob, use_dropout);

  const auto query_md = PlainDesc(
      {batch, heads, queries, head_size},
      {heads * queries * head_size, queries * head_size, head_size, 1});
  ExecuteMatmul(ctx, engine, stream, score_grad, score_md, key, key_value_md,
                query_grad, query_md);
  ExecuteMatmul(ctx, engine, stream, score_grad, score_transposed_md, query,
                query_md, key_grad, key_value_md);
}

}  // namespace generic_mha
}  // namespace itex

#endif  // ITEX_CORE_KERNELS_GPU_GENERIC_MHA_KERNELS_H_

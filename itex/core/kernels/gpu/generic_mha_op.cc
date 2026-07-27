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

#include <string>

#include "itex/core/kernels/gpu/generic_mha_kernels.h"
#include "itex/core/utils/errors.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/utils/types.h"

namespace itex {
namespace {

Status ValidateProjectedInputs(const Tensor& query, const Tensor& key,
                               const Tensor& value) {
  if (query.dims() != 4 || key.dims() != 4 || value.dims() != 4) {
    return errors::InvalidArgument(
        "query, key, and value must all have rank 4");
  }
  if (query.dim_size(0) != key.dim_size(0) ||
      query.dim_size(0) != value.dim_size(0) ||
      query.dim_size(1) != key.dim_size(1) ||
      query.dim_size(1) != value.dim_size(1) ||
      key.dim_size(2) != value.dim_size(2) ||
      query.dim_size(3) != key.dim_size(3) ||
      query.dim_size(3) != value.dim_size(3)) {
    return errors::InvalidArgument(
        "projected attention dimensions are inconsistent: query=",
        query.shape().DebugString(), ", key=", key.shape().DebugString(),
        ", value=", value.shape().DebugString());
  }
  return Status::OK();
}

class GenericScaledDotProductAttentionOp : public OpKernel {
 public:
  explicit GenericScaledDotProductAttentionOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_mask", &use_mask_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_dropout", &use_dropout_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout_prob", &dropout_prob_));
    OP_REQUIRES(
        ctx, dropout_prob_ >= 0.0f && dropout_prob_ < 1.0f,
        errors::InvalidArgument("dropout_prob must be in the interval [0, 1)"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& query = ctx->input(0);
    const Tensor& key = ctx->input(1);
    const Tensor& value = ctx->input(2);
    const Tensor& attention_mask = ctx->input(3);
    const Tensor& dropout_mask = ctx->input(4);
    OP_REQUIRES_OK(ctx, ValidateProjectedInputs(query, key, value));

    const int batch = query.dim_size(0);
    const int heads = query.dim_size(1);
    const int queries = query.dim_size(2);
    const int head_size = query.dim_size(3);
    const int keys = key.dim_size(2);

    if (use_mask_) {
      OP_REQUIRES(ctx, attention_mask.dims() == 4,
                  errors::InvalidArgument("attention mask must have rank 4"));
      OP_REQUIRES(ctx,
                  (attention_mask.dim_size(0) == 1 ||
                   attention_mask.dim_size(0) == batch) &&
                      (attention_mask.dim_size(1) == 1 ||
                       attention_mask.dim_size(1) == heads) &&
                      (attention_mask.dim_size(2) == 1 ||
                       attention_mask.dim_size(2) == queries) &&
                      attention_mask.dim_size(3) == keys,
                  errors::InvalidArgument(
                      "attention mask is not broadcast-compatible with [",
                      batch, ", ", heads, ", ", queries, ", ", keys,
                      "]: ", attention_mask.shape().DebugString()));
    }
    if (use_dropout_) {
      OP_REQUIRES(
          ctx,
          dropout_mask.shape() == TensorShape({batch, heads, queries, keys}),
          errors::InvalidArgument("dropout mask must have shape [", batch, ", ",
                                  heads, ", ", queries, ", ", keys,
                                  "]: ", dropout_mask.shape().DebugString()));
    }

    Tensor* output = nullptr;
    Tensor* softmax = nullptr;
    Tensor* dropped_softmax = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 0, TensorShape({batch, queries, heads, head_size}), &output));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({batch, heads, queries, keys}),
                                  &softmax));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2, TensorShape({batch, heads, queries, keys}),
                                  &dropped_softmax));

    try {
      generic_mha::Forward(ctx, query, key, value, attention_mask, dropout_mask,
                           output, softmax, dropped_softmax, batch, heads,
                           queries, keys, head_size, dropout_prob_, use_mask_,
                           use_dropout_);
    } catch (const dnnl::error& error) {
      OP_REQUIRES_OK(
          ctx, errors::Aborted("generic scaled-dot-product attention failed: ",
                               error.message));
    }
  }

 private:
  float dropout_prob_ = 0.0f;
  bool use_mask_ = false;
  bool use_dropout_ = false;
};

class GenericScaledDotProductAttentionGradOp : public OpKernel {
 public:
  explicit GenericScaledDotProductAttentionGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout_prob", &dropout_prob_));
    OP_REQUIRES(
        ctx, dropout_prob_ >= 0.0f && dropout_prob_ < 1.0f,
        errors::InvalidArgument("dropout_prob must be in the interval [0, 1)"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& query = ctx->input(0);
    const Tensor& key = ctx->input(1);
    const Tensor& value = ctx->input(2);
    const Tensor& dropout_mask = ctx->input(3);
    const Tensor& softmax = ctx->input(4);
    const Tensor& dropped_softmax = ctx->input(5);
    const Tensor& output_grad = ctx->input(6);
    OP_REQUIRES_OK(ctx, ValidateProjectedInputs(query, key, value));

    const int batch = query.dim_size(0);
    const int heads = query.dim_size(1);
    const int queries = query.dim_size(2);
    const int head_size = query.dim_size(3);
    const int keys = key.dim_size(2);
    const TensorShape score_shape({batch, heads, queries, keys});
    OP_REQUIRES(ctx,
                softmax.shape() == score_shape &&
                    dropped_softmax.shape() == score_shape,
                errors::InvalidArgument(
                    "saved attention probabilities have invalid shapes"));
    OP_REQUIRES(
        ctx,
        output_grad.shape() == TensorShape({batch, queries, heads, head_size}),
        errors::InvalidArgument("output gradient has invalid shape: ",
                                output_grad.shape().DebugString()));
    if (dropout_prob_ != 0.0f) {
      OP_REQUIRES(ctx, dropout_mask.shape() == score_shape,
                  errors::InvalidArgument("dropout mask has invalid shape: ",
                                          dropout_mask.shape().DebugString()));
    }

    Tensor* query_grad = nullptr;
    Tensor* key_grad = nullptr;
    Tensor* value_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, query.shape(), &query_grad));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, key.shape(), &key_grad));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, value.shape(), &value_grad));

    try {
      generic_mha::Backward(ctx, query, key, value, dropout_mask, softmax,
                            dropped_softmax, output_grad, query_grad, key_grad,
                            value_grad, batch, heads, queries, keys, head_size,
                            dropout_prob_);
    } catch (const dnnl::error& error) {
      OP_REQUIRES_OK(
          ctx, errors::Aborted(
                   "generic scaled-dot-product attention gradient failed: ",
                   error.message));
    }
  }

 private:
  float dropout_prob_ = 0.0f;
};

REGISTER_KERNEL_BUILDER(Name("ScaledDotProductAttention")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        GenericScaledDotProductAttentionOp);
REGISTER_KERNEL_BUILDER(Name("ScaledDotProductAttentionGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        GenericScaledDotProductAttentionGradOp);

}  // namespace
}  // namespace itex

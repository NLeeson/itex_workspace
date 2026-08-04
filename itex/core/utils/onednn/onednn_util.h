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

#ifndef ITEX_CORE_UTILS_ONEDNN_ONEDNN_UTIL_H_
#define ITEX_CORE_UTILS_ONEDNN_ONEDNN_UTIL_H_

#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "dnnl.hpp"  // NOLINT(build/include_subdir)

#ifndef INTEL_CPU_ONLY
#include "dnnl_sycl.hpp"  // NOLINT(build/include_subdir)
#endif                    // INTEL_CPU_ONLY

#include "itex/core/utils/logging.h"
#include "itex/core/utils/onednn/mkl_threadpool.h"
#include "itex/core/utils/op_kernel.h"
#include "itex/core/utils/op_requires.h"
#include "itex/core/utils/status.h"
#include "itex/core/utils/strcat.h"
#include "itex/core/utils/tensor_format.h"
#include "itex/core/utils/tensor_shape.h"
#include "itex/core/wrapper/itex_cpu_wrapper.h"

namespace itex {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

#ifndef INTEL_CPU_ONLY
const int MAX_NDIMS = 6;
#else
const int MAX_NDIMS = DNNL_MAX_NDIMS;
#endif

#ifndef INTEL_CPU_ONLY
static dnnl::engine& FindOrCreateEngine(ITEX_GPUStream* stream) {
  static std::map<ITEX_GPUStream*, dnnl::engine> stream_engine_map;
  auto iter = stream_engine_map.find(stream);
  if (iter != stream_engine_map.end()) return iter->second;

  dnnl::engine engine;
  engine = dnnl::sycl_interop::make_engine(stream->get_device(),
                                           stream->get_context());
  return stream_engine_map
      .insert(std::pair<ITEX_GPUStream*, dnnl::engine>(stream, engine))
      .first->second;
}
#endif

typedef enum {
  Dim_N = 0,
  Dim_C = 1,
  Dim_H = 2,
  Dim_W = 3,
  Dim_O = 0,
  Dim_I = 1
} DimensionIndex;

typedef enum {
  Dim3d_N = 0,
  Dim3d_C = 1,
  Dim3d_D = 2,
  Dim3d_H = 3,
  Dim3d_W = 4,
  Dim3d_O = 0,
  Dim3d_I = 1
} DimensionIndex3D;

enum class OneDnnTensorFormat {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
  FORMAT_NDHWC = 2,
  FORMAT_NCDHW = 3,
  FORMAT_X = 4,
  FORMAT_NC = 5,
  FORMAT_TNC = 6,
  FORMAT_INVALID = 7,
};

typedef enum {
  TF_2DFILTER_DIM_H = 0,
  TF_2DFILTER_DIM_W = 1,
  TF_2DFILTER_DIM_I = 2,
  TF_2DFILTER_DIM_O = 3
} TFFilterDims2d;

typedef enum {
  TF_3DFILTER_DIM_P = 0,
  TF_3DFILTER_DIM_H = 1,
  TF_3DFILTER_DIM_W = 2,
  TF_3DFILTER_DIM_I = 3,
  TF_3DFILTER_DIM_O = 4
} TFFilterDims3d;

typedef enum {
  GROUP_FILTER_DIM_G = 0,
  GROUP_FILTER_DIM_O = 1,
  GROUP_FILTER_DIM_I = 2,
  GROUP_FILTER_DIM_H = 3,
  GROUP_FILTER_DIM_W = 4
} FilterGroupDims;

template <typename T>
inline dnnl::memory::data_type OneDnnType();

template <>
inline dnnl::memory::data_type OneDnnType<float>() {
  return dnnl::memory::data_type::f32;
}

template <>
inline dnnl::memory::data_type OneDnnType<double>() {
  return dnnl::memory::data_type::f64;
}

template <>
inline dnnl::memory::data_type OneDnnType<Eigen::half>() {
  return dnnl::memory::data_type::f16;
}

template <>
inline dnnl::memory::data_type OneDnnType<quint8>() {
  return dnnl::memory::data_type::u8;
}

template <>
inline dnnl::memory::data_type OneDnnType<uint8>() {
  return dnnl::memory::data_type::u8;
}

template <>
inline dnnl::memory::data_type OneDnnType<qint8>() {
  return dnnl::memory::data_type::s8;
}

template <>
inline dnnl::memory::data_type OneDnnType<int8>() {
  return dnnl::memory::data_type::s8;
}

template <>
inline dnnl::memory::data_type OneDnnType<qint32>() {
  return dnnl::memory::data_type::s32;
}

template <>
inline dnnl::memory::data_type OneDnnType<Eigen::bfloat16>() {
  return dnnl::memory::data_type::bf16;
}

#ifndef ITEX_BUILD_JAX
template <typename Device>
inline dnnl::engine& CreateDnnlEngine(const OpKernelContext& ctx);

#ifndef INTEL_CPU_ONLY
template <>
inline dnnl::engine& CreateDnnlEngine<GPUDevice>(const OpKernelContext& ctx) {
  auto* ITEX_GPU_stream = ctx.GetDeviceStream();
  return FindOrCreateEngine(ITEX_GPU_stream);
}
#endif  // INTEL_CPU_ONLY

inline dnnl::engine& GetCPUDnnlEngine() {
  static dnnl::engine cpu_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
  return cpu_engine;
}

template <>
inline dnnl::engine& CreateDnnlEngine<CPUDevice>(
    const OpKernelContext& ctx) {
  ITEX_CHECK(::ITEX_GetTensorFlowThreadPool(
                 const_cast<OpKernelContext*>(&ctx)->Get()) != nullptr)
      << "TensorFlow Eigen CPU threadpool is unavailable";
  return GetCPUDnnlEngine();
}

inline dnnl::stream CreateDnnlStream(const OpKernelContext& ctx,
                                     const dnnl::engine& engine,
                                     int num_thread = -1) {
#ifndef INTEL_CPU_ONLY
  ITEX_CHECK(engine.get_kind() == dnnl::engine::kind::gpu)
      << "Create oneDNN stream for unsupported engine.";
  auto* ITEX_GPU_stream = ctx.GetDeviceStream();
  return dnnl::sycl_interop::make_stream(engine, *ITEX_GPU_stream);
#else
#ifndef CC_BUILD
  ITEX_CHECK(engine.get_kind() == dnnl::engine::kind::cpu)
      << "Create oneDNN stream for unsupported engine.";
  MklDnnThreadPool* eigen_tp = GetMklDnnThreadPool(&ctx, num_thread);
  dnnl::stream tp_stream =
      dnnl::stream(dnnl::threadpool_interop::make_stream(engine, eigen_tp));
  return tp_stream;
#else
#ifdef CC_THREADPOOL_BUILD
  if (num_thread == 1) return dnnl::stream(engine);
  MklDnnThreadPool* eigen_tp = GetMklDnnThreadPool(&ctx, num_thread);
  dnnl::stream tp_stream =
      dnnl::stream(dnnl::threadpool_interop::make_stream(engine, eigen_tp));
  return tp_stream;
#else
  ITEX_CHECK(engine.get_kind() == dnnl::engine::kind::cpu)
      << "Create oneDNN stream for unsupported engine.";
  return dnnl::stream(engine);
#endif  // CC_THREADPOOL_BUILD
#endif  // CC_BUILD
#endif  // INTEL_CPU_ONLY
}

#endif  // ITEX_BUILD_JAX
inline bool HasDnnlScratchpad(const dnnl::memory::desc& scratchpad_md) {
  return scratchpad_md.get_size() != 0;
}

inline dnnl::memory CreateDnnlMemory(const dnnl::memory::desc& md,
                                     const dnnl::engine& engine,
                                     void* data_handle = nullptr) {
#ifndef INTEL_CPU_ONLY
  if (engine.get_kind() == dnnl::engine::kind::gpu) {
    auto kind = dnnl::sycl_interop::memory_kind::usm;
    if (data_handle == nullptr)
      return dnnl::sycl_interop::make_memory(md, engine, kind,
                                             DNNL_MEMORY_ALLOCATE);
    else
      return dnnl::sycl_interop::make_memory(md, engine, kind, data_handle);
  }
#endif  // INTEL_CPU_ONLY

  ITEX_CHECK(engine.get_kind() == dnnl::engine::kind::cpu)
      << "Create oneDNN memory for unsupported engine.";
  if (data_handle == nullptr)
    return dnnl::memory(md, engine);
  else
    return dnnl::memory(md, engine, data_handle);
}

inline dnnl::memory::format_tag OneDnnTensorFormatToTag(
    OneDnnTensorFormat format) {
  if (format == OneDnnTensorFormat::FORMAT_NHWC)
    return dnnl::memory::format_tag::nhwc;
  if (format == OneDnnTensorFormat::FORMAT_NCHW)
    return dnnl::memory::format_tag::nchw;
  if (format == OneDnnTensorFormat::FORMAT_NDHWC)
    return dnnl::memory::format_tag::ndhwc;
  if (format == OneDnnTensorFormat::FORMAT_NCDHW)
    return dnnl::memory::format_tag::ncdhw;
  if (format == OneDnnTensorFormat::FORMAT_X)
    return dnnl::memory::format_tag::x;
  if (format == OneDnnTensorFormat::FORMAT_NC)
    return dnnl::memory::format_tag::nc;
  if (format == OneDnnTensorFormat::FORMAT_TNC)
    return dnnl::memory::format_tag::tnc;
  return dnnl::memory::format_tag::undef;
}

inline OneDnnTensorFormat TFDataFormatToOneDnnDataFormat(TensorFormat format,
                                                         bool is_2d = true) {
  if (is_2d) {
    if (format == FORMAT_NHWC) return OneDnnTensorFormat::FORMAT_NHWC;
    if (format == FORMAT_NCHW) return OneDnnTensorFormat::FORMAT_NCHW;
  } else {
    if (format == FORMAT_NHWC) return OneDnnTensorFormat::FORMAT_NDHWC;
    if (format == FORMAT_NCHW) return OneDnnTensorFormat::FORMAT_NCDHW;
  }

  ITEX_CHECK_OK(Status(TF_INVALID_ARGUMENT, "Unsupported data format"));
  return OneDnnTensorFormat::FORMAT_INVALID;
}

inline TensorFormat OneDnnDataFormatToTFDataFormat(OneDnnTensorFormat format) {
  if (format == OneDnnTensorFormat::FORMAT_NHWC ||
      format == OneDnnTensorFormat::FORMAT_NDHWC)
    return FORMAT_NHWC;
  if (format == OneDnnTensorFormat::FORMAT_NCHW ||
      format == OneDnnTensorFormat::FORMAT_NCDHW)
    return FORMAT_NCHW;
  ITEX_CHECK_OK(Status(TF_INVALID_ARGUMENT, "Unsupported data format"));
  return FORMAT_NHWC;
}

inline dnnl::memory::dims TFShapeToOneDnnDims(const TensorShape& shape) {
  if (shape.dims() == 0) {
    dnnl::memory::dims dims{shape.num_elements()};
    return dims;
  }
  dnnl::memory::dims dims(shape.dims());
  for (int d = 0; d < shape.dims(); ++d) {
    dims[d] = shape.dim_size(d);
  }
  return dims;
}

inline dnnl::memory::dims TFShapeToOneDnnDimsInNC(const TensorShape& shape,
                                                  TensorFormat format,
                                                  bool is_2d = true) {
  ITEX_DCHECK_NE(
      static_cast<int>(TFDataFormatToOneDnnDataFormat(format, is_2d)),
      static_cast<int>(OneDnnTensorFormat::FORMAT_INVALID));

  if (is_2d) {
    int n = shape.dim_size(GetTensorDimIndex(format, 'N'));
    int c = shape.dim_size(GetTensorDimIndex(format, 'C'));
    int h = shape.dim_size(GetTensorDimIndex(format, 'H'));
    int w = shape.dim_size(GetTensorDimIndex(format, 'W'));
    return dnnl::memory::dims({n, c, h, w});
  } else {
    int n = shape.dim_size(GetTensorDimIndex<3>(format, 'N'));
    int c = shape.dim_size(GetTensorDimIndex<3>(format, 'C'));
    int d = shape.dim_size(GetTensorDimIndex<3>(format, '0'));
    int h = shape.dim_size(GetTensorDimIndex<3>(format, '1'));
    int w = shape.dim_size(GetTensorDimIndex<3>(format, '2'));
    return dnnl::memory::dims({n, c, d, h, w});
  }
}

inline dnnl::memory::dims OneDnnDimsInNC(const dnnl::memory::dims& in_dims,
                                         TensorFormat format,
                                         bool is_2d = true) {
  ITEX_DCHECK_NE(
      static_cast<int>(TFDataFormatToOneDnnDataFormat(format, is_2d)),
      static_cast<int>(OneDnnTensorFormat::FORMAT_INVALID));

  if (is_2d) {
    int n = in_dims[GetTensorDimIndex(format, 'N')];
    int c = in_dims[GetTensorDimIndex(format, 'C')];
    int h = in_dims[GetTensorDimIndex(format, 'H')];
    int w = in_dims[GetTensorDimIndex(format, 'W')];
    return dnnl::memory::dims({n, c, h, w});
  } else {
    int n = in_dims[GetTensorDimIndex<3>(format, 'N')];
    int c = in_dims[GetTensorDimIndex<3>(format, 'C')];
    int d = in_dims[GetTensorDimIndex<3>(format, '0')];
    int h = in_dims[GetTensorDimIndex<3>(format, '1')];
    int w = in_dims[GetTensorDimIndex<3>(format, '2')];
    return dnnl::memory::dims({n, c, d, h, w});
  }
}

inline TensorShape OneDnnDimsToTFShape(const dnnl::memory::dims& dims) {
  std::vector<int32> shape(dims.size(), -1);
  for (size_t d = 0; d < dims.size(); d++) {
    shape[d] = dims[d];
  }

  TensorShape ret;
  ITEX_CHECK_EQ(TensorShapeUtils::MakeShape(shape, &ret).ok(), true);
  return ret;
}

inline dnnl::memory::dims CalculateTFStrides(
    const dnnl::memory::dims& dims_tf_order) {
  ITEX_CHECK_GT(dims_tf_order.size(), 0);
  dnnl::memory::dims strides(dims_tf_order.size(), 1);
  for (int d = strides.size() - 2; d >= 0; d--) {
    strides[d] = strides[d + 1] * dims_tf_order[d + 1];
  }
  return strides;
}

#ifndef ITEX_BUILD_JAX
template <typename T>
inline void* GetTensorBuffer(const Tensor* tensor) {
  ITEX_CHECK_NOTNULL(tensor);
  return const_cast<void*>(static_cast<const void*>(tensor->flat<T>().data()));
}

template <typename T>
inline dnnl::memory::desc CreatePlainMemDescWithFormatTag(
    const dnnl::memory::dims& onednn_dims) {
  if (onednn_dims.size() > MAX_NDIMS)
    ITEX_LOG(FATAL) << "Max dims for current device is " << MAX_NDIMS;

  if (onednn_dims.size() == 1)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::a);
  else if (onednn_dims.size() == 2)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::ab);
  else if (onednn_dims.size() == 3)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abc);
  else if (onednn_dims.size() == 4)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcd);
  else if (onednn_dims.size() == 5)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcde);
  else if (onednn_dims.size() == 6)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdef);
  else if (onednn_dims.size() == 7)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefg);
  else if (onednn_dims.size() == 8)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefgh);
  else if (onednn_dims.size() == 9)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefghi);
  else if (onednn_dims.size() == 10)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefghij);
  else if (onednn_dims.size() == 11)
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefghijk);
  else
    return dnnl::memory::desc(onednn_dims, OneDnnType<T>(),
                              dnnl::memory::format_tag::abcdefghijkl);
}

void ReorderMemoryInternal(const dnnl::memory* src_memory,
                           dnnl::memory* reorder_memory,
                           dnnl::stream& onednn_stream);

void ReorderMemory(const OpKernelContext& context,
                   const dnnl::memory* src_memory, dnnl::memory* reorder_memory,
                   const dnnl::engine& onednn_engine);

template <typename T>
class WeightCacheManager {
 public:
  WeightCacheManager() = default;
  ~WeightCacheManager() = default;

  bool IsEmpty() TF_LOCKS_EXCLUDED(mu_);

  void SetCache(OpKernelContext* context,
                const dnnl::memory::desc& weight_original_md,
                const dnnl::memory::desc& weight_expected_md, void* weight_data,
                const dnnl::engine& onednn_engine) TF_LOCKS_EXCLUDED(mu_);

  T* GetCache(OpKernelContext* context, const dnnl::memory::desc& expected_md)
      TF_LOCKS_EXCLUDED(mu_);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WeightCacheManager);

  mutex mu_;
  PersistentTensor weight_cached_data_ TF_GUARDED_BY(mu_);
  PersistentTensor weight_cached_md_ TF_GUARDED_BY(mu_);
};

template <typename T>
class BiasCacheManager {
 public:
  BiasCacheManager() = default;
  ~BiasCacheManager() = default;

  bool IsEmpty() TF_LOCKS_EXCLUDED(mu_);

  void SetCache(OpKernelContext* context, const dnnl::memory::desc& bias_md,
                const dnnl::primitive_attr& bias_attr, void* bias_data,
                const dnnl::engine& onednn_engine,
                const dnnl::memory& scales_mem = dnnl::memory())
      TF_LOCKS_EXCLUDED(mu_);

  T* GetCache(OpKernelContext* context) TF_LOCKS_EXCLUDED(mu_);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BiasCacheManager);

  mutex mu_;
  PersistentTensor bias_cached_data_ TF_GUARDED_BY(mu_);
};
#endif

template <typename Device>
inline dnnl::fpmath_mode GetFP32MathMode() {
  std::string fp32_math_mode = "fp32";
  ITEX_CHECK_OK(
      ReadStringFromEnvVar("ITEX_FP32_MATH_MODE", "fp32", &fp32_math_mode));
  fp32_math_mode = str_util::Lowercase(fp32_math_mode);
  if (fp32_math_mode == "fp32") {
    return dnnl::fpmath_mode::strict;
  }
  if (fp32_math_mode == "tf32") {
    if (std::is_same<Device, CPUDevice>::value) {
      ITEX_LOG(FATAL) << "Did not support TF32 math mode on CPU ";
    }
    return dnnl::fpmath_mode::tf32;
  }
  if (fp32_math_mode == "bf32") {
    if (std::is_same<Device, GPUDevice>::value) {
      ITEX_LOG(FATAL) << "Did not support BF32 math mode on GPU ";
    }
    return dnnl::fpmath_mode::bf16;
  }
  ITEX_LOG(FATAL)
      << "Invalid ITEX_FP32_MATH_MODE, should be FP32, TF32 or BF32, but got "
      << fp32_math_mode;
}

}  // namespace itex
#endif  // ITEX_CORE_UTILS_ONEDNN_ONEDNN_UTIL_H_

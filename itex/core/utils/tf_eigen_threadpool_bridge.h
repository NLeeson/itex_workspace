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

#ifndef ITEX_CORE_UTILS_TF_EIGEN_THREADPOOL_BRIDGE_H_
#define ITEX_CORE_UTILS_TF_EIGEN_THREADPOOL_BRIDGE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ITEX_TensorFlowThreadPoolTask)(void* argument);

// Returns the Eigen intra-op threadpool used by the TensorFlow device for the
// current OpKernelContext. TensorFlow owns the returned object.
void* ITEX_GetTensorFlowThreadPool(void* opaque_context);

int ITEX_TensorFlowThreadPoolNumThreads(void* opaque_threadpool);
int ITEX_TensorFlowThreadPoolCurrentThreadId(void* opaque_threadpool);

void ITEX_TensorFlowThreadPoolScheduleWithHint(
    void* opaque_threadpool, ITEX_TensorFlowThreadPoolTask task,
    void* argument, int start, int end);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ITEX_CORE_UTILS_TF_EIGEN_THREADPOOL_BRIDGE_H_

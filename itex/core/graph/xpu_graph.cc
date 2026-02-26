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

#ifndef CC_BUILD
#include "itex/core/graph/xpu_graph.h"
#endif

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "itex/core/devices/xpu_device_util.h"
#include "itex/core/graph/config_util.h"
#include "itex/core/graph/optimizer_config.h"
#include "itex/core/graph/xpu_optimizer.h"
#include "itex/core/utils/cpu_info.h"
#include "itex/core/utils/env_var.h"
#include "itex/core/utils/hw_info.h"
#include "itex/core/utils/logging.h"
#include "itex/core/utils/numbers.h"
#include "itex/core/utils/tf_version.h"
#include "itex/core/version.h"
#include "tensorflow/c/experimental/grappler/grappler.h"

extern bool itex::isxehpc_value;
extern bool itex::hasxmx_value;

void InitGlobalSetting(const OptimizerConfigFlags& config) {
  using env_pair = std::pair<std::string, bool>;

  // TODO(itex): Read config name/value from proto instead of hard code here.
  static std::vector<env_pair> env_list = {
      {"ITEX_ONEDNN_GRAPH", config.enable_onednn_graph},
      {"ITEX_REMAPPER", config.enable_remapper},
      {"ITEX_LAYOUT_OPT", config.enable_layout_opt},
      {"ITEX_AUTO_MIXED_PRECISION", config.enable_auto_mixed_precision},
      {"ITEX_CACHE_ONEDNN_OBJECT", false},
      {"_ITEX_ONEDNN_GRAPH_ALL_TYPE", config.enable_onednn_graph_all_type},
      {"_ITEX_ONEDNN_GRAPH_COMPILER_BACKEND",
       config.enable_onednn_graph_compiler_backend},
      {"_ITEX_ONEDNN_GRAPH_DNNL_BACKEND",
       config.enable_onednn_graph_dnnl_backend},
      {"_ITEX_TEST_MODE", config.enable_test_mode},
  };

  // set ITEX_CACHE_ONEDNN_OBJECT to 1 if this env did not set.
  setenv("ITEX_CACHE_ONEDNN_OBJECT", "1", 0);
  for (env_pair ep : env_list) {
    bool is_enabled = false;
    std::string& str = ep.first;

    ITEX_CHECK_OK(itex::ReadBoolFromEnvVar(str, ep.second, &is_enabled));
    transform(str.begin(), str.end(), str.begin(), ::tolower);
    std::string statues = is_enabled ? "ON." : "OFF.";
    ITEX_VLOG(1) << "ITEX config " << str.substr(5) << " is " << statues;
  }

  // Print Stock TF and ITEX info.
  itex::TensorFlowVersion tf_version;
  auto* itex_version = GetITEXVersion();
  ITEX_VLOG(1) << "Stock Tensorflow version: " << tf_version;
  ITEX_VLOG(1) << "Intel Extension for Tensorflow version: "
               << itex_version->major << "." << itex_version->minor << "."
               << itex_version->patch << ", commit: " << itex_version->hash;

#ifndef INTEL_CPU_ONLY
  itex::isxehpc_value = IsXeHPC();
  itex::hasxmx_value = HasXMX();
#endif

#ifdef INTEL_CPU_ONLY
  bool requested_omp = false;
  ITEX_CHECK_OK(
      itex::ReadBoolFromEnvVar("ITEX_OMP_THREADPOOL", false, &requested_omp));
  if (requested_omp) {
    ITEX_LOG(WARNING)
        << "ITEX_OMP_THREADPOOL=1 is deprecated and ignored. "
        << "Using oneDNN CPU THREADPOOL runtime.";
  }

  // Initialize CPU allocator:
  //   For stock TF version >= 2.9, stock TF will enable MklCPUAllocator by
  //   default. but for TF version < 2.9, need users manually enable it.
  bool enable_onednn = true;
  ITEX_CHECK_OK(
      itex::ReadBoolFromEnvVar("TF_ENABLE_ONEDNN_OPTS", true, &enable_onednn));
  if (enable_onednn) {
    if (tf_version < "2.9.0") {
      ITEX_LOG(INFO) << "For stock TF version < 2.9.0, please manually enable "
                        "`TF_ENABLE_ONEDNN_OPTS` ."
                     << "So can benefit from the optimization of the "
                        "MklCPUAllocator memory allocation.";
    }
  } else {
    ITEX_LOG(INFO) << "Please enable TF_ENABLE_ONEDNN_OPTS."
                   << "So can benefit from the optimization of the "
                      "MklCPUAllocator memory allocation.";
  }
#endif  // INTEL_CPU_ONLY
}

#ifndef CC_BUILD
void TF_InitGraph_Internal(TP_OptimizerRegistrationParams* params,
                           TF_Status* status) {
#else
void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status) {
#endif
  params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->optimizer_configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
  params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;

  // Define some configs to turn off existing optimizers.
  params->optimizer_configs->remapping = TF_TriState_Off;
  params->optimizer_configs->layout_optimizer = TF_TriState_Off;
  // Disable tensorflow auto mixed precision when enable auto mixed precision
  // on itex.
  if (GetOptimizerConfigFlags().enable_auto_mixed_precision) {
    params->optimizer_configs->auto_mixed_precision = TF_TriState_Off;
    params->optimizer_configs->auto_mixed_precision_mkl = TF_TriState_Off;
    params->optimizer_configs->auto_mixed_precision_onednn_bfloat16 =
        TF_TriState_Off;
  }

  // ITEX + oneDNN Graph INT8 pass doesn't support constant folding pass
  if (!GetOptimizerConfigFlags().enable_tf_constant_folding) {
    params->optimizer_configs->constant_folding = TF_TriState_Off;
  }
  // Set functions to create a new optimizer.
  params->optimizer->optimize_func = (itex::graph::Optimizer_Optimize);
  params->optimizer->destroy_func = (itex::graph::Optimizer_Destroy);

#ifdef INTEL_CPU_ONLY
  params->device_type = itex::DEVICE_CPU;
  params->optimizer->create_func = (itex::graph::Optimizer_CPU_Create);
#else
  params->device_type = itex::DEVICE_XPU;
  params->optimizer->create_func = (itex::graph::Optimizer_XPU_Create);
#endif  // INTEL_CPU_ONLY
  ITEX_LOG(INFO) << "ITEX_INIT Graph optimizer registered for device_type="
                 << params->device_type;

  // Initialize and print global settings.
  InitGlobalSetting(GetOptimizerConfigFlags());
}

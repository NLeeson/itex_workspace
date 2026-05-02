package(default_visibility = ["//visibility:public"])

load(":platform.bzl", "sycl_library_path")
load("@local_config_sycl//sycl:build_defs.bzl", "if_sycl")
load(
    "@intel_extension_for_tensorflow//itex:itex.bzl",
    "if_using_nextpluggable_device",
    itex_cc_library = "cc_library",
)

config_setting(
    name = "using_sycl",
    values = {
        "define": "using_sycl=true",
    },
)

config_setting(
    name = "using_xetla",
    values = {
        "define": "using_xetla=true",
    },
)

config_setting(
    name = "using_nextpluggable",
    values = {
        "define": "using_nextpluggable=true",
    },
)

itex_cc_library(
    name = "itex_gpu_headers",
    hdrs = glob([
        "runtime/itex_gpu_runtime.h",
    ]),
    copts = if_using_nextpluggable_device(
      ["-DUSING_NEXTPLUGGABLE_DEVICE"],
    ),
    includes = [
        ".",
        "include",
    ],
)

cc_library(
    name = "sycl_runtime",
    linkopts = if_sycl([
        "-L%{SYCL_ROOT_DIR}/lib",
        "-Wl,-rpath,%{SYCL_ROOT_DIR}/lib",
        "-Wl,--no-as-needed",
        "-lsycl",
    ]),
)

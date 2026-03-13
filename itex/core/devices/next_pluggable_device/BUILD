package(default_visibility = ["//visibility:public"])

load("//itex:itex.bzl", "gpu_cc_library", "if_using_nextpluggable_device")

gpu_cc_library(
    name = "next_pluggable_device",
    srcs = ["gpu_device_plugin.cc"],
    hdrs = ["gpu_device_plugin.h"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/devices/gpu:itex_gpu_runtime_imp",
        "//itex/core/utils:libintel_extension_for_openxla",
        "//third_party/build_option/sycl:itex_gpu_header",
    ],
    alwayslink = True,
)

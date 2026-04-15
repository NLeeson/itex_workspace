package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_import(
    name = "tbb_prebuilt",
    shared_library = "lib/libtbb.so",
)

cc_library(
    name = "tbb",
    hdrs = glob([
        "include/oneapi/**/*.h",
        "include/tbb/**/*.h",
    ]),
    includes = [
        "include",
    ],
    deps = [
        ":tbb_prebuilt",
    ],
)

cc_import(
    name = "tbbmalloc_prebuilt",
    shared_library = "lib/libtbbmalloc.so",
)

cc_library(
    name = "tbbmalloc",
    hdrs = glob([
        "include/oneapi/**/*.h",
        "include/tbb/**/*.h",
    ]),
    includes = [
        "include",
    ],
    deps = [
        ":tbbmalloc_prebuilt",
    ],
)

cc_import(
    name = "tbbmalloc_proxy_prebuilt",
    shared_library = "lib/libtbbmalloc_proxy.so",
)

cc_library(
    name = "tbbmalloc_proxy",
    hdrs = glob([
        "include/oneapi/**/*.h",
        "include/tbb/**/*.h",
    ]),
    includes = [
        "include",
    ],
    deps = [
        ":tbbmalloc_proxy_prebuilt",
        ":tbbmalloc",
    ],
)

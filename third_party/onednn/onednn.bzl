def convert_cl_to_cpp(name, src, cl_list, **kwargs):
    """Create a miniature of the src image.
    The generated file is prefixed with 'small_'.
    """
    cpp_list = [cl.replace(".cl", "_kernel.cpp") for cl in cl_list]
    kernel_list = src.replace(".in", "")
    cpp_list.append(kernel_list)

    tool = "@intel_extension_for_tensorflow//third_party/onednn:gen_gpu_kernel_list"

    native.genrule(
        name = name,
        srcs = [src],
        outs = cpp_list,
        tools = [tool],
        cmd = "$(location {}) ".format(tool) + "--in=$< --out=$(@D) --header=False",
        **kwargs
    )

def convert_header_to_cpp(name, src, header_list, **kwargs):
    """Create a miniature of the src image.
    The generated file is prefixed with 'small_'.
    """
    cpp_list = []
    h_list = []
    for h in header_list:
        if h.endswith(".h"):
            h_list.append(h.replace(".h", "_header.cpp"))
    cpp_list.extend(h_list)

    tool = "@intel_extension_for_tensorflow//third_party/onednn:gen_gpu_kernel_list"

    native.genrule(
        name = name,
        srcs = [src],
        outs = cpp_list,
        tools = [tool],
        cmd = "$(location {}) ".format(tool) + "--in=$< --out=$(@D) --header=True",
        **kwargs
    )

def gen_onednn_version(name, header_in, header_out, **kwargs):
    tool = "@intel_extension_for_tensorflow//third_party/onednn:gen_onednn_version"

    native.genrule(
        name = name,
        srcs = [header_in],
        outs = [header_out],
        tools = [tool],
        cmd = "$(location {}) ".format(tool) + "--in=$< " + "--out=$@",
        **kwargs
    )

def _gen_onednn_config_impl(ctx):
    args = ctx.actions.args()
    args.add("--src=%s" % ctx.file.src.path)
    args.add("--out=%s" % ctx.outputs.out.path)
    for key, value in sorted(ctx.attr.substitutions.items()):
        args.add("--substitution")
        args.add(key)
        args.add(value)

    ctx.actions.run(
        inputs = [ctx.file.src],
        outputs = [ctx.outputs.out],
        executable = ctx.executable._tool,
        arguments = [args],
    )

_gen_onednn_config = rule(
    implementation = _gen_onednn_config_impl,
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "substitutions": attr.string_dict(mandatory = True),
        "out": attr.output(mandatory = True),
        "_tool": attr.label(
            default = Label("@intel_extension_for_tensorflow//third_party/onednn:gen_onednn_config"),
            executable = True,
            cfg = "exec",
        ),
    },
)

def gen_onednn_config(name, src, out, substitutions, **kwargs):
    _gen_onednn_config(
        name = name,
        src = src,
        out = out,
        substitutions = substitutions,
        **kwargs
    )

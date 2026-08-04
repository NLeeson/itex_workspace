"""Setup TensorFlow as external dependency"""

_TF_HEADER_DIR = "TF_HEADER_DIR"
_TF_SHARED_LIBRARY_DIR = "TF_SHARED_LIBRARY_DIR"

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl
    repository_ctx.template(
        out,
        Label("//third_party/tf_dependency:%s.tpl" % tpl),
        substitutions,
    )

def _fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sPython Configuration Error:%s %s\n" % (red, no_color, msg))

def _is_windows(repository_ctx):
    """Returns true if the host operating system is windows."""
    os_name = repository_ctx.os.name.lower()
    if os_name.find("windows") != -1:
        return True
    return False

def _execute(
        repository_ctx,
        cmdline,
        error_msg = None,
        error_details = None,
        empty_stdout_fine = False):
    """Executes an arbitrary shell command.

    Helper for executes an arbitrary shell command.

    Args:
      repository_ctx: The context object that is passed to the implementation
        function of the repository rule.
      cmdline: The command to execute.
      error_msg: The error message to print on failure.
      error_details: Additional error details to print on failure.
      empty_stdout_fine: Whether empty stdout is accepted.

    Returns:
      The result of repository_ctx.execute.
    """
    result = repository_ctx.execute(cmdline)
    if result.stderr or not (empty_stdout_fine or result.stdout):
        _fail("\n".join([
            error_msg.strip() if error_msg else "Repository command failed",
            result.stderr.strip(),
            error_details if error_details else "",
        ]))
    return result

def _read_dir(repository_ctx, src_dir):
    """Returns a string with all files in a directory."""
    if _is_windows(repository_ctx):
        src_dir = src_dir.replace("/", "\\")
        find_result = _execute(
            repository_ctx,
            ["cmd.exe", "/c", "dir", src_dir, "/b", "/s", "/a-d"],
            empty_stdout_fine = True,
        )
        result = find_result.stdout.replace("\\", "/")
    else:
        find_result = _execute(
            repository_ctx,
            ["find", src_dir, "-follow", "-type", "f"],
            empty_stdout_fine = True,
        )
        result = find_result.stdout
    return result

def _dir_exists(repository_ctx, src_dir):
    result = repository_ctx.execute(
        ["ls", src_dir],
    )
    return result.stdout

def _genrule(genrule_name, command, outs):
    """Returns a genrule target definition."""
    return (
        "genrule(\n" +
        '    name = "' +
        genrule_name + '",\n' +
        "    outs = [\n" +
        outs +
        "\n    ],\n" +
        '    cmd = """\n' +
        command +
        '\n   """,\n' +
        ")\n"
    )

def _norm_path(path):
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def _symlink_genrule_for_dir(
        repository_ctx,
        src_dir,
        dest_dir,
        genrule_name,
        src_files = [],
        dest_files = [],
        tf_pip_dir_rename_pair = [],
        filter_dependency_headers = True):
    """Returns a genrule that copies a directory into the external repo.

    The normal TensorFlow header target intentionally filters dependency
    headers so they do not collide with ITEX-owned Abseil and Eigen. A private
    implementation-only target may disable that filter for translation units
    that must compile against TensorFlow's private C++ ABI.
    """
    tf_pip_dir_rename_pair_len = len(tf_pip_dir_rename_pair)
    if tf_pip_dir_rename_pair_len != 0 and tf_pip_dir_rename_pair_len != 2:
        _fail("The size of argument tf_pip_dir_rename_pair should be either 0 or 2, but %d is given." % tf_pip_dir_rename_pair_len)

    if src_dir != None:
        src_dir = _norm_path(src_dir)
        dest_dir = _norm_path(dest_dir)
        files = "\n".join(sorted(_read_dir(repository_ctx, src_dir).splitlines()))

        if tf_pip_dir_rename_pair_len:
            dest_files = files.replace(src_dir, "").replace(tf_pip_dir_rename_pair[0], tf_pip_dir_rename_pair[1]).splitlines()
        else:
            dest_files = files.replace(src_dir, "").splitlines()
        src_files = files.splitlines()

    command = []
    outs = []

    for i in range(len(dest_files)):
        dest_file_lower = dest_files[i].lower()
        should_copy = dest_files[i] != ""
        if filter_dependency_headers:
            should_copy = should_copy and (
                dest_file_lower.rfind("third_party") == -1 and
                dest_file_lower.rfind("external") == -1 and
                dest_file_lower.rfind("absl") == -1 and
                dest_file_lower.rfind("eigen") == -1
            )

        if should_copy:
            dest = "$(@D)/" + dest_dir + dest_files[i] if len(dest_files) != 1 else "$(@D)/" + dest_files[i]
            command.append('cp -f "%s" "%s"' % (src_files[i], dest))
            outs.append('        "' + dest_dir + dest_files[i] + '",')

    return _genrule(
        genrule_name,
        ";\n".join(command),
        "\n".join(outs),
    )

def _tf_pip_impl(repository_ctx):
    tf_header_dir = repository_ctx.os.environ[_TF_HEADER_DIR]
    tf_header_rule = ""
    tf_private_header_rule = ""

    tf_shared_library_name = "dummy"
    tf_shared_library_rule = ""
    if _dir_exists(repository_ctx, tf_header_dir):
        tf_header_rule = _symlink_genrule_for_dir(
            repository_ctx,
            tf_header_dir,
            "include",
            "tf_header_include",
            tf_pip_dir_rename_pair = ["tensorflow_core", "tensorflow"],
        )

        tf_private_header_genrule = _symlink_genrule_for_dir(
            repository_ctx,
            tf_header_dir,
            "private_include",
            "tf_private_header_include",
            tf_pip_dir_rename_pair = ["tensorflow_core", "tensorflow"],
            filter_dependency_headers = False,
        )
        tf_private_header_rule = (
            "\ncc_library(\n" +
            '    name = "tf_private_header_lib",\n' +
            '    hdrs = [":tf_private_header_include"],\n' +
            '    includes = ["private_include"],\n' +
            '    visibility = ["//visibility:public"],\n' +
            ")\n\n" +
            tf_private_header_genrule
        )

        tf_shared_library_dir = repository_ctx.os.environ[_TF_SHARED_LIBRARY_DIR]
        tf_shared_library_name = "_pywrap_tensorflow_internal.so"
        tf_shared_library_path = "%s/python/%s" % (
            tf_shared_library_dir,
            tf_shared_library_name,
        )

        tf_shared_library_rule = _symlink_genrule_for_dir(
            repository_ctx,
            None,
            "",
            tf_shared_library_name,
            [tf_shared_library_path],
            ["lib_tensorflow_internal.so"],
        )

    _tpl(repository_ctx, "BUILD", {
        "%{TF_HEADER_GENRULE}": tf_header_rule + tf_private_header_rule,
        "%{TF_SHARED_LIBRARY_GENRULE}": tf_shared_library_rule,
        "%{TF_SHARED_LIBRARY_NAME}": tf_shared_library_name,
    })

tf_configure = repository_rule(
    environ = [
        _TF_HEADER_DIR,
    ],
    implementation = _tf_pip_impl,
)

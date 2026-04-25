import os
import sys


def parse_args(argv):
  args = {"--substitution": []}
  i = 0
  while i < len(argv):
    arg = argv[i]
    if arg == "--substitution":
      if i + 2 >= len(argv):
        raise ValueError("missing key/value for --substitution")
      args["--substitution"].append((argv[i + 1], argv[i + 2]))
      i += 3
      continue
    if "=" in arg:
      k, v = arg.split("=", 1)
      args[k] = v
      i += 1
      continue
    raise ValueError("unsupported arg: %s" % arg)
  return args


def rewrite_line(line, substitutions):
  stripped = line.rstrip("\n")
  replacement = substitutions.get(stripped)
  if replacement is None:
    return line
  if line.endswith("\n"):
    return replacement + "\n"
  return replacement


def generate_config(src, out, substitutions):
  with open(os.path.expanduser(src), "r", encoding="utf-8") as inf:
    lines = inf.readlines()

  content = "".join(rewrite_line(line, substitutions) for line in lines)

  out = os.path.expanduser(out)
  out_dir = os.path.dirname(out)
  if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

  with open(out, "w", encoding="utf-8") as outf:
    outf.write(content)


def main():
  args = parse_args(sys.argv[1:])
  generate_config(
      args["--src"],
      args["--out"],
      dict(args["--substitution"]),
  )


if __name__ == "__main__":
  main()

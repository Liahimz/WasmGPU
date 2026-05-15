#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def symbol_name(path):
    stem = re.sub(r"[^0-9A-Za-z_]", "_", path.name)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem.upper()


def raw_literal(data):
    delimiter = "wgsl_shader"
    while f"){delimiter}\"" in data:
        delimiter += "_x"
    return f'R"{delimiter}({data}){delimiter}"'


def main():
    parser = argparse.ArgumentParser(description="Embed WGSL shader files into a C++ header.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shaders = sorted(path for path in args.input_dir.glob("*.wgsl") if path.is_file())
    rows = []
    for path in shaders:
        rows.append(f"static constexpr const char* {symbol_name(path)} = {raw_literal(path.read_text())};")

    if not rows:
        rows.append('static constexpr const char* EMPTY_WGSL = "";')

    template = args.template.read_text()
    output = template.replace("@EMBED_SHADER_ROWS@", "\n\n".join(rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)


if __name__ == "__main__":
    main()

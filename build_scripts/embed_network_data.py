#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def symbol_name(path):
    stem = re.sub(r"[^0-9A-Za-z_]", "_", path.as_posix())
    stem = re.sub(r"_+", "_", stem).strip("_")
    if stem and stem[0].isdigit():
        stem = "_" + stem
    return "NETWORK_" + stem.upper()


def format_bytes(data):
    if not data:
        return "    0x00"

    rows = []
    for offset in range(0, len(data), 12):
        chunk = data[offset:offset + 12]
        rows.append("    " + ", ".join(f"0x{value:02x}" for value in chunk))
    return ",\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Embed network_data files into a C++ header.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true", help="Embed files from nested model-data directories.")
    parser.add_argument("--exclude-dir", action="append", default=[], help="Relative directory to skip while embedding. Can be repeated.")
    args = parser.parse_args()

    files = []
    for pattern in ("*.bin", "*.json"):
        glob = args.input_dir.rglob if args.recursive else args.input_dir.glob
        files.extend(glob(pattern))
    excluded_dirs = {Path(value) for value in args.exclude_dir}
    files = sorted(
        path for path in files
        if path.is_file() and not any(
            excluded == path.relative_to(args.input_dir) or excluded in path.relative_to(args.input_dir).parents
            for excluded in excluded_dirs
        )
    )

    array_rows = []
    blob_rows = []

    for path in files:
        relative_path = path.relative_to(args.input_dir)
        data = path.read_bytes()
        symbol = symbol_name(relative_path)
        array_rows.append(
            f"static constexpr uint8_t {symbol}[] = {{\n"
            f"{format_bytes(data)}\n"
            f"}};"
        )
        blob_rows.append(f'    {{"{relative_path.as_posix()}", {symbol}, sizeof({symbol})}}')

    if not array_rows:
        array_rows.append("static constexpr uint8_t NETWORK_EMPTY[] = { 0x00 };")

    template = args.template.read_text()
    output = template.replace("@EMBED_ARRAY_ROWS@", "\n\n".join(array_rows))
    output = output.replace("@EMBED_BLOB_ROWS@", ",\n".join(blob_rows))
    output = output.replace("@EMBED_BLOB_COUNT@", str(len(blob_rows)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)


if __name__ == "__main__":
    main()

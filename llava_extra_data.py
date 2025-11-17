#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------
# I/O helpers
# ---------------------------

def read_json_any(path: Path) -> List[Dict[str, Any]]:
    """Read either array-style JSON or JSONL."""
    with path.open("r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        # JSON Lines
        return [json.loads(line) for line in f if line.strip()]

def write_json_array(path: Path, items: List[Dict[str, Any]], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(items, f, ensure_ascii=False, indent=2)
        else:
            json.dump(items, f, ensure_ascii=False)

# ---------------------------
# Operations
# ---------------------------

def op_sample(files: List[Path], output: Path, per_file: int, seed: int, pretty: bool) -> None:
    random.seed(seed)
    combined: List[Dict[str, Any]] = []

    for p in files:
        data = read_json_any(p)
        k = min(per_file, len(data))
        if k < per_file:
            print(f"Warning: {p} has only {len(data)} items; sampling {k}.")
        sampled = random.sample(data, k) if k > 0 else []
        combined.extend(sampled)

    random.shuffle(combined)
    for item in combined:
        if "image" in item:
            item["image"] = f"train2017/{item['image']}"
    write_json_array(output, combined, pretty=pretty)
    print(f"[sample] Wrote {len(combined)} items to {output}")

def op_merge(files: List[Path], output: Path, seed: int, pretty: bool, unique_by_id: bool) -> None:
    random.seed(seed)
    combined: List[Dict[str, Any]] = []
    for p in files:
        combined.extend(read_json_any(p))

    if unique_by_id:
        seen = set()
        deduped = []
        for item in combined:
            item_id = item.get("id")
            if item_id is None or item_id not in seen:
                deduped.append(item)
                if item_id is not None:
                    seen.add(item_id)
        combined = deduped

    random.shuffle(combined)
    write_json_array(output, combined, pretty=pretty)
    print(f"[merge] Wrote {len(combined)} items to {output}")

# ---------------------------
# CLI
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dataset tools: sample from multiple JSON files or merge them, then shuffle and save."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # sample subcommand
    p_sample = subparsers.add_parser(
        "sample",
        help="Sample K items from each input JSON/JSONL file, combine, shuffle, and save."
    )
    p_sample.add_argument("files", type=Path, nargs="+",
                          help="Input JSON/JSONL files (2 or more; commonly 4).")
    p_sample.add_argument("-o", "--output", type=Path, required=True, help="Output JSON file.")
    p_sample.add_argument("--per-file", type=int, default=10_000,
                          help="Samples per file (default: 10000).")
    p_sample.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p_sample.add_argument("--no-indent", action="store_true", help="Write compact JSON (no pretty indent).")

    # merge subcommand
    p_merge = subparsers.add_parser(
        "merge",
        help="Merge two or more JSON/JSONL files, optionally dedupe by 'id', shuffle, and save."
    )
    p_merge.add_argument("files", type=Path, nargs="+",
                         help="Input JSON/JSONL files (at least 2).")
    p_merge.add_argument("-o", "--output", type=Path, required=True, help="Output JSON file.")
    p_merge.add_argument("--unique-by-id", action="store_true",
                         help="Remove duplicates by 'id' after merging (keeps first occurrence).")
    p_merge.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p_merge.add_argument("--no-indent", action="store_true", help="Write compact JSON (no pretty indent).")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "sample":
        if len(args.files) < 2:
            parser.error("sample: provide at least two input files.")
        op_sample(
            files=args.files,
            output=args.output,
            per_file=args.per_file,
            seed=args.seed,
            pretty=not args.no_indent,
        )
    elif args.cmd == "merge":
        if len(args.files) < 2:
            parser.error("merge: provide at least two input files.")
        op_merge(
            files=args.files,
            output=args.output,
            seed=args.seed,
            pretty=not args.no_indent,
            unique_by_id=args.unique_by_id,
        )
    else:
        parser.error("unknown command")

if __name__ == "__main__":
    main()

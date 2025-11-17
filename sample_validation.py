#!/usr/bin/env python3
import json
import argparse
import random
from collections import defaultdict
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Sample n questions per question_type and save all to one JSON file."
    )
    parser.add_argument("input_json", type=Path, help="Path to the input JSON file.")
    parser.add_argument("n", type=int, help="Number of samples per question_type.")
    parser.add_argument(
        "-o", "--outfile", type=Path, default=Path("sampled_all.json"),
        help="Output JSON file (single file with all sampled records)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle the final combined list before saving."
    )
    args = parser.parse_args()

    # Load input
    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be a list of records.")

    # Group by question_type
    groups = defaultdict(list)
    for item in data:
        qt = item.get("question_type")
        if qt is None:
            continue
        groups[qt].append(item)

    # Sample from each group
    random.seed(args.seed)
    sampled_all = []
    for qt, items in groups.items():
        k = min(args.n, len(items))
        sampled = random.sample(items, k) if k > 0 else []
        sampled_all.extend(sampled)

    # Optionally shuffle combined output
    if args.shuffle:
        random.shuffle(sampled_all)

    # Write single JSON file (same format: list of records)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with args.outfile.open("w", encoding="utf-8") as f:
        json.dump(sampled_all, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(sampled_all)} sampled items to {args.outfile}")

if __name__ == "__main__":
    main()

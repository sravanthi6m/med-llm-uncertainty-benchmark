#!/usr/bin/env python3
"""
Blend two parallel JSON‑array files by taking 50 % of the ids
from 'original' and the other 50 % from 'modified'.

Each JSON file must be an array of objects and share the same
unique 'id' field (default key is "id").

For every pair in `file_pairs`, this script:
  • loads both arrays
  • randomly divides the ids in half
  • copies the corresponding objects from their respective files
  • writes one blended JSON file to OUTPUT_DIR

Author: ChatGPT, 2025‑07‑21
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------
# 1) EDIT YOUR PAIRS HERE  – add as many (orig, mod) tuples as you need
# ------------------------------------------------------------
file_pairs = [
    (
        "/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/few_shot_pool_medqa_1_train_randabst.json",
        "/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/perturbed_few_shot_pool_medqa_1_train_randabst.json",
    )
    # ("data/typeA_orig.json", "data/typeA_mod.json"),
    # ("data/typeB_orig.json", "data/typeB_mod.json"),
]
# ------------------------------------------------------------
OUTPUT_DIR = Path("./blended")  # folder where outputs go
ID_KEY = "id"  # change if your id field is named differently
# ------------------------------------------------------------


def load_by_id(path: Path, id_key: str):
    """Return a dict mapping id → object for a JSON‑array file."""
    with path.open("r", encoding="utf‑8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array.")
    mapping = {}
    for obj in data:
        if id_key not in obj:
            raise KeyError(f"{path}: object missing '{id_key}' key.")
        mapping[obj[id_key]] = obj
    return mapping


def blend_pair(orig_path: str, mod_path: str, seed=None):
    orig_path = Path(orig_path)
    mod_path = Path(mod_path)

    orig_map = load_by_id(orig_path, ID_KEY)
    mod_map = load_by_id(mod_path, ID_KEY)

    orig_ids = set(orig_map)
    mod_ids = set(mod_map)
    if orig_ids != mod_ids:
        miss_orig = mod_ids - orig_ids
        miss_mod = orig_ids - mod_ids
        raise ValueError(
            f"Id mismatch between {orig_path.name} and {mod_path.name}\n"
            f"  only in orig : {sorted(list(miss_mod))[:5]}\n"
            f"  only in mod  : {sorted(list(miss_orig))[:5]}"
        )

    ids = list(orig_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    k = len(ids) // 2  # floor(n/2)
    ids_from_orig = set(ids[:k])
    ids_from_mod = set(ids[k:])  # remainder (>= k)

    # If you want the extra id to come from *orig* instead, just swap the sets above.

    blended = [orig_map[i] for i in ids_from_orig] + [mod_map[i] for i in ids_from_mod]
    blended.sort(key=lambda obj: obj["id"])
    # rng.shuffle(blended)  # optional: mix the order

    out_name = f"{orig_path.stem}_blended.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / out_name

    with out_file.open("w", encoding="utf‑8") as f:
        json.dump(blended, f, indent=2, ensure_ascii=False)

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"{out_file.name}: {len(blended)} items "
        f"({len(ids_from_orig)} from orig, {len(ids_from_mod)} from mod)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Blend 50 % ids from each of two JSON files."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible splits."
    )
    args = parser.parse_args()

    if not file_pairs:
        parser.error(
            "file_pairs list is empty – edit blend_halves.py to add your input files."
        )

    for orig, mod in file_pairs:
        blend_pair(orig, mod, seed=args.seed)


if __name__ == "__main__":
    main()

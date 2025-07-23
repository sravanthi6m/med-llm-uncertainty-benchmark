#!/usr/bin/env python3
"""
Blend two parallel JSON‑array files.

• 50 % of ids → copy object from original file
• 50 % of ids → copy object from modified file
   – of those, 50 % have their "answer" replaced with the
     option letter whose text equals the REFUSAL_TEXT.

Outputs combined arrays (sorted by id) to ./blended/.
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------------------------
file_pairs = [
    (
        "/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/few_shot_pool_medqa_1_train_randabst.json",
        "/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/perturbed_few_shot_pool_medqa_1_train_randabst.json",
    ),
]
# ----------------------------------------------------------------------
OUTPUT_DIR = Path("./blended")
ID_KEY = "id"
REFUSAL_TEXT = "I am unable to answer this question for certain reasons."
# ----------------------------------------------------------------------


def load_by_id(path: Path, id_key: str):
    with path.open("r", encoding="utf-8") as f:
        arr = json.load(f)
    if not isinstance(arr, list):
        raise ValueError(f"{path} is not a JSON array")
    return {obj[id_key]: obj for obj in arr if id_key in obj}


def blend_pair(orig_path: str, mod_path: str, rng: random.Random):
    orig_path, mod_path = Path(orig_path), Path(mod_path)

    orig_map = load_by_id(orig_path, ID_KEY)
    mod_map = load_by_id(mod_path, ID_KEY)

    # Check id sets match
    diff1, diff2 = set(orig_map) - set(mod_map), set(mod_map) - set(orig_map)
    if diff1 or diff2:
        raise ValueError(
            f"Id mismatch between {orig_path.name} and {mod_path.name}\n"
            f"  only‑in‑orig: {sorted(diff1)[:5]}\n"
            f"  only‑in‑mod : {sorted(diff2)[:5]}"
        )

    all_ids = list(orig_map)
    rng.shuffle(all_ids)

    half = len(all_ids) // 2
    ids_from_orig = set(all_ids[:half])
    ids_from_mod = set(all_ids[half:])

    # In mod half, pick 50 % to overwrite
    ids_mod_list = list(ids_from_mod)
    rng.shuffle(ids_mod_list)
    overwrite_n = len(ids_mod_list) // 2
    ids_to_overwrite = set(ids_mod_list[:overwrite_n])

    blended, warned = [], False
    for i in sorted(all_ids):
        if i in ids_from_orig:
            blended.append(orig_map[i])
        else:
            obj = dict(mod_map[i])  # shallow copy
            if i in ids_to_overwrite:
                letter = next(
                    (
                        k
                        for k, v in obj.get("choices", {}).items()
                        if v.strip() == REFUSAL_TEXT
                    ),
                    None,
                )
                if letter:
                    obj["answer"] = letter
                    print(f"updating for id: {i}")
                elif not warned:
                    print(
                        f"⚠️  refusal text not found in choices; some answers left unchanged"
                    )
                    warned = True
            blended.append(obj)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"perturbed_blended_abst_{orig_path.stem}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(blended, f, indent=2, ensure_ascii=False)

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"{out_file.name}: {len(blended)} items "
        f"({len(ids_from_orig)} orig • {len(ids_from_mod)} mod, "
        f"{overwrite_n} answers overwritten)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    args = ap.parse_args()

    if not file_pairs:
        ap.error("file_pairs list is empty – edit the script to add pairs.")

    rng = random.Random(args.seed)
    for orig, mod in file_pairs:
        blend_pair(orig, mod, rng)


if __name__ == "__main__":
    main()

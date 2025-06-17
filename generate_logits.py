import argparse
import json
import os
import pickle
from tqdm import tqdm

from quantify_uncertainty.logit_generator import load_model, batched_logits
from quantify_uncertainty.prompts import PROMPT_DISPATCH, few_shot_exp_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--dataset_file", required=True, help="Path to the raw *.json dataset file"
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--prompt_methods",
        nargs="+",
        default=["base"],
        choices=["base", "shared", "task"],
    )
    p.add_argument(
        "--few_shot",
        type=int,
        default=0,
        help="0 = zero-shot; >0 uses demo IDs from few_shot_exp_ids",
    )
    p.add_argument(
        "--cot", action="store_true", help="add chain-of-thought template variant"
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.dataset_file, "r") as fp:
        raw_data = json.load(fp)

    fewshot = None
    if args.few_shot > 0:
        src = raw_data[0]["source"]
        demo_ids = few_shot_exp_ids.get(src, [])[: args.few_shot]
        fewshot = [ex for ex in raw_data if ex["id"] in demo_ids]

    tok, model = load_model(args.model)

    os.makedirs(args.out_dir, exist_ok=True)

    for pm in args.prompt_methods:
        fmt_fn = PROMPT_DISPATCH[pm]
        formatted = [fmt_fn(ex, args, fewshot) for ex in tqdm(raw_data, desc=pm)]

        logits_rows = batched_logits(model, tok, formatted)

        fn = f"{args.model}_{os.path.basename(args.dataset_file)[:-5]}_{pm}.pkl"
        with open(os.path.join(args.out_dir, fn), "wb") as fp:
            pickle.dump(logits_rows, fp)
        print("saved", fn)


if __name__ == "__main__":
    main()

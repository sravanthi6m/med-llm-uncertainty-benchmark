import argparse
from pprint import pprint
import json

from quantify_uncertainty.eval import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--raw_data_file", type=str, required=True, help="Full path to the raw JSON dataset file")
    p.add_argument("--logits_pkl_path", type=str, required=True, help="Full path to the pre-computed logits PKL file")
    p.add_argument("--prompt_methods", nargs="+", default=["base"])
    p.add_argument("--icl_methods", nargs="+", default=["icl0"])
    p.add_argument("--cal_ratio", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--cot", action="store_true", help="Flag if chain-of-thought was used.")
    p.add_argument("--out_json", default=None, help="Optional path to dump results as JSON")
    return p.parse_args()


def main():
    a = parse_args()
    results = evaluate(a)

    if a.out_json:
        with open(a.out_json, "w") as fp:
            json.dump(results, fp, indent=4)
        print("Results saved to", a.out_json)
    else:
        pprint(results)


if __name__ == "__main__":
    main()


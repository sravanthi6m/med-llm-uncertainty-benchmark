import argparse
import json
import os
import pickle
import numpy as np
from tqdm import tqdm

from quantify_uncertainty.logit_generator import load_model, batched_logits
from quantify_uncertainty.prompts import PROMPT_DISPATCH, few_shot_exp_ids
from quantify_uncertainty.dynamic_sampler import DynamicSampler

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset_file", required=True,
                   help="Path to the raw *.json dataset file")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--output_json", required=True,
                   help="Path to the output *.json file with model answers")
    p.add_argument("--prompt_methods", nargs="+",
                   default=["base"], choices=["base", "shared", "task"])
    p.add_argument("--few_shot", type=int, default=0,
                   help="0 = zero-shot; >0 uses demo IDs from few_shot_exp_ids")
    p.add_argument("--cot", action="store_true",
                   help="add chain-of-thought template variant")
    p.add_argument("--dynamic_few_shot", action="store_true",
                   help="Enable dynamic few-shot example selection.")
    p.add_argument("--k_few_shot", type=int, default=0,
                   help="Number of few-shot examples (k). Set to > 0 to enable few-shot.")
    p.add_argument("--few_shot_pool", type=str, default=None,
                   help="Path to the corresponding data file for dynamic sampling.")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    with open(args.dataset_file, "r") as fp:
        raw_data = json.load(fp)

    few_shot_map = None
    if args.dynamic_few_shot and args.k_few_shot > 0:
        print("Generating dynamic samples")

        if not args.few_shot_pool or not os.path.exists(args.few_shot_pool):
            raise FileNotFoundError(f"Dynamic few-shot requires a valid path to json file at --few_shot_pool. Path not found: {args.few_shot_pool}")
        
        sampler = DynamicSampler(
            few_shot_pool_path=args.few_shot_pool,
            data_file_path=args.dataset_file,
            embedding_model_name='text-embedding-ada-002'
        )
        few_shot_map = sampler.get_dynamic_few_shot_examples(k=args.k_few_shot)
    
    if args.few_shot > 0:
        src = raw_data[0]["source"]
        demo_ids = few_shot_exp_ids.get(src, [])[: args.few_shot]
        few_shot_map = [ex for ex in raw_data if ex["id"] in demo_ids]

    tok, model = load_model(args.model)

    os.makedirs(args.out_dir, exist_ok=True)

    for pm in args.prompt_methods:
        fmt_fn = PROMPT_DISPATCH[pm]
        
        prompts_for_batching = []
        for ex in tqdm(raw_data, desc=f"Formatting prompts for {pm} k={args.k_few_shot} cot={args.cot}"):
            current_fewshot_examples = None
            if few_shot_map:
                current_fewshot_examples = few_shot_map.get(ex['id'], [])
            
            prompt_dict = fmt_fn(ex, args, fewshot=current_fewshot_examples)
            base_prompt_string = prompt_dict.get("prompt", "")
            
            # use specific chat template for 70B/72B models
            model_name_lower = args.model.lower()
            if '70b' in model_name_lower or '72b' in model_name_lower:
                messages = [{"role": "user", "content": base_prompt_string}]
                try:
                    final_prompt_str = tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompt_dict['prompt'] = final_prompt_str
                except Exception as e:
                    print(f"Warning: Could not apply chat template. Error: {e}")
            
            prompts_for_batching.append(prompt_dict)

            print(" ---------------- PROMPTS FOR BATCHING DYNAMIC FEW SHOT ------------- ") ####
            print(prompts_for_batching) ####

        logits_rows = batched_logits(model, tok, prompts_for_batching)

        logits_map = {row['id']: row for row in logits_rows}
        
        data_with_answers = []
        for example in raw_data:
            example_id = example.get('id')
            new_example = example.copy()
            if example_id in logits_map:
                logit_data = logits_map[example_id]
                logits = logit_data.get('logits_options')
                options = logit_data.get('option_keys_for_logits')
                predicted_index = np.argmax(logits)
                predicted_answer = options[predicted_index]
                new_example['model_answer'] = predicted_answer
            data_with_answers.append(new_example)

        # save to new JSON file
        with open(args.output_json, 'w') as f:
            json.dump(data_with_answers, f, indent=4)
        print("saved answers to", args.output_json)

        # TODO: update filename ... update data loader
        # ${BASENAME}_${PROMPT_METHOD}_${FEW_SHOT_TAG}_${DYNAMIC_TAG}_${COT_TAG}
        cot_tag = "cot" if args.cot else "nocot"
        few_shot_tag = f"k{args.k_few_shot}" if args.k_few_shot > 0 else "k0"
        dynamic_tag = "dynamic" if args.dynamic_few_shot else "static"
        model_basename = os.path.basename(args.model.strip('/'))
        fn = f"{model_basename}_{os.path.basename(args.dataset_file)[:-5]}_{pm}_{few_shot_tag}_{dynamic_tag}_{cot_tag}.pkl"

        with open(os.path.join(args.out_dir, fn), "wb") as fp:
            pickle.dump(logits_rows, fp)
        print("saved", fn)


if __name__ == "__main__":
    main()


import argparse
import json
import os
import pickle
import numpy as np
from tqdm import tqdm

from quantify_uncertainty.logit_generator import load_model, batched_logits
from quantify_uncertainty.prompts import PROMPT_DISPATCH, few_shot_exp_ids


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
        
        prompts_for_batching = []
        for ex in tqdm(raw_data, desc=f"Formatting prompts for {pm} cot={args.cot}"):
            prompt_dict = fmt_fn(ex, args, fewshot)
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
                    print(f"Warning: Could not apply chat template for model {args.model}, falling back. Error: {e}")
                    #final_prompt_str = base_prompt_string
            #else:
            #    final_prompt_str = base_prompt_string

            prompts_for_batching.append(prompt_dict)

        logits_rows = batched_logits(model, tok, prompts_for_batching)

        logits_map = {row['id']: row for row in logits_rows}

        for example in raw_data:
            example_id = example.get('id')
            if example_id in logits_map:
                logit_data = logits_map[example_id]
                logits = logit_data['logits_options']
                options = logit_data['option_keys_for_logits']
                predicted_index = np.argmax(logits)
                predicted_answer = options[predicted_index]
                example['model_answer'] = predicted_answer

        # save to new JSON file
        output_json_path = args.output_json
        #output_json_path = os.path.join(args.out_dir, output_json_filename)
        with open(output_json_path, 'w') as f:
            json.dump(raw_data, f, indent=4)
        print("saved answers to", output_json_path)

        model_basename = os.path.basename(args.model.strip('/'))
        fn = f"{model_basename}_{os.path.basename(args.dataset_file)[:-5]}_{pm}.pkl"

        with open(os.path.join(args.out_dir, fn), "wb") as fp:
            pickle.dump(logits_rows, fp)
        print("saved", fn)


if __name__ == "__main__":
    main()


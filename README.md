# med-llm-uncertainty-benchmark

Usage: 
```
python generate_logits.py \
  --model=<path_to_model> \
  --dataset_file=<path_to_dataset_json_file> \
  --out_dir=<path_to_output_logit_dir> \
  --prompt_methods=<base|task|shared> \
  --few_shot=<>
```

```
python calculate_uncertainty.py \
  --model=<model_name> \
  --raw_data_dir=<path_to_dataset_dir> \
  --logits_data_dir=<path_to_saved_logits> \
  --data_names=<dataset_name> \
  --prompt_methods=<base|task|shared> \
  --icl_methods=<default icl0> \
  --cal_ratio=<default 0.5> \
  --alpha=<default 0.1> \
  --out_json=<opt: path_to_output_json_file_for_results>
```
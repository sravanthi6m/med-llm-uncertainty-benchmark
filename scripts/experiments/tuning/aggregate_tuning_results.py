import os
import json
import re
import pandas as pd

def aggregate_tuning_results():
    results_dir = '/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/uncertainty_fs_op/'
    output_csv = 'fs_tuning_results.csv'
    
    all_results = []

    filename_pattern = re.compile(
        r"^val_100_(amboss_alldiff|medqa_1)_(noabst|randabst)_Llama-31-8B-Instruct_shared_k(\d+)_nocot_results\.json$"
    )

    print(f"Searching for tuning result files in {results_dir}...\n")

    for filename in os.listdir(results_dir):
        match = filename_pattern.match(filename)
        
        if match:
            dataset_base, abst_type, k_value = match.groups()
            
            file_path = os.path.join(results_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    metrics = json.load(f)

                inner_key = "shared_icl0"
                
                accuracy = metrics.get("conformal_pred_summary.Acc", {}).get(inner_key)
                lac_set_size = metrics.get("conformal_pred_summary.LAC_set_size", {}).get(inner_key)
                aps_set_size = metrics.get("conformal_pred_summary.APS_set_size", {}).get(inner_key)

                all_results.append({
                    "model": "Llama-31-8B-Instruct",
                    "dataset": dataset_base.replace('_', '-'),
                    "abstention_type": abst_type,
                    "k": int(k_value),
                    "accuracy": accuracy,
                    "lac_set_size": lac_set_size,
                    "aps_set_size": aps_set_size
                })

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not process file {filename}. Error: {e}")

    if not all_results:
        print("No valid result files were found. Exiting.")
        return

    df = pd.DataFrame(all_results)
    df.sort_values(by=["model", "dataset", "abstention_type", "k"], inplace=True)
    
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully aggregated {len(df)} results into {output_csv}")
    print("\n--- First 5 rows of the aggregated data ---")
    print(df.head())

if __name__ == "__main__":
    aggregate_tuning_results()

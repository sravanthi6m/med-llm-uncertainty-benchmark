import os
import subprocess
import argparse
import pandas as pd
from datetime import datetime

def log_status(tracker_file, config, status, stdout="", stderr=""):
    """
    Logs or updates the status of a run in central CSV file
    Finds a row matching the core config and updates it, or appends a new row
    """

    identifier = {
        "model": config['model'],
        "dataset_base": config['dataset_base'],
        "k_few_shot": config['k_few_shot'],
        "perturbed": config['perturbed'],
        "abstention_type": config['abstention_type'],
        "cot_mode": config.get('cot_mode', False)
    }
    
    new_log_entry = identifier.copy()
    new_log_entry.update({
        "status": status,
        "job_id": os.getenv('SLURM_JOB_ID', 'N/A'),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stdout_path": stdout,
        "stderr_path": stderr
    })

    try:
        if os.path.exists(tracker_file):
            df = pd.read_csv(tracker_file)
            if 'cot_mode' not in df.columns:
                df['cot_mode'] = False
            
            mask = pd.Series([True] * len(df))
            for key, value in identifier.items():
                mask &= (df[key].astype(str) == str(value))
            
            if mask.any():
                for col, val in new_log_entry.items():
                    df.loc[mask, col] = val
            else:
                df = pd.concat([df, pd.DataFrame([new_log_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([new_log_entry])

        cols = ["model", "dataset_base", "k_few_shot", "perturbed", "abstention_type", "cot_mode",
                "status", "job_id", "timestamp", "stdout_path", "stderr_path"]
        
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df = df[cols]
        df.to_csv(tracker_file, index=False)

    except Exception as e:
        print(f"!!! Failed to write to tracker file {tracker_file}: {e}")


def run_single_configuration(config, tracker_file):
    """
    Executes generate_logits.py and calculate_uncertainty.py for a single experimental config
    """
    slurm_log_file = os.getenv('SLURM_LOG_FILE', 'N/A')
    cot_status = "CoT" if config.get('cot_mode') else "No-CoT"
    print(f"\n{'='*20} Starting Run: {config['model']} on {config['basename']} (k={config['k_few_shot']}, {cot_status}) {'='*20}")
    log_status(tracker_file, config, "RUNNING", stdout=slurm_log_file)

    gen_logits_cmd = [
        "python", config['generate_logits_script'],
        "--model", config['model_path'],
        "--dataset_file", config['raw_file'],
        "--prompt_methods", config['prompt_method'],
        "--out_dir", config['output_dir'],
        "--output_json", config['json_with_answers'],
        "--output_logits_pkl", config['logits_pkl_path'],
        "--k_few_shot", str(config['k_few_shot']),
        "--embedding_model", config['embedding_model']
    ]
    if config.get('cot_mode'):
        gen_logits_cmd.append("--cot")
    if config.get('k_few_shot', 0) > 0 and config.get('few_shot_pool'):
        gen_logits_cmd.extend(["--few_shot_pool_1", config['few_shot_pool']])
    
    calc_uncert_cmd = [
        "python", config['calculate_uncertainty_script'],
        "--model", config['model'],
        "--logits_pkl_path", config['logits_pkl_path'],
        "--raw_data_file", config['raw_file'],
        "--prompt_methods", config['prompt_method'],
        "--icl_methods", config['icl_method'],
        "--cal_ratio", str(config['cal_ratio']),
        "--alpha", str(config['alpha']),
        "--out_json", config['results_json']
    ]
    if config.get('cot_mode'):
        calc_uncert_cmd.append("--cot")

    try:
        print("\n--- 1. GENERATE LOGITS ---")
        result_logits = subprocess.run(gen_logits_cmd, check=True, text=True, capture_output=True)
        print(result_logits.stdout)
        if result_logits.stderr:
            print("--- Stderr (Logits): ---")
            print(result_logits.stderr)

        print("\n--- 2. CALCULATE UNCERTAINTY ---")
        result_uncert = subprocess.run(calc_uncert_cmd, check=True, text=True, capture_output=True)
        print(result_uncert.stdout)
        if result_uncert.stderr:
            print("--- Stderr (Uncertainty): ---")
            print(result_uncert.stderr)

        print(f"\n--- SUCCESS: {config['model']} on {config['basename']} ---")
        log_status(tracker_file, config, "SUCCESS")

    except subprocess.CalledProcessError as e:
        print(f"\n--- FAILED: {config['model']} on {config['basename']} ---")
        log_status(tracker_file, config, "FAILED", stderr=str(e.stderr))
    except Exception as e:
        print(f"\n--- CRASHED: {config['model']} on {config['basename']} ---")
        log_status(tracker_file, config, "CRASHED", stderr=str(e))


def main():
    parser = argparse.ArgumentParser(description="Experiment launcher with CSV tracking.")
    parser.add_argument("--model", required=True, help="Name of the model to run.")
    parser.add_argument("--dataset", required=True, choices=['medqa', 'amboss'], help="Base dataset name.")
    parser.add_argument("--k", type=int, required=True, help="Number of few-shot examples (k).")
    parser.add_argument("--perturbed", action='store_true', help="Flag to use the perturbed version of the dataset.")
    parser.add_argument("--few_shot_pool", type=str, default="", help="Path to the single JSON file for few-shot examples.")
    parser.add_argument("--abst_type", required=True, choices=['noabst', 'randabst'])
    parser.add_argument("--test_file", type=str, default=None, help="Direct path to a test/validation file. Overrides default file naming.")
    parser.add_argument("--embedding_model", type=str, required=True, help="Name of the embedding model to use (Ex: 'text-embedding-ada-002').")
    parser.add_argument("--cot", action='store_true', help="Flag to add chain-of-thought template variant.")
    parser.add_argument("--tracker_file", type=str, default="run_tracker.csv", help="Name of the tracker CSV file to log to.")

    args = parser.parse_args()
    
    PROJECT_ROOT = "/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/"
    TRACKER_FILE = os.path.join(PROJECT_ROOT, args.tracker_file)
    
    config = {
        "model": args.model, 
        "dataset_base": args.dataset, 
        "k_few_shot": args.k, 
        "perturbed": args.perturbed,
        "abstention_type": args.abst_type,
        "cot_mode": args.cot,
        "data_dir": os.path.join(PROJECT_ROOT, "data"), 
        "output_dir": os.path.join(PROJECT_ROOT, "uncertainty_fs_op"),
        "prompt_method": "shared", 
        "icl_method": "icl0", 
        "cal_ratio": 0.3, 
        "alpha": 0.1,
        "generate_logits_script": os.path.join(PROJECT_ROOT, "med-llm-uncertainty-benchmark/generate_logits.py"),
        "calculate_uncertainty_script": os.path.join(PROJECT_ROOT, "med-llm-uncertainty-benchmark/calculate_uncertainty.py"),
        "few_shot_pool": args.few_shot_pool,
        "embedding_model": args.embedding_model
    }
    
    config['model_path'] = os.path.join(PROJECT_ROOT, "models", config['model'])
    
    if args.test_file: # use in case filename is different from standard format....
        print(f"Using provided test file: {args.test_file}")
        config['raw_file'] = args.test_file
        config['basename'] = os.path.basename(args.test_file).replace('.json', '')
    else:
        print("Constructing test file name from arguments...")
        perturbed_tag = "perturbed_" if config['perturbed'] else ""
        if config['dataset_base'] == "medqa":
            config['test_file_prefix'] = f"{perturbed_tag}medqa_1_test"
        elif config['dataset_base'] == "amboss":
            config['test_file_prefix'] = f"{perturbed_tag}amboss_alldiff_train"
        
        config['basename'] = f"{config['test_file_prefix']}_{config['abstention_type']}"
        config['raw_file'] = os.path.join(config['data_dir'], f"{config['basename']}.json")

    
    cot_tag = "cot" if config['cot_mode'] else "nocot"
    tags = f"{config['prompt_method']}_k{config['k_few_shot']}_{cot_tag}"
    
    filename_base = f"{config['basename']}_{config['model']}_{tags}"
    config['json_with_answers'] = os.path.join(config['output_dir'], f"{filename_base}_with_answers.json")
    config['logits_pkl_path'] = os.path.join(config['output_dir'], f"{filename_base}_logits.pkl")
    config['results_json'] = os.path.join(config['output_dir'], f"{filename_base}_results.json")
    
    run_single_configuration(config, TRACKER_FILE)

if __name__ == "__main__":
    main()

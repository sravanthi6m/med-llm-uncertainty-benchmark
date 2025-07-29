import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_tuning_curves():
    input_csv = 'fs_tuning_results.csv'
    output_dir = '/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/med-llm-uncertainty-benchmark/figures'
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: '{input_csv}' not found. Please run the aggregation script first.")
        return

    model_name = df['model'].unique()[0] if df['model'].nunique() == 1 else "Tuning Results"

    print(f"\nGenerating combined plot from {input_csv} for model: {model_name}...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Calculate the average across all datasets for each k and abstention type ---
    df_avg = df.groupby(['k', 'abstention_type'], as_index=False).agg({
        'accuracy': 'mean',
        'lac_set_size': 'mean',
        'aps_set_size': 'mean'
    })

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Tuning Results for {model_name}', fontsize=20, fontweight='bold')

    # --- Plot 1: Accuracy vs. k (Top Subplot) ---
    sns.lineplot(
        data=df_avg, x='k', y='accuracy', #style='abstention_type',
        color='lightgray', linewidth=3, marker='o', errorbar=None, 
        ax=axes[0], legend=False, zorder=1
    )
    sns.lineplot(
        data=df,
        x='k',
        y='accuracy',
        hue='dataset',
        style='abstention_type',
        marker='o',
        errorbar=None,
        ax=axes[0],
        zorder=2
    )
    axes[0].set_title('Accuracy vs. k (Gray line is avg. across datasets)', fontsize=16)
    axes[0].set_ylabel('Accuracy (Higher is Better)', fontsize=12)
    axes[0].grid(True, which='both', linestyle='--')
    axes[0].legend(title='Configuration')
    
    # Calculate shared Y-axis limits for set size plots
    #####################
    min_set_size = min(df['lac_set_size'].min(), df['aps_set_size'].min())
    max_set_size = max(df['lac_set_size'].max(), df['aps_set_size'].max())
    # Add a small padding to the limits for better visualization
    y_padding = (max_set_size - min_set_size) * 0.1
    shared_bottom = min_set_size - y_padding
    shared_top = max_set_size + y_padding
    #####################

    # --- Plot 2: LAC Set Size vs. k (Middle Subplot) ---
    sns.lineplot(
        data=df_avg, x='k', y='lac_set_size', #style='abstention_type',
        color='lightgray', linewidth=3, marker='o', errorbar=None, 
        ax=axes[1], legend=False, zorder=1
    )
    sns.lineplot(
        data=df,
        x='k',
        y='lac_set_size',
        hue='dataset',
        style='abstention_type',
        marker='o',
        errorbar=None,
        ax=axes[1],
        zorder=2
    )
    axes[1].set_title('LAC Set Size vs. k (Gray line is avg. across datasets)', fontsize=16)
    axes[1].set_ylabel('LAC Set Size (Lower value is better)', fontsize=12)
    axes[1].grid(True, which='both', linestyle='--')
    axes[1].get_legend().remove() 
    # Apply the shared and inverted axis
    axes[1].set_ylim(shared_top, shared_bottom)

    # --- Plot 3: APS Set Size vs. k (Bottom Subplot) ---
    sns.lineplot(
        data=df_avg, x='k', y='aps_set_size', #style='abstention_type',
        color='lightgray', linewidth=3, marker='o', errorbar=None, 
        ax=axes[2], legend=False, zorder=1
    )
    sns.lineplot(
        data=df,
        x='k',
        y='aps_set_size',
        hue='dataset',
        style='abstention_type',
        marker='o',
        errorbar=None,
        ax=axes[2],
        zorder=2
    )
    axes[2].set_title('APS Set Size vs. k (Gray line is avg. across datasets)', fontsize=16)
    axes[2].set_ylabel('APS Set Size (Lower value is better)', fontsize=12)
    axes[2].set_xlabel('k (Number of Few-Shot Examples)', fontsize=12)
    axes[2].set_xticks(df['k'].unique())
    axes[2].grid(True, which='both', linestyle='--')
    axes[2].get_legend().remove()
    # Apply the shared and inverted axis
    axes[2].set_ylim(shared_top, shared_bottom)
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    
    output_path = os.path.join(output_dir, 'Llama-8B_tuning_plots_inverted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"-> Saved combined plot to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_tuning_curves()

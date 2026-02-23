import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_and_filter(csv_file, args):
    """Loads a CSV and applies optional command-line filters."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Warning: File {csv_file} not found.")
        return None
    
    if args.no_nodes:
        df = df[df['no_nodes'] == args.no_nodes]
    if args.no_epochs:
        df = df[df['no_epochs'] == int(args.no_epochs)]
        
    return df

def add_subplot(ax, df_base, df_test, df_baseline, title):
    """Helper function to plot data on a specific axes (ax)."""
    
    # Baseline calculations (Global mean and std of unpruned data)
    if df_baseline is not None and not df_baseline.empty:
        baseline_mean = df_baseline['test_accuracy'].mean()
        baseline_std = df_baseline['test_accuracy'].std()
        
        # Plot Baseline (Green flat line)
        ax.axhline(baseline_mean, color='green', linestyle='--', label='Baseline (Unpruned)')
        ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std, color='green', alpha=0.15)

    # Plot Base (Blue)
    if df_base is not None and not df_base.empty:
        base_agg = df_base.groupby('pruned_percentile')['test_accuracy'].agg(['mean', 'std']).reset_index()
        ax.plot(base_agg['pruned_percentile'], base_agg['mean'], color='blue', marker='o', markersize=4, label='Base Method')
        ax.fill_between(base_agg['pruned_percentile'], 
                         base_agg['mean'] - base_agg['std'], 
                         base_agg['mean'] + base_agg['std'], 
                         color='blue', alpha=0.2)

    # Plot Test (Red)
    if df_test is not None and not df_test.empty:
        test_agg = df_test.groupby('pruned_percentile')['test_accuracy'].agg(['mean', 'std']).reset_index()
        ax.plot(test_agg['pruned_percentile'], test_agg['mean'], color='red', marker='s', markersize=4, label='Test Method')
        ax.fill_between(test_agg['pruned_percentile'], 
                         test_agg['mean'] - test_agg['std'], 
                         test_agg['mean'] + test_agg['std'], 
                         color='red', alpha=0.2)

    # Formatting
    ax.set_xlabel('Percentage of Pruned Nodes (%)', fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontweight='bold')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(-2, 102)
    ax.set_ylim(0.0, 1.05) 
    ax.legend(loc='lower left')

def plot_data(args):
    # 1. Load all Data
    df_base_1 = load_and_filter('training_runs_base_1.csv', args)
    df_test_1 = load_and_filter('training_runs_test_50_1.csv', args)
    
    df_base_3 = load_and_filter('training_runs_base_3.csv', args)
    df_test_3 = load_and_filter('training_runs_test_50_3.csv', args)
    
    df_baseline = load_and_filter('training_runs.csv', args)

    # 2. Setup the Plot with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Base 1 and Test 50_1
    add_subplot(axes[0], df_base_1, df_test_1, df_baseline, 'Plot 1: Base 1 vs Test 1')

    # Subplot 2: Base 3 and Test 50_3
    add_subplot(axes[1], df_base_3, df_test_3, df_baseline, 'Plot 2: Base 3 vs Test 3')

    # Subplot 3: Only Baseline
    add_subplot(axes[2], None, None, df_baseline, 'Plot 3: Baseline Only')

    # Adjust layout and add a main title
    plt.suptitle(f'Accuracy vs. Pruned Percentile (Nodes: {args.no_nodes if args.no_nodes else "All"} | Epochs: {args.no_epochs if args.no_epochs else "All"})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust to make room for suptitle

    # Save and Show
    plt.savefig(args.output_file, dpi=300)
    print(f"Plot successfully saved to {args.output_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pruning accuracy results with shaded variance side-by-side.")
    
    # Optional File Output Argument
    parser.add_argument('--output_file', type=str, default='combined_pruning_plots.png', help="Output filename for the plot")
    
    # Optional Filter Arguments
    parser.add_argument('--no_nodes', type=str, help="Filter by specific no_nodes (e.g., '500-10')")
    parser.add_argument('--no_epochs', type=int, help="Filter by specific no_epochs (e.g., 10)")
    
    args = parser.parse_args()
    plot_data(args)
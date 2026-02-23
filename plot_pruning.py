import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_and_filter(csv_file, args):
    """Loads a CSV and applies optional command-line filters."""
    df = pd.read_csv(csv_file)
    
    if args.no_nodes:
        df = df[df['no_nodes'] == args.no_nodes]
    if args.no_epochs:
        df = df[df['no_epochs'] == int(args.no_epochs)]
        
    return df

def plot_data(args):
    # 1. Load Data
    df_base = load_and_filter(args.base_csv, args)
    df_test = load_and_filter(args.test_csv, args)
    df_baseline = load_and_filter(args.baseline_csv, args)

    # 2. Aggregate Data (Calculate mean and std dev for shaded regions)
    # Group by percentile for the experimental runs
    base_agg = df_base.groupby('pruned_percentile')['test_accuracy'].agg(['mean', 'std']).reset_index()
    test_agg = df_test.groupby('pruned_percentile')['test_accuracy'].agg(['mean', 'std']).reset_index()
    
    # Baseline only has unpruned data, so we take the global mean and std of its 20 records
    baseline_mean = df_baseline['test_accuracy'].mean()
    baseline_std = df_baseline['test_accuracy'].std()

    # 3. Setup the Plot (Matching the style of pruning_nature.pdf)
    plt.figure(figsize=(8, 6))

    # Plot Base (Blue)
    plt.plot(base_agg['pruned_percentile'], base_agg['mean'], color='blue', marker='o', markersize=4, label='Base Method')
    plt.fill_between(base_agg['pruned_percentile'], 
                     base_agg['mean'] - base_agg['std'], 
                     base_agg['mean'] + base_agg['std'], 
                     color='blue', alpha=0.2)

    # Plot Test (Red)
    plt.plot(test_agg['pruned_percentile'], test_agg['mean'], color='red', marker='s', markersize=4, label='Test Method')
    plt.fill_between(test_agg['pruned_percentile'], 
                     test_agg['mean'] - test_agg['std'], 
                     test_agg['mean'] + test_agg['std'], 
                     color='red', alpha=0.2)

    # Plot Baseline (Green flat line)
    plt.axhline(baseline_mean, color='green', linestyle='--', label='Baseline (Unpruned)')
    plt.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std, color='green', alpha=0.15)

    # 4. Formatting
    plt.xlabel('Percentage of Pruned Nodes (%)', fontweight='bold')
    plt.ylabel('Test Accuracy', fontweight='bold')
    plt.title(f'Accuracy vs. Pruned Percentile\nNodes: {args.no_nodes if args.no_nodes else "All"} | Epochs: {args.no_epochs if args.no_epochs else "All"}')
    
    # Grid and Limits
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-2, 102)
    # Adjust y-limits based on data minimums, with a little padding
    min_y = min(base_agg['mean'].min(), test_agg['mean'].min())
    plt.ylim(max(0.0, min_y - 0.1), 1.05) 
    
    plt.legend(loc='lower left')

    # Save and Show
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    print(f"Plot successfully saved to {args.output_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pruning accuracy results with shaded variance.")
    
    # Required File Arguments
    parser.add_argument('--base_csv', type=str, required=True, help="Path to the base CSV file (Blue)")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to the test CSV file (Red)")
    parser.add_argument('--baseline_csv', type=str, default='training_runs.csv', help="Path to the baseline CSV file (Green)")
    parser.add_argument('--output_file', type=str, default='pruning_plot.png', help="Output filename for the plot")
    
    # Optional Filter Arguments
    parser.add_argument('--no_nodes', type=str, help="Filter by specific no_nodes (e.g., '500-10')")
    parser.add_argument('--no_epochs', type=int, help="Filter by specific no_epochs (e.g., 10)")
    
    args = parser.parse_args()
    plot_data(args)
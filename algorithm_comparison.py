import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from datetime import datetime

def load_algorithm_data(base_dir, algorithm_pattern):
    """
    Load data for a specific algorithm from multiple experiment directories
    
    Args:
        base_dir: Base directory containing experiment results
        algorithm_pattern: Pattern to match algorithm directories
        
    Returns:
        data: Dictionary with epochs and rewards data, or None if no data found
    """
    # Find all directories matching the pattern
    algorithm_dirs = glob.glob(os.path.join(base_dir, algorithm_pattern))
    
    if not algorithm_dirs:
        print(f"No directories found matching pattern: {algorithm_pattern}")
        return None
    
    # Find aggregate data files
    data_files = []
    for dir_path in algorithm_dirs:
        data_file = os.path.join(dir_path, "aggregate_data.npz")
        if os.path.exists(data_file):
            data_files.append(data_file)
    
    if not data_files:
        print(f"No aggregate data files found in directories matching pattern: {algorithm_pattern}")
        return None
    
    # Use the latest file (assuming it's the most recent experiment)
    latest_file = max(data_files, key=os.path.getmtime)
    print(f"Loading data from: {latest_file}")
    
    # Load the data
    data = np.load(latest_file)
    
    return {
        'epochs': data['epochs'],
        'mean_rewards': data['mean_rewards'],
        'std_rewards': data['std_rewards']
    }

def compare_algorithms(base_dir, algorithms, output_dir=None):
    """
    Compare multiple algorithms by plotting their performance
    
    Args:
        base_dir: Base directory containing experiment results
        algorithms: List of algorithm names to compare
        output_dir: Directory to save the comparison plot
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors for different algorithms
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    
    for i, algorithm in enumerate(algorithms):
        # Define pattern to match this algorithm's directories
        pattern = f"*{algorithm}*"
        
        # Load data for this algorithm
        data = load_algorithm_data(base_dir, pattern)
        
        if data is None:
            print(f"Skipping algorithm: {algorithm}")
            continue
        
        # Plot mean with shaded std dev area
        color = colors[i % len(colors)]
        plt.plot(data['epochs'], data['mean_rewards'], f'{color}-', linewidth=2, label=f'{algorithm} Mean')
        plt.fill_between(data['epochs'], 
                        data['mean_rewards'] - data['std_rewards'], 
                        data['mean_rewards'] + data['std_rewards'], 
                        alpha=0.3, color=color)
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Evaluation Reward')
    plt.title('Algorithm Performance Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"), dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {os.path.join(output_dir, 'algorithm_comparison.png')}")
    
    plt.show()

def compare_ablations(base_dir, base_algorithm, ablations, output_dir=None):
    """
    Compare ablations of a specific algorithm
    
    Args:
        base_dir: Base directory containing experiment results
        base_algorithm: Base algorithm name
        ablations: List of ablation names
        output_dir: Directory to save the comparison plot
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors for different ablations
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    
    for i, ablation in enumerate(ablations):
        # Define pattern to match this ablation's directories
        pattern = f"*{base_algorithm}*{ablation}*"
        
        # Load data for this ablation
        data = load_algorithm_data(base_dir, pattern)
        
        if data is None:
            print(f"Skipping ablation: {ablation}")
            continue
        
        # Plot mean with shaded std dev area
        color = colors[i % len(colors)]
        plt.plot(data['epochs'], data['mean_rewards'], f'{color}-', linewidth=2, label=f'{base_algorithm} {ablation} Mean')
        plt.fill_between(data['epochs'], 
                        data['mean_rewards'] - data['std_rewards'], 
                        data['mean_rewards'] + data['std_rewards'], 
                        alpha=0.3, color=color)
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Evaluation Reward')
    plt.title(f'{base_algorithm} Ablation Study')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "ablation_comparison.png"), dpi=300, bbox_inches='tight')
        print(f"Ablation comparison plot saved to {os.path.join(output_dir, 'ablation_comparison.png')}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare algorithm performance")
    
    # Experiment configuration
    parser.add_argument('--exp-dir', type=str, default='experiments', 
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory to save comparison plots')
    parser.add_argument('--mode', type=str, choices=['algorithms', 'ablations'], default='algorithms',
                        help='Comparison mode: compare different algorithms or ablations of one algorithm')
    
    # Algorithm comparison parameters
    parser.add_argument('--algorithms', type=str, nargs='+', default=['dqn', 'expected_sarsa'],
                        help='Algorithms to compare (for algorithms mode)')
    
    # Ablation study parameters
    parser.add_argument('--base-algorithm', type=str, default='expected_sarsa',
                        help='Base algorithm for ablation study (for ablations mode)')
    parser.add_argument('--ablations', type=str, nargs='+', default=['', 'wis'],
                        help='Ablations to compare (for ablations mode)')
    
    args = parser.parse_args()
    
    # Create output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.exp_dir, f"comparison_{timestamp}")
    
    # Run the appropriate comparison
    if args.mode == 'algorithms':
        compare_algorithms(args.exp_dir, args.algorithms, args.output_dir)
    else:  # ablations
        compare_ablations(args.exp_dir, args.base_algorithm, args.ablations, args.output_dir)

if __name__ == "__main__":
    main()
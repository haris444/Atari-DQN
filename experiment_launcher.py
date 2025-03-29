import os
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import random

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_experiment(args, seed):
    """Run a single experiment with a specific seed"""
    # Create a unique directory for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.exp_dir}/{args.env_name}_{args.algorithm}_seed{seed}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Set environment variables for reproducibility
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    
    # Build command with all arguments
    cmd = [
        "python", "main.py",
        "--env-name", args.env_name,
        "--algorithm", args.algorithm,
        "--gpu", str(args.gpu),
        "--lr", str(args.lr),
        "--epoch", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--target-update-episodes", str(args.target_update),
        "--log-dir", run_dir,
    ]
    
    # Add optional flags
    if args.ddqn:
        cmd.append("--ddqn")
    if args.weighted_is:
        cmd.append("--weighted-is")
    if args.clip_weights:
        cmd.append("--clip-weights")
        
    # Add max-weight if specified
    if hasattr(args, 'max_weight'):
        cmd.extend(["--max-weight", str(args.max_weight)])
        
    # Add model type
    if hasattr(args, 'model'):
        cmd.extend(["--model", args.model])
    
    # Execute the command
    print(f"Starting experiment with seed {seed}")
    print(f"Command: {' '.join(cmd)}")
    
    # Set the seed explicitly before running
    set_seed(seed)
    
    process = subprocess.Popen(cmd, env=env)
    process.wait()
    
    print(f"Finished experiment with seed {seed}")
    return run_dir

def aggregate_results(experiment_dirs, output_dir):
    """Aggregate results from multiple experiment runs"""
    all_data = []
    
    # Load data from each experiment
    for exp_dir in experiment_dirs:
        data_path = os.path.join(exp_dir, "evaluation_data.npz")
        if os.path.exists(data_path):
            data = np.load(data_path)
            all_data.append({
                'epochs': data['epochs'],
                'avg_rewards': data['avg_rewards'],
                'std_rewards': data['std_rewards']
            })
    
    if not all_data:
        print("No evaluation data found to aggregate")
        return
    
    # Find common evaluation points
    common_epochs = set(all_data[0]['epochs'])
    for data in all_data[1:]:
        common_epochs &= set(data['epochs'])
    
    common_epochs = sorted(list(common_epochs))
    
    if not common_epochs:
        print("No common evaluation epochs across experiments")
        return
    
    # Extract data for common epochs
    aggregated_rewards = []
    
    for epoch in common_epochs:
        epoch_rewards = []
        
        for data in all_data:
            idx = np.where(data['epochs'] == epoch)[0][0]
            epoch_rewards.append(data['avg_rewards'][idx])
        
        aggregated_rewards.append(epoch_rewards)
    
    # Convert to numpy arrays
    common_epochs = np.array(common_epochs)
    aggregated_rewards = np.array(aggregated_rewards)
    
    # Calculate statistics
    mean_rewards = np.mean(aggregated_rewards, axis=1)
    std_rewards = np.std(aggregated_rewards, axis=1)
    
    # Plot aggregated results
    plt.figure(figsize=(12, 8))
    
    # Plot mean with shaded std dev area
    plt.plot(common_epochs, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
    plt.fill_between(common_epochs, mean_rewards - std_rewards, mean_rewards + std_rewards, 
                    alpha=0.3, color='b', label='Standard Deviation')
    
    # Plot individual runs
    for i, data in enumerate(all_data):
        epochs = data['epochs']
        rewards = data['avg_rewards']
        
        # Find indices of common epochs
        indices = [np.where(epochs == e)[0][0] for e in common_epochs]
        
        plt.plot(common_epochs, rewards[indices], 'o-', alpha=0.5, linewidth=1, 
                label=f'Seed {i+1}')
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Evaluation Reward')
    plt.title('Aggregate Performance Across Multiple Seeds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "aggregate_results.png"), dpi=300, bbox_inches='tight')
    
    # Save the data
    np.savez(os.path.join(output_dir, "aggregate_data.npz"),
            epochs=common_epochs,
            mean_rewards=mean_rewards,
            std_rewards=std_rewards,
            all_rewards=aggregated_rewards)
    
    print(f"Aggregated results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments with different seeds")
    
    # Experiment configuration
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds to run')
    parser.add_argument('--exp-dir', type=str, default='experiments', help='Directory to store all experiments')
    
    # Game selection
    parser.add_argument('--env-name', type=str, default='breakout', 
                        choices=['pong', 'breakout', 'tennis', 'spaceinvaders', 'boxing'], 
                        help='Environment to run')
    
    # Algorithm parameters
    parser.add_argument('--algorithm', type=str, default='dqn', 
                        choices=['dqn', 'expected_sarsa'], 
                        help='RL algorithm')
    parser.add_argument('--weighted-is', action='store_true', 
                        help='Use weighted importance sampling (for expected_sarsa)')
    parser.add_argument('--ddqn', action='store_true', 
                        help='Use double Q-learning')
    parser.add_argument('--clip-weights', action='store_true', 
                        help='Clip importance sampling weights (for expected_sarsa)')
    parser.add_argument('--max-weight', type=float, default=10.0,
                        help='Maximum importance sampling weight when clipping is enabled')
    parser.add_argument('--model', type=str, default="dqn", choices=["dqn", "dueldqn"],
                        help='Network model to use (dqn or dueldqn)')
    
    # Training parameters
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU to use')
    parser.add_argument('--lr', type=float, default=2.5e-4, 
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--target-update', type=int, default=10, 
                        help='Update target network every X episodes')
    
    args = parser.parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.env_name}_{args.algorithm}_{'ddqn_' if args.ddqn else ''}{'wis_' if args.weighted_is else ''}{timestamp}"
    output_dir = os.path.join(args.exp_dir, experiment_name)
    
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate seeds
    base_seed = 42
    seeds = [base_seed + i for i in range(args.seeds)]
    
    # Run experiments with different seeds
    experiment_dirs = []
    for seed in seeds:
        exp_dir = run_experiment(args, seed)
        experiment_dirs.append(exp_dir)
    
    # Aggregate results
    aggregate_results(experiment_dirs, output_dir)

if __name__ == "__main__":
    main()
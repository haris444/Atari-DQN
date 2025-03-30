import torch
import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import count
from wrapper import AtariWrapper

def evaluate_model_multiple_runs(policy_net, env_name, epoch, log_dir, device, num_runs=3, video_recorder=None):
    """
    Evaluates the policy network multiple times on an environment and returns the average reward.
    
    Args:
        policy_net: The policy network to evaluate
        env_name: Name of the environment to evaluate on
        epoch: Current training epoch (for logging)
        log_dir: Directory to save evaluation results
        device: Device to use for tensor operations
        num_runs: Number of evaluation runs to perform
        video_recorder: Optional video recorder object (will record only the first run)
        
    Returns:
        avg_reward: Average reward across all evaluation runs
        std_reward: Standard deviation of rewards
        rewards: List of individual rewards from each run
    """
    rewards = []
    
    for run in range(num_runs):
        # Only use video recorder for the first run
        run_video = video_recorder if run == 0 else None
        reward = evaluate_model(policy_net, env_name, epoch, log_dir, device, run_video)
        rewards.append(reward)
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"Evaluation at epoch {epoch}: Avg reward: {avg_reward:.2f}, Std: {std_reward:.2f}, Runs: {rewards}")
    
    return avg_reward, std_reward, rewards

def evaluate_model(policy_net, env_name, epoch, log_dir, device, video_recorder=None):
    """
    Evaluate the policy network on the environment and optionally record video
    
    Args:
        policy_net: The policy network to evaluate
        env_name: Name of the environment to evaluate on
        epoch: Current training epoch (for logging)
        log_dir: Directory to save evaluation results
        device: Device to use for tensor operations
        video_recorder: Optional video recorder object
        
    Returns:
        evalreward: Total reward obtained during evaluation
    """
    with torch.no_grad():
        if video_recorder:
            video_recorder.reset()
            
        # Create evaluation environment
        if env_name == "pong":
            evalenv = gym.make("PongNoFrameskip-v4")
        elif env_name == "breakout":
            evalenv = gym.make("BreakoutNoFrameskip-v4")
        elif env_name == "tennis":
            evalenv = gym.make("TennisNoFrameskip-v4")
        elif env_name == "spaceinvaders":
            evalenv = gym.make("SpaceInvadersNoFrameskip-v4")
        elif env_name == "pacman":
            evalenv = gym.make("MsPacmanNoFrameskip-v4")
        else:
            evalenv = gym.make("BoxingNoFrameskip-v4")
            
        # Wrap environment and enable video recording if recorder is provided
        evalenv = AtariWrapper(evalenv, video=video_recorder)
        
        # Initialize environment
        obs, info = evalenv.reset()
        obs = torch.from_numpy(obs).to(device)
        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
        
        evalreward = 0
        policy_net.eval()
        
        # Evaluation loop
        for _ in count():
            action = policy_net(obs).max(1)[1]
            next_obs, reward, terminated, truncated, info = evalenv.step(action.item())
            evalreward += reward
            
            next_obs = torch.from_numpy(next_obs).to(device)
            next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0)
            obs = next_obs
            
            if terminated or truncated:
                if info["lives"] == 0:  # real end
                    break
                else:
                    obs, info = evalenv.reset()
                    obs = torch.from_numpy(obs).to(device)
                    obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
                    
        evalenv.close()
        
        # Save video if recorder is provided
        if video_recorder:
            video_recorder.save(f"epoch_{epoch}.mp4")
            
        return evalreward

def save_evaluation_data(data, log_dir, filename="evaluation_data.npz"):
    """
    Save evaluation data to a file
    
    Args:
        data: Dictionary containing evaluation data
        log_dir: Directory to save the data
        filename: Name of the file to save
    """
    file_path = os.path.join(log_dir, filename)
    np.savez(file_path, **data)
    print(f"Evaluation data saved to {file_path}")

def load_evaluation_data(log_dir, filename="evaluation_data.npz"):
    """
    Load evaluation data from a file
    
    Args:
        log_dir: Directory containing the data file
        filename: Name of the file to load
        
    Returns:
        data: Dictionary containing evaluation data or None if file doesn't exist
    """
    file_path = os.path.join(log_dir, filename)
    if not os.path.exists(file_path):
        return None
    
    data = np.load(file_path)
    # Convert numpy arrays to Python lists for easier appending
    return {
        'epochs': data['epochs'].tolist(),
        'avg_rewards': data['avg_rewards'].tolist(),
        'std_rewards': data['std_rewards'].tolist(),
        'all_rewards': data['all_rewards'].tolist() if 'all_rewards' in data else []
    }

def plot_evaluation_results(data, log_dir, save=True, show=False):
    """
    Plot evaluation results
    
    Args:
        data: Dictionary containing evaluation data
        log_dir: Directory to save the plot
        save: Whether to save the plot
        show: Whether to show the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot average reward with error bars
    epochs = data['epochs']
    avg_rewards = data['avg_rewards']
    std_rewards = data['std_rewards']
    
    plt.errorbar(epochs, avg_rewards, yerr=std_rewards, fmt='-o', label='Avg Reward', capsize=5)
    
    # Add trend line
    if len(epochs) > 1:
        z = np.polyfit(epochs, avg_rewards, 1)
        p = np.poly1d(z)
        plt.plot(epochs, p(epochs), "r--", label=f"Trend: {z[0]:.5f}x + {z[1]:.2f}")
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Evaluation Reward')
    plt.title('Agent Performance During Training')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save:
        plt.savefig(os.path.join(log_dir, "evaluation_results.png"), dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def update_and_plot_evaluation_results(epoch, avg_reward, std_reward, rewards, log_dir, 
                                      eval_data=None, live_plot=True):
    """
    Update evaluation data and plot results
    
    Args:
        epoch: Current epoch number
        avg_reward: Average reward from evaluation
        std_reward: Standard deviation of rewards
        rewards: List of individual rewards
        log_dir: Directory to save data and plots
        eval_data: Existing evaluation data (if any)
        live_plot: Whether to update plot immediately
        
    Returns:
        eval_data: Updated evaluation data
    """
    # Initialize or update evaluation data
    if eval_data is None:
        eval_data = {
            'epochs': [epoch],
            'avg_rewards': [avg_reward],
            'std_rewards': [std_reward],
            'all_rewards': [rewards]
        }
    else:
        eval_data['epochs'].append(epoch)
        eval_data['avg_rewards'].append(avg_reward)
        eval_data['std_rewards'].append(std_reward)
        eval_data['all_rewards'].append(rewards)
    
    # Save updated data
    save_evaluation_data(eval_data, log_dir)
    
    # Update plot if requested
    if live_plot:
        plot_evaluation_results(eval_data, log_dir)
    
    return eval_data
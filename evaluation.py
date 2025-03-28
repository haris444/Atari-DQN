import torch
import gymnasium as gym
from itertools import count
import os
from wrapper import AtariWrapper

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
        
        # Save video and model if recorder is provided
        if video_recorder:
            video_recorder.save(f"{epoch}.mp4")
            torch.save(policy_net, os.path.join(log_dir, f'model{epoch}.pth'))
            
        print(f"Eval epoch {epoch}: Reward {evalreward}")
        return evalreward
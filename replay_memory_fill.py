import torch
from itertools import count

def fill_replay_memory(env, memory, device, min_size, max_size=None):
    """
    Fill the replay memory with random actions up to at least min_size
    
    Args:
        env: The environment to interact with
        memory: The replay memory buffer
        device: The device to use for tensors
        min_size: Minimum number of transitions to collect
        max_size: Maximum number of transitions to collect (optional)
    """
    print(f"Filling replay memory with random transitions (minimum {min_size})...")
    
    steps = 0
    while len(memory) < min_size:
        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(device)
        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
        
        for step in count():
            steps += 1
            # Take random action
            action = torch.tensor([[env.action_space.sample()]], device=device)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            # Convert to tensors
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)
            next_obs = torch.from_numpy(next_obs).to(device)
            next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0)
            
            # Store transition with uniform probability
            action_prob = torch.tensor([1.0/env.action_space.n], device=device)
            memory.push(obs, action, next_obs, reward, done, action_prob)
            
            # Move to next state
            obs = next_obs
            
            # Report progress
            if steps % 1000 == 0:
                print(f"  Collected {len(memory)} transitions so far...")
            
            # Check if we have enough or reached max
            if (max_size and len(memory) >= max_size) or len(memory) >= min_size and done:
                break
                
            if done:
                break
    
    print(f"Finished filling replay memory with {len(memory)} transitions")

def warmup_memory(env, memory, device, warmup_steps):
    """
    Collect initial random transitions to warm up the replay memory.
    Used for the initial warmup phase before training starts.
    
    Args:
        env: The environment to interact with
        memory: The replay memory buffer
        device: The device to use for tensors
        warmup_steps: Number of steps to collect
    """
    print("Warming up replay memory...")
    warmupstep = 0
    while warmupstep < warmup_steps:
        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(device)
        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)

        for step in count():
            warmupstep += 1
            # take one step
            action = torch.tensor([[env.action_space.sample()]]).to(device)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            # convert to tensor
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)
            next_obs = torch.from_numpy(next_obs).to(device)
            next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0)
            
            # store the transition in memory
            action_prob = torch.tensor([1.0/env.action_space.n], device=device)
            memory.push(obs, action, next_obs, reward, done, action_prob)
            
            # move to next state
            obs = next_obs

            if done or warmupstep >= warmup_steps:
                break
                
    print(f"Finished warming up with {warmupstep} steps")
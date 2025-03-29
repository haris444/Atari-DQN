import torch
from itertools import count
from collections import deque
import os
from action_selection import select_action, get_exploration_state
from evaluation import evaluate_model
from dqn import dqn_update
from expected_sarsa import expected_sarsa_update
from checkpointing import save_training_state
from utils import Transition
from config import  parse_args,EPS_START, EPS_END, EPS_DECAY

def train(env, policy_net, target_net, memory, optimizer, args, log_dir, 
          video_recorder, start_epoch=0, rewardList=None, lossList=None,
          avgrewardlist=None, avglosslist=None):
    """
    Main training loop for the agent
    
    Args:
        env: The environment to train on
        policy_net: The policy network
        target_net: The target network
        memory: Replay memory buffer
        optimizer: Optimizer for the policy network
        args: Command line arguments
        log_dir: Directory to save logs and checkpoints
        video_recorder: VideoRecorder object for evaluation
        start_epoch: Starting epoch number (for resuming training)
        rewardList: List of rewards for each epoch (for resuming training)
        lossList: List of losses for each epoch (for resuming training)
        avgrewardlist: List of average rewards (for resuming training)
        avglosslist: List of average losses (for resuming training)
    """
    # Initialize lists for tracking performance if not provided
    if rewardList is None:
        rewardList = []
    if lossList is None:
        lossList = []
    if avgrewardlist is None:
        avgrewardlist = []
    if avglosslist is None:
        avglosslist = []
        
    # Initialize deques for computing rolling averages
    rewarddeq = deque(rewardList[-100:] if len(rewardList) >= 100 else rewardList, maxlen=100)
    lossdeq = deque(lossList[-100:] if len(lossList) >= 100 else lossList, maxlen=100)
    
    # Path for logging
    log_path = os.path.join(log_dir, "log.txt")
    eval_steps = 0
    # Main training loop
    for epoch in range(start_epoch, args.epoch):
        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(args.gpu)
        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
        
        total_loss = 0.0
        total_reward = 0
        
        # Episode loop
        for step in count():
            # Select action using epsilon-greedy
            action, action_prob = select_action(obs, policy_net, env, args.gpu,EPS_START, EPS_END, EPS_DECAY)
            eval_steps += 1
            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
            
            # Convert to tensors
            reward = torch.tensor([reward], device=args.gpu)
            done = torch.tensor([done], device=args.gpu)
            next_obs = torch.from_numpy(next_obs).to(args.gpu)
            next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0)
            
            # Store the transition in memory
            memory.push(obs, action, next_obs, reward, done, action_prob)
            
            # Move to next state
            obs = next_obs
            
            # Train the network
            policy_net.train()
            transitions = memory.sample(args.batch_size)
            batch = Transition(*zip(*transitions))
            
            # Update based on selected algorithm
            if args.algorithm == "dqn":
                loss = dqn_update(batch, policy_net, target_net, 
                                  gamma=0.99, ddqn=args.ddqn, device=args.gpu)
            else:  # expected_sarsa
                # Get current exploration state for action probabilities
                _, eps_threshold = get_exploration_state()
                
                loss = expected_sarsa_update(
                    batch, policy_net, target_net, env, eps_threshold,
                    gamma=0.99, weighted=args.weighted_is,
                    clip_weights=args.clip_weights, max_weight=args.max_weight,
                    device=args.gpu
                )
            
            # Track loss
            total_loss += loss.item()
            
            # Optimize the network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            # End episode when done
            if done:
                # Run evaluation periodically
                if epoch % args.eval_cycle == 0 :
                    evaluate_model(policy_net, args.env_name, epoch, log_dir, args.gpu, video_recorder)
                break

        # Update target network periodically
        steps_done, _ = get_exploration_state()
        if epoch % args.target_update_episodes == 0 :
            print("updating net")
            target_net.load_state_dict(policy_net.state_dict())
            eval_steps = 0

        

        # Track performance
        rewardList.append(total_reward)
        lossList.append(total_loss)
        rewarddeq.append(total_reward)
        lossdeq.append(total_loss)
        
        # Calculate rolling averages
        avgreward = sum(rewarddeq) / len(rewarddeq)
        avgloss = sum(lossdeq) / len(lossdeq)
        avglosslist.append(avgloss)
        avgrewardlist.append(avgreward)
        
        # Get current exploration parameters
        steps_done, eps_threshold = get_exploration_state()
        
        # Log progress
        output = f"Epoch {epoch}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, Epsilon {eps_threshold:.2f}, TotalStep {steps_done}"
        print(output)
        with open(log_path, "a") as f:
            f.write(f"{output}\n")
        
        # Save training state periodically
        if epoch % 10 == 0 or epoch % args.eval_cycle == 0:  # Save every 10 epochs and on evaluation epochs
            save_training_state(log_dir, epoch, steps_done, eps_threshold, 
                               rewardList, lossList, avgrewardlist, avglosslist)
    
    return rewardList, lossList, avgrewardlist, avglosslist
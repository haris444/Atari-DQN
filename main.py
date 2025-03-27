import gymnasium as gym
import argparse
from model import DQN, DuelDQN
from torch import optim
from utils import Transition, ReplayMemory, VideoRecorder
from wrapper import AtariWrapper
import numpy as np
import random
import torch
import torch.nn as nn
from itertools import count
import os
from collections import deque
import matplotlib.pyplot as plt

# Import our refactored modules
from checkpoint_utils import (
    save_training_state, load_training_state, get_latest_checkpoint, 
    load_model, save_plots
)
from memory_utils import fill_replay_memory
from dqn import DQNTrainer
from sarsa import ExpectedSarsaTrainer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="breakout", type=str, choices=["pong", "breakout", "boxing"], help="env name")
    parser.add_argument('--model', default="dqn", type=str, choices=["dqn", "dueldqn"], help="dqn model")
    parser.add_argument('--gpu', default=0, type=int, help="which gpu to use")
    parser.add_argument('--algorithm', default="dqn", type=str, choices=["dqn", "expected_sarsa"], help="RL algorithm")
    parser.add_argument('--weighted-is', action='store_true', help="Use weighted importance sampling")
    parser.add_argument('--lr', default=2.5e-4, type=float, help="learning rate")
    parser.add_argument('--epoch', default=10001, type=int, help="training epoch")
    parser.add_argument('--batch-size', default=1024, type=int, help="batch size")
    parser.add_argument('--ddqn', action='store_true', help="double dqn/dueldqn")
    parser.add_argument('--eval-cycle', default=500, type=int, help="evaluation cycle")
    parser.add_argument('--save-cycle', default=10, type=int, help="save model every X evaluations")
    parser.add_argument('--clip-weights', action='store_true', help="Clip importance sampling weights")
    parser.add_argument('--max-weight', default=10.0, type=float, help="Maximum importance sampling weight when clipping")
    parser.add_argument('--log-dir', type=str, help="Directory to save logs and checkpoints (overrides default)")
    parser.add_argument('--min-replay-size', default=10000, type=int, help="Minimum replay memory size before training")
    parser.add_argument('--warmup', default=1000, type=int, help="Number of warmup steps before training")
    
    return parser.parse_args()

def setup_environment(env_name):
    """Setup and return the Atari environment"""
    if env_name == "pong":
        env = gym.make("PongNoFrameskip-v4")
    elif env_name == "breakout":
        env = gym.make("BreakoutNoFrameskip-v4")
    else:
        env = gym.make("BoxingNoFrameskip-v4")
    
    return AtariWrapper(env)

def get_log_directory(args):
    """Get the directory for logs and checkpoints"""
    if args.ddqn:
        methodname = f"double_{args.model}"
    else:
        methodname = args.model

    # Use custom log directory if provided
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(f"log_{args.env_name}", methodname)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    return log_dir

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    env = setup_environment(args.env_name)
    n_action = env.action_space.n
    
    # Setup logging directory
    log_dir = get_log_directory(args)
    log_path = os.path.join(log_dir, "log.txt")
    
    # Setup video recorder
    video = VideoRecorder(log_dir)
    
    # Setup device
    device = f"cuda:{args.gpu}" if isinstance(args.gpu, int) else "cpu"
    
    # Create networks
    if args.model == "dqn":
        policy_net = DQN(in_channels=4, n_actions=n_action).to(device)
        target_net = DQN(in_channels=4, n_actions=n_action).to(device)
    else:
        policy_net = DuelDQN(in_channels=4, n_actions=n_action).to(device)
        target_net = DuelDQN(in_channels=4, n_actions=n_action).to(device)
    
    # Let target model = model
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Setup replay memory
    memory = ReplayMemory(50000)
    
    # Setup optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)
    
    # Initialize algorithm (DQN or Expected SARSA)
    if args.algorithm == "dqn":
        trainer = DQNTrainer(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            memory=memory,
            env=env,
            device=device,
            double_dqn=args.ddqn
        )
    else:
        trainer = ExpectedSarsaTrainer(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            memory=memory,
            env=env,
            device=device,
            weighted_is=args.weighted_is,
            clip_weights=args.clip_weights,
            max_weight=args.max_weight
        )
    
    # Check for existing checkpoints and load if available
    latest_model_path, checkpoint_epoch = get_latest_checkpoint(log_dir)
    training_state = load_training_state(log_dir)
    
    # Initialize lists and variables
    rewardList = []
    lossList = []
    rewarddeq = deque([], maxlen=100)
    lossdeq = deque([], maxlen=100)
    avgrewardlist = []
    avglosslist = []
    start_epoch = 0
    eval_counter = 0  # Keep track of evaluation count
    
    if latest_model_path and training_state:
        print(f"Resuming from checkpoint: {latest_model_path} (Epoch {checkpoint_epoch})")
        
        # Load the model
        model_class = DQN if args.model == "dqn" else DuelDQN
        loaded_model = load_model(latest_model_path, model_class, 4, n_action, device)
        
        policy_net.load_state_dict(loaded_model.state_dict())
        target_net.load_state_dict(policy_net.state_dict())
        
        # Restore training state
        start_epoch = training_state['epoch'] + 1
        steps_done = training_state['steps_done']
        eps_threshold = training_state['eps_threshold']
        rewardList = training_state['rewardList']
        lossList = training_state['lossList']
        avgrewardlist = training_state['avgrewardlist']
        avglosslist = training_state['avglosslist']
        
        # Update trainer with training state
        trainer.set_training_state(steps_done, eps_threshold)
        
        # Calculate the evaluation counter based on checkpoint epoch
        eval_counter = checkpoint_epoch // args.eval_cycle
        
        # Reinitialize the deques with the correct values
        rewarddeq = deque(rewardList[-100:] if len(rewardList) >= 100 else rewardList, maxlen=100)
        lossdeq = deque(lossList[-100:] if len(lossList) >= 100 else lossList, maxlen=100)
        
        print(f"Resuming training from epoch {start_epoch}, with {steps_done} total steps")
        
        # Fill replay memory before resuming if needed
        if len(memory) < args.batch_size:
            fill_replay_memory(env, memory, device, args.batch_size)
    else:
        print("Starting training from scratch")
        # Warming up with random actions
        print("Warming up...")
        warmupstep = 0
        for epoch in count():
            obs, info = env.reset()
            obs = torch.from_numpy(obs).to(device)
            obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)

            # step loop
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

                if done:
                    break

            if warmupstep > args.warmup:
                break
    
    # Check if replay memory is empty or not enough samples
    if len(memory) < args.batch_size:
        print(f"Replay memory has only {len(memory)} samples, filling to minimum batch size...")
        fill_replay_memory(env, memory, device, args.batch_size)
    
    # Main training loop
    for epoch in range(start_epoch, args.epoch):
        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(device)
        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)

        total_loss = 0.0
        total_reward = 0

        # Step loop for each epoch
        for step in count():
            # Take one step
            action, action_prob = trainer.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
            
            # Convert to tensor
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)
            next_obs = torch.from_numpy(next_obs).to(device)
            next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0)
            
            # Store the transition in memory
            memory.push(obs, action, next_obs, reward, done, action_prob)

            # Move to next state
            obs = next_obs

            # Train
            policy_net.train()
            
            # Double-check there are enough samples in memory before sampling
            if len(memory) < args.batch_size:
                print(f"WARNING: Not enough samples in replay memory ({len(memory)}/{args.batch_size}). Collecting more...")
                break
                
            # Sample transitions from memory
            transitions = memory.sample(args.batch_size)
            batch = Transition(*zip(*transitions))
            
            # Update network
            loss = trainer.update(batch)
            total_loss += loss
            
            # Periodically sync target network
            if trainer.steps_done % 1000 == 0:
                trainer.sync_target_network()

            # Check if episode is done
            if done:
                # Run evaluation if needed
                if epoch % args.eval_cycle == 0:
                    # Increment evaluation counter
                    eval_counter += 1
                    
                    # Setup evaluation environment
                    video.reset()
                    evalenv = setup_environment(args.env_name)
                    evalenv = AtariWrapper(evalenv, video=video)
                    
                    # Run evaluation
                    evalreward = trainer.run_evaluation(evalenv)
                    evalenv.close()
                    video.save(f"{epoch}.mp4")
                    
                    # Save model
                    if eval_counter % args.save_cycle == 0:
                        torch.save(policy_net, os.path.join(log_dir, f'model{epoch}.pth'))
                        print(f"Eval #{eval_counter} (epoch {epoch}): Reward {evalreward} - Model saved")
                    else:
                        print(f"Eval #{eval_counter} (epoch {epoch}): Reward {evalreward}")
                break
        
        # Update metrics
        rewardList.append(total_reward)
        lossList.append(total_loss)
        rewarddeq.append(total_reward)
        lossdeq.append(total_loss)
        avgreward = sum(rewarddeq)/len(rewarddeq)
        avgloss = sum(lossdeq)/len(lossdeq)
        avglosslist.append(avgloss)
        avgrewardlist.append(avgreward)

        # Log progress
        output = f"Epoch {epoch}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, Epsilon {trainer.eps_threshold:.2f}, TotalStep {trainer.steps_done}"
        print(output)
        with open(log_path, "a") as f:
            f.write(f"{output}\n")

        # Save training state periodically
        if epoch % 10 == 0 or epoch % args.eval_cycle == 0:
            save_training_state(
                log_dir, epoch, trainer.steps_done, trainer.eps_threshold, 
                rewardList, lossList, avgrewardlist, avglosslist
            )

    # Close environment
    env.close()
    
    # Save final plots
    save_plots(log_dir, lossList, avglosslist, rewardList, avgrewardlist)

if __name__ == "__main__":
    main()
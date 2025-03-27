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
import matplotlib.pyplot as plt
import math
from collections import deque
import json
import glob
import re

# Functions for saving and loading training state
def save_training_state(log_dir, epoch, steps_done, eps_threshold, 
                         rewardList, lossList, avgrewardlist, avglosslist):
    """
    Save the current training state to a file
    """
    training_state = {
        'epoch': epoch,
        'steps_done': steps_done,
        'eps_threshold': float(eps_threshold),
        'rewardList': rewardList,
        'lossList': lossList,
        'avgrewardlist': avgrewardlist,
        'avglosslist': avglosslist
    }
    
    # Save to a JSON file
    with open(os.path.join(log_dir, 'training_state.json'), 'w') as f:
        json.dump(training_state, f)
    
    print(f"Training state saved at epoch {epoch}")

def load_training_state(log_dir):
    """
    Load the training state from a file if it exists
    """
    state_path = os.path.join(log_dir, 'training_state.json')
    
    if not os.path.exists(state_path):
        return None
    
    with open(state_path, 'r') as f:
        training_state = json.load(f)
    
    return training_state

def get_latest_checkpoint(log_dir):
    """
    Find the latest model checkpoint and its corresponding epoch
    """
    model_files = glob.glob(os.path.join(log_dir, "model*.pth"))
    if not model_files:
        return None, 0
    
    # Extract epoch numbers from filenames
    epoch_numbers = []
    for file in model_files:
        match = re.search(r'model(\d+)\.pth', file)
        if match:
            epoch_numbers.append(int(match.group(1)))
    
    if not epoch_numbers:
        return None, 0
    
    latest_epoch = max(epoch_numbers)
    latest_model = os.path.join(log_dir, f"model{latest_epoch}.pth")
    
    return latest_model, latest_epoch

#comment
# parser
parser = argparse.ArgumentParser()
parser.add_argument('--env-name',default="breakout",type=str,choices=["pong","breakout","boxing"], help="env name")
parser.add_argument('--model', default="dqn", type=str, choices=["dqn","dueldqn"], help="dqn model")
parser.add_argument('--gpu',default=0,type=int,help="which gpu to use")
#parser.add_argument('--gpu',default="cpu",type=str,help="which device to use")
parser.add_argument('--algorithm', default="dqn", type=str, choices=["dqn", "expected_sarsa"], help="RL algorithm")
parser.add_argument('--weighted-is', action='store_true', help="Use weighted importance sampling")
parser.add_argument('--lr', default=2.5e-4, type=float, help="learning rate")
parser.add_argument('--epoch', default=10001, type=int, help="training epoch")
parser.add_argument('--batch-size', default=32, type=int, help="batch size")
parser.add_argument('--ddqn',action='store_true', help="double dqn/dueldqn")
parser.add_argument('--eval-cycle', default=500, type=int, help="evaluation cycle")
parser.add_argument('--clip-weights', action='store_true', help="Clip importance sampling weights")
parser.add_argument('--max-weight', default=10.0, type=float, help="Maximum importance sampling weight when clipping")
parser.add_argument('--log-dir', type=str, help="Directory to save logs and checkpoints (overrides default)")
args = parser.parse_args()

# some hyperparameters
GAMMA = 0.99 # bellman function
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 50000
WARMUP = 1000 # don't update net until WARMUP steps

steps_done = 0
eps_threshold = EPS_START
def select_action(state:torch.Tensor)->torch.Tensor:
    '''
    epsilon greedy
    - epsilon: choose random action
    - 1-epsilon: argmax Q(a,s)

    Input: state shape (1,4,84,84)

    Output: action shape (1,1)
    '''
    global eps_threshold
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # Calculate action probabilities (epsilon-greedy)
    action_probs = torch.ones(env.action_space.n, device=args.gpu) * eps_threshold / env.action_space.n
    
    # In the select_action function, ensure action_probs are consistently shaped
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            best_action = q_values.max(1)[1].view(1, 1)
            # Return the probability as a 1D tensor
            return best_action, action_probs[best_action].view(-1)
    else:
        action = torch.tensor([[env.action_space.sample()]], device=args.gpu)
        # Return the probability as a 1D tensor
        return action, action_probs[action].view(-1)



    # if sample > eps_threshold:
    #     with torch.no_grad():
    #         return policy_net(state).max(1)[1].view(1, 1)
    # else:
    #     return torch.tensor([[env.action_space.sample()]]).to(args.gpu)

def expected_sarsa_update(batch, weighted=True, clip_weights=True, max_weight=10.0, epsilon=1e-6):
    """
    Perform an Expected Sarsa update with importance sampling.
    
    Args:
        batch: A batch of transitions from the replay buffer
        weighted: Whether to use weighted importance sampling
        clip_weights: Whether to clip importance weights to prevent extreme values
        max_weight: Maximum importance weight value when clipping is enabled
        epsilon: Small value to prevent division by zero
        
    Returns:
        loss: The calculated loss value
    """
    state_batch = torch.cat(batch.state)  # (bs,4,84,84)
    next_state_batch = torch.cat(batch.next_state)  # (bs,4,84,84)
    action_batch = torch.cat(batch.action)  # (bs,1)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)  # (bs,1)
    done_batch = torch.cat(batch.done).unsqueeze(1)  # (bs,1)
    
    # Make sure all action_prob tensors have the same shape before concatenating
    action_prob_list = []
    for prob in batch.action_prob:
        # Ensure each prob is a 1D tensor
        action_prob_list.append(prob.view(-1))
    
    action_prob_batch = torch.cat(action_prob_list).unsqueeze(1)  # (bs,1)
    # Add epsilon to prevent division by zero
    action_prob_batch = action_prob_batch + epsilon
    
    # Q(st,a) from policy network
    state_qvalues = policy_net(state_batch)  # (bs,n_actions)
    selected_state_qvalue = state_qvalues.gather(1, action_batch)  # (bs,1)
    
    # Calculate current policy probabilities for all actions
    current_probs = torch.ones(state_batch.size(0), env.action_space.n, device=args.gpu) * eps_threshold / env.action_space.n
    best_actions = state_qvalues.max(1)[1].unsqueeze(1)
    row_indices = torch.arange(state_batch.size(0), device=args.gpu)
    current_probs[row_indices, best_actions.squeeze()] = 1 - eps_threshold + (eps_threshold / env.action_space.n)
    
    # Calculate importance weights (current policy / behavior policy)
    importance_weights = current_probs.gather(1, action_batch) / action_prob_batch
    
    # Optional clipping of importance weights
    if clip_weights:
        importance_weights = torch.clamp(importance_weights, max=max_weight)
    
    with torch.no_grad():
        # Calculate expected value of next state under current policy
        next_state_qvalues = target_net(next_state_batch)  # (bs,n_actions)
        
        # Calculate expected value using current policy probabilities
        # E[Q(s',a')] = sum_a' π(a'|s') * Q(s',a')
        expected_qvalues = (next_state_qvalues * current_probs).sum(1).unsqueeze(1)
        
        # TD target
        expected_state_values = expected_qvalues * GAMMA * ~done_batch + reward_batch  # (bs,1)
    
    # Apply importance sampling to the loss
    criterion = nn.SmoothL1Loss(reduction='none')
    loss_per_sample = criterion(selected_state_qvalue, expected_state_values)
    
    if weighted:
        # Properly normalize weights for each state separately (group by state)
        # For batch learning, we use all weights in the batch from the same minibatch update
        # This is an approximation, as ideally we would group by identical states
        batch_weights = importance_weights.detach().clone()
        
        # Compute the sum of importance weights for normalization (across the batch)
        # Keep dimensions for proper broadcasting
        sum_weights = batch_weights.sum()
        if sum_weights > epsilon:  # Avoid division by zero
            normalized_weights = batch_weights / sum_weights
        else:
            normalized_weights = torch.ones_like(batch_weights) / batch_weights.size(0)
            
        # Apply normalized weights to loss
        loss = (loss_per_sample * normalized_weights).sum()
    else:
        # Regular importance sampling (not weighted)
        loss = (loss_per_sample * importance_weights).mean()
    
    return loss
# environment
if args.env_name == "pong":
    env = gym.make("PongNoFrameskip-v4")
elif args.env_name == "breakout":
    env = gym.make("BreakoutNoFrameskip-v4")
else:
    env = gym.make("BoxingNoFrameskip-v4")
env = AtariWrapper(env)

n_action = env.action_space.n # pong:6; breakout:4; boxing:18

# make dir to store result
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
log_path = os.path.join(log_dir,"log.txt")


# video
video = VideoRecorder(log_dir)

# create network and target network
if args.model == "dqn":
    policy_net = DQN(in_channels=4, n_actions=n_action).to(args.gpu)
    target_net = DQN(in_channels=4, n_actions=n_action).to(args.gpu)
else:
    policy_net = DuelDQN(in_channels=4, n_actions=n_action).to(args.gpu)
    target_net = DuelDQN(in_channels=4, n_actions=n_action).to(args.gpu)
# let target model = model
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# replay memory
memory = ReplayMemory(50000)

# optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)

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

if latest_model_path and training_state:
    print(f"Resuming from checkpoint: {latest_model_path} (Epoch {checkpoint_epoch})")
    # Load model
    device = f"cuda:{args.gpu}" if isinstance(args.gpu, int) else "cpu"
    loaded_model = torch.load(latest_model_path, map_location=device)
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
    
    # Reinitialize the deques with the correct values
    rewarddeq = deque(rewardList[-100:] if len(rewardList) >= 100 else rewardList, maxlen=100)
    lossdeq = deque(lossList[-100:] if len(lossList) >= 100 else lossList, maxlen=100)
    
    print(f"Resuming training from epoch {start_epoch}, with {steps_done} total steps")
    
    # Skip warmup if resuming training
    warmupstep = WARMUP + 1
else:
    print("Starting training from scratch")
    # warming up
    print("Warming up...")
    warmupstep = 0
    for epoch in count():
        obs, info = env.reset() # (84,84)
        obs = torch.from_numpy(obs).to(args.gpu) #(84,84)
        # stack four frames together, hoping to learn temporal info
        obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0) #(1,4,84,84)

        # step loop
        for step in count():
            warmupstep += 1
            # take one step
            action = torch.tensor([[env.action_space.sample()]]).to(args.gpu)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            # convert to tensor
            reward = torch.tensor([reward],device=args.gpu) # (1)
            done = torch.tensor([done],device=args.gpu) # (1)
            next_obs = torch.from_numpy(next_obs).to(args.gpu) # (84,84)
            next_obs = torch.stack((next_obs,obs[0][0],obs[0][1],obs[0][2])).unsqueeze(0) # (1,4,84,84)
            
            # store the transition in memory
            #memory.push(obs,action,next_obs,reward,done)
            action_prob = torch.tensor([1.0/env.action_space.n], device=args.gpu)  # Uniform probability for random actions
            memory.push(obs, action, next_obs, reward, done, action_prob)
            # move to next state
            obs = next_obs

            if done:
                break

        if warmupstep > WARMUP:
            break

# epoch loop 
for epoch in range(start_epoch, args.epoch):
    obs, info = env.reset() # (84,84)
    obs = torch.from_numpy(obs).to(args.gpu) #(84,84)
    # stack four frames together, hoping to learn temporal info
    obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0) #(1,4,84,84)

    total_loss = 0.0
    total_reward = 0

    # step loop
    for step in count():
        # take one step
        #action = select_action(obs)
        action, action_prob = select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        done = terminated or truncated
        
        # convert to tensor
        reward = torch.tensor([reward],device=args.gpu) # (1)
        done = torch.tensor([done],device=args.gpu) # (1)
        next_obs = torch.from_numpy(next_obs).to(args.gpu) # (84,84)
        next_obs = torch.stack((next_obs,obs[0][0],obs[0][1],obs[0][2])).unsqueeze(0) # (1,4,84,84)
        
        # store the transition in memory
        #memory.push(obs,action,next_obs,reward,done)
        memory.push(obs, action, next_obs, reward, done, action_prob)

        # move to next state
        obs = next_obs

        # train
        policy_net.train()
        transitions = memory.sample(args.batch_size)
        batch = Transition(*zip(*transitions))  # batch-array of Transitions -> Transition of batch-arrays.

        if args.algorithm == "dqn":
            # Original DQN update
            state_batch = torch.cat(batch.state)  # (bs,4,84,84)
            next_state_batch = torch.cat(batch.next_state)  # (bs,4,84,84)
            action_batch = torch.cat(batch.action)  # (bs,1)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)  # (bs,1)
            done_batch = torch.cat(batch.done).unsqueeze(1)  #(bs,1)

            # Q(st,a)
            state_qvalues = policy_net(state_batch)  # (bs,n_actions)
            selected_state_qvalue = state_qvalues.gather(1, action_batch)  # (bs,1)
            
            with torch.no_grad():
                # Q'(st+1,a)
                next_state_target_qvalues = target_net(next_state_batch)  # (bs,n_actions)
                if args.ddqn:
                    # Q(st+1,a)
                    next_state_qvalues = policy_net(next_state_batch)  # (bs,n_actions)
                    # argmax Q(st+1,a)
                    next_state_selected_action = next_state_qvalues.max(1, keepdim=True)[1]  # (bs,1)
                    # Q'(st+1,argmax_a Q(st+1,a))
                    next_state_selected_qvalue = next_state_target_qvalues.gather(1, next_state_selected_action)  # (bs,1)
                else:
                    # max_a Q'(st+1,a)
                    next_state_selected_qvalue = next_state_target_qvalues.max(1, keepdim=True)[0]  # (bs,1)

            # td target
            tdtarget = next_state_selected_qvalue * GAMMA * ~done_batch + reward_batch  # (bs,1)

            # optimize
            criterion = nn.SmoothL1Loss()
            loss = criterion(selected_state_qvalue, tdtarget)
            
        else:
            # Expected Sarsa update
            loss = expected_sarsa_update(
            batch, 
            weighted=args.weighted_is,
            clip_weights=args.clip_weights,
            max_weight=args.max_weight
)

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # let target_net = policy_net every 1000 steps
        if steps_done % 1000 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            # eval
            if epoch % args.eval_cycle == 0:
                with torch.no_grad():
                    video.reset()
                    if args.env_name == "pong":
                        evalenv = gym.make("PongNoFrameskip-v4")
                    elif args.env_name == "breakout":
                        evalenv = gym.make("BreakoutNoFrameskip-v4")
                    else:
                        evalenv = gym.make("BoxingNoFrameskip-v4")
                    evalenv = AtariWrapper(evalenv,video=video)
                    obs, info = evalenv.reset()
                    obs = torch.from_numpy(obs).to(args.gpu)
                    obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0)
                    evalreward = 0
                    policy_net.eval()
                    for _ in count():
                        action = policy_net(obs).max(1)[1]
                        next_obs, reward, terminated, truncated, info = evalenv.step(action.item())
                        evalreward += reward
                        next_obs = torch.from_numpy(next_obs).to(args.gpu) # (84,84)
                        next_obs = torch.stack((next_obs,obs[0][0],obs[0][1],obs[0][2])).unsqueeze(0) # (1,4,84,84)
                        obs = next_obs
                        if terminated or truncated:
                            if info["lives"] == 0: # real end
                                break
                            else:
                                obs, info = evalenv.reset()
                                obs = torch.from_numpy(obs).to(args.gpu)
                                obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0)
                    evalenv.close()
                    video.save(f"{epoch}.mp4")
                    torch.save(policy_net, os.path.join(log_dir,f'model{epoch}.pth')) 
                    print(f"Eval epoch {epoch}: Reward {evalreward}")
            break
    
    rewardList.append(total_reward)
    lossList.append(total_loss)
    rewarddeq.append(total_reward)
    lossdeq.append(total_loss)
    avgreward = sum(rewarddeq)/len(rewarddeq)
    avgloss = sum(lossdeq)/len(lossdeq)
    avglosslist.append(avgloss)
    avgrewardlist.append(avgreward)

    output = f"Epoch {epoch}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, Epsilon {eps_threshold:.2f}, TotalStep {steps_done}"
    print(output)
    with open(log_path,"a") as f:
        f.write(f"{output}\n")

    # Save training state periodically
    if epoch % 10 == 0 or epoch % args.eval_cycle == 0:  # Save every 10 epochs and on evaluation epochs
        save_training_state(log_dir, epoch, steps_done, eps_threshold, 
                           rewardList, lossList, avgrewardlist, avglosslist)

env.close()


# plot loss-epoch and reward-epoch
plt.figure(1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(range(len(lossList)),lossList,label="loss")
plt.plot(range(len(lossList)),avglosslist,label="avg")
plt.legend()
plt.savefig(os.path.join(log_dir,"loss.png"))

plt.figure(2)
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.plot(range(len(rewardList)),rewardList,label="reward")
plt.plot(range(len(rewardList)),avgrewardlist, label="avg")
plt.legend()
plt.savefig(os.path.join(log_dir,"reward.png"))
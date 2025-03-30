import gymnasium as gym
import torch
from torch import optim
import os
import matplotlib.pyplot as plt
from collections import deque

# Import custom modules
from model import DQN, DuelDQN
from utils import Transition, ReplayMemory, VideoRecorder
from wrapper import AtariWrapper
from config import parse_args, GAMMA, EPS_START, EPS_END, EPS_DECAY, WARMUP
from checkpointing import get_latest_checkpoint, load_training_state
from replay_memory_fill import fill_replay_memory, warmup_memory
from action_selection import init_exploration
from train import train
from seed_utils import set_seed  # Import the new seed utilities

# Add this at the top to make your custom classes safe for loading with PyTorch 2.6+
from torch.serialization import add_safe_globals
add_safe_globals(['model.DQN', 'model.DuelDQN'])

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed, deterministic=args.deterministic)
    
    # Create environment
    if args.env_name == "pong":
        env = gym.make("PongNoFrameskip-v4")
    elif args.env_name == "breakout":
        env = gym.make("BreakoutNoFrameskip-v4")
    elif args.env_name == "spaceinvaders":
        env = gym.make("SpaceInvadersNoFrameskip-v4")
    elif args.env_name == "pacman":
        env = gym.make("MsPacmanNoFrameskip-v4")
    else:
        env = gym.make("BoxingNoFrameskip-v4")
    
    # Set the environment seed
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)
    
    env = AtariWrapper(env, seed=args.seed)
    
    n_action = env.action_space.n  # pong:6; breakout:4; boxing:18
    
    # Create directory for logs and checkpoints
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
    
    # Create video recorder
    video = VideoRecorder(log_dir)
    
    # Create networks
    if args.model == "dqn":
        policy_net = DQN(in_channels=4, n_actions=n_action).to(args.gpu)
        target_net = DQN(in_channels=4, n_actions=n_action).to(args.gpu)
    else:
        policy_net = DuelDQN(in_channels=4, n_actions=n_action).to(args.gpu)
        target_net = DuelDQN(in_channels=4, n_actions=n_action).to(args.gpu)
    
    # Initialize target network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Create replay memory
    memory = ReplayMemory(50000, seed=args.seed)
    
    # Create optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)
    
    # Check for existing checkpoints and load if available
    latest_model_path, checkpoint_epoch = get_latest_checkpoint(log_dir)
    training_state = load_training_state(log_dir)
    
    # Initialize lists and variables
    rewardList = []
    lossList = []
    avgrewardlist = []
    avglosslist = []
    start_epoch = 0
    
    if latest_model_path and training_state:
        print(f"Resuming from checkpoint: {latest_model_path} (Epoch {checkpoint_epoch})")
        # Load model
        device = f"cuda:{args.gpu}" if isinstance(args.gpu, int) else "cpu"
        try:
            # First try loading with weights_only=False (which was the default before PyTorch 2.6)
            loaded_model = torch.load(latest_model_path, map_location=device, weights_only=False)
            print("Successfully loaded checkpoint")
            policy_net = loaded_model
        except Exception as e:
            # If that fails, try to recreate the model architecture and load state_dict
            if args.model == "dqn":
                loaded_model = DQN(in_channels=4, n_actions=n_action).to(device)
            else:
                loaded_model = DuelDQN(in_channels=4, n_actions=n_action).to(device)
            
            # Load just the state dictionary
            state_dict = torch.load(latest_model_path, map_location=device, weights_only=True)
            loaded_model.load_state_dict(state_dict)
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
        
        # Initialize exploration parameters
        init_exploration(EPS_START, EPS_END, EPS_DECAY, steps_done)
        
        print(f"Resuming training from epoch {start_epoch}, with {steps_done} total steps")
        
        # Fill replay memory before resuming
        print("Filling replay memory before resuming training...")
        min_replay_size = args.min_replay_size  # Use command line arg or default to 10000
        fill_replay_memory(env, memory, args.gpu, min_replay_size, seed=args.seed)
        
    else:
        print("Starting training from scratch")
        # Initialize exploration parameters
        init_exploration(EPS_START, EPS_END, EPS_DECAY)
        
        # Warm up replay memory
        warmup_memory(env, memory, args.gpu, WARMUP, seed=args.seed)
    
    # Run training
    rewardList, lossList, avgrewardlist, avglosslist = train(
        env, policy_net, target_net, memory, optimizer, args, log_dir, 
        video, start_epoch, rewardList, lossList, avgrewardlist, avglosslist, 
        seed=args.seed
    )
    
    # Close environment
    env.close()
    
    # Plot training results
    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(len(lossList)), lossList, label="loss")
    plt.plot(range(len(lossList)), avglosslist, label="avg")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss.png"))
    
    plt.figure(2)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.plot(range(len(rewardList)), rewardList, label="reward")
    plt.plot(range(len(rewardList)), avgrewardlist, label="avg")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "reward.png"))

if __name__ == "__main__":
    main()
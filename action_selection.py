import torch
import random
import math

# Global variables for tracking exploration
steps_done = 0
eps_threshold = 1.0

def init_exploration(start_eps, end_eps, decay_steps, initial_steps=0):
    """
    Initialize exploration parameters
    
    Args:
        start_eps: Initial epsilon value
        end_eps: Final epsilon value
        decay_steps: Number of steps for epsilon decay
        initial_steps: Initial value for steps_done (for resuming training)
    """
    global steps_done, eps_threshold
    
    steps_done = initial_steps
    eps_threshold = end_eps + (start_eps - end_eps) * math.exp(-1. * steps_done / decay_steps)
    
    return steps_done, eps_threshold

def select_action(state, policy_net, env, device, eps_start=1.0, eps_end=0.05, eps_decay=50000):
    """
    Epsilon greedy action selection
    - epsilon: choose random action
    - 1-epsilon: argmax Q(a,s)

    Args:
        state: Current state tensor shape (1,4,84,84)
        policy_net: Policy network
        env: Environment (for action space)
        device: Device to use for tensor operations
        eps_start: Starting epsilon value
        eps_end: Final epsilon value
        eps_decay: Decay rate for epsilon
        
    Returns:
        action: Selected action shape (1,1)
        action_prob: Probability of selecting the action
    """
    global eps_threshold
    global steps_done
    
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1

    # Calculate action probabilities (epsilon-greedy)
    action_probs = torch.ones(env.action_space.n, device=device) * eps_threshold / env.action_space.n
    
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            best_action = q_values.max(1)[1].view(1, 1)
            # Return the probability as a 1D tensor
            return best_action, action_probs[best_action].view(-1)
    else:
        action = torch.tensor([[env.action_space.sample()]], device=device)
        # Return the probability as a 1D tensor
        return action, action_probs[action].view(-1)

def get_exploration_state():
    """Get the current state of exploration parameters"""
    global steps_done, eps_threshold
    return steps_done, eps_threshold
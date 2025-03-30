import torch
import torch.nn as nn

def expected_sarsa_update(batch, policy_net, target_net, env, eps_threshold, 
                          gamma=0.99, weighted=True, clip_weights=True, 
                          max_weight=10.0, epsilon=1e-6, device=None):
    """
    Perform an Expected SARSA update with importance sampling for off-policy learning.
    
    This function implements the Expected SARSA algorithm with importance sampling 
    to correct for the difference between behavior policy which gathered the data
    and target policy which we're learning. It supports both regular and weighted
    importance sampling techniques.
    
    Args:
        batch: A batch of transitions from the replay buffer containing states, actions,
               rewards, next states, done flags, and action probabilities
        policy_net: The policy network that we're updating (online network)
        target_net: The target network used for more stable TD targets
        env: The environment (needed for action space information)
        eps_threshold: Current exploration rate for the ε-greedy policy
        gamma: Discount factor for future rewards (default: 0.99)
        weighted: Whether to use weighted importance sampling (default: True)
                 - True: More stable but potentially biased estimates
                 - False: Unbiased but potentially higher variance
        clip_weights: Whether to clip importance weights to prevent extreme values (default: True)
        max_weight: Maximum allowed importance weight when clipping is enabled (default: 10.0)
        epsilon: Small value to prevent division by zero in calculations (default: 1e-6)
        device: Device to use for tensor operations (CPU/GPU)
        
    Returns:
        loss: The calculated loss value after importance sampling adjustment
    """
    # Extract batch components and ensure proper dimensions
    state_batch = torch.cat(batch.state)  # Shape: (batch_size, channels, height, width)
    next_state_batch = torch.cat(batch.next_state)  # Shape: (batch_size, channels, height, width)
    action_batch = torch.cat(batch.action)  # Shape: (batch_size, 1)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)  # Shape: (batch_size, 1)
    done_batch = torch.cat(batch.done).unsqueeze(1)  # Shape: (batch_size, 1)
    
    # Get behavior policy action probabilities from the batch
    # These represent the probabilities used when the actions were originally selected
    action_prob_batch = torch.cat(batch.action_prob).unsqueeze(1)  # Shape: (batch_size, 1)
    # Add small epsilon to prevent division by zero in importance weight calculation
    action_prob_batch = action_prob_batch + epsilon
    
    # q values
    state_qvalues = policy_net(state_batch)  # Shape: (batch_size, n_actions)
    # taken action q values
    selected_state_qvalue = state_qvalues.gather(1, action_batch)  # Shape: (batch_size, 1)
    
    # Calculate current target policy probabilities for all actions (ε-greedy policy)
    # Initialize with exploration probability distributed evenly across all actions
    current_probs = torch.ones(state_batch.size(0), env.action_space.n, device=device) * eps_threshold / env.action_space.n
    # Find greedy actions based on current Q-values
    best_actions = state_qvalues.max(1)[1].unsqueeze(1)
    # Create indices for batch elements
    row_indices = torch.arange(state_batch.size(0), device=device)
    # Set higher probability for greedy actions (1-ε + ε/|A| for the greedy action)
    current_probs[row_indices, best_actions.squeeze()] = 1 - eps_threshold + (eps_threshold / env.action_space.n)
    
    # Calculate importance weights: ratio of target policy to behavior policy probabilities
    # For the specific actions that were taken in the batch
    importance_weights = current_probs.gather(1, action_batch) / action_prob_batch
    
    # Optional clipping of importance weights to prevent extreme values destabilizing training
    if clip_weights:
        importance_weights = torch.clamp(importance_weights, max=max_weight)
    
    with torch.no_grad():  # No need to track gradients for target computation
        # Calculate Q-values for next states using target network for stability
        next_state_qvalues = target_net(next_state_batch)  # Shape: (batch_size, n_actions)
        
        # Calculate expected value using current policy probabilities
        # E[Q(s',a')] = Σ_a' π(a'|s') * Q(s',a')
        # This is the key difference between Expected SARSA and Q-learning (which uses max)
        expected_qvalues = (next_state_qvalues * current_probs).sum(1).unsqueeze(1)
        
        # Compute TD target: r + γ * E[Q(s',a')] for non-terminal states
        # For terminal states (done=True), we only consider the immediate reward
        expected_state_values = expected_qvalues * gamma * (~done_batch) + reward_batch  # Shape: (batch_size, 1)
    
    # Calculate TD error using Smooth L1 Loss (Huber loss)
    # This is more robust to outliers than MSE loss
    criterion = nn.SmoothL1Loss(reduction='none')
    loss_per_sample = criterion(selected_state_qvalue, expected_state_values)
    
    if weighted:
        # Apply weighted importance sampling (WIS)
        # This normalizes the weights to sum to 1, which can reduce variance but introduces bias
        batch_weights = importance_weights.detach().clone()
        
        # Compute the sum of importance weights for normalization
        sum_weights = batch_weights.sum()
        if sum_weights > epsilon:  # Avoid division by zero
            normalized_weights = batch_weights / sum_weights
        else:
            # If weights sum is too small, use uniform weights
            normalized_weights = torch.ones_like(batch_weights) / batch_weights.size(0)
            
        # Apply normalized weights to loss (each sample weighted by its normalized importance)
        loss = (loss_per_sample * normalized_weights).sum()
    else:
        # Use regular importance sampling (IS)
        # This is unbiased but may have higher variance than WIS
        loss = (loss_per_sample * importance_weights).mean()
    
    return loss
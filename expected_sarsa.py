import torch
import torch.nn as nn

def expected_sarsa_update(batch, policy_net, target_net, env, eps_threshold, 
                          gamma=0.99, weighted=True, clip_weights=True, 
                          max_weight=10.0, epsilon=1e-6, device=None):
    """
    expected SARSA algorithm with importance sampling 
    to correct for the difference between behavior policy which gathered the data
    and target policy which were learning. Regular and weighted
    importance sampling.
    
    Args:
        
        eps_threshold: Current exploration rate for the ε-greedy policy
        clip_weights: Whether to clip importance weights to prevent extreme values (default: True)
        max_weight: Maximum allowed importance weight when clipping is enabled (default: 10.0)
        epsilon: Small value to prevent division by zero in calculations (default: 1e-6)
        
    """
    # Extract batch components and ensure proper dimensions
    state_batch = torch.cat(batch.state)  #Shape: (batch_size, channels, height, width)
    next_state_batch = torch.cat(batch.next_state)  # Shape: (batch_size, channels, height, width)
    action_batch = torch.cat(batch.action)  # Shape: (batch_size, 1)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)  #Shape: (batch_size, 1)
    done_batch = torch.cat(batch.done).unsqueeze(1)  #Shape: (batch_size, 1)
    
    #get behavior policy action probabilities from the batch
    #recreate probabilities like when originally selected
    action_prob_batch = torch.cat(batch.action_prob).unsqueeze(1)  # Shape: (batch_size, 1)
    #prevent division by zero in importance weight calculation
    action_prob_batch = action_prob_batch + epsilon
    
    # q values
    state_qvalues = policy_net(state_batch)  # Shape: (batch_size, n_actions)
    # taken action q values
    selected_state_qvalue = state_qvalues.gather(1, action_batch)  # Shape: (batch_size, 1)
    
    # Calculate current target policy probabilities for all actions (e-greedy policy)
    # initialize with exploration probability
    current_probs = torch.ones(state_batch.size(0), env.action_space.n, device=device) * eps_threshold / env.action_space.n
    # Find greedy actions based on current Q-values
    best_actions = state_qvalues.max(1)[1].unsqueeze(1)
    row_indices = torch.arange(state_batch.size(0), device=device)
    # Set probability for greedy actions (1-ε + ε/|A| for the greedy action)
    current_probs[row_indices, best_actions.squeeze()] = 1 - eps_threshold + (eps_threshold / env.action_space.n)
    
    # Calculate importance weights as ratio of target policy to behavior policy probabilities
    # For the specific actions that were taken in the batch
    importance_weights = current_probs.gather(1, action_batch) / action_prob_batch
    
    # clipping of importance weights to prevent extreme values destabilizing training
    if clip_weights:
        importance_weights = torch.clamp(importance_weights, max=max_weight)
    
    with torch.no_grad():  
        # Calculate Q-values for next states using target network for stability
        next_state_qvalues = target_net(next_state_batch)  # Shape: (batch_size, n_actions)
        
        # Calculate expected value using current policy probabilities
        # E[Q(s',a')] = Σ_a' π(a'|s') * Q(s',a')
        # difference between Expected SARSA and Q-learning (which uses max)
        expected_qvalues = (next_state_qvalues * current_probs).sum(1).unsqueeze(1)
        
        # Compute TD target: r + γ * E[Q(s',a')] for non-terminal states
        # for terminal states done stops calculation
        expected_state_values = expected_qvalues * gamma * (~done_batch) + reward_batch  # Shape: (batch_size, 1)
    
    # Calculate TD error using Smooth L1 Loss (Huber loss)
    criterion = nn.SmoothL1Loss(reduction='none')
    loss_per_sample = criterion(selected_state_qvalue, expected_state_values)
    
    if weighted:
        # Apply weighted importance sampling
        # less variance but bias
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
        # Use regular importance sampling 
        loss = (loss_per_sample * importance_weights).mean()
    
    return loss
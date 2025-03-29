import torch
import torch.nn as nn

def expected_sarsa_update(batch, policy_net, target_net, env, eps_threshold, 
                          gamma=0.99, weighted=True, clip_weights=True, 
                          max_weight=10.0, epsilon=1e-6, device=None):
    """
    Perform an Expected Sarsa update with importance sampling.
    
    Args:
        batch: A batch of transitions from the replay buffer
        policy_net: The policy network
        target_net: The target network
        env: The environment (needed for action space)
        eps_threshold: Current exploration rate
        gamma: Discount factor
        weighted: Whether to use weighted importance sampling
        clip_weights: Whether to clip importance weights to prevent extreme values
        max_weight: Maximum importance weight value when clipping is enabled
        epsilon: Small value to prevent division by zero
        device: Device to use for tensor operations
        
    Returns:
        loss: The calculated loss value
    """
    # Extract batch components
    state_batch = torch.cat(batch.state)  # (bs,4,84,84)
    next_state_batch = torch.cat(batch.next_state)  # (bs,4,84,84)
    action_batch = torch.cat(batch.action)  # (bs,1)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)  # (bs,1)
    done_batch = torch.cat(batch.done).unsqueeze(1)  # (bs,1)
    
    # Get action probabilities as a batch
    action_prob_batch = torch.cat(batch.action_prob).unsqueeze(1)  # (bs,1)
    # Add epsilon to prevent division by zero
    action_prob_batch = action_prob_batch + epsilon
    
    # Q(st,a) from policy network
    state_qvalues = policy_net(state_batch)  # (bs,n_actions)
    selected_state_qvalue = state_qvalues.gather(1, action_batch)  # (bs,1)
    
    # Calculate current policy probabilities for all actions
    current_probs = torch.ones(state_batch.size(0), env.action_space.n, device=device) * eps_threshold / env.action_space.n
    best_actions = state_qvalues.max(1)[1].unsqueeze(1)
    row_indices = torch.arange(state_batch.size(0), device=device)
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
        expected_state_values = expected_qvalues * gamma * ~done_batch + reward_batch  # (bs,1)
    
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
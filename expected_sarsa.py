import torch
import torch.nn as nn

def expected_sarsa_update(batch, policy_net, target_net, env, eps_threshold, 
                          gamma=0.99, weighted=True, clip_weights=True, 
                          max_weight=10.0, epsilon=1e-6, device=None):
    """
    Perform an Expected Sarsa update with importance sampling, properly normalized by state.
    
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
        # ===== IMPROVED STATE-WISE NORMALIZATION IMPLEMENTATION =====
        # For high-dimensional states like Atari frames, we need a fingerprinting approach
        # to identify similar states
        
        # Create state fingerprints for clustering similar states
        if state_batch.dim() == 4:  # Image-based states (bs, channels, height, width)
            # Create a compact representation by:
            # 1. Downsampling spatial dimensions (average pooling)
            # 2. Taking channel means to create a spatial pattern fingerprint
            
            # Average pooling to reduce dimensionality (84x84 -> 8x8)
            pooled = torch.nn.functional.avg_pool2d(state_batch, kernel_size=10)
            
            # Further compress by taking channel means
            if pooled.size(1) > 1:  # If multiple channels
                fingerprints = pooled.mean(dim=1).flatten(1)  # (bs, reduced_h*reduced_w)
            else:
                fingerprints = pooled.flatten(1)  # (bs, channels*reduced_h*reduced_w)
        else:
            # For vector states, use the state directly as fingerprint
            fingerprints = state_batch.detach().clone()
        
        # Normalize fingerprints for better clustering (important for cosine similarity)
        fingerprint_norms = torch.norm(fingerprints, p=2, dim=1, keepdim=True)
        normalized_fingerprints = fingerprints / (fingerprint_norms + epsilon)
        
        # Compute pairwise similarities between state fingerprints
        # Using cosine similarity which is more robust for high-dimensional data
        similarities = torch.mm(normalized_fingerprints, normalized_fingerprints.t())  # (bs, bs)
        
        # Threshold for considering states as "same cluster"
        # Higher values = more strict clustering (fewer states per cluster)
        similarity_threshold = 0.95  # Tunable parameter
        
        # Initialize for state grouping
        batch_size = state_batch.size(0)
        cluster_indices = -torch.ones(batch_size, dtype=torch.long, device=device)
        current_cluster = 0
        
        # Group states into clusters
        for i in range(batch_size):
            if cluster_indices[i] >= 0:
                continue  # Already assigned to a cluster
                
            # Find all states similar to state i
            similar_states = (similarities[i] >= similarity_threshold).nonzero().squeeze(1)
            
            # Assign all similar states to the current cluster
            cluster_indices[similar_states] = current_cluster
            current_cluster += 1
        
        # Count number of clusters for debugging
        num_clusters = cluster_indices.max().item() + 1
        
        # Normalize weights within each cluster
        normalized_weights = torch.zeros_like(importance_weights)
        
        for cluster_idx in range(num_clusters):
            # Get indices of states in this cluster
            cluster_mask = (cluster_indices == cluster_idx)
            if not cluster_mask.any():
                continue
                
            # Get indices as a list
            cluster_state_indices = cluster_mask.nonzero().squeeze(1)
            
            # Get weights for this state cluster
            cluster_weights = importance_weights[cluster_state_indices]
            
            # Normalize weights within this cluster
            cluster_sum = cluster_weights.sum()
            if cluster_sum > epsilon:
                cluster_normalized = cluster_weights / cluster_sum
            else:
                # Fallback to uniform weights if sum is too small
                cluster_normalized = torch.ones_like(cluster_weights) / cluster_weights.size(0)
            
            # Assign normalized weights back
            normalized_weights[cluster_state_indices] = cluster_normalized
            
        # Apply normalized weights to loss
        loss = (loss_per_sample * normalized_weights).sum()
        
    else:
        # Regular importance sampling (not weighted)
        loss = (loss_per_sample * importance_weights).mean()
    
    return loss
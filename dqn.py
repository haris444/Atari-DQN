import torch
import torch.nn as nn

def dqn_update(batch, policy_net, target_net, gamma=0.99, ddqn=False, device=None):
    """
    DQN and Double DQN update implementation
    
    Args:
        batch: A batch of transitions from the replay buffer
        policy_net: The policy network
        target_net: The target network
        gamma: Discount factor
        ddqn: Whether to use Double DQN
        device: The device to use (for tensor operations)
        
    Returns:
        loss: The calculated loss value
    """
    # Extract batch components
    state_batch = torch.cat(batch.state)  # (bs,4,84,84)
    next_state_batch = torch.cat(batch.next_state)  # (bs,4,84,84)
    action_batch = torch.cat(batch.action)  # (bs,1)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)  # (bs,1)
    done_batch = torch.cat(batch.done).unsqueeze(1)  # (bs,1)

    # Q(st,a)
    state_qvalues = policy_net(state_batch)  # (bs,n_actions)
    selected_state_qvalue = state_qvalues.gather(1, action_batch)  # (bs,1)
    
    with torch.no_grad():
        # Q'(st+1,a)
        next_state_target_qvalues = target_net(next_state_batch)  # (bs,n_actions)
        if ddqn:
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
        tdtarget = next_state_selected_qvalue * gamma * ~done_batch + reward_batch  # (bs,1)

    # Calculate loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(selected_state_qvalue, tdtarget)
    
    return loss
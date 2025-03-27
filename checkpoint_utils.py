import os
import json
import glob
import re
import torch
from torch.serialization import add_safe_globals

# Add this to make your custom classes safe for loading with PyTorch 2.6+
add_safe_globals(['model.DQN', 'model.DuelDQN'])

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
    state_path = os.path.join(log_dir, 'training_state.json')
    with open(state_path, 'w') as f:
        json.dump(training_state, f)

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

def load_model(model_path, model_class, in_channels, n_action, device):
    """
    Load a model from a checkpoint with handling for PyTorch 2.6+ security changes
    """
    try:
        # First try loading with weights_only=False (which was the default before PyTorch 2.6)
        loaded_model = torch.load(model_path, map_location=device, weights_only=False)
        print("Successfully loaded checkpoint")
    except Exception as e:
        # If that fails, try to recreate the model architecture and load state_dict
        loaded_model = model_class(in_channels=in_channels, n_actions=n_action).to(device)
        
        # Load just the state dictionary
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        loaded_model.load_state_dict(state_dict)
    
    return loaded_model

def save_plots(log_dir, lossList, avglosslist, rewardList, avgrewardlist):
    """
    Save plots of training metrics
    """
    import matplotlib.pyplot as plt
    
    # Plot loss-epoch
    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(len(lossList)), lossList, label="loss")
    plt.plot(range(len(lossList)), avglosslist, label="avg")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss.png"))
    plt.close()

    # Plot reward-epoch
    plt.figure(2)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.plot(range(len(rewardList)), rewardList, label="reward")
    plt.plot(range(len(rewardList)), avgrewardlist, label="avg")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "reward.png"))
    plt.close()
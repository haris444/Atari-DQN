import os
import json
import torch
import glob
import re

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
    
    state_path = os.path.join(log_dir, 'training_state.json')
    with open(state_path, 'w') as f:
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
import torch
import torch.nn as nn
import random
import math
from itertools import count

class DQNTrainer:
    def __init__(self, policy_net, target_net, optimizer, memory, env, device, 
                 gamma=0.99, eps_start=1, eps_end=0.05, eps_decay=250000,
                 double_dqn=False):
        """
        Initialize the DQN Trainer
        
        Args:
            policy_net: The policy network
            target_net: The target network
            optimizer: The optimizer for the policy network
            memory: The replay memory
            env: The environment
            device: The device to use (CPU/GPU)
            gamma: Discount factor
            eps_start: Initial epsilon value for epsilon-greedy
            eps_end: Final epsilon value for epsilon-greedy
            eps_decay: Decay rate for epsilon
            double_dqn: Whether to use Double DQN
        """
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = memory
        self.env = env
        self.device = device
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.double_dqn = double_dqn
        
        # Initialize epsilon and steps
        self.steps_done = 0
        self.eps_threshold = eps_start
        
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: The current state
            
        Returns:
            action: The selected action
            action_prob: The probability of selecting the action
        """
        sample = random.random()
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # Calculate action probabilities (epsilon-greedy)
        action_probs = torch.ones(self.env.action_space.n, device=self.device) * self.eps_threshold / self.env.action_space.n
        
        if sample > self.eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state)
                best_action = q_values.max(1)[1].view(1, 1)
                # Return the probability as a 1D tensor
                return best_action, action_probs[best_action].view(-1)
        else:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device)
            # Return the probability as a 1D tensor
            return action, action_probs[action].view(-1)
    
    def update(self, batch):
        """
        Perform a DQN update
        
        Args:
            batch: A batch of transitions from the replay buffer
            
        Returns:
            loss: The calculated loss value
        """
        state_batch = torch.cat(batch.state)  # (bs,4,84,84)
        next_state_batch = torch.cat(batch.next_state)  # (bs,4,84,84)
        action_batch = torch.cat(batch.action)  # (bs,1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)  # (bs,1)
        done_batch = torch.cat(batch.done).unsqueeze(1)  # (bs,1)

        # Q(st,a)
        state_qvalues = self.policy_net(state_batch)  # (bs,n_actions)
        selected_state_qvalue = state_qvalues.gather(1, action_batch)  # (bs,1)
        
        with torch.no_grad():
            # Q'(st+1,a)
            next_state_target_qvalues = self.target_net(next_state_batch)  # (bs,n_actions)
            if self.double_dqn:
                # Q(st+1,a)
                next_state_qvalues = self.policy_net(next_state_batch)  # (bs,n_actions)
                # argmax Q(st+1,a)
                next_state_selected_action = next_state_qvalues.max(1, keepdim=True)[1]  # (bs,1)
                # Q'(st+1,argmax_a Q(st+1,a))
                next_state_selected_qvalue = next_state_target_qvalues.gather(1, next_state_selected_action)  # (bs,1)
            else:
                # max_a Q'(st+1,a)
                next_state_selected_qvalue = next_state_target_qvalues.max(1, keepdim=True)[0]  # (bs,1)

        # td target
        tdtarget = next_state_selected_qvalue * self.gamma * ~done_batch + reward_batch  # (bs,1)

        # optimize
        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_state_qvalue, tdtarget)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def sync_target_network(self):
        """
        Synchronize the target network with the policy network
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def run_evaluation(self, video_recorder=None):
        """
        Run an evaluation episode
        
        Args:
            video_recorder: Optional video recorder for saving episode
            
        Returns:
            eval_reward: Total reward accumulated during evaluation
        """
        with torch.no_grad():
            self.policy_net.eval()
            obs, info = self.env.reset()
            obs = torch.from_numpy(obs).to(self.device)
            obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
            eval_reward = 0
            
            for _ in count():
                action = self.policy_net(obs).max(1)[1]
                next_obs, reward, terminated, truncated, info = self.env.step(action.item())
                eval_reward += reward
                next_obs = torch.from_numpy(next_obs).to(self.device)
                next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0)
                obs = next_obs
                
                if terminated or truncated:
                    if "lives" in info and info["lives"] == 0:  # real end
                        break
                    else:
                        obs, info = self.env.reset()
                        obs = torch.from_numpy(obs).to(self.device)
                        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
            
            self.policy_net.train()
            return eval_reward
    
    def set_training_state(self, steps_done, eps_threshold):
        """
        Set the training state (for resuming training)
        
        Args:
            steps_done: Number of steps already done
            eps_threshold: Current epsilon threshold
        """
        self.steps_done = steps_done
        self.eps_threshold = eps_threshold
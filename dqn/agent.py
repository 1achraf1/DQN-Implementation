import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from ReplayBuffer import ReplayBuffer
from model import Qnetwork


class DQN:
    def __init__(self, state_size, action_size, buffer_size, batch_size, device, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize Networks
        self.policy_network = Qnetwork(state_size, action_size).to(self.device)  # FIX: Move to device
        self.target_network = Qnetwork(state_size, action_size).to(self.device)  # FIX: Move to device

        # Sync weights
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # Optional: Set target to eval mode as we never calculate gradients for it

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            # Convert state to tensor on device
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_network(state)
                return torch.argmax(q_values).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, n_states, dones = self.memory.sample()

        # Get Q values for current states
        # gather selects the Q-value corresponding to the action taken
        q_values = self.policy_network(states).gather(1, actions)

        # Get max Q values for next states from Target Network
        with torch.no_grad():
            # FIX: Use n_states (next states), NOT states
            q_next = self.target_network(n_states).max(1)[0].unsqueeze(1)

        # Compute target Q values
        target_q = rewards + (self.gamma * q_next * (1 - dones))
        
        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    #method to update the target network
    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

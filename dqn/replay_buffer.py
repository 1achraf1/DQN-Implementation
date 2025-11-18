import numpy as np
import random
import torch
from collections import deque, namedtuple

Transition = namedtuple('Transition',['state','action','reward','next_state','done'])

class ReplayBuffer:
    def __init__(self,buffer_size, batch_size,device):
        self.buffer = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.device = device

    #method to add transition to the replay buffer
    def push(self,s,a,r,next_s,d):
        T = Transition(s,a,r,next_s,d)
        self.buffer.append(T)
      
    #method to get a training sample from the buffer
    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        actions = torch.from_numpy(np.array(batch.action)).long().to(self.device).unsqueeze(-1)
        rewards = torch.from_numpy(np.array(batch.reward)).float().to(self.device).unsqueeze(-1)
        n_states = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        dones = torch.from_numpy(np.array(batch.done).astype(np.uint8)).float().to(self.device).unsqueeze(-1)

        return states, actions, rewards, n_states, dones
    def __len__(self):
        return len(self.buffer)
      

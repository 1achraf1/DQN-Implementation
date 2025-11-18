import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnetwork(nn.Module):
    def __init__(self,state_size,action_size):
        super().__init__()
        self.layer1 = nn.Linear(state_size,64)
        self.layer2 = nn.Linear(64,action_size)

    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

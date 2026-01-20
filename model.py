import torch
import torch.nn as nn
import torch.nn.functional as F

class SquadroNet(nn.Module):
    def __init__(self):
        super(SquadroNet, self).__init__()
        # Input: 5+5+5+5+5+5+1 = 31 features
        self.fc1 = nn.Linear(31, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Policy Head (Actor): Probability of moving pieces 0-4
        self.policy_head = nn.Linear(128, 5)
        
        # Value Head (Critic): Win probability (-1 to 1)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value
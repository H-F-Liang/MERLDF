from collections import namedtuple
import torch as T
import torch.optim as optim

from torch.nn import functional as F
import torch
import torch.nn as nn


class Dueling_DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Dueling_DQN, self).__init__()
        self.embedding = nn.Embedding(state_dim[0], 4)
        # input [66, 3] 序列长度66 特征维度3
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.Dropout(0.5),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 16 * state_dim[0], 2048),
            nn.ReLU())

        self.V_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU())

        self.A_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU())

        self.V = nn.Linear(1024, 1)
        self.A = nn.Linear(1024, action_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(dim=0)
        elif x.dim() == 4:
            x = x.squeeze(dim=1)
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        V_fc = self.V_fc(x)

        A_fc = self.A_fc(x)

        V = self.V(V_fc)

        A = self.A(A_fc)

        A_ave = torch.mean(A, dim=1, keepdim=True)

        x = A + V - A_ave
        return x

    def predict(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()


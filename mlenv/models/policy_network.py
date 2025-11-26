# mlenv/models/policy_network.py
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Простая MLP-сеть для DQN: Q(s, a) для каждого курьера.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, obs_dim)
        return self.net(x)

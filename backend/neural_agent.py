# backend/neural_agent.py

import torch
import torch.nn as nn
import numpy as np

from pathlib import Path


class NeuralPolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden=(128, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NeuralCourierAgent:
    def __init__(self, model_path: str, obs_dim: int, num_couriers: int, device="cpu"):
        self.obs_dim = obs_dim
        self.num_actions = num_couriers
        self.device = torch.device(device)

        self.net = NeuralPolicyNetwork(obs_dim, num_couriers).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.eval()

    def choose(self, obs: np.ndarray, available_actions):
        """
        obs: numpy vector [obs_dim]
        available_actions: список ID курьеров, которые доступны
        """

        # forward pass
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.net(x)[0].cpu().numpy()

        # availability mask
        masked_q = {a: q[a] for a in available_actions}

        # select action with maximum Q
        best = max(masked_q.items(), key=lambda x: x[1])[0]
        return int(best)

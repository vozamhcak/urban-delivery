# mlenv/models/dqn_agent.py
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .policy_network import PolicyNetwork


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = torch.device(device)

        self.q_net = PolicyNetwork(obs_dim, num_actions).to(self.device)
        self.target_net = PolicyNetwork(obs_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t)[0].cpu().numpy()
        return int(np.argmax(q_values))

    def update(
        self,
        batch_obs: np.ndarray,
        batch_actions: np.ndarray,
        batch_rewards: np.ndarray,
        batch_next_obs: np.ndarray,
        batch_dones: np.ndarray,
    ) -> float:
        """
        Один шаг обучения по mini-batch.
        """
        obs_t = torch.tensor(batch_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(batch_next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(batch_dones, dtype=torch.float32, device=self.device)

        # Q(s, a) для выбранных действий
        q_values = self.q_net(obs_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # таргеты: r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_net(next_obs_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

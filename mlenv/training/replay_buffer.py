# mlenv/training/replay_buffer.py
from typing import Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.obs_dim = obs_dim

        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity,), dtype=np.int64)
        self.rews_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)

        self.size = 0
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = action
        self.rews_buf[self.ptr] = reward
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs_buf[idxs],
            self.acts_buf[idxs],
            self.rews_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs],
        )

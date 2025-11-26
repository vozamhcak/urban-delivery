# mlenv/utils/config.py
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    num_couriers: int = 10
    orders_per_minute: float = 4.0
    max_episode_orders: int = 100
    max_episode_time: float = 1800.0

    total_episodes: int = 500
    max_steps_per_episode: int = 200  # защитный лимит

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 100_000
    start_learning_after: int = 1_000
    train_every_step: int = 1
    target_update_every: int = 500

    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_episodes: int = 300

    device: str = "cpu"
    invalid_action_penalty: float = -1.0

    checkpoint_path: str = "mlenv_checkpoints/dqn_policy.pt"

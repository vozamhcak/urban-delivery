# mlenv/env/delivery_env.py
import math
import random
from pathlib import Path
from typing import Dict, Set, Tuple, Any

import numpy as np

from backend.models import Config, OrderStatus   # ОСТАВЛЯЕМ — это безопасно
from .observation_builder import build_observation


class DeliveryEnv:
    """
    RL-среда поверх Simulation.
    Агент принимает решение: какому курьеру назначить новый заказ.
    """

    def __init__(
        self,
        num_couriers: int = 10,
        orders_per_minute: float = 4.0,
        max_episode_orders: int = 100,
        max_episode_time: float = 1800.0,
        courier_speed: float = 1.5,
        invalid_action_penalty: float = -1.0,
        gamma: float = 0.99,
    ):
        self.num_couriers = num_couriers
        self.orders_per_minute = orders_per_minute
        self.max_episode_orders = max_episode_orders
        self.max_episode_time = max_episode_time
        self.courier_speed = courier_speed
        self.invalid_action_penalty = invalid_action_penalty
        self.gamma = gamma

        self.observation_dim = 5 + 7 * self.num_couriers
        self.num_actions = self.num_couriers

        self.sim = None
        self.current_time = 0.0
        self.current_order = None
        self.episode_orders_generated = 0
        self.done = False

        self.order_creation_time: Dict[int, float] = {}
        self.completed_orders: Set[int] = set()

        self._data_dir = Path(__file__).resolve().parents[2] / "backend" / "data"

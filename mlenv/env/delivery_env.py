# mlenv/env/delivery_env.py
import math
import random
from pathlib import Path
from typing import Dict, Set, Tuple, Any

import numpy as np

from backend.models import Config, OrderStatus
from .observation_builder import build_observation


class DeliveryEnv:
    """
    RL environment on top of Simulation.
    Agent makes decision: which courier to assign the order to.
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

        # observation/action dimensions
        self.observation_dim = 5 + 7 * self.num_couriers
        self.num_actions = self.num_couriers

        self.sim = None
        self.current_time = 0.0
        self.current_order = None
        self.episode_orders_generated = 0
        self.done = False

        self.order_creation_time: Dict[int, float] = {}
        self.completed_orders: Set[int] = set()

        # path to backend/data
        self._data_dir = Path(__file__).resolve().parents[2] / "backend" / "data"

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def reset(self) -> np.ndarray:
        """Start of new episode."""

        # IMPORT ONLY HERE — THIS BREAKS THE CYCLE FROM simulation.py
        from backend.simulation import Simulation

        self.sim = Simulation(self._data_dir)

        cfg = Config(
            num_couriers=self.num_couriers,
            courier_speed=self.courier_speed,
            orders_per_minute=0.0,  # we generate orders ourselves
        )
        self.sim.set_config(cfg)

        self.sim.orders = []
        self.current_time = 0.0
        self.episode_orders_generated = 0
        self.done = False

        self.order_creation_time.clear()
        self.completed_orders.clear()

        # create first order
        self.current_order = self._create_random_order()
        obs = build_observation(self.sim, self.current_order, self.num_couriers)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        One environment step:
        1) assign order to action-courier
        2) advance simulation
        3) collect rewards
        4) create new order
        """
        if self.done:
            raise RuntimeError("Episode finished — call reset().")

        total_reward = 0.0
        info = {}

        # 1) assignment
        self._assign_order_with_action(
            action,
            invalid_flag_out=lambda bad: info.update({"invalid_action": bad})
        )
        if info.get("invalid_action", False):
            total_reward += self.invalid_action_penalty

        # 2) advance simulation
        dt = self._sample_interarrival_time()
        total_reward += self._advance_and_collect_rewards(dt)
        self.current_time += dt

        self.episode_orders_generated += 1

        # 3) episode end
        if (
            self.episode_orders_generated >= self.max_episode_orders
            or self.current_time >= self.max_episode_time
        ):
            self.done = True
            return np.zeros(self.observation_dim, dtype=np.float32), total_reward, True, info

        # 4) create new order
        self.current_order = self._create_random_order()
        next_obs = build_observation(self.sim, self.current_order, self.num_couriers)
        return next_obs, total_reward, False, info

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _sample_interarrival_time(self) -> float:
        """Exponential distribution between orders."""
        lam = self.orders_per_minute / 60.0
        if lam <= 0:
            return 10.0
        return random.expovariate(lam)

    def _advance_and_collect_rewards(self, dt_total: float) -> float:
        """Advance simulation and collect rewards."""
        reward_sum = 0.0
        dt_step = 1.0
        t = dt_total

        while t > 0:
            dt = min(dt_step, t)
            t -= dt

            # move couriers
            for c in self.sim.couriers:
                self.sim._move_courier_along_path(c, dt)
                self.sim._update_courier_state(c)

            reward_sum += self._collect_completion_rewards()

        return reward_sum

    def _collect_completion_rewards(self) -> float:
        """Reward for completed orders."""
        reward = 0.0
        for o in self.sim.orders:
            if o.status == "done" and o.id not in self.completed_orders:
                self.completed_orders.add(o.id)

                created = self.order_creation_time.get(o.id, 0.0)
                delivery_time = max(0, self.current_time - created)

                # MAKE REWARD STRONGER — BETTER FOR GRAPHS
                base = 20.0
                time_penalty = -0.2 * (delivery_time / 60.0)

                reward += base + time_penalty

        return reward

    def _create_random_order(self) -> OrderStatus:
        """Create order without auto-assignment."""
        shop = random.choice(self.sim.shops)
        house = random.choice(self.sim.houses)
        weight = max(0.5, random.gauss(3.0, 1.0))

        order = OrderStatus(
            id=self.sim.next_order_id,
            shop_id=shop.id,
            house_id=house.id,
            weight=weight,
        )
        self.sim.next_order_id += 1
        self.sim.orders.append(order)
        self.order_creation_time[order.id] = self.current_time
        return order

    def _assign_order_with_action(self, action: int, invalid_flag_out):
        """Assigns order to courier, checks correctness."""
        order = self.current_order
        weight = order.weight or 0.0

        eligible = []
        for idx, c in enumerate(self.sim.couriers):
            cap = getattr(c, "max_capacity", 10.0)
            load = getattr(c, "current_load", 0.0)
            if cap - load >= weight:
                eligible.append(idx)

        if not eligible:
            invalid_flag_out(True)
            return

        if action not in eligible:
            invalid_flag_out(True)
            chosen = random.choice(eligible)
        else:
            invalid_flag_out(False)
            chosen = action

        courier = self.sim.couriers[chosen]

        if not hasattr(courier, "current_load"):
            courier.current_load = 0.0
        courier.current_load = min(courier.max_capacity, courier.current_load + weight)

        order.assigned_courier_id = courier.id

        if courier.current_order_id is None:
            self.sim._start_order_for_courier(courier, order)
        else:
            order.status = "waiting"

import random
import time
import math
from pathlib import Path
from typing import List
import numpy as np

from backend.models import (
    House,
    Shop,
    Courier,
    OrderStatus,
    LogEntry,
    Config,
    StateResponse,
    QLearningAgent
)
from backend.graph import Graph, load_road_network, load_points, attach_nearest_nodes

from mlenv.env.observation_builder import build_observation

from backend.neural_agent import NeuralCourierAgent
from mlenv.env.observation_builder import build_observation

class Simulation:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

        self.road_network = load_road_network(self.data_dir / "roads.json")
        self.graph = Graph(self.road_network)
        self.houses: List[House] = load_points(self.data_dir / "houses.json", House)
        self.shops: List[Shop] = load_points(self.data_dir / "shops.json", Shop)
        attach_nearest_nodes(self.graph, self.houses, self.shops)

        self.config = Config()
        self.couriers: List[Courier] = []
        self.orders: List[OrderStatus] = []
        self.logs: List[LogEntry] = []
        
        # RL agent
        # self.agent = QLearningAgent()
        # ...
        self.agent = None  # will replace below
        self.use_neural_agent = False
        self.neural_agent = None

        
        # Statistics for rewards
        self.order_statistics = {
            'total_orders': 0,
            'completed_orders': 0,
            'total_delivery_time': 0.0,
            'order_start_times': {}  # order_id -> start_time
        }

        self.next_order_id = 1
        self._init_couriers()
        self.last_time = time.time()
        
    def _build_neural_observation(self, order: OrderStatus) -> np.ndarray:
        """
        Builds observation in mlenv format so that the neural agent
        makes decisions identically to how it was trained.
        """
        return build_observation(
            self,
            order,
            num_couriers=self.config.num_couriers
        )
        
    def load_neural_agent(self, path: str, obs_dim: int, num_couriers: int):
        self.neural_agent = NeuralCourierAgent(
            model_path=path,
            obs_dim=obs_dim,
            num_couriers=num_couriers,
            device="cpu"
        )
        self.use_neural_agent = True
        self.log("Neural agent loaded.")


    def _init_couriers(self):
        self.couriers = []

        base_shop = self.shops[0]
        for i in range(self.config.num_couriers):
            self.couriers.append(
                Courier(
                    id=i + 1,
                    x=base_shop.x,
                    y=base_shop.y,
                    state="idle",
                    current_order_id=None,
                    path=[],
                    path_index=0,
                )
            )

    def set_config(self, cfg: Config):
        self.config = cfg

        # Initialize couriers
        self._init_couriers()

        # If old QLearningAgent is used — update its parameters
        if self.agent is not None:
            if hasattr(self.agent, "learning_rate"):
                self.agent.learning_rate = cfg.learning_rate
            if hasattr(self.agent, "discount_factor"):
                self.agent.discount_factor = cfg.discount_factor
            if hasattr(self.agent, "exploration_rate"):
                self.agent.exploration_rate = cfg.exploration_rate

        self.log(
            f"Configuration updated: couriers={cfg.num_couriers}, speed={cfg.courier_speed} m/s, "
            f"orders/min={cfg.orders_per_minute:.2f}"
        )


    def log(self, msg: str):
        self.logs.append(LogEntry(ts=time.time(), message=msg))
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]

    def maybe_generate_orders(self, dt: float):
        lam = self.config.orders_per_minute
        if lam <= 0:
            return
        p = lam * dt / 60.0
        if random.random() < p:
            self.create_random_order()

    def create_random_order(self):
        shop = random.choice(self.shops)
        house = random.choice(self.houses)
        
        # NEW: random weight ~ 2–4 kg, with rare outliers
        weight = max(0.5, random.gauss(3.0, 1.0))  # mean ~3 kg, sd=1, but not less than 0.5
        
        order = OrderStatus(
            id=self.next_order_id,
            shop_id=shop.id,
            house_id=house.id,
            weight=weight,
        )
        self.next_order_id += 1
        self.orders.append(order)
        self.order_statistics['total_orders'] += 1
        self.order_statistics['order_start_times'][order.id] = time.time()
        
        self.log(
            f"Order #{order.id} created: shop {order.shop_id} → house {order.house_id}, "
            f"weight ≈{order.weight:.1f} kg"
        )

        # Assign order considering capacity
        self.assign_order_with_rl(order)

    def _assign_order_to_courier(self, courier: Courier, order: OrderStatus):
        """Carefully assigns an order to a specific courier."""
        weight = getattr(order, "weight", 0.0)

        order.assigned_courier_id = courier.id
        courier.current_load += weight

        # If courier is free → immediately start the trip for the order
        if courier.current_order_id is None:
            self._start_order_for_courier(courier, order)
        else:
            # Otherwise add order to courier's queue
            order.status = "waiting"
            self.log(
                f"Courier #{courier.id} added order #{order.id} to queue "
                f"(load {courier.current_load:.1f}/{courier.max_capacity:.1f} kg)"
            )


    def _calculate_distance_cost(self, courier: Courier, order: OrderStatus) -> float:
        """Calculates the cost of assigning a courier to an order based on distances"""
        shop = self._shop_by_id(order.shop_id)
        house = self._house_by_id(order.house_id)
        
        # Current courier position
        courier_node_id = self.graph.nearest_node_id(courier.x, courier.y)
        
        # Distance from courier to shop
        dist_to_shop = self.graph.shortest_path_length(courier_node_id, shop.node_id)
        
        # Distance from shop to house
        dist_shop_to_house = self.graph.shortest_path_length(shop.node_id, house.node_id)
        
        return dist_to_shop + dist_shop_to_house

    def assign_order_with_rl(self, order: OrderStatus):
        """
        Courier selection.
        If neural agent is enabled — it is used.
        Otherwise — fallback: random among those who fit by capacity.
        """
        # If order is already assigned or completed — do nothing
        if order.assigned_courier_id is not None or order.status == "done":
            return

        weight = getattr(order, "weight", 0.0)

        # 1. Couriers who can take the order by capacity
        eligible = []
        for c in self.couriers:
            if getattr(c, "max_capacity", 10.0) - getattr(c, "current_load", 0.0) >= weight:
                eligible.append(c)

        if not eligible:
            self.log(
                f"Order #{order.id} (≈{order.weight:.1f} kg) waiting for courier "
                f"(no free capacity)"
            )
            return

        # ----------------------------------------------------------------------
        # 2. NEURAL AGENT
        # ----------------------------------------------------------------------
        if self.use_neural_agent and self.neural_agent is not None:
            obs = self._build_neural_observation(order)

            # neural network chooses action in space {0..N-1}
            available_actions = [c.id - 1 for c in eligible]   # convert courier.id → index
            chosen_action = self.neural_agent.choose(obs, available_actions)

            # convert back to courier.id
            chosen_courier = next(c for c in self.couriers if c.id == chosen_action + 1)

            self._assign_order_to_courier(chosen_courier, order)
            return

        # ----------------------------------------------------------------------
        # 3. FALLBACK: random selection of suitable courier
        # ----------------------------------------------------------------------
        courier = random.choice(eligible)
        self._assign_order_to_courier(courier, order)


    def _calculate_reward(self, order: OrderStatus, delivery_time: float) -> float:
        """Calculates reward for order delivery"""
        # Base reward for completing order
        base_reward = 10.0
        
        # Time penalty (faster is better)
        time_penalty = -delivery_time / 60.0  # normalize to minutes
        
        # Bonus for fast delivery
        speed_bonus = max(0, 5.0 - delivery_time / 60.0)
        
        return base_reward + time_penalty + speed_bonus

    def _update_rl_agent(self, order: OrderStatus):
        """Updates RL agent after order completion"""
        if not hasattr(order, 'rl_state_key') or not hasattr(order, 'rl_chosen_courier'):
            return
            
        delivery_time = time.time() - self.order_statistics['order_start_times'].get(order.id, time.time())
        reward = self._calculate_reward(order, delivery_time)
        
        # Next state (current courier positions)
        idle_couriers = [c for c in self.couriers if c.state == "idle"]
        next_available_couriers = [c.id for c in idle_couriers]
        
        # Simplified next state
        next_state_key = "default_state"
        
        self.agent.update(
            order.rl_state_key,
            order.rl_chosen_courier,
            reward,
            next_state_key,
            next_available_couriers
        )
        
        # Decrease exploration rate
        self.agent.exploration_rate *= self.config.exploration_decay

    def _shop_by_id(self, shop_id: int) -> Shop:
        for s in self.shops:
            if s.id == shop_id:
                return s
        raise KeyError(shop_id)

    def _house_by_id(self, house_id: int) -> House:
        for h in self.houses:
            if h.id == house_id:
                return h
        raise KeyError(house_id)

    def _road_node_by_id(self, node_id: int):
        for n in self.road_network.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def _move_courier_along_path(self, courier: Courier, dt: float):
        if not courier.path or courier.path_index >= len(courier.path):
            return

        speed = self.config.courier_speed
        remaining = speed * dt

        while remaining > 0 and courier.path_index < len(courier.path):
            target_node_id = courier.path[courier.path_index]
            target_node = self._road_node_by_id(target_node_id)
            dx = target_node.x - courier.x
            dy = target_node.y - courier.y
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                courier.path_index += 1
                continue
            if dist <= remaining:
                courier.x = target_node.x
                courier.y = target_node.y
                remaining -= dist
                courier.path_index += 1
            else:
                ratio = remaining / dist
                courier.x += dx * ratio
                courier.y += dy * ratio
                remaining = 0

    def _update_courier_state(self, courier: Courier):
        if courier.current_order_id is None:
            return
        try:
            order = next(o for o in self.orders if o.id == courier.current_order_id)
        except StopIteration:
            return

        if courier.path_index < len(courier.path):
            return

        if order.status == "to_shop":
            house = self._house_by_id(order.house_id)
            start_node = self.graph.nearest_node_id(courier.x, courier.y)
            assert house.node_id is not None
            path = self.graph.shortest_path(start_node, house.node_id)
            courier.path = path
            courier.path_index = 0
            order.status = "to_house"
            self.log(
                f"Courier #{courier.id} picked up order #{order.id} from shop {order.shop_id}, heading to house {order.house_id}"
            )
        elif order.status == "to_house":
            order.status = "done"
            delivery_time = time.time() - self.order_statistics['order_start_times'].get(order.id, time.time())
            
            self.log(
                f"Courier #{courier.id} delivered order #{order.id} to house {order.house_id} "
                f"in {delivery_time:.1f}s"
            )
            
            # RL agent (might not update anything yet, that's ok)
            self._update_rl_agent(order)
            
            self.order_statistics['completed_orders'] += 1
            self.order_statistics['total_delivery_time'] += delivery_time
            
            # Clear temporary data
            if order.id in self.order_statistics['order_start_times']:
                del self.order_statistics['order_start_times'][order.id]

            # NEW: free up part of courier's capacity
            if hasattr(order, "weight"):
                courier.current_load = max(0.0, courier.current_load - order.weight)

            courier.current_order_id = None
            courier.path = []
            courier.path_index = 0

            # NEW: are there more orders already assigned to this courier but not started?
            my_waiting_orders = [
                o
                for o in self.orders
                if o.assigned_courier_id == courier.id and o.status == "waiting"
            ]
            if my_waiting_orders:
                # Take the earliest by id (roughly FIFO)
                next_order = sorted(my_waiting_orders, key=lambda o: o.id)[0]
                self._start_order_for_courier(courier, next_order)
                return  # don't consider courier idle, already heading for next order

            # If no personal queue — courier is truly free
            courier.state = "idle"

            # NEW: after freeing courier, try to distribute other waiting orders
            self._assign_waiting_orders()


    def step(self):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 0.01
        self.last_time = now

        self.maybe_generate_orders(dt)

        for courier in self.couriers:
            self._move_courier_along_path(courier, dt)
            self._update_courier_state(courier)

        self.orders = self.orders[-200:]

    def get_rl_stats(self) -> dict:
        """Returns RL/neural agent statistics"""
        if self.agent is not None and hasattr(self.agent, "q_table"):
            exploration_rate = getattr(self.agent, "exploration_rate", 0.0)
            q_table_size = len(self.agent.q_table)
        else:
            # for neural agent show placeholders
            exploration_rate = 0.0
            q_table_size = 0

        return {
            'exploration_rate': exploration_rate,
            'q_table_size': q_table_size,
            'completed_orders': self.order_statistics['completed_orders'],
            'total_orders': self.order_statistics['total_orders'],
            'avg_delivery_time': (
                self.order_statistics['total_delivery_time'] / self.order_statistics['completed_orders']
                if self.order_statistics['completed_orders'] > 0 else 0
            )
        }


    def state(self) -> StateResponse:
        return StateResponse(
            houses=self.houses,
            shops=self.shops,
            couriers=self.couriers,
            orders=self.orders,
            logs=self.logs[-100:],
            road_nodes=self.road_network.nodes,
        )
        
    def _start_order_for_courier(self, courier: Courier, order: OrderStatus):
        """Starts actual order execution by courier (heading to shop)."""
        shop = self._shop_by_id(order.shop_id)
        assert shop.node_id is not None

        start_node = self.graph.nearest_node_id(courier.x, courier.y)
        path = self.graph.shortest_path(start_node, shop.node_id)

        courier.state = "to_shop"
        courier.current_order_id = order.id
        courier.path = path
        courier.path_index = 0
        order.status = "to_shop"

        self.log(
            f"Courier #{courier.id} assigned to order #{order.id}: "
            f"shop {order.shop_id} → house {order.house_id} "
            f"(load {courier.current_load:.1f}/{courier.max_capacity:.1f} kg)"
        )

    def _assign_waiting_orders(self):
        """Try to distribute all unassigned orders to couriers considering capacity."""
        waiting_orders = [
            o for o in self.orders
            if o.status == "waiting" and o.assigned_courier_id is None
        ]
        for order in waiting_orders:
            self.assign_order_with_rl(order)

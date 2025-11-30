from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel
import numpy as np


class Point(BaseModel):
    x: float
    y: float


class House(BaseModel):
    id: int
    x: float
    y: float
    node_id: Optional[int] = None


class Shop(BaseModel):
    id: int
    x: float
    y: float
    node_id: Optional[int] = None


class RoadNode(BaseModel):
    id: int
    x: float
    y: float


class RoadNetwork(BaseModel):
    nodes: List[RoadNode]
    edges: List[List[int]]  # [from_id, to_id]


CourierState = Literal["idle", "to_shop", "to_house"]


class Courier(BaseModel):
    id: int
    x: float
    y: float
    state: CourierState = "idle"
    current_order_id: Optional[int] = None
    path: List[int] = []       # list of road node IDs along the path
    path_index: int = 0        # index of current target in path

    # NEW: capacity and current load
    max_capacity: float = 10.0  # kg
    current_load: float = 0.0   # kg



class OrderStatus(BaseModel):
    id: int
    shop_id: int
    house_id: int
    assigned_courier_id: Optional[int] = None
    status: Literal["waiting", "to_shop", "to_house", "done"] = "waiting"
    
    # NEW: order weight
    weight: float = 0.0



class LogEntry(BaseModel):
    ts: float
    message: str


class Config(BaseModel):
    num_couriers: int = 5
    courier_speed: float = 1.5  # m/s
    orders_per_minute: float = 2.0
    # RL parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995


class StateResponse(BaseModel):
    houses: List[House]
    shops: List[Shop]
    couriers: List[Courier]
    orders: List[OrderStatus]
    logs: List[LogEntry]
    road_nodes: List[RoadNode]


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table: Dict[str, Dict[int, float]] = {}
        
    def get_state_key(self, courier_locations, shop_location, house_location):
        """Creates a state key based on positions"""
        # Simplified state representation - nearest nodes to key points
        return f"{shop_location}_{house_location}"
    
    def choose_action(self, state_key: str, available_couriers: List[int]) -> int:
        """Action (courier) selection with exploration/exploitation"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {courier_id: 0.0 for courier_id in available_couriers}
        
        # Exploration: random selection
        if np.random.random() < self.exploration_rate:
            return np.random.choice(available_couriers)
        
        # Exploitation: choose best action
        q_values = self.q_table[state_key]
        # Filter only available couriers
        available_q = {k: v for k, v in q_values.items() if k in available_couriers}
        if not available_q:
            return np.random.choice(available_couriers)
        
        max_q = max(available_q.values())
        # Select among all with maximum Q-value
        best_actions = [courier_id for courier_id, q_val in available_q.items() if q_val == max_q]
        return np.random.choice(best_actions)
    
    def update(self, state_key: str, action: int, reward: float, next_state_key: str, next_available_couriers: List[int]):
        """Updates Q-table based on received reward"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
            
        # Maximum Q-value for next state
        next_max = 0.0
        if next_state_key in self.q_table and next_available_couriers:
            next_q_values = {k: v for k, v in self.q_table[next_state_key].items() if k in next_available_couriers}
            if next_q_values:
                next_max = max(next_q_values.values())
        
        # Q-learning formula
        current_q = self.q_table[state_key][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max - current_q)
        self.q_table[state_key][action] = new_q
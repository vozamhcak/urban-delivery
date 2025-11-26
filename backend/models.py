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
    path: List[int] = []       # список id road nodes по пути
    path_index: int = 0        # индекс текущей цели в path

    # NEW: вместимость и текущая нагрузка
    max_capacity: float = 10.0  # кг
    current_load: float = 0.0   # кг



class OrderStatus(BaseModel):
    id: int
    shop_id: int
    house_id: int
    assigned_courier_id: Optional[int] = None
    status: Literal["waiting", "to_shop", "to_house", "done"] = "waiting"
    
    # NEW: вес заказа
    weight: float = 0.0



class LogEntry(BaseModel):
    ts: float
    message: str


class Config(BaseModel):
    num_couriers: int = 5
    courier_speed: float = 1.5  # м/с
    orders_per_minute: float = 2.0
    # RL параметры
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
        """Создает ключ состояния на основе позиций"""
        # Упрощенное представление состояния - ближайшие узлы к ключевым точкам
        return f"{shop_location}_{house_location}"
    
    def choose_action(self, state_key: str, available_couriers: List[int]) -> int:
        """Выбор действия (курьера) с учетом exploration/exploitation"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {courier_id: 0.0 for courier_id in available_couriers}
        
        # Exploration: случайный выбор
        if np.random.random() < self.exploration_rate:
            return np.random.choice(available_couriers)
        
        # Exploitation: выбор лучшего действия
        q_values = self.q_table[state_key]
        # Фильтруем только доступных курьеров
        available_q = {k: v for k, v in q_values.items() if k in available_couriers}
        if not available_q:
            return np.random.choice(available_couriers)
        
        max_q = max(available_q.values())
        # Выбираем среди всех с максимальным Q-value
        best_actions = [courier_id for courier_id, q_val in available_q.items() if q_val == max_q]
        return np.random.choice(best_actions)
    
    def update(self, state_key: str, action: int, reward: float, next_state_key: str, next_available_couriers: List[int]):
        """Обновление Q-table на основе полученной награды"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
            
        # Максимальное Q-value для следующего состояния
        next_max = 0.0
        if next_state_key in self.q_table and next_available_couriers:
            next_q_values = {k: v for k, v in self.q_table[next_state_key].items() if k in next_available_couriers}
            if next_q_values:
                next_max = max(next_q_values.values())
        
        # Q-learning формула
        current_q = self.q_table[state_key][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max - current_q)
        self.q_table[state_key][action] = new_q
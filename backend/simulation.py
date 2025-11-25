import random
import time
import math
from pathlib import Path
from typing import List
import numpy as np

from models import (
    House,
    Shop,
    Courier,
    OrderStatus,
    LogEntry,
    Config,
    StateResponse,
    QLearningAgent
)
from graph import Graph, load_road_network, load_points, attach_nearest_nodes


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
        
        # RL агент
        self.agent = QLearningAgent()
        
        # Статистика для наград
        self.order_statistics = {
            'total_orders': 0,
            'completed_orders': 0,
            'total_delivery_time': 0.0,
            'order_start_times': {}  # order_id -> start_time
        }

        self.next_order_id = 1
        self._init_couriers()
        self.last_time = time.time()

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
        # Обновляем параметры агента
        self.agent.learning_rate = cfg.learning_rate
        self.agent.discount_factor = cfg.discount_factor
        self.agent.exploration_rate = cfg.exploration_rate
        
        self._init_couriers()
        self.log(
            f"Конфигурация обновлена: курьеров={cfg.num_couriers}, v={cfg.courier_speed} м/с, "
            f"заказов/мин={cfg.orders_per_minute:.2f}"
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
        order = OrderStatus(
            id=self.next_order_id,
            shop_id=shop.id,
            house_id=house.id,
        )
        self.next_order_id += 1
        self.orders.append(order)
        self.order_statistics['total_orders'] += 1
        self.order_statistics['order_start_times'][order.id] = time.time()
        
        self.log(
            f"Создан заказ #{order.id}: магазин {order.shop_id} → дом {order.house_id}"
        )
        self.assign_order_with_rl(order)

    def _calculate_distance_cost(self, courier: Courier, order: OrderStatus) -> float:
        """Рассчитывает стоимость назначения курьера на заказ на основе расстояний"""
        shop = self._shop_by_id(order.shop_id)
        house = self._house_by_id(order.house_id)
        
        # Текущая позиция курьера
        courier_node_id = self.graph.nearest_node_id(courier.x, courier.y)
        
        # Расстояние от курьера до магазина
        dist_to_shop = self.graph.shortest_path_length(courier_node_id, shop.node_id)
        
        # Расстояние от магазина до дома
        dist_shop_to_house = self.graph.shortest_path_length(shop.node_id, house.node_id)
        
        return dist_to_shop + dist_shop_to_house

    def assign_order_with_rl(self, order: OrderStatus):
        """Распределение заказа с использованием RL"""
        idle_couriers = [c for c in self.couriers if c.state == "idle"]
        if not idle_couriers:
            self.log(f"Заказ #{order.id} ожидает свободного курьера")
            return

        shop = self._shop_by_id(order.shop_id)
        house = self._house_by_id(order.house_id)
        
        # Создаем ключ состояния
        state_key = self.agent.get_state_key(
            [self.graph.nearest_node_id(c.x, c.y) for c in idle_couriers],
            shop.node_id,
            house.node_id
        )
        
        available_courier_ids = [c.id for c in idle_couriers]
        
        # Выбираем курьера с помощью RL
        chosen_courier_id = self.agent.choose_action(state_key, available_courier_ids)
        courier = next(c for c in idle_couriers if c.id == chosen_courier_id)
        
        # Сохраняем информацию для будущего обновления Q-table
        order.rl_state_key = state_key
        order.rl_chosen_courier = chosen_courier_id
        
        # Назначаем заказ
        order.assigned_courier_id = courier.id
        order.status = "to_shop"

        shop = self._shop_by_id(order.shop_id)
        assert shop.node_id is not None
        start_node = self.graph.nearest_node_id(courier.x, courier.y)
        path = self.graph.shortest_path(start_node, shop.node_id)

        courier.state = "to_shop"
        courier.current_order_id = order.id
        courier.path = path
        courier.path_index = 0
        
        self.log(
            f"RL назначил курьера #{courier.id} на заказ #{order.id} (расстояние: {self._calculate_distance_cost(courier, order):.1f}м)"
        )

    def _calculate_reward(self, order: OrderStatus, delivery_time: float) -> float:
        """Рассчитывает награду за доставку заказа"""
        # Базовая награда за завершение заказа
        base_reward = 10.0
        
        # Штраф за время доставки (чем быстрее, тем лучше)
        time_penalty = -delivery_time / 60.0  # нормализуем к минутам
        
        # Бонус за быструю доставку
        speed_bonus = max(0, 5.0 - delivery_time / 60.0)
        
        return base_reward + time_penalty + speed_bonus

    def _update_rl_agent(self, order: OrderStatus):
        """Обновляет RL агента после завершения заказа"""
        if not hasattr(order, 'rl_state_key') or not hasattr(order, 'rl_chosen_courier'):
            return
            
        delivery_time = time.time() - self.order_statistics['order_start_times'].get(order.id, time.time())
        reward = self._calculate_reward(order, delivery_time)
        
        # Следующее состояние (текущие позиции курьеров)
        idle_couriers = [c for c in self.couriers if c.state == "idle"]
        next_available_couriers = [c.id for c in idle_couriers]
        
        # Упрощенное следующее состояние
        next_state_key = "default_state"
        
        self.agent.update(
            order.rl_state_key,
            order.rl_chosen_courier,
            reward,
            next_state_key,
            next_available_couriers
        )
        
        # Уменьшаем exploration rate
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
                f"Курьер #{courier.id} забрал заказ #{order.id} из магазина {order.shop_id}, едет в дом {order.house_id}"
            )
        elif order.status == "to_house":
            order.status = "done"
            delivery_time = time.time() - self.order_statistics['order_start_times'].get(order.id, time.time())
            
            self.log(
                f"Курьер #{courier.id} доставил заказ #{order.id} в дом {order.house_id} за {delivery_time:.1f}с"
            )
            
            # Обновляем RL агента
            self._update_rl_agent(order)
            
            self.order_statistics['completed_orders'] += 1
            self.order_statistics['total_delivery_time'] += delivery_time
            
            # Очищаем временные данные
            if order.id in self.order_statistics['order_start_times']:
                del self.order_statistics['order_start_times'][order.id]
            
            courier.state = "idle"
            courier.current_order_id = None
            courier.path = []
            courier.path_index = 0

            # Назначаем следующий ожидающий заказ
            waiting_orders = [
                o
                for o in self.orders
                if o.status == "waiting" and o.assigned_courier_id is None
            ]
            if waiting_orders:
                self.assign_order_with_rl(waiting_orders[0])

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
        """Возвращает статистику RL обучения"""
        return {
            'exploration_rate': self.agent.exploration_rate,
            'q_table_size': len(self.agent.q_table),
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
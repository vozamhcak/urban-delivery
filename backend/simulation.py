import random
import time
import math
from pathlib import Path
from typing import List

from models import (
    House,
    Shop,
    Courier,
    OrderStatus,
    LogEntry,
    Config,
    StateResponse,
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
        self.log(
            f"Создан заказ #{order.id}: магазин {order.shop_id} → дом {order.house_id}"
        )
        self.assign_order(order)


    def assign_order(self, order: OrderStatus):
        idle_couriers = [c for c in self.couriers if c.state == "idle"]
        if not idle_couriers:
            return
        courier = random.choice(idle_couriers)  # пока просто рандомный свободный
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
          f"Курьер #{courier.id} назначен на заказ #{order.id}, движется к магазину {order.shop_id}"
        )

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
            self.log(
                f"Курьер #{courier.id} доставил заказ #{order.id} в дом {order.house_id}"
            )
            courier.state = "idle"
            courier.current_order_id = None
            courier.path = []
            courier.path_index = 0

            waiting_orders = [
                o
                for o in self.orders
                if o.status == "waiting" and o.assigned_courier_id is None
            ]
            if waiting_orders:
                self.assign_order(waiting_orders[0])


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


    def state(self) -> StateResponse:
        return StateResponse(
            houses=self.houses,
            shops=self.shops,
            couriers=self.couriers,
            orders=self.orders,
            logs=self.logs[-100:],
            road_nodes=self.road_network.nodes,
        )

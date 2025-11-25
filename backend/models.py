from typing import List, Optional, Literal
from pydantic import BaseModel


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


class OrderStatus(BaseModel):
    id: int
    shop_id: int
    house_id: int
    assigned_courier_id: Optional[int] = None
    status: Literal["waiting", "to_shop", "to_house", "done"] = "waiting"


class LogEntry(BaseModel):
    ts: float
    message: str


class Config(BaseModel):
    num_couriers: int = 5
    courier_speed: float = 1.5  # м/с
    orders_per_minute: float = 2.0


class StateResponse(BaseModel):
    houses: List[House]
    shops: List[Shop]
    couriers: List[Courier]
    orders: List[OrderStatus]
    logs: List[LogEntry]
    road_nodes: List[RoadNode]  # НОВОЕ поле: узлы дорог (для отрисовки маршрутов)

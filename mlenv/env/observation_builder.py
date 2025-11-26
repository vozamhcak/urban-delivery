# mlenv/env/observation_builder.py
from typing import List
import numpy as np


def build_observation(sim, order, num_couriers: int) -> np.ndarray:
    """
    Строит вектор наблюдения фиксированной длины.
    Формат:
    [ order_features (5),
      courier_1_features (7),
      ...,
      courier_N_features (7) ]
    """

    MAP_NORM = 900.0
    CAPACITY_NORM = 10.0
    DIST_NORM = 1000.0

    shop = next(s for s in sim.shops if s.id == order.shop_id)
    house = next(h for h in sim.houses if h.id == order.house_id)

    order_feats = [
        shop.x / MAP_NORM,
        shop.y / MAP_NORM,
        house.x / MAP_NORM,
        house.y / MAP_NORM,
        (order.weight or 0.0) / CAPACITY_NORM,
    ]

    courier_feats: List[float] = []
    for i in range(num_couriers):
        if i < len(sim.couriers):
            c = sim.couriers[i]

            idle = 1.0 if c.state == "idle" else 0.0
            to_shop = 1.0 if c.state == "to_shop" else 0.0
            to_house = 1.0 if c.state == "to_house" else 0.0

            max_cap = getattr(c, "max_capacity", CAPACITY_NORM)
            cur_load = getattr(c, "current_load", 0.0)
            load_ratio = min(1.0, max(0.0, cur_load / max_cap))

            courier_node = sim.graph.nearest_node_id(c.x, c.y)
            dist_to_shop = sim.graph.shortest_path_length(courier_node, shop.node_id) / DIST_NORM
            dist_to_house = sim.graph.shortest_path_length(courier_node, house.node_id) / DIST_NORM

            queue_len = sum(
                1 for o in sim.orders
                if o.assigned_courier_id == c.id and o.status == "waiting"
            )
            queue_ratio = min(1.0, queue_len / 10.0)

            courier_feats.extend([
                idle,
                to_shop,
                to_house,
                load_ratio,
                dist_to_shop,
                dist_to_house,
                queue_ratio,
            ])
        else:
            courier_feats.extend([0.0] * 7)

    obs = np.array(order_feats + courier_feats, dtype=np.float32)
    return obs

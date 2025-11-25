import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from models import RoadNetwork, RoadNode, House, Shop


class Graph:
    def __init__(self, network: RoadNetwork):
        self.network = network
        self.adj: Dict[int, List[Tuple[int, float]]] = {}
        self._build()

    def _build(self):
        nodes_by_id = {n.id: n for n in self.network.nodes}
        for n in self.network.nodes:
            self.adj[n.id] = []

        for u_id, v_id in self.network.edges:
            u = nodes_by_id[u_id]
            v = nodes_by_id[v_id]
            dist = math.dist((u.x, u.y), (v.x, v.y))
            self.adj[u_id].append((v_id, dist))
            self.adj[v_id].append((u_id, dist))  # считаем дороги двусторонними

    def nearest_node_id(self, x: float, y: float) -> int:
        best_id = self.network.nodes[0].id
        best_dist = float("inf")
        for n in self.network.nodes:
            d = (n.x - x) ** 2 + (n.y - y) ** 2
            if d < best_dist:
                best_dist = d
                best_id = n.id
        return best_id

    def shortest_path(self, start_id: int, end_id: int) -> List[int]:
        import heapq

        dist: Dict[int, float] = {node.id: float("inf") for node in self.network.nodes}
        prev: Dict[int, Optional[int]] = {node.id: None for node in self.network.nodes}
        dist[start_id] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, start_id)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == end_id:
                break
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist[end_id] == float("inf"):
            return [start_id]

        path = []
        cur = end_id
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def shortest_path_length(self, start_id: int, end_id: int) -> float:
        """Вычисляет длину кратчайшего пути между двумя узлами"""
        import heapq

        dist: Dict[int, float] = {node.id: float("inf") for node in self.network.nodes}
        dist[start_id] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, start_id)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == end_id:
                return d
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        return dist.get(end_id, float("inf"))


def load_road_network(path: Path) -> RoadNetwork:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return RoadNetwork(**data)


def load_points(path: Path, cls):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [cls(**item) for item in data]


def attach_nearest_nodes(
    graph: Graph, houses: List[House], shops: List[Shop]
) -> None:
    for h in houses:
        h.node_id = graph.nearest_node_id(h.x, h.y)
    for s in shops:
        s.node_id = graph.nearest_node_id(s.x, s.y)
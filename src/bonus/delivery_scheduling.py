"""Bonus Task 2: Real-World Last-Mile Delivery Routing with Time Windows (TSP-TW).

A single urban delivery vehicle starts at a depot (0, 0) at time t=0 and must
deliver M customer orders, each with coordinates (x_i, y_i), service duration,
and delivery time window [earliest_i, latest_i], then return to the depot.

Objective: minimize Total Cost = Travel Distance + Penalty * Total Lateness.

Compares:
1. Nearest-Neighbor Greedy
2. Earliest-Deadline-First (EDF) Heuristic
3. Exact Branch-and-Bound (optimal reference on small instances)
4. Single-Player MCTS-UCT with Random Rollout
5. Single-Player MCTS-UCT with Urgency-Distance Heuristic Rollout
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Order:
    order_id: int
    x: float
    y: float
    earliest: float
    latest: float
    service_time: float = 1.0


@dataclass
class DeliveryInstance:
    depot: tuple[float, float]
    orders: list[Order]
    speed: float = 1.0
    late_penalty: float = 4.0

    def distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def evaluate_route(self, route: list[int]) -> tuple[float, float, float, int]:
        """Return (total_cost, travel_distance, total_lateness, late_orders)."""
        curr_pos = self.depot
        curr_time = 0.0
        total_dist = 0.0
        total_late = 0.0
        late_count = 0

        for idx in route:
            order = self.orders[idx]
            pos = (order.x, order.y)
            d = self.distance(curr_pos, pos)
            total_dist += d
            arrival = curr_time + d / self.speed
            start_service = max(arrival, order.earliest)
            if arrival > order.latest:
                total_late += arrival - order.latest
                late_count += 1
            curr_time = start_service + order.service_time
            curr_pos = pos

        # Return to depot
        total_dist += self.distance(curr_pos, self.depot)
        total_cost = total_dist + self.late_penalty * total_late
        return total_cost, total_dist, total_late, late_count


@dataclass
class RouteSolution:
    method: str
    route: list[int]
    total_cost: float
    travel_distance: float
    total_lateness: float
    late_orders: int
    elapsed_ms: float
    nodes_explored: int = 0


def generate_instance(num_orders: int = 12, seed: int = 42, grid_size: float = 20.0) -> DeliveryInstance:
    """Generate a reproducible urban last-mile delivery instance with realistic time windows."""
    rng = random.Random(seed)
    depot = (grid_size / 2.0, grid_size / 2.0)
    orders: list[Order] = []
    # Distribute orders around city clusters with staggered time windows
    for i in range(num_orders):
        x = rng.uniform(0.0, grid_size)
        y = rng.uniform(0.0, grid_size)
        dist_from_depot = math.hypot(x - depot[0], y - depot[1])
        # Realistic staggered windows across the shift
        window_center = dist_from_depot + rng.uniform(4.0, num_orders * 4.5)
        half_width = rng.uniform(4.0, 10.0)
        earliest = max(0.0, window_center - half_width)
        latest = window_center + half_width
        orders.append(Order(order_id=i, x=x, y=y, earliest=earliest, latest=latest, service_time=1.0))
    return DeliveryInstance(depot=depot, orders=orders, speed=1.0, late_penalty=4.0)


def solve_nearest_neighbor(inst: DeliveryInstance) -> RouteSolution:
    start = time.perf_counter()
    remaining = set(range(len(inst.orders)))
    curr_pos = inst.depot
    route: list[int] = []
    while remaining:
        nxt = min(
            remaining,
            key=lambda i: inst.distance(curr_pos, (inst.orders[i].x, inst.orders[i].y)),
        )
        route.append(nxt)
        remaining.remove(nxt)
        curr_pos = (inst.orders[nxt].x, inst.orders[nxt].y)
    cost, dist, late, late_cnt = inst.evaluate_route(route)
    return RouteSolution(
        method="NearestNeighbor",
        route=route,
        total_cost=cost,
        travel_distance=dist,
        total_lateness=late,
        late_orders=late_cnt,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        nodes_explored=len(route),
    )


def solve_edf(inst: DeliveryInstance) -> RouteSolution:
    """Earliest-Deadline-First with distance tie-breaking."""
    start = time.perf_counter()
    remaining = set(range(len(inst.orders)))
    curr_pos = inst.depot
    curr_time = 0.0
    route: list[int] = []
    while remaining:
        def urgency(i: int) -> float:
            o = inst.orders[i]
            d = inst.distance(curr_pos, (o.x, o.y))
            slack = o.latest - (curr_time + d / inst.speed)
            return slack + 0.6 * d

        nxt = min(remaining, key=urgency)
        o = inst.orders[nxt]
        d = inst.distance(curr_pos, (o.x, o.y))
        curr_time = max(curr_time + d / inst.speed, o.earliest) + o.service_time
        curr_pos = (o.x, o.y)
        route.append(nxt)
        remaining.remove(nxt)

    cost, dist, late, late_cnt = inst.evaluate_route(route)
    return RouteSolution(
        method="EarliestDeadline",
        route=route,
        total_cost=cost,
        travel_distance=dist,
        total_lateness=late,
        late_orders=late_cnt,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        nodes_explored=len(route),
    )


def solve_branch_and_bound(inst: DeliveryInstance, max_nodes: int = 200_000) -> RouteSolution:
    """Exact Branch-and-Bound solver (feasible for small instances M <= 10)."""
    start = time.perf_counter()
    # Warm-start upper bound with EDF solution
    edf = solve_edf(inst)
    best_cost = edf.total_cost
    best_route = list(edf.route)
    nodes = 0
    n = len(inst.orders)
    coords = [(o.x, o.y) for o in inst.orders]
    dist_depot = [inst.distance(inst.depot, p) for p in coords]
    dist_mat = [[inst.distance(coords[i], coords[j]) for j in range(n)] for i in range(n)]

    def dfs(
        curr_idx: int,  # -1 means depot
        curr_time: float,
        curr_dist: float,
        curr_late: float,
        visited_mask: int,
        path: list[int],
    ) -> None:
        nonlocal best_cost, best_route, nodes
        if nodes >= max_nodes:
            return
        nodes += 1

        if len(path) == n:
            total_dist = curr_dist + dist_depot[curr_idx]
            total_cost = total_dist + inst.late_penalty * curr_late
            if total_cost < best_cost:
                best_cost = total_cost
                best_route = path.copy()
            return

        # Admissible lower bound: current cost + min return to depot from remaining nodes
        lb = curr_dist + inst.late_penalty * curr_late
        if lb >= best_cost:
            return

        # Order unvisited children by urgency to tighten bound early
        candidates = []
        for nxt in range(n):
            if not (visited_mask & (1 << nxt)):
                d = dist_depot[nxt] if curr_idx == -1 else dist_mat[curr_idx][nxt]
                o = inst.orders[nxt]
                arr = curr_time + d / inst.speed
                late_inc = max(0.0, arr - o.latest)
                candidates.append((d + inst.late_penalty * late_inc, nxt, d, arr, late_inc))
        candidates.sort(key=lambda t: t[0])

        for _, nxt, d, arr, late_inc in candidates:
            new_dist = curr_dist + d
            new_late = curr_late + late_inc
            if new_dist + inst.late_penalty * new_late >= best_cost:
                continue
            o = inst.orders[nxt]
            next_time = max(arr, o.earliest) + o.service_time
            path.append(nxt)
            dfs(nxt, next_time, new_dist, new_late, visited_mask | (1 << nxt), path)
            path.pop()

    dfs(-1, 0.0, 0.0, 0.0, 0, [])
    cost, dist, late, late_cnt = inst.evaluate_route(best_route)
    return RouteSolution(
        method="ExactBnB",
        route=best_route,
        total_cost=cost,
        travel_distance=dist,
        total_lateness=late,
        late_orders=late_cnt,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        nodes_explored=nodes,
    )


class _SPMCTSNode:
    __slots__ = ("route", "remaining", "parent", "action", "children", "untried", "visits", "total_reward")

    def __init__(
        self,
        route: tuple[int, ...],
        remaining: list[int],
        parent: Optional["_SPMCTSNode"] = None,
        action: Optional[int] = None,
    ) -> None:
        self.route = route
        self.remaining = remaining
        self.parent = parent
        self.action = action
        self.children: list[_SPMCTSNode] = []
        self.untried: list[int] = list(remaining)
        self.visits: int = 0
        self.total_reward: float = 0.0

    def uct(self, exploration: float) -> float:
        if self.visits == 0:
            return math.inf
        n_p = max(self.parent.visits if self.parent else 1, 1)
        return (self.total_reward / self.visits) + exploration * math.sqrt(math.log(n_p) / self.visits)


def solve_mcts(
    inst: DeliveryInstance,
    iterations: int = 2000,
    exploration: float = math.sqrt(2.0),
    heuristic_rollout: bool = True,
    seed: int = 42,
) -> RouteSolution:
    """Single-Player MCTS-UCT for Last-Mile Delivery Scheduling with Time Windows."""
    start = time.perf_counter()
    rng = random.Random(seed)
    n = len(inst.orders)

    # Reference cost for normalizing reward into (0, 1]
    ref_sol = solve_nearest_neighbor(inst)
    ref_cost = max(ref_sol.total_cost, 1.0)
    best_cost = ref_sol.total_cost
    best_route = list(ref_sol.route)

    root = _SPMCTSNode(route=(), remaining=list(range(n)))
    tree_nodes = 1

    def rollout_from(prefix: tuple[int, ...], rem: list[int]) -> list[int]:
        route = list(prefix)
        left = list(rem)
        if not heuristic_rollout:
            rng.shuffle(left)
            route.extend(left)
            return route

        # Urgency-distance randomized greedy rollout (top-3 soft selection)
        curr_pos = inst.depot
        curr_time = 0.0
        for idx in route:
            o = inst.orders[idx]
            d = inst.distance(curr_pos, (o.x, o.y))
            curr_time = max(curr_time + d / inst.speed, o.earliest) + o.service_time
            curr_pos = (o.x, o.y)

        while left:
            scored = []
            for idx in left:
                o = inst.orders[idx]
                d = inst.distance(curr_pos, (o.x, o.y))
                arr = curr_time + d / inst.speed
                late_pen = max(0.0, arr - o.latest) * inst.late_penalty
                wait_pen = max(0.0, o.earliest - arr) * 0.4
                slack = max(0.0, o.latest - arr)
                score = d + late_pen + wait_pen + 0.35 * slack
                scored.append((score, idx))
            scored.sort(key=lambda t: t[0])
            # Pick probabilistically among top-2 to keep rollout diversity
            k = min(2, len(scored))
            pick = scored[0][1] if (k == 1 or rng.random() < 0.78) else scored[1][1]
            left.remove(pick)
            route.append(pick)
            o = inst.orders[pick]
            d = inst.distance(curr_pos, (o.x, o.y))
            curr_time = max(curr_time + d / inst.speed, o.earliest) + o.service_time
            curr_pos = (o.x, o.y)
        return route

    for _ in range(iterations):
        node = root
        # 1. Selection
        while not node.untried and node.children:
            node = max(node.children, key=lambda ch: ch.uct(exploration))

        # 2. Expansion
        if node.untried:
            idx = rng.randrange(len(node.untried))
            node.untried[idx], node.untried[-1] = node.untried[-1], node.untried[idx]
            act = node.untried.pop()
            next_rem = [x for x in node.remaining if x != act]
            child = _SPMCTSNode(route=node.route + (act,), remaining=next_rem, parent=node, action=act)
            node.children.append(child)
            node = child
            tree_nodes += 1

        # 3. Simulation
        full_route = rollout_from(node.route, node.remaining)
        cost, _, _, _ = inst.evaluate_route(full_route)
        if cost < best_cost:
            best_cost = cost
            best_route = full_route

        # Smooth bounded reward in (0, 1] inversely related to cost/ref_cost
        reward = math.exp(-max(0.0, (cost / ref_cost) - 0.5))
        reward = min(1.0, max(0.0, reward))

        # 4. Backpropagation
        curr: Optional[_SPMCTSNode] = node
        while curr is not None:
            curr.visits += 1
            curr.total_reward += reward
            curr = curr.parent

    cost, dist, late, late_cnt = inst.evaluate_route(best_route)
    tag = "MCTS-Heuristic" if heuristic_rollout else "MCTS-Random"
    return RouteSolution(
        method=f"{tag}(it={iterations})",
        route=best_route,
        total_cost=cost,
        travel_distance=dist,
        total_lateness=late,
        late_orders=late_cnt,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        nodes_explored=tree_nodes,
    )

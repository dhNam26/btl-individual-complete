"""Bonus real-world AI applications."""

from .delivery_scheduling import (
    DeliveryInstance,
    Order,
    RouteSolution,
    generate_instance,
    solve_branch_and_bound,
    solve_edf,
    solve_mcts,
    solve_nearest_neighbor,
)

__all__ = [
    "Order",
    "DeliveryInstance",
    "RouteSolution",
    "generate_instance",
    "solve_nearest_neighbor",
    "solve_edf",
    "solve_branch_and_bound",
    "solve_mcts",
]

"""Minimax and Alpha-Beta Pruning with Move Ordering and optional Transposition Table."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..games.base import GameState


@dataclass
class SearchResult:
    """Structured output of a Minimax or Alpha-Beta search."""

    best_move: Any
    best_value: float
    nodes_visited: int
    cutoffs: int
    tt_hits: int
    elapsed_ms: float
    root_values: list[tuple[Any, float]] = field(default_factory=list)


def minimax_search(
    state: GameState,
    depth: int,
    use_move_ordering: bool = False,
) -> SearchResult:
    """Plain depth-limited Minimax (no pruning) for baseline comparison."""
    start = time.perf_counter()
    perspective = state.current_player
    stats = {"nodes": 0}

    def recurse(node: GameState, d: int, maximizing: bool) -> float:
        stats["nodes"] += 1
        if d == 0 or node.is_terminal():
            return node.evaluate(perspective)
        moves = node.ordered_moves() if use_move_ordering else node.legal_moves()
        if not moves:
            return node.evaluate(perspective)
        if maximizing:
            best = -math.inf
            for m in moves:
                v = recurse(node.apply_move(m), d - 1, False)
                if v > best:
                    best = v
            return best
        else:
            best = math.inf
            for m in moves:
                v = recurse(node.apply_move(m), d - 1, True)
                if v < best:
                    best = v
            return best

    moves = state.ordered_moves() if use_move_ordering else state.legal_moves()
    if not moves:
        raise RuntimeError("No legal moves available for Minimax search.")

    stats["nodes"] += 1  # count root
    best_move = moves[0]
    best_val = -math.inf
    root_values: list[tuple[Any, float]] = []

    for m in moves:
        v = recurse(state.apply_move(m), depth - 1, False)
        root_values.append((m, v))
        if v > best_val:
            best_val = v
            best_move = m

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SearchResult(
        best_move=best_move,
        best_value=best_val,
        nodes_visited=stats["nodes"],
        cutoffs=0,
        tt_hits=0,
        elapsed_ms=elapsed_ms,
        root_values=root_values,
    )


def alphabeta_search(
    state: GameState,
    depth: int,
    use_move_ordering: bool = True,
    use_tt: bool = False,
) -> SearchResult:
    """Minimax with Alpha-Beta pruning, optional heuristic move ordering, and optional TT."""
    start = time.perf_counter()
    perspective = state.current_player
    stats = {"nodes": 0, "cutoffs": 0, "tt_hits": 0}
    # TT entry: (depth, flag, value) where flag in {0: EXACT, 1: LOWERBOUND, 2: UPPERBOUND}
    tt: dict[Any, tuple[int, int, float]] = {}

    def recurse(
        node: GameState,
        d: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> float:
        stats["nodes"] += 1
        alpha_orig = alpha
        beta_orig = beta

        if use_tt:
            key = node.state_key()
            entry = tt.get(key)
            if entry is not None and entry[0] >= d:
                _, flag, val = entry
                if flag == 0:
                    stats["tt_hits"] += 1
                    return val
                elif flag == 1 and val >= beta:
                    stats["tt_hits"] += 1
                    return val
                elif flag == 2 and val <= alpha:
                    stats["tt_hits"] += 1
                    return val

        if d == 0 or node.is_terminal():
            return node.evaluate(perspective)

        moves = node.ordered_moves() if use_move_ordering else node.legal_moves()
        if not moves:
            return node.evaluate(perspective)

        if maximizing:
            value = -math.inf
            for m in moves:
                v = recurse(node.apply_move(m), d - 1, alpha, beta, False)
                if v > value:
                    value = v
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    stats["cutoffs"] += 1
                    break
        else:
            value = math.inf
            for m in moves:
                v = recurse(node.apply_move(m), d - 1, alpha, beta, True)
                if v < value:
                    value = v
                if value < beta:
                    beta = value
                if alpha >= beta:
                    stats["cutoffs"] += 1
                    break

        if use_tt:
            if value <= alpha_orig:
                flag = 2  # UPPERBOUND
            elif value >= beta_orig:
                flag = 1  # LOWERBOUND
            else:
                flag = 0  # EXACT
            tt[node.state_key()] = (d, flag, value)

        return value

    moves = state.ordered_moves() if use_move_ordering else state.legal_moves()
    if not moves:
        raise RuntimeError("No legal moves available for Alpha-Beta search.")

    stats["nodes"] += 1  # count root
    best_move = moves[0]
    best_val = -math.inf
    alpha = -math.inf
    beta = math.inf
    root_values: list[tuple[Any, float]] = []

    for m in moves:
        v = recurse(state.apply_move(m), depth - 1, alpha, beta, False)
        root_values.append((m, v))
        if v > best_val:
            best_val = v
            best_move = m
        if v > alpha:
            alpha = v

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SearchResult(
        best_move=best_move,
        best_value=best_val,
        nodes_visited=stats["nodes"],
        cutoffs=stats["cutoffs"],
        tt_hits=stats["tt_hits"],
        elapsed_ms=elapsed_ms,
        root_values=root_values,
    )

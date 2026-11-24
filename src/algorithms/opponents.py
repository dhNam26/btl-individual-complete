"""Unified Agent wrappers for head-to-head tournaments and demos."""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from ..games.base import GameState
from .mcts import MCTSResult, mcts_search
from .minimax import SearchResult, alphabeta_search, minimax_search


class Agent:
    name: str = "Agent"

    def select_move(self, state: GameState) -> tuple[Any, Any]:
        """Return (move, metadata)."""
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.name = "Random"

    def select_move(self, state: GameState) -> tuple[Any, dict[str, Any]]:
        return self.rng.choice(state.legal_moves()), {}


class GreedyAgent(Agent):
    """Tactical 1-ply greedy agent using the game's heuristic rollout policy."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.name = "GreedyHeuristic"

    def select_move(self, state: GameState) -> tuple[Any, dict[str, Any]]:
        moves = state.ordered_moves()
        # Deterministic top heuristic move
        return moves[0], {}


class MinimaxAgent(Agent):
    def __init__(
        self,
        depth: int = 2,
        use_alphabeta: bool = True,
        use_move_ordering: bool = True,
        use_tt: bool = True,
    ) -> None:
        self.depth = depth
        self.use_alphabeta = use_alphabeta
        self.use_move_ordering = use_move_ordering
        self.use_tt = use_tt
        algo = "AlphaBeta" if use_alphabeta else "Minimax"
        self.name = f"{algo}(d={depth})"

    def select_move(self, state: GameState) -> tuple[Any, SearchResult]:
        if self.use_alphabeta:
            res = alphabeta_search(
                state,
                depth=self.depth,
                use_move_ordering=self.use_move_ordering,
                use_tt=self.use_tt,
            )
        else:
            res = minimax_search(
                state,
                depth=self.depth,
                use_move_ordering=self.use_move_ordering,
            )
        return res.best_move, res


class MCTSAgent(Agent):
    def __init__(
        self,
        iterations: int = 1000,
        exploration: float = math.sqrt(2.0),
        uct_variant: str = "uct",
        rollout_policy: str = "random",
        final_selection: str = "robust",
        seed: Optional[int] = None,
        max_rollout_steps: int = 100,
        cutoff_steps: int = 6,
    ) -> None:
        self.iterations = iterations
        self.exploration = exploration
        self.uct_variant = uct_variant
        self.rollout_policy = rollout_policy
        self.final_selection = final_selection
        self.rng = random.Random(seed)
        self.max_rollout_steps = max_rollout_steps
        self.cutoff_steps = cutoff_steps
        self.name = (
            f"MCTS({uct_variant}, {rollout_policy}, c={exploration:.2f}, it={iterations})"
        )

    def select_move(self, state: GameState) -> tuple[Any, MCTSResult]:
        move_seed = self.rng.randint(0, 2**31 - 1)
        res = mcts_search(
            state,
            iterations=self.iterations,
            exploration=self.exploration,
            uct_variant=self.uct_variant,
            rollout_policy=self.rollout_policy,
            final_selection=self.final_selection,
            seed=move_seed,
            max_rollout_steps=self.max_rollout_steps,
            cutoff_steps=self.cutoff_steps,
        )
        return res.best_move, res

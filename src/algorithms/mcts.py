"""Monte Carlo Tree Search (MCTS) with UCT, UCT Formula Variants, and Improved Rollouts.

Supported Selection Formulas (`uct_variant`):
- "uct"        : Standard UCT = W/N + c * sqrt(ln(N_p) / N)
- "no_sqrt"    : Ablation without square root = W/N + c * (ln(N_p) / N)
- "no_ln"      : Ablation without logarithm   = W/N + c * sqrt(1 / N)
- "linear"     : Linear decay exploration     = W/N + c / N
- "greedy"     : Pure exploitation (c = 0)    = W/N
- "ucb1_tuned" : Variance-aware UCB1-Tuned (Auer et al., 2002)
- "puct"       : Heuristic-Prior UCT (AlphaZero-style PUCT)

Supported Simulation Policies (`rollout_policy`):
- "random"       : Uniform random playout to terminal state
- "heuristic"    : Tactical heuristic playout (win/block/threat/capture)
- "value_cutoff" : Short tactical playout + static value evaluation (Value-Network precursor)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..games.base import GameState


@dataclass
class RootChildStats:
    """Per-move statistics at the MCTS root for reporting and analysis."""

    move: Any
    wins: float
    visits: int
    win_rate: float
    uct_score: float
    prior: float


@dataclass
class MCTSResult:
    """Structured output of an MCTS search."""

    best_move: Any
    root_visits: int
    tree_nodes: int
    elapsed_ms: float
    root_stats: list[RootChildStats] = field(default_factory=list)


class MCTSNode:
    """Tree node storing statistics from the perspective of `player_just_moved`."""

    __slots__ = (
        "state",
        "parent",
        "move",
        "player_just_moved",
        "prior",
        "children",
        "untried_moves",
        "untried_priors",
        "visits",
        "wins",
        "sum_sq",
    )

    def __init__(
        self,
        state: GameState,
        parent: Optional["MCTSNode"] = None,
        move: Any = None,
        player_just_moved: Optional[int] = None,
        prior: float = 1.0,
        use_priors: bool = False,
    ) -> None:
        self.state = state
        self.parent = parent
        self.move = move
        self.player_just_moved = player_just_moved
        self.prior = prior
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.wins: float = 0.0
        self.sum_sq: float = 0.0

        if state.is_terminal():
            self.untried_moves: list[Any] = []
            self.untried_priors: list[float] = []
        elif use_priors:
            ordered = state.ordered_moves()
            n = len(ordered)
            # Rank-based geometric prior favoring top heuristic moves while remaining strictly > 0
            weights = [math.exp(-0.35 * rank) for rank in range(n)]
            total_w = sum(weights) or 1.0
            # Store reversed so .pop() expands highest-prior moves first in O(1)
            self.untried_moves = list(reversed(ordered))
            self.untried_priors = list(reversed([w / total_w for w in weights]))
        else:
            self.untried_moves = list(state.legal_moves())
            self.untried_priors = []

    def selection_score(self, exploration: float, variant: str = "uct") -> float:
        if self.visits == 0:
            return math.inf

        n_i = self.visits
        q_i = self.wins / n_i
        n_p = max(self.parent.visits if self.parent is not None else 1, 1)

        if variant == "uct":
            return q_i + exploration * math.sqrt(math.log(n_p) / n_i)
        elif variant == "no_sqrt":
            return q_i + exploration * (math.log(n_p) / n_i)
        elif variant == "no_ln":
            return q_i + exploration * math.sqrt(1.0 / n_i)
        elif variant == "linear":
            return q_i + exploration / n_i
        elif variant == "greedy":
            return q_i
        elif variant == "ucb1_tuned":
            ln_p = math.log(n_p)
            empirical_var = max(0.0, (self.sum_sq / n_i) - q_i * q_i)
            v_i = empirical_var + math.sqrt(2.0 * ln_p / n_i)
            # Standard UCB1-Tuned uses factor min(1/4, V_i); scale by exploration/sqrt(2) so c=sqrt(2) is canonical
            scale = exploration / math.sqrt(2.0)
            return q_i + scale * math.sqrt((ln_p / n_i) * min(0.25, v_i))
        elif variant == "puct":
            # AlphaZero-style PUCT: Q + c * P(s,a) * sqrt(N_p) / (1 + N_i)
            return q_i + exploration * self.prior * math.sqrt(n_p) / (1.0 + n_i)
        else:
            raise ValueError(f"Unknown UCT variant: {variant}")


def _simulate(
    state: GameState,
    rollout_policy: str,
    rng: random.Random,
    max_rollout_steps: int,
    cutoff_steps: int,
) -> float:
    """Return reward in [0, 1] from player +1's perspective."""
    fast_fn = getattr(state, "fast_rollout", None)
    if fast_fn is not None:
        return fast_fn(
            policy=rollout_policy,
            rng=rng,
            max_steps=max_rollout_steps,
            cutoff_steps=cutoff_steps,
        )

    curr = state
    steps = 0

    if rollout_policy == "value_cutoff":
        while not curr.is_terminal() and steps < cutoff_steps:
            move = curr.rollout_move(rng)
            curr = curr.apply_move(move)
            steps += 1
        return curr.rollout_value(perspective=1)

    while not curr.is_terminal() and steps < max_rollout_steps:
        if rollout_policy == "heuristic":
            move = curr.rollout_move(rng)
        elif rollout_policy == "random":
            moves = curr.legal_moves()
            move = rng.choice(moves)
        else:
            raise ValueError(f"Unknown rollout policy: {rollout_policy}")
        curr = curr.apply_move(move)
        steps += 1

    w = curr.winner()
    if w is None:
        return curr.rollout_value(perspective=1) if rollout_policy == "heuristic" else 0.5
    if w == 0:
        return 0.5
    return 1.0 if w == 1 else 0.0


def mcts_search(
    root_state: GameState,
    iterations: int = 1000,
    exploration: float = math.sqrt(2.0),
    uct_variant: str = "uct",
    rollout_policy: str = "random",
    final_selection: str = "robust",
    seed: Optional[int] = None,
    max_rollout_steps: int = 100,
    cutoff_steps: int = 6,
) -> MCTSResult:
    """Run MCTS for `iterations` simulations and return the selected move and root stats."""
    start = time.perf_counter()
    rng = random.Random(seed)
    use_priors = uct_variant == "puct"

    root = MCTSNode(root_state, use_priors=use_priors)
    if root_state.is_terminal() or not root.untried_moves:
        raise RuntimeError("No legal moves available for MCTS search.")

    tree_nodes = 1

    for _ in range(iterations):
        node = root

        # 1. Selection
        while not node.untried_moves and node.children and not node.state.is_terminal():
            node = max(
                node.children,
                key=lambda ch: ch.selection_score(exploration, uct_variant),
            )

        # 2. Expansion
        if node.untried_moves and not node.state.is_terminal():
            if use_priors:
                move = node.untried_moves.pop()
                prior = node.untried_priors.pop()
            else:
                idx = rng.randrange(len(node.untried_moves))
                # O(1) swap-and-pop
                node.untried_moves[idx], node.untried_moves[-1] = (
                    node.untried_moves[-1],
                    node.untried_moves[idx],
                )
                move = node.untried_moves.pop()
                prior = 1.0 / max(len(node.children) + len(node.untried_moves) + 1, 1)

            next_state = node.state.apply_move(move)
            child = MCTSNode(
                state=next_state,
                parent=node,
                move=move,
                player_just_moved=node.state.current_player,
                prior=prior,
                use_priors=use_priors,
            )
            node.children.append(child)
            node = child
            tree_nodes += 1

        # 3. Simulation (reward from player +1's perspective)
        reward_p1 = _simulate(
            node.state,
            rollout_policy=rollout_policy,
            rng=rng,
            max_rollout_steps=max_rollout_steps,
            cutoff_steps=cutoff_steps,
        )

        # 4. Backpropagation
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visits += 1
            if curr.player_just_moved is not None:
                r = reward_p1 if curr.player_just_moved == 1 else (1.0 - reward_p1)
                curr.wins += r
                curr.sum_sq += r * r
            curr = curr.parent

    if not root.children:
        raise RuntimeError("MCTS did not expand any child from root.")

    if final_selection == "robust":
        best_child = max(root.children, key=lambda ch: (ch.visits, ch.wins / max(ch.visits, 1)))
    elif final_selection == "max_winrate":
        best_child = max(root.children, key=lambda ch: (ch.wins / max(ch.visits, 1), ch.visits))
    else:
        raise ValueError(f"Unknown final_selection: {final_selection}")

    root_stats = [
        RootChildStats(
            move=ch.move,
            wins=ch.wins,
            visits=ch.visits,
            win_rate=ch.wins / ch.visits if ch.visits else 0.0,
            uct_score=ch.selection_score(exploration, uct_variant),
            prior=ch.prior,
        )
        for ch in sorted(root.children, key=lambda c: c.visits, reverse=True)
    ]

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return MCTSResult(
        best_move=best_child.move,
        root_visits=root.visits,
        tree_nodes=tree_nodes,
        elapsed_ms=elapsed_ms,
        root_stats=root_stats,
    )

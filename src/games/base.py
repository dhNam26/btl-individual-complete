"""Abstract interface for two-player zero-sum deterministic perfect-information games."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Hashable, Optional


class GameState(ABC):
    """Base class for all board games used with Minimax, Alpha-Beta, and MCTS."""

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Return the player to move: +1 (First/X/Red) or -1 (Second/O/Black)."""

    @abstractmethod
    def legal_moves(self) -> list[Any]:
        """Return all legal moves in the current state."""

    @abstractmethod
    def ordered_moves(self) -> list[Any]:
        """Return legal moves ordered heuristically to improve Alpha-Beta pruning."""

    @abstractmethod
    def apply_move(self, move: Any) -> "GameState":
        """Return a new GameState after applying `move` (non-mutating)."""

    @abstractmethod
    def winner(self) -> Optional[int]:
        """Return +1 or -1 if a player has won, 0 for draw, or None if non-terminal."""

    def is_terminal(self) -> bool:
        """Return True if the game has ended (win, loss, or draw)."""
        return self.winner() is not None

    @abstractmethod
    def evaluate(self, perspective: int) -> float:
        """Heuristic evaluation from `perspective`'s point of view (larger is better)."""

    @abstractmethod
    def state_key(self) -> Hashable:
        """Hashable key uniquely identifying this state for transposition tables."""

    def rollout_move(self, rng: Any) -> Any:
        """Tactical heuristic move used during MCTS heuristic rollouts.

        Subclasses should override with fast domain-specific tactical priorities
        (e.g., win immediately, block immediate loss, tactical captures/threats).
        """
        moves = self.legal_moves()
        return rng.choice(moves)

    def rollout_value(self, perspective: int) -> float:
        """Normalized heuristic value in [0, 1] for value-cutoff MCTS simulation."""
        w = self.winner()
        if w is not None:
            if w == 0:
                return 0.5
            return 1.0 if w == perspective else 0.0
        # Default logistic squashing of heuristic score
        import math
        score = self.evaluate(perspective)
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score / 500.0))))

    @abstractmethod
    def render(self) -> str:
        """Return a human-readable ASCII representation of the board."""

    @staticmethod
    def player_name(player: int) -> str:
        return "X" if player == 1 else "O"

"""Exact 3x3 Tic-Tac-Toe implementation for verification and benchmarking."""

from __future__ import annotations

import random
from typing import Optional

from .base import GameState

WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

# Center first, then corners, then edges -> maximizes Alpha-Beta cutoffs
PREFERRED_ORDER: tuple[int, ...] = (4, 0, 2, 6, 8, 1, 3, 5, 7)

SYMBOL = {1: "X", -1: "O", 0: "."}


class TicTacToeState(GameState):
    """3x3 Tic-Tac-Toe state using a compact 9-tuple of {1, -1, 0}."""

    __slots__ = ("board", "_player", "_winner", "_moves_played")

    def __init__(
        self,
        board: Optional[tuple[int, ...]] = None,
        player: int = 1,
        winner: Optional[int] = None,
        moves_played: Optional[int] = None,
    ) -> None:
        self.board: tuple[int, ...] = board if board is not None else (0,) * 9
        self._player: int = player
        self._moves_played: int = (
            moves_played if moves_played is not None else sum(1 for c in self.board if c != 0)
        )
        self._winner: Optional[int] = winner if winner is not None else self._compute_winner()

    def _compute_winner(self) -> Optional[int]:
        b = self.board
        for a, c, d in WIN_LINES:
            v = b[a]
            if v != 0 and v == b[c] == b[d]:
                return v
        if self._moves_played >= 9:
            return 0
        return None

    @property
    def current_player(self) -> int:
        return self._player

    def legal_moves(self) -> list[int]:
        if self._winner is not None:
            return []
        b = self.board
        return [i for i in range(9) if b[i] == 0]

    def ordered_moves(self) -> list[int]:
        if self._winner is not None:
            return []
        b = self.board
        return [i for i in PREFERRED_ORDER if b[i] == 0]

    def apply_move(self, move: int) -> "TicTacToeState":
        if not (0 <= move < 9) or self.board[move] != 0:
            raise ValueError(f"Illegal Tic-Tac-Toe move: {move}")
        new_board = list(self.board)
        new_board[move] = self._player
        return TicTacToeState(
            board=tuple(new_board),
            player=-self._player,
            moves_played=self._moves_played + 1,
        )

    def winner(self) -> Optional[int]:
        return self._winner

    def evaluate(self, perspective: int) -> float:
        if self._winner is not None:
            if self._winner == 0:
                return 0.0
            # Prefer faster wins and slower losses via slight ply bonus
            bonus = (10 - self._moves_played) * 0.01
            return (100.0 + bonus) if self._winner == perspective else -(100.0 + bonus)
        # Non-terminal heuristic (if depth-limited)
        b = self.board
        score = 0.0
        for a, c, d in WIN_LINES:
            line = (b[a], b[c], b[d])
            own = line.count(perspective)
            opp = line.count(-perspective)
            if own > 0 and opp == 0:
                score += 10.0 if own == 2 else 1.0
            elif opp > 0 and own == 0:
                score -= 10.0 if opp == 2 else 1.0
        return score

    def state_key(self) -> tuple[tuple[int, ...], int]:
        return (self.board, self._player)

    def is_winning_move(self, move: int, player: int) -> bool:
        b = self.board
        if b[move] != 0:
            return False
        for a, c, d in WIN_LINES:
            if move == a and b[c] == player and b[d] == player:
                return True
            if move == c and b[a] == player and b[d] == player:
                return True
            if move == d and b[a] == player and b[c] == player:
                return True
        return False

    def rollout_move(self, rng: random.Random) -> int:
        moves = self.legal_moves()
        p = self._player
        # 1. Immediate win
        for m in moves:
            if self.is_winning_move(m, p):
                return m
        # 2. Immediate block
        for m in moves:
            if self.is_winning_move(m, -p):
                return m
        # 3. Prefer center (with 70% probability to retain rollout diversity)
        if 4 in moves and rng.random() < 0.7:
            return 4
        return rng.choice(moves)

    def rollout_value(self, perspective: int) -> float:
        if self._winner is not None:
            if self._winner == 0:
                return 0.5
            return 1.0 if self._winner == perspective else 0.0
        import math
        s = self.evaluate(perspective)
        return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, s / 10.0))))

    def fast_rollout(
        self,
        policy: str,
        rng: random.Random,
        max_steps: int = 9,
        cutoff_steps: int = 4,
    ) -> float:
        """Fast in-place 3x3 Tic-Tac-Toe rollout returning reward in [0, 1] for player +1."""
        if self._winner is not None:
            return 0.5 if self._winner == 0 else (1.0 if self._winner == 1 else 0.0)
        b = list(self.board)
        empty = [i for i in range(9) if b[i] == 0]
        p = self._player

        if policy == "random":
            rng.shuffle(empty)
            for m in empty:
                b[m] = p
                for a, c, d in WIN_LINES:
                    v = b[a]
                    if v != 0 and v == b[c] == b[d]:
                        return 1.0 if v == 1 else 0.0
                p = -p
            return 0.5

        # Heuristic or value_cutoff
        limit = cutoff_steps if policy == "value_cutoff" else len(empty)
        for _ in range(min(limit, len(empty))):
            chosen = -1
            for target_p in (p, -p):
                for m in empty:
                    b[m] = target_p
                    for a, c, d in WIN_LINES:
                        if b[a] == target_p and b[c] == target_p and b[d] == target_p:
                            chosen = m
                            break
                    b[m] = 0
                    if chosen != -1:
                        break
                if chosen != -1:
                    break
            if chosen == -1:
                chosen = rng.choice(empty)
            b[chosen] = p
            empty.remove(chosen)
            for a, c, d in WIN_LINES:
                v = b[a]
                if v != 0 and v == b[c] == b[d]:
                    return 1.0 if v == 1 else 0.0
            p = -p
        if not empty:
            return 0.5
        leaf = TicTacToeState(tuple(b), p, moves_played=9 - len(empty))
        return leaf.rollout_value(1)

    def render(self) -> str:
        rows = []
        for r in range(3):
            cells = [SYMBOL[self.board[3 * r + c]] for c in range(3)]
            rows.append(f" {cells[0]} | {cells[1]} | {cells[2]} ")
        return "\n---+---+---\n".join(rows)

    def format_move(self, move: int) -> str:
        return f"({move // 3},{move % 3}) [cell {move + 1}]"

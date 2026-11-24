"""Configurable n x n Gomoku (5-in-a-row Tic-Tac-Toe) game engine.

Uses a flat 1D board representation, O(k) incremental win detection around the
last move, Chebyshev-neighborhood candidate generation, and pattern-based
threat evaluation for both Minimax/Alpha-Beta and Heuristic MCTS rollouts.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from .base import GameState

DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))
SYMBOL = {1: "X", -1: "O", 0: "."}

# Heuristic pattern scores indexed by (consecutive_count, open_ends)
# Strongly differentiates live-4, live-3, dead-4, live-2, etc.
PATTERN_SCORES: dict[tuple[int, int], float] = {
    (5, 2): 1_000_000.0,
    (5, 1): 1_000_000.0,
    (5, 0): 1_000_000.0,
    (4, 2): 100_000.0,   # Unstoppable open four
    (4, 1): 12_000.0,    # Half-open four (must block)
    (3, 2): 8_000.0,     # Live three (creates open four next)
    (3, 1): 600.0,       # Half-open three
    (2, 2): 300.0,       # Live two
    (2, 1): 40.0,        # Half-open two
    (1, 2): 15.0,
    (1, 1): 3.0,
}


class GomokuState(GameState):
    """n x n board where `win_length` (default 5) in a row wins."""

    __slots__ = (
        "size",
        "win_length",
        "candidate_radius",
        "board",
        "_player",
        "last_move",
        "moves_played",
        "_winner",
    )

    def __init__(
        self,
        size: int = 9,
        win_length: int = 5,
        candidate_radius: Optional[int] = 2,
        board: Optional[list[int]] = None,
        player: int = 1,
        last_move: Optional[int] = None,
        moves_played: int = 0,
        winner: Optional[int] = None,
    ) -> None:
        if size < win_length:
            raise ValueError(f"Board size ({size}) must be >= win_length ({win_length})")
        self.size = size
        self.win_length = win_length
        self.candidate_radius = candidate_radius
        self.board: list[int] = board if board is not None else [0] * (size * size)
        self._player = player
        self.last_move = last_move
        self.moves_played = moves_played
        self._winner = winner

    @property
    def current_player(self) -> int:
        return self._player

    def coord(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.size)

    def idx(self, row: int, col: int) -> int:
        return row * self.size + col

    def _check_win_at(self, idx: int, player: int) -> bool:
        n = self.size
        k = self.win_length
        b = self.board
        r0, c0 = divmod(idx, n)
        for dr, dc in DIRECTIONS:
            count = 1
            # Forward direction
            r, c = r0 + dr, c0 + dc
            while 0 <= r < n and 0 <= c < n and b[r * n + c] == player:
                count += 1
                if count >= k:
                    return True
                r += dr
                c += dc
            # Backward direction
            r, c = r0 - dr, c0 - dc
            while 0 <= r < n and 0 <= c < n and b[r * n + c] == player:
                count += 1
                if count >= k:
                    return True
                r -= dr
                c -= dc
        return False

    def is_winning_move(self, move: int, player: int) -> bool:
        """O(win_length) check whether placing `player` at `move` immediately wins."""
        if self.board[move] != 0:
            return False
        self.board[move] = player
        won = self._check_win_at(move, player)
        self.board[move] = 0
        return won

    def _line_threat_score(self, move: int, player: int) -> float:
        """Fast local pattern score if `player` plays at `move` (no board copy)."""
        n = self.size
        k = self.win_length
        b = self.board
        r0, c0 = divmod(move, n)
        total = 0.0
        for dr, dc in DIRECTIONS:
            count = 1
            open_ends = 0
            r, c = r0 + dr, c0 + dc
            while 0 <= r < n and 0 <= c < n and b[r * n + c] == player:
                count += 1
                r += dr
                c += dc
            if 0 <= r < n and 0 <= c < n and b[r * n + c] == 0:
                open_ends += 1

            r, c = r0 - dr, c0 - dc
            while 0 <= r < n and 0 <= c < n and b[r * n + c] == player:
                count += 1
                r -= dr
                c -= dc
            if 0 <= r < n and 0 <= c < n and b[r * n + c] == 0:
                open_ends += 1

            if count >= k:
                return 1_000_000.0
            total += PATTERN_SCORES.get((count, open_ends), 0.0)
        return total

    def legal_moves(self) -> list[int]:
        if self._winner is not None:
            return []
        n = self.size
        b = self.board
        if self.moves_played == 0:
            center = (n // 2) * n + (n // 2)
            return [center]

        rad = self.candidate_radius
        if rad is None:
            return [i for i, v in enumerate(b) if v == 0]

        seen = bytearray(n * n)
        moves: list[int] = []
        for idx, val in enumerate(b):
            if val != 0:
                r0, c0 = divmod(idx, n)
                r_lo = max(0, r0 - rad)
                r_hi = min(n - 1, r0 + rad)
                c_lo = max(0, c0 - rad)
                c_hi = min(n - 1, c0 + rad)
                for r in range(r_lo, r_hi + 1):
                    row_off = r * n
                    for c in range(c_lo, c_hi + 1):
                        pos = row_off + c
                        if b[pos] == 0 and not seen[pos]:
                            seen[pos] = 1
                            moves.append(pos)
        return moves

    def all_empty_moves(self) -> list[int]:
        """Return every empty square on the board (for human UI input validation)."""
        if self._winner is not None:
            return []
        return [i for i, v in enumerate(self.board) if v == 0]

    def ordered_moves(self) -> list[int]:
        """Order candidate moves by combined offensive + defensive threat score."""
        moves = self.legal_moves()
        if len(moves) <= 1:
            return moves
        p = self._player
        opp = -p
        n = self.size
        center = (n - 1) / 2.0

        def move_priority(m: int) -> float:
            atk = self._line_threat_score(m, p)
            dfn = self._line_threat_score(m, opp)
            r, c = divmod(m, n)
            dist = abs(r - center) + abs(c - center)
            return atk * 1.1 + dfn - dist * 0.5

        moves.sort(key=move_priority, reverse=True)
        return moves

    def apply_move(self, move: int) -> "GomokuState":
        if not (0 <= move < self.size * self.size) or self.board[move] != 0:
            raise ValueError(f"Illegal Gomoku move: {move}")
        new_board = self.board.copy()
        p = self._player
        new_board[move] = p
        new_moves = self.moves_played + 1
        next_state = GomokuState(
            size=self.size,
            win_length=self.win_length,
            candidate_radius=self.candidate_radius,
            board=new_board,
            player=-p,
            last_move=move,
            moves_played=new_moves,
            winner=None,
        )
        if next_state._check_win_at(move, p):
            next_state._winner = p
        elif new_moves >= self.size * self.size:
            next_state._winner = 0
        return next_state

    def winner(self) -> Optional[int]:
        return self._winner

    def evaluate(self, perspective: int) -> float:
        """Pattern-based board evaluation considering consecutive runs and open ends."""
        if self._winner is not None:
            if self._winner == 0:
                return 0.0
            bonus = (self.size * self.size - self.moves_played) * 10.0
            return (1_000_000.0 + bonus) if self._winner == perspective else -(1_000_000.0 + bonus)

        n = self.size
        k = self.win_length
        b = self.board
        total = 0.0
        opp = -perspective

        # Scan maximal contiguous runs along all 4 directions
        for r in range(n):
            row_off = r * n
            for c in range(n):
                val = b[row_off + c]
                if val == 0:
                    continue
                for dr, dc in DIRECTIONS:
                    # Only start counting at the beginning of a maximal run
                    pr, pc = r - dr, c - dc
                    if 0 <= pr < n and 0 <= pc < n and b[pr * n + pc] == val:
                        continue
                    open_ends = 0
                    if 0 <= pr < n and 0 <= pc < n and b[pr * n + pc] == 0:
                        open_ends += 1
                    length = 1
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < n and 0 <= nc < n and b[nr * n + nc] == val:
                        length += 1
                        nr += dr
                        nc += dc
                    if 0 <= nr < n and 0 <= nc < n and b[nr * n + nc] == 0:
                        open_ends += 1

                    if length >= k:
                        score = 1_000_000.0
                    elif open_ends == 0:
                        score = 0.0
                    else:
                        score = PATTERN_SCORES.get((length, open_ends), 0.0)

                    # Tempo weight: opponent's unblocked threats are slightly more dangerous
                    if val == perspective:
                        total += score * (1.15 if self._player == perspective else 1.0)
                    elif val == opp:
                        total -= score * (1.25 if self._player == opp else 1.0)

        return total

    def state_key(self) -> tuple[bytes, int]:
        # Compact immutable key for transposition table
        return (bytes(c + 1 for c in self.board), self._player)

    def rollout_move(self, rng: random.Random) -> int:
        """Tactical heuristic rollout policy (bug-free win/block + open-line preference).

        Priority:
        1. Immediate 5-in-a-row winning move.
        2. Immediate block of opponent's 5-in-a-row threat.
        3. High-threat tactical moves (open-4 / live-3 / blocking opponent live-3),
           sampled softly from top candidates to preserve exploration diversity.
        4. Fallback to random local candidate move.
        """
        moves = self.legal_moves()
        if len(moves) == 1:
            return moves[0]
        p = self._player
        opp = -p

        # 1. Immediate win
        for m in moves:
            if self.is_winning_move(m, p):
                return m

        # 2. Immediate block
        for m in moves:
            if self.is_winning_move(m, opp):
                return m

        # 3. Tactical scoring (with 80% greedy-top-k, 20% random to prevent rollout bias)
        if rng.random() < 0.85:
            best_score = -1.0
            top_moves: list[int] = []
            for m in moves:
                s = self._line_threat_score(m, p) * 1.1 + self._line_threat_score(m, opp)
                if s > best_score:
                    best_score = s
                    top_moves = [m]
                elif s == best_score:
                    top_moves.append(m)
            if best_score >= 300.0:
                return rng.choice(top_moves)

        return rng.choice(moves)

    def rollout_value(self, perspective: int) -> float:
        if self._winner is not None:
            if self._winner == 0:
                return 0.5
            return 1.0 if self._winner == perspective else 0.0
        score = self.evaluate(perspective)
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score / 4000.0))))

    def fast_rollout(
        self,
        policy: str,
        rng: random.Random,
        max_steps: int = 80,
        cutoff_steps: int = 6,
    ) -> float:
        """Zero-allocation in-place MCTS simulation returning reward in [0, 1] for player +1."""
        if self._winner is not None:
            return 0.5 if self._winner == 0 else (1.0 if self._winner == 1 else 0.0)

        n = self.size
        k = self.win_length
        b = self.board.copy()
        player = self._player
        moves_played = self.moves_played
        total_cells = n * n

        def check_win(idx: int, p: int) -> bool:
            r0, c0 = divmod(idx, n)
            for dr, dc in DIRECTIONS:
                cnt = 1
                r, c = r0 + dr, c0 + dc
                while 0 <= r < n and 0 <= c < n and b[r * n + c] == p:
                    cnt += 1
                    if cnt >= k:
                        return True
                    r += dr
                    c += dc
                r, c = r0 - dr, c0 - dc
                while 0 <= r < n and 0 <= c < n and b[r * n + c] == p:
                    cnt += 1
                    if cnt >= k:
                        return True
                    r -= dr
                    c -= dc
            return False

        if policy == "random":
            empty = [i for i, v in enumerate(b) if v == 0]
            rng.shuffle(empty)
            steps = 0
            for m in empty:
                if steps >= max_steps:
                    return 0.5
                b[m] = player
                moves_played += 1
                if check_win(m, player):
                    return 1.0 if player == 1 else 0.0
                if moves_played >= total_cells:
                    return 0.5
                player = -player
                steps += 1
            return 0.5

        # Tactical heuristic or value_cutoff rollout using an incrementally updated frontier set
        rad = 1 if n >= 9 else 2
        frontier: set[int] = set()
        if moves_played == 0:
            frontier.add((n // 2) * n + (n // 2))
        else:
            for idx, v in enumerate(b):
                if v != 0:
                    r0, c0 = divmod(idx, n)
                    for r in range(max(0, r0 - rad), min(n, r0 + rad + 1)):
                        off = r * n
                        for c in range(max(0, c0 - rad), min(n, c0 + rad + 1)):
                            pos = off + c
                            if b[pos] == 0:
                                frontier.add(pos)

        def quick_threat(m: int, p: int) -> int:
            r0, c0 = divmod(m, n)
            score = 0
            for dr, dc in DIRECTIONS:
                cnt = 1
                opens = 0
                r, c = r0 + dr, c0 + dc
                while 0 <= r < n and 0 <= c < n and b[r * n + c] == p:
                    cnt += 1
                    r += dr
                    c += dc
                if 0 <= r < n and 0 <= c < n and b[r * n + c] == 0:
                    opens += 1
                r, c = r0 - dr, c0 - dc
                while 0 <= r < n and 0 <= c < n and b[r * n + c] == p:
                    cnt += 1
                    r -= dr
                    c -= dc
                if 0 <= r < n and 0 <= c < n and b[r * n + c] == 0:
                    opens += 1
                if cnt >= k:
                    return 100000
                if cnt == 4:
                    score += 5000 if opens == 2 else (1200 if opens == 1 else 0)
                elif cnt == 3:
                    score += 800 if opens == 2 else (80 if opens == 1 else 0)
                elif cnt == 2 and opens == 2:
                    score += 40
            return score

        limit = cutoff_steps if policy == "value_cutoff" else max_steps
        for _ in range(limit):
            if not frontier:
                break
            opp = -player
            chosen = -1
            # 1. Immediate win
            for m in frontier:
                b[m] = player
                won = check_win(m, player)
                b[m] = 0
                if won:
                    return 1.0 if player == 1 else 0.0
            # 2. Immediate block
            for m in frontier:
                b[m] = opp
                opp_won = check_win(m, opp)
                b[m] = 0
                if opp_won:
                    chosen = m
                    break
            # 3. Tactical threat or random
            if chosen == -1:
                if rng.random() < 0.80:
                    best_s = 0
                    best_m = -1
                    for m in frontier:
                        s = quick_threat(m, player) + quick_threat(m, opp)
                        if s > best_s:
                            best_s = s
                            best_m = m
                    if best_s >= 40:
                        chosen = best_m
                if chosen == -1:
                    chosen = rng.choice(tuple(frontier))

            b[chosen] = player
            frontier.discard(chosen)
            moves_played += 1
            if moves_played >= total_cells:
                return 0.5
            r0, c0 = divmod(chosen, n)
            for r in range(max(0, r0 - rad), min(n, r0 + rad + 1)):
                off = r * n
                for c in range(max(0, c0 - rad), min(n, c0 + rad + 1)):
                    pos = off + c
                    if b[pos] == 0:
                        frontier.add(pos)
            player = opp

        # Reached cutoff or max_steps without terminal win -> evaluate leaf
        leaf = GomokuState(
            size=n,
            win_length=k,
            candidate_radius=self.candidate_radius,
            board=b,
            player=player,
            moves_played=moves_played,
            winner=None,
        )
        return leaf.rollout_value(perspective=1)

    def render(self) -> str:
        n = self.size
        header = "   " + " ".join(f"{c:2d}" for c in range(n))
        lines = [header]
        for r in range(n):
            row_str = " ".join(f" {SYMBOL[self.board[r * n + c]]}" for c in range(n))
            lines.append(f"{r:2d} {row_str}")
        return "\n".join(lines)

    def format_move(self, move: int) -> str:
        r, c = divmod(move, self.size)
        return f"({r},{c})"

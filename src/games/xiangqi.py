"""Complete Xiangqi (Chinese Chess / Co Tuong) game engine.

Board coordinates: 10 rows (0..9) x 9 columns (0..8).
- Row 0..4: Black side (player = -1, lowercase: k, a, e, h, r, c, p)
- Row 5..9: Red side   (player = +1, uppercase: K, A, E, H, R, C, P)
- River lies between Row 4 and Row 5.
- Palaces: cols 3..5; Black rows 0..2, Red rows 7..9.

Implements all standard rules:
- Exact movement for General, Advisor, Elephant (blocking eye & river),
  Horse (blocking leg), Chariot, Cannon (screen jump to capture), Soldier (river crossing).
- Flying General rule (generals cannot face each other on an open file).
- Strict check legality (cannot leave own general in check).
- Xiangqi terminal rule: having no legal moves (checkmate OR stalemate) is a LOSS.
- Draw by max plies or insufficient attacking material.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from .base import GameState

ROWS = 10
COLS = 9

# Piece encoding: positive = Red (+1), negative = Black (-1)
EMPTY = 0
GENERAL = 1   # K / k (Tuong)
ADVISOR = 2   # A / a (Si)
ELEPHANT = 3  # E / e (Tuong / Voi)
HORSE = 4     # H / h (Ma)
CHARIOT = 5   # R / r (Xe)
CANNON = 6    # C / c (Phao)
SOLDIER = 7   # P / p (Tot)

PIECE_CHAR = {
    0: ".",
    1: "K", 2: "A", 3: "E", 4: "H", 5: "R", 6: "C", 7: "P",
    -1: "k", -2: "a", -3: "e", -4: "h", -5: "r", -6: "c", -7: "p",
}
CHAR_PIECE = {v: k for k, v in PIECE_CHAR.items()}

PIECE_UNICODE = {
    0: "·",
    1: "帥", 2: "仕", 3: "相", 4: "傌", 5: "俥", 6: "炮", 7: "兵",
    -1: "將", -2: "士", -3: "象", -4: "馬", -5: "車", -6: "砲", -7: "卒",
}

PIECE_VALUE = {
    EMPTY: 0.0,
    GENERAL: 10000.0,
    ADVISOR: 200.0,
    ELEPHANT: 200.0,
    HORSE: 400.0,
    CHARIOT: 900.0,
    CANNON: 450.0,
    SOLDIER: 100.0,
}

# Piece-Square Tables from Red's perspective (row 0 = Black back rank, row 9 = Red back rank)
PST_SOLDIER = (
    (9,  9,  9, 11, 13, 11,  9,  9,  9),
    (19, 24, 34, 42, 44, 42, 34, 24, 19),
    (19, 24, 32, 37, 37, 37, 32, 24, 19),
    (19, 23, 27, 29, 30, 29, 27, 23, 19),
    (14, 18, 20, 27, 29, 27, 20, 18, 14),
    (7,   0, 13,  0, 16,  0, 13,  0,  7),
    (7,   0,  7,  0, 15,  0,  7,  0,  7),
    (0,   0,  0,  0,  0,  0,  0,  0,  0),
    (0,   0,  0,  0,  0,  0,  0,  0,  0),
    (0,   0,  0,  0,  0,  0,  0,  0,  0),
)

PST_HORSE = (
    (4,  8, 16, 12,  4, 12, 16,  8,  4),
    (4, 10, 28, 16,  8, 16, 28, 10,  4),
    (12, 14, 16, 20, 18, 20, 16, 14, 12),
    (8, 24, 18, 24, 20, 24, 18, 24,  8),
    (6, 16, 14, 18, 16, 18, 14, 16,  6),
    (4, 12, 16, 14, 12, 14, 16, 12,  4),
    (2,  6,  8,  6, 10,  6,  8,  6,  2),
    (4,  2,  8,  8,  4,  8,  8,  2,  4),
    (0,  2,  4,  4, -2,  4,  4,  2,  0),
    (0, -4,  0,  0,  0,  0,  0, -4,  0),
)

PST_CHARIOT = (
    (14, 14, 12, 18, 16, 18, 12, 14, 14),
    (16, 20, 18, 24, 26, 24, 18, 20, 16),
    (12, 12, 12, 18, 18, 18, 12, 12, 12),
    (12, 18, 16, 22, 22, 22, 16, 18, 12),
    (12, 14, 12, 18, 18, 18, 12, 14, 12),
    (12, 16, 14, 20, 20, 20, 14, 16, 12),
    (6,  10,  8, 14, 14, 14,  8, 10,  6),
    (4,   8,  6, 14, 12, 14,  6,  8,  4),
    (8,   4,  8, 16,  8, 16,  8,  4,  8),
    (-2, 10,  6, 14, 12, 14,  6, 10, -2),
)

PST_CANNON = (
    (6,  4,  0, -10, -12, -10,  0,  4,  6),
    (2,  2,  0,  -4, -14,  -4,  0,  2,  2),
    (2,  2,  0, -10,  -8, -10,  0,  2,  2),
    (0,  0, -2,   4,  10,   4, -2,  0,  0),
    (0,  0,  0,   2,   8,   2,  0,  0,  0),
    (-2, 0,  4,   2,   6,   2,  4,  0, -2),
    (0,  0,  0,   2,   4,   2,  0,  0,  0),
    (4,  0,  8,   6,  10,   6,  8,  0,  4),
    (0,  2,  4,   6,   6,   6,  4,  2,  0),
    (0,  0,  2,   6,   6,   6,  2,  0,  0),
)

STANDARD_START_LAYOUT = (
    "rheakaehr",
    ".........",
    ".c.....c.",
    "p.p.p.p.p",
    ".........",
    ".........",
    "P.P.P.P.P",
    ".C.....C.",
    ".........",
    "RHEAKAEHR",
)

# Balanced midgame/tactical skirmish layout for fast multi-game tournament evaluation
SKIRMISH_LAYOUT = (
    "...aka...",
    "....r....",
    "..h...c..",
    "p...p...p",
    ".........",
    ".........",
    "P...P...P",
    "..C...H..",
    "....R....",
    "...AKA...",
)

Move = tuple[int, int, int, int]  # (r1, c1, r2, c2)


def _layout_to_board(layout: tuple[str, ...]) -> list[int]:
    board = [0] * (ROWS * COLS)
    for r, row_str in enumerate(layout):
        for c, ch in enumerate(row_str):
            board[r * COLS + c] = CHAR_PIECE[ch]
    return board


class XiangqiState(GameState):
    """Xiangqi (Chinese Chess) state with full rule enforcement."""

    __slots__ = ("board", "_player", "plies", "max_plies", "_legal_cache", "_winner_cache")

    def __init__(
        self,
        board: Optional[list[int]] = None,
        player: int = 1,
        plies: int = 0,
        max_plies: int = 120,
    ) -> None:
        self.board: list[int] = board if board is not None else _layout_to_board(STANDARD_START_LAYOUT)
        self._player = player
        self.plies = plies
        self.max_plies = max_plies
        self._legal_cache: Optional[list[Move]] = None
        self._winner_cache: Optional[int] = None

    @classmethod
    def from_layout(cls, layout: tuple[str, ...], player: int = 1, max_plies: int = 120) -> "XiangqiState":
        return cls(board=_layout_to_board(layout), player=player, plies=0, max_plies=max_plies)

    @classmethod
    def skirmish(cls, max_plies: int = 80) -> "XiangqiState":
        """Reduced-piece symmetric tactical starting position for rapid benchmarking."""
        return cls.from_layout(SKIRMISH_LAYOUT, player=1, max_plies=max_plies)

    @property
    def current_player(self) -> int:
        return self._player

    def _find_general(self, player: int) -> Optional[tuple[int, int]]:
        target = GENERAL * player
        b = self.board
        r_range = range(7, 10) if player == 1 else range(0, 3)
        for r in r_range:
            off = r * COLS
            for c in range(3, 6):
                if b[off + c] == target:
                    return (r, c)
        return None

    def is_in_check(self, player: int) -> bool:
        """Return True if `player`'s General is attacked or facing the enemy General."""
        pos = self._find_general(player)
        if pos is None:
            return True
        kr, kc = pos
        b = self.board
        opp = -player

        # 1. Orthogonal rays: Chariot, Cannon, and Flying General
        opp_chariot = CHARIOT * opp
        opp_cannon = CANNON * opp
        opp_general = GENERAL * opp

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = kr + dr, kc + dc
            screen_seen = False
            while 0 <= r < ROWS and 0 <= c < COLS:
                piece = b[r * COLS + c]
                if piece != 0:
                    if not screen_seen:
                        if piece == opp_chariot or piece == opp_general:
                            return True
                        screen_seen = True
                    else:
                        if piece == opp_cannon:
                            return True
                        break
                r += dr
                c += dc

        # 2. Horse attacks (reverse-check horse leg!)
        opp_horse = HORSE * opp
        # Horse at (hr, hc) attacks (kr, kc); its leg is orthogonally adjacent to (hr, hc)
        # Equivalently, leg is diagonally adjacent to (kr, kc)!
        for dr, dc, ldr, ldc in (
            (-2, -1, -1, -1), (-2, 1, -1, 1),
            (2, -1, 1, -1),   (2, 1, 1, 1),
            (-1, -2, -1, -1), (1, -2, 1, -1),
            (-1, 2, -1, 1),   (1, 2, 1, 1),
        ):
            hr, hc = kr + dr, kc + dc
            if 0 <= hr < ROWS and 0 <= hc < COLS and b[hr * COLS + hc] == opp_horse:
                lr, lc = kr + ldr, kc + ldc
                if b[lr * COLS + lc] == 0:
                    return True

        # 3. Soldier attacks
        opp_soldier = SOLDIER * opp
        # Red general is attacked by Black soldier from above (kr - 1) or sides;
        # Black general is attacked by Red soldier from below (kr + 1) or sides.
        front_r = kr - player
        if 0 <= front_r < ROWS and b[front_r * COLS + kc] == opp_soldier:
            return True
        for sc in (kc - 1, kc + 1):
            if 0 <= sc < COLS and b[kr * COLS + sc] == opp_soldier:
                return True

        return False

    def _pseudo_legal_moves(self, player: int) -> list[Move]:
        moves: list[Move] = []
        b = self.board
        for idx, piece in enumerate(b):
            if piece * player <= 0:
                continue
            r, c = divmod(idx, COLS)
            ptype = abs(piece)

            if ptype == GENERAL:
                r_lo, r_hi = (7, 9) if player == 1 else (0, 2)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if r_lo <= nr <= r_hi and 3 <= nc <= 5:
                        if b[nr * COLS + nc] * player <= 0:
                            moves.append((r, c, nr, nc))

            elif ptype == ADVISOR:
                r_lo, r_hi = (7, 9) if player == 1 else (0, 2)
                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    nr, nc = r + dr, c + dc
                    if r_lo <= nr <= r_hi and 3 <= nc <= 5:
                        if b[nr * COLS + nc] * player <= 0:
                            moves.append((r, c, nr, nc))

            elif ptype == ELEPHANT:
                r_lo, r_hi = (5, 9) if player == 1 else (0, 4)
                for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
                    nr, nc = r + dr, c + dc
                    if r_lo <= nr <= r_hi and 0 <= nc < COLS:
                        eye_r, eye_c = r + dr // 2, c + dc // 2
                        if b[eye_r * COLS + eye_c] == 0 and b[nr * COLS + nc] * player <= 0:
                            moves.append((r, c, nr, nc))

            elif ptype == HORSE:
                for leg_dr, leg_dc, targets in (
                    (-1, 0, ((-2, -1), (-2, 1))),
                    (1, 0, ((2, -1), (2, 1))),
                    (0, -1, ((-1, -2), (1, -2))),
                    (0, 1, ((-1, 2), (1, 2))),
                ):
                    lr, lc = r + leg_dr, c + leg_dc
                    if 0 <= lr < ROWS and 0 <= lc < COLS and b[lr * COLS + lc] == 0:
                        for dr, dc in targets:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < ROWS and 0 <= nc < COLS:
                                if b[nr * COLS + nc] * player <= 0:
                                    moves.append((r, c, nr, nc))

            elif ptype == CHARIOT:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < ROWS and 0 <= nc < COLS:
                        target = b[nr * COLS + nc]
                        if target == 0:
                            moves.append((r, c, nr, nc))
                        else:
                            if target * player < 0:
                                moves.append((r, c, nr, nc))
                            break
                        nr += dr
                        nc += dc

            elif ptype == CANNON:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    jumped = False
                    while 0 <= nr < ROWS and 0 <= nc < COLS:
                        target = b[nr * COLS + nc]
                        if not jumped:
                            if target == 0:
                                moves.append((r, c, nr, nc))
                            else:
                                jumped = True
                        else:
                            if target != 0:
                                if target * player < 0:
                                    moves.append((r, c, nr, nc))
                                break
                        nr += dr
                        nc += dc

            elif ptype == SOLDIER:
                forward = -1 if player == 1 else 1
                nr = r + forward
                if 0 <= nr < ROWS and b[nr * COLS + c] * player <= 0:
                    moves.append((r, c, nr, c))
                crossed = (r <= 4) if player == 1 else (r >= 5)
                if crossed:
                    for dc in (-1, 1):
                        nc = c + dc
                        if 0 <= nc < COLS and b[r * COLS + nc] * player <= 0:
                            moves.append((r, c, r, nc))

        return moves

    def _has_sufficient_material(self) -> bool:
        for p in self.board:
            ap = abs(p)
            if ap in (CHARIOT, CANNON, HORSE, SOLDIER):
                return True
        return False

    def legal_moves(self) -> list[Move]:
        if self._legal_cache is not None:
            return self._legal_cache
        if self.plies >= self.max_plies or not self._has_sufficient_material():
            self._legal_cache = []
            return self._legal_cache

        p = self._player
        b = self.board
        legal: list[Move] = []
        for move in self._pseudo_legal_moves(p):
            r1, c1, r2, c2 = move
            i1, i2 = r1 * COLS + c1, r2 * COLS + c2
            captured = b[i2]
            moved = b[i1]
            b[i2] = moved
            b[i1] = 0
            if not self.is_in_check(p):
                legal.append(move)
            b[i1] = moved
            b[i2] = captured
        self._legal_cache = legal
        return legal

    def ordered_moves(self) -> list[Move]:
        """Order moves with MVV-LVA (Most Valuable Victim - Least Valuable Attacker)."""
        moves = list(self.legal_moves())
        if len(moves) <= 1:
            return moves
        b = self.board

        def score_move(m: Move) -> float:
            r1, c1, r2, c2 = m
            victim = abs(b[r2 * COLS + c2])
            attacker = abs(b[r1 * COLS + c1])
            s = 0.0
            if victim != 0:
                s += 10.0 * PIECE_VALUE[victim] - PIECE_VALUE[attacker]
            # Slight bonus for central files and advancing
            s += (4 - abs(c2 - 4)) * 2.0
            return s

        moves.sort(key=score_move, reverse=True)
        return moves

    def apply_move(self, move: Move) -> "XiangqiState":
        r1, c1, r2, c2 = move
        i1, i2 = r1 * COLS + c1, r2 * COLS + c2
        if self.board[i1] * self._player <= 0:
            raise ValueError(f"Illegal Xiangqi move: {move}")
        new_board = self.board.copy()
        new_board[i2] = new_board[i1]
        new_board[i1] = 0
        return XiangqiState(
            board=new_board,
            player=-self._player,
            plies=self.plies + 1,
            max_plies=self.max_plies,
        )

    def winner(self) -> Optional[int]:
        if self._winner_cache is not None:
            return self._winner_cache
        if self._find_general(1) is None:
            self._winner_cache = -1
            return -1
        if self._find_general(-1) is None:
            self._winner_cache = 1
            return 1
        if self.plies >= self.max_plies or not self._has_sufficient_material():
            self._winner_cache = 0
            return 0
        if not self.legal_moves():
            # In Xiangqi, both checkmate and stalemate are a LOSS for the side to move
            self._winner_cache = -self._player
            return self._winner_cache
        return None

    def evaluate(self, perspective: int) -> float:
        w = self.winner()
        if w is not None:
            if w == 0:
                return 0.0
            bonus = (self.max_plies - self.plies) * 2.0
            return (100_000.0 + bonus) if w == perspective else -(100_000.0 + bonus)

        b = self.board
        score = 0.0
        for idx, piece in enumerate(b):
            if piece == 0:
                continue
            r, c = divmod(idx, COLS)
            sign = 1.0 if piece > 0 else -1.0
            ptype = abs(piece)
            val = PIECE_VALUE[ptype]
            pst_r = r if piece > 0 else (9 - r)
            if ptype == SOLDIER:
                val += PST_SOLDIER[pst_r][c]
                if pst_r <= 4:
                    val += 80.0
            elif ptype == HORSE:
                val += PST_HORSE[pst_r][c]
            elif ptype == CHARIOT:
                val += PST_CHARIOT[pst_r][c]
            elif ptype == CANNON:
                val += PST_CANNON[pst_r][c]
            score += sign * val

        return score if perspective == 1 else -score

    def state_key(self) -> tuple[bytes, int]:
        return (bytes(p + 10 for p in self.board), self._player)

    def rollout_move(self, rng: random.Random) -> Move:
        """Tactical rollout: prioritize high-value captures, else random legal move."""
        moves = self.legal_moves()
        b = self.board
        if rng.random() < 0.85:
            best_cap_val = 0.0
            best_caps: list[Move] = []
            for m in moves:
                victim = abs(b[m[2] * COLS + m[3]])
                if victim != 0:
                    val = PIECE_VALUE[victim]
                    if val > best_cap_val:
                        best_cap_val = val
                        best_caps = [m]
                    elif val == best_cap_val:
                        best_caps.append(m)
            if best_caps:
                return rng.choice(best_caps)
        return rng.choice(moves)

    def rollout_value(self, perspective: int) -> float:
        w = self.winner()
        if w is not None:
            if w == 0:
                return 0.5
            return 1.0 if w == perspective else 0.0
        score = self.evaluate(perspective)
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score / 600.0))))

    def render(self) -> str:
        header = "    " + " ".join(f"{c}" for c in range(COLS))
        lines = [header, "   +" + "--" * COLS + "+"]
        for r in range(ROWS):
            row_chars = " ".join(PIECE_CHAR[self.board[r * COLS + c]] for c in range(COLS))
            lines.append(f" {r} |{row_chars}|")
            if r == 4:
                lines.append("   |~~~ RIVER ~~~ ~~~|")
        lines.append("   +" + "--" * COLS + "+")
        return "\n".join(lines)

    def format_move(self, move: Move) -> str:
        r1, c1, r2, c2 = move
        p = PIECE_CHAR[self.board[r1 * COLS + c1]]
        return f"{p}({r1},{c1})->({r2},{c2})"

    @staticmethod
    def player_name(player: int) -> str:
        return "Red" if player == 1 else "Black"

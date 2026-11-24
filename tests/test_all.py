"""Comprehensive verification test suite for all games, algorithms, and bonus tasks."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithms.mcts import mcts_search
from src.algorithms.minimax import alphabeta_search, minimax_search
from src.bonus.delivery_scheduling import (
    generate_instance,
    solve_branch_and_bound,
    solve_edf,
    solve_mcts,
    solve_nearest_neighbor,
)
from src.games.gomoku import GomokuState
from src.games.tictactoe import TicTacToeState
from src.games.xiangqi import XiangqiState


class TestTicTacToe(unittest.TestCase):
    def test_minimax_alphabeta_equivalence_and_pruning(self) -> None:
        s = TicTacToeState().apply_move(4).apply_move(0)
        mm = minimax_search(s, depth=9, use_move_ordering=False)
        ab = alphabeta_search(s, depth=9, use_move_ordering=True, use_tt=False)
        ab_tt = alphabeta_search(s, depth=9, use_move_ordering=True, use_tt=True)
        self.assertAlmostEqual(mm.best_value, ab.best_value, places=5)
        self.assertAlmostEqual(mm.best_value, ab_tt.best_value, places=5)
        self.assertLess(ab.nodes_visited, mm.nodes_visited)
        self.assertGreater(ab.cutoffs, 0)
        self.assertLessEqual(ab_tt.nodes_visited, ab.nodes_visited)

    def test_empty_board_exact_draw_value(self) -> None:
        s = TicTacToeState()
        ab = alphabeta_search(s, depth=9, use_move_ordering=True, use_tt=True)
        self.assertAlmostEqual(ab.best_value, 0.0, places=5)

    def test_mcts_never_loses_against_optimal_minimax(self) -> None:
        # Play 6 full games (3 as X, 3 as O) vs exact Alpha-Beta
        for game_idx in range(6):
            state = TicTacToeState()
            mcts_side = 1 if game_idx % 2 == 0 else -1
            while not state.is_terminal():
                if state.current_player == mcts_side:
                    res = mcts_search(
                        state,
                        iterations=800,
                        exploration=math.sqrt(2.0),
                        rollout_policy="random",
                        seed=100 + game_idx * 17 + state._moves_played,
                    )
                    move = res.best_move
                else:
                    move = alphabeta_search(state, depth=9, use_tt=True).best_move
                state = state.apply_move(move)
            self.assertNotEqual(state.winner(), -mcts_side, "MCTS should never lose 3x3 TTT!")


class TestGomoku5InARow(unittest.TestCase):
    def test_four_direction_win_detection(self) -> None:
        # Horizontal, vertical, main diag, anti-diag
        patterns = [
            [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5)],
            [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4)],
            [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
            [(1, 7), (2, 6), (3, 5), (4, 4), (5, 3)],
        ]
        for coords in patterns:
            s = GomokuState(size=9, win_length=5)
            # Interleave dummy O moves far away
            o_moves = [(0, 0), (0, 1), (0, 2), (0, 3)]
            for idx, (r, c) in enumerate(coords):
                s = s.apply_move(s.idx(r, c))
                if idx < 4:
                    self.assertIsNone(s.winner())
                    s = s.apply_move(s.idx(*o_moves[idx]))
            self.assertEqual(s.winner(), 1)

    def test_heuristic_rollout_blocks_opponent_immediate_win(self) -> None:
        # X has 4 in a row at row 4 cols 2..5; O must block at (4,1) or (4,6)
        s = GomokuState(size=9, win_length=5)
        seq = [(4, 2), (0, 0), (4, 3), (0, 1), (4, 4), (0, 2), (4, 5)]
        for r, c in seq:
            s = s.apply_move(s.idx(r, c))
        self.assertEqual(s.current_player, -1)
        move = s.rollout_move(random.Random(1))
        self.assertIn(move, {s.idx(4, 1), s.idx(4, 6)})

    def test_minimax_alphabeta_exact_equivalence(self) -> None:
        s = GomokuState(size=7, win_length=5, candidate_radius=1)
        s = s.apply_move(s.idx(3, 3)).apply_move(s.idx(3, 4)).apply_move(s.idx(4, 3))
        mm = minimax_search(s, depth=2, use_move_ordering=False)
        ab = alphabeta_search(s, depth=2, use_move_ordering=True, use_tt=False)
        self.assertAlmostEqual(mm.best_value, ab.best_value, places=5)
        self.assertLess(ab.nodes_visited, mm.nodes_visited)

    def test_all_uct_variants_and_rollouts(self) -> None:
        s = GomokuState(size=7, win_length=5)
        s = s.apply_move(s.idx(3, 3)).apply_move(s.idx(3, 4))
        for variant in ("uct", "no_sqrt", "no_ln", "linear", "greedy", "ucb1_tuned", "puct"):
            for policy in ("random", "heuristic", "value_cutoff"):
                res = mcts_search(
                    s,
                    iterations=40,
                    exploration=1.4,
                    uct_variant=variant,
                    rollout_policy=policy,
                    seed=7,
                )
                self.assertIn(res.best_move, s.legal_moves())
                self.assertEqual(res.root_visits, 40)


class TestXiangqi(unittest.TestCase):
    def test_initial_legal_moves_count(self) -> None:
        # Standard Xiangqi starting position has exactly 44 legal moves for Red!
        s = XiangqiState()
        self.assertEqual(len(s.legal_moves()), 44)

    def test_flying_general_prohibited(self) -> None:
        # Two generals facing each other with a single Rook between them;
        # Red Rook cannot move sideways off file 4 because it exposes Flying General!
        layout = (
            "....k....",
            ".........",
            ".........",
            ".........",
            ".........",
            "....R....",
            ".........",
            ".........",
            ".........",
            "....K....",
        )
        s = XiangqiState.from_layout(layout, player=1)
        rook_moves = [m for m in s.legal_moves() if m[0] == 5 and m[1] == 4]
        for r1, c1, r2, c2 in rook_moves:
            self.assertEqual(c2, 4, "Rook cannot leave file 4 due to Flying General rule")

    def test_horse_leg_blocking(self) -> None:
        # Red horse at (5,4) blocked above at (4,4) by a soldier -> cannot jump to (3,3) or (3,5)
        layout = (
            "....k....",
            ".........",
            ".........",
            ".........",
            "....P....",
            "....H....",
            ".........",
            ".........",
            ".........",
            "...K.....",
        )
        s = XiangqiState.from_layout(layout, player=1)
        horse_targets = {(m[2], m[3]) for m in s.legal_moves() if m[0] == 5 and m[1] == 4}
        self.assertNotIn((3, 3), horse_targets)
        self.assertNotIn((3, 5), horse_targets)
        self.assertIn((7, 3), horse_targets)

    def test_cannon_jump_capture(self) -> None:
        # Red cannon at (5,4), screen soldier at (3,4), black chariot at (1,4) -> can capture (1,4)!
        layout = (
            "...k.....",
            "....r....",
            ".........",
            "....P....",
            ".........",
            "....C....",
            ".........",
            ".........",
            ".........",
            "....K....",
        )
        s = XiangqiState.from_layout(layout, player=1)
        cannon_targets = {(m[2], m[3]) for m in s.legal_moves() if m[0] == 5 and m[1] == 4}
        self.assertIn((1, 4), cannon_targets)
        self.assertNotIn((2, 4), cannon_targets)  # cannot land on empty square behind screen

    def test_minimax_alphabeta_xiangqi_equivalence(self) -> None:
        s = XiangqiState.skirmish()
        mm = minimax_search(s, depth=2, use_move_ordering=False)
        ab = alphabeta_search(s, depth=2, use_move_ordering=True, use_tt=False)
        self.assertAlmostEqual(mm.best_value, ab.best_value, places=4)
        self.assertLess(ab.nodes_visited, mm.nodes_visited)


class TestBonusDelivery(unittest.TestCase):
    def test_mcts_improves_over_greedy_and_approaches_optimal(self) -> None:
        inst = generate_instance(num_orders=8, seed=42)
        nn = solve_nearest_neighbor(inst)
        edf = solve_edf(inst)
        bnb = solve_branch_and_bound(inst)
        mcts_h = solve_mcts(inst, iterations=1000, heuristic_rollout=True, seed=42)
        self.assertLessEqual(bnb.total_cost, nn.total_cost + 1e-6)
        self.assertLessEqual(bnb.total_cost, edf.total_cost + 1e-6)
        self.assertLessEqual(mcts_h.total_cost, nn.total_cost + 1e-6)
        self.assertEqual(len(set(mcts_h.route)), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Interactive CLI Demo for Tic-Tac-Toe 3x3, Gomoku 5-in-a-row, Xiangqi, and Delivery Bonus.

Usage examples:
    python -m demo.cli --game tictactoe --algo mcts --iterations 1000 --human
    python -m demo.cli --game gomoku --size 9 --algo mcts --rollout heuristic --iterations 1000 --human
    python -m demo.cli --game xiangqi --algo alphabeta --depth 2 --human
    python -m demo.cli --game delivery --orders 12 --iterations 2000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithms.mcts import MCTSResult, mcts_search
from src.algorithms.minimax import SearchResult, alphabeta_search, minimax_search
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


def print_mcts_stats(state, res: MCTSResult, top_k: int = 8) -> None:
    print(
        f"  [MCTS] root_visits={res.root_visits}, tree_nodes={res.tree_nodes}, "
        f"time={res.elapsed_ms:.1f} ms"
    )
    print(f"  {'Move':<14} {'W':>8} {'N':>6} {'W/N':>8} {'UCT':>8}")
    print("  " + "-" * 48)
    for st in res.root_stats[:top_k]:
        m_str = state.format_move(st.move) if hasattr(state, "format_move") else str(st.move)
        print(f"  {m_str:<14} {st.wins:8.1f} {st.visits:6d} {st.win_rate:8.4f} {st.uct_score:8.4f}")


def print_minimax_stats(state, res: SearchResult) -> None:
    m_str = state.format_move(res.best_move) if hasattr(state, "format_move") else str(res.best_move)
    print(
        f"  [Search] best={m_str}, value={res.best_value:.2f}, "
        f"nodes={res.nodes_visited}, cutoffs={res.cutoffs}, time={res.elapsed_ms:.1f} ms"
    )


def get_ai_move(state, args: argparse.Namespace):
    if args.algo == "minimax":
        res = minimax_search(state, depth=args.depth, use_move_ordering=False)
        print_minimax_stats(state, res)
        return res.best_move
    elif args.algo == "alphabeta":
        res = alphabeta_search(state, depth=args.depth, use_move_ordering=True, use_tt=True)
        print_minimax_stats(state, res)
        return res.best_move
    else:
        res = mcts_search(
            state,
            iterations=args.iterations,
            exploration=args.c,
            uct_variant=args.uct_variant,
            rollout_policy=args.rollout,
            seed=args.seed,
        )
        print_mcts_stats(state, res)
        return res.best_move


def prompt_human_move(state, game_name: str):
    legal = state.all_empty_moves() if hasattr(state, "all_empty_moves") else state.legal_moves()
    legal_set = set(legal)
    while True:
        try:
            if game_name == "tictactoe":
                raw = input("Your move (cell 1..9): ").strip()
                move = int(raw) - 1
            elif game_name == "gomoku":
                raw = input(f"Your move (row col, 0..{state.size - 1}): ").strip()
                r, c = map(int, raw.split())
                move = r * state.size + c
            else:
                raw = input("Your move (r1 c1 r2 c2): ").strip()
                r1, c1, r2, c2 = map(int, raw.split())
                move = (r1, c1, r2, c2)
            if move in legal_set:
                return move
            print("Illegal move! Try again.")
        except (ValueError, IndexError):
            print("Invalid input format! Try again.")


def run_board_demo(args: argparse.Namespace) -> None:
    if args.game == "tictactoe":
        state = TicTacToeState()
    elif args.game == "gomoku":
        state = GomokuState(size=args.size, win_length=5)
    elif args.game == "xiangqi":
        state = XiangqiState.skirmish() if args.skirmish else XiangqiState()
    else:
        raise ValueError(args.game)

    human_side = -1 if args.human_second else 1
    print(f"=== Interactive Demo: {args.game.upper()} ({args.algo}) ===")
    while not state.is_terminal():
        print("\n" + state.render())
        pname = state.player_name(state.current_player)
        print(f"\nTurn: {pname}")
        if args.human and state.current_player == human_side:
            move = prompt_human_move(state, args.game)
        else:
            move = get_ai_move(state, args)
            m_str = state.format_move(move) if hasattr(state, "format_move") else str(move)
            print(f"  -> AI ({pname}) plays {m_str}")
        state = state.apply_move(move)

    print("\n" + state.render())
    w = state.winner()
    if w == 0:
        print("\nResult: DRAW!")
    else:
        print(f"\nResult: {state.player_name(w)} WINS!")


def run_delivery_demo(args: argparse.Namespace) -> None:
    inst = generate_instance(num_orders=args.orders, seed=args.seed)
    print(f"=== Bonus Task 2: Last-Mile Delivery Routing ({args.orders} orders, seed={args.seed}) ===")
    sols = [
        solve_nearest_neighbor(inst),
        solve_edf(inst),
        solve_mcts(inst, iterations=args.iterations, exploration=args.c, heuristic_rollout=False, seed=args.seed),
        solve_mcts(inst, iterations=args.iterations, exploration=args.c, heuristic_rollout=True, seed=args.seed),
    ]
    if args.orders <= 9:
        sols.append(solve_branch_and_bound(inst))

    print(f"{'Method':<26} {'Total Cost':>11} {'Distance':>10} {'Lateness':>10} {'Late #':>7} {'Time(ms)':>10}")
    print("-" * 78)
    for s in sols:
        print(
            f"{s.method:<26} {s.total_cost:11.2f} {s.travel_distance:10.2f} "
            f"{s.total_lateness:10.2f} {s.late_orders:7d} {s.elapsed_ms:10.1f}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AI BTL Interactive Demo")
    ap.add_argument("--game", choices=["tictactoe", "gomoku", "xiangqi", "delivery"], default="gomoku")
    ap.add_argument("--algo", choices=["minimax", "alphabeta", "mcts"], default="mcts")
    ap.add_argument("--size", type=int, default=9, help="Gomoku board size")
    ap.add_argument("--depth", type=int, default=2, help="Minimax/Alpha-Beta depth")
    ap.add_argument("--iterations", type=int, default=1000, help="MCTS iterations")
    ap.add_argument("--c", type=float, default=math.sqrt(2.0), help="UCT exploration constant")
    ap.add_argument("--uct-variant", default="uct", choices=["uct", "no_sqrt", "no_ln", "linear", "greedy", "ucb1_tuned", "puct"])
    ap.add_argument("--rollout", default="heuristic", choices=["random", "heuristic", "value_cutoff"])
    ap.add_argument("--human", action="store_true", help="Play as human against AI")
    ap.add_argument("--human-second", action="store_true", help="Human plays O / Black (second)")
    ap.add_argument("--skirmish", action="store_true", help="Use Xiangqi skirmish preset")
    ap.add_argument("--orders", type=int, default=12, help="Number of delivery orders")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.game == "delivery":
        run_delivery_demo(args)
    else:
        run_board_demo(args)


if __name__ == "__main__":
    main()

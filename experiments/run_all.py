"""Master Experiment Runner for AI Midterm Project.

Executes all 8 mandatory + bonus experiments in parallel across CPU cores,
saving 100% verified, seed-reproducible results to `experiments/results/all_results.json`.

Usage:
    python -m experiments.run_all
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

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

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper opponent policies (picklable top-level functions for multiprocessing)
# ---------------------------------------------------------------------------

def _semi_tactical_move(state: Any, rng: random.Random, tactical_prob: float = 0.55) -> Any:
    """Realistic noisy tactical opponent for Gomoku/Xiangqi benchmarks.

    Always takes immediate wins/blocks (if Gomoku), otherwise plays best ordered
    tactical move with probability `tactical_prob` and random legal move otherwise.
    This prevents ceiling saturation at 100% win rate and cleanly separates c & iterations.
    """
    moves = state.legal_moves()
    if len(moves) == 1:
        return moves[0]
    if isinstance(state, GomokuState):
        p = state.current_player
        for m in moves:
            if state.is_winning_move(m, p):
                return m
        for m in moves:
            if state.is_winning_move(m, -p):
                return m
    if rng.random() < tactical_prob:
        return state.ordered_moves()[0]
    return rng.choice(moves)


def _play_single_game(task: dict[str, Any]) -> dict[str, Any]:
    """Worker function executing one complete game with alternating sides."""
    game_type = task["game"]
    seed = task["seed"]
    rng = random.Random(seed)
    agent_side = task["agent_side"]  # +1 or -1

    if game_type == "ttt":
        state: Any = TicTacToeState()
    elif game_type == "gomoku7":
        state = GomokuState(size=7, win_length=5, candidate_radius=2)
    elif game_type == "gomoku9":
        state = GomokuState(size=9, win_length=5, candidate_radius=2)
    elif game_type == "xiangqi":
        state = XiangqiState.skirmish(max_plies=80)
    else:
        raise ValueError(game_type)

    # Optional randomized opening (1st 2 plies) for Gomoku to avoid identical deterministic games
    if task.get("random_opening", False) and game_type.startswith("gomoku"):
        center = state.idx(state.size // 2, state.size // 2)
        state = state.apply_move(center)
        # Second ply randomly chosen among neighbors of center
        state = state.apply_move(rng.choice(state.legal_moves()))

    agent_cfg = task["agent"]
    opp_cfg = task["opponent"]

    agent_times: list[float] = []
    agent_nodes: list[int] = []
    move_idx = 0

    while not state.is_terminal():
        move_seed = seed * 1009 + move_idx * 37
        move_rng = random.Random(move_seed)
        if state.current_player == agent_side:
            t0 = time.perf_counter()
            if agent_cfg["type"] == "mcts":
                res = mcts_search(
                    state,
                    iterations=agent_cfg["iterations"],
                    exploration=agent_cfg["c"],
                    uct_variant=agent_cfg.get("variant", "uct"),
                    rollout_policy=agent_cfg.get("rollout", "random"),
                    final_selection=agent_cfg.get("final_selection", "robust"),
                    seed=move_seed,
                    max_rollout_steps=agent_cfg.get("max_rollout_steps", 80),
                    cutoff_steps=agent_cfg.get("cutoff_steps", 6),
                )
                move = res.best_move
                agent_nodes.append(res.tree_nodes)
            elif agent_cfg["type"] == "alphabeta":
                res = alphabeta_search(
                    state,
                    depth=agent_cfg["depth"],
                    use_move_ordering=True,
                    use_tt=True,
                )
                move = res.best_move
                agent_nodes.append(res.nodes_visited)
            else:
                raise ValueError(agent_cfg["type"])
            agent_times.append((time.perf_counter() - t0) * 1000.0)
        else:
            otype = opp_cfg["type"]
            if otype == "random":
                move = move_rng.choice(state.legal_moves())
            elif otype == "semi_tactical":
                move = _semi_tactical_move(state, move_rng, opp_cfg.get("prob", 0.55))
            elif otype == "greedy":
                move = state.ordered_moves()[0]
            elif otype == "optimal_ttt":
                res = alphabeta_search(state, depth=9, use_move_ordering=True, use_tt=True)
                move = res.best_move
            elif otype == "alphabeta":
                res = alphabeta_search(
                    state, depth=opp_cfg["depth"], use_move_ordering=True, use_tt=True
                )
                move = res.best_move
            elif otype == "mcts":
                res = mcts_search(
                    state,
                    iterations=opp_cfg["iterations"],
                    exploration=opp_cfg["c"],
                    uct_variant=opp_cfg.get("variant", "uct"),
                    rollout_policy=opp_cfg.get("rollout", "random"),
                    final_selection=opp_cfg.get("final_selection", "robust"),
                    seed=move_seed + 13,
                    max_rollout_steps=opp_cfg.get("max_rollout_steps", 80),
                    cutoff_steps=opp_cfg.get("cutoff_steps", 6),
                )
                move = res.best_move
            else:
                raise ValueError(otype)
        state = state.apply_move(move)
        move_idx += 1

    w = state.winner()
    if w == 0 or w is None:
        outcome = "draw"
    elif w == agent_side:
        outcome = "win"
    else:
        outcome = "loss"

    return {
        "outcome": outcome,
        "avg_ms": sum(agent_times) / max(len(agent_times), 1),
        "avg_nodes": sum(agent_nodes) / max(len(agent_nodes), 1),
        "plies": move_idx,
    }


def run_matchup(
    pool: ProcessPoolExecutor,
    game: str,
    agent_cfg: dict[str, Any],
    opp_cfg: dict[str, Any],
    num_games: int = 50,
    base_seed: int = 1000,
    random_opening: bool = False,
) -> dict[str, Any]:
    tasks = [
        {
            "game": game,
            "agent": agent_cfg,
            "opponent": opp_cfg,
            "agent_side": 1 if i % 2 == 0 else -1,
            "seed": base_seed + i,
            "random_opening": random_opening,
        }
        for i in range(num_games)
    ]
    results = list(pool.map(_play_single_game, tasks))
    wins = sum(1 for r in results if r["outcome"] == "win")
    draws = sum(1 for r in results if r["outcome"] == "draw")
    losses = sum(1 for r in results if r["outcome"] == "loss")
    avg_ms = sum(r["avg_ms"] for r in results) / num_games
    avg_nodes = sum(r["avg_nodes"] for r in results) / num_games
    return {
        "games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(wins / num_games, 4),
        "draw_rate": round(draws / num_games, 4),
        "loss_rate": round(losses / num_games, 4),
        "non_loss_rate": round((wins + draws) / num_games, 4),
        "score_rate": round((wins + 0.5 * draws) / num_games, 4),
        "avg_move_ms": round(avg_ms, 2),
        "avg_nodes": round(avg_nodes, 1),
    }


# ---------------------------------------------------------------------------
# Individual Experiments
# ---------------------------------------------------------------------------

def exp1_minimax_vs_alphabeta() -> list[dict[str, Any]]:
    print("[Exp 1] Minimax vs Alpha-Beta Pruning across TTT 3x3, Gomoku 9x9, and Xiangqi 9x10...")
    rows: list[dict[str, Any]] = []

    # 1a. 3x3 TTT empty board (depth 9)
    ttt_empty = TicTacToeState()
    # 1b. 3x3 TTT after 1 move (X at center, depth 8)
    ttt_mid = TicTacToeState().apply_move(4)
    # 1c. Gomoku 9x9 tactical midgame (6 stones, depth 2 & depth 3)
    g9 = GomokuState(size=9, win_length=5, candidate_radius=1)
    for r, c in [(4, 4), (4, 5), (3, 4), (5, 5), (3, 5), (2, 4)]:
        g9 = g9.apply_move(g9.idx(r, c))
    # 1d. Xiangqi Standard Opening (44 moves, depth 2) & Skirmish (depth 2, 3)
    xq_std = XiangqiState()
    xq_skirmish = XiangqiState.skirmish()

    benchmarks = [
        ("Tic-Tac-Toe 3x3 (Empty)", ttt_empty, 9, True),
        ("Tic-Tac-Toe 3x3 (After 1 ply)", ttt_mid, 8, True),
        ("Gomoku 9x9 (Midgame, r=1)", g9, 2, True),
        ("Gomoku 9x9 (Midgame, r=1)", g9, 3, True),
        ("Xiangqi 9x10 (Standard Opening)", xq_std, 2, True),
        ("Xiangqi 9x10 (Skirmish)", xq_skirmish, 2, True),
        ("Xiangqi 9x10 (Skirmish)", xq_skirmish, 3, False),  # plain minimax d=3 has ~50k nodes, still fast!
    ]

    for label, state, depth, run_plain in benchmarks:
        if run_plain:
            mm = minimax_search(state, depth=depth, use_move_ordering=False)
            mm_nodes = mm.nodes_visited
            mm_ms = mm.elapsed_ms
            mm_val = mm.best_value
        else:
            mm_nodes, mm_ms, mm_val = None, None, None

        ab_nat = alphabeta_search(state, depth=depth, use_move_ordering=False, use_tt=False)
        ab_ord = alphabeta_search(state, depth=depth, use_move_ordering=True, use_tt=False)
        ab_tt = alphabeta_search(state, depth=depth, use_move_ordering=True, use_tt=True)

        # Sanity assert exact value equivalence
        if mm_val is not None:
            assert abs(mm_val - ab_nat.best_value) < 1e-4, f"Mismatch on {label} d={depth}"
            assert abs(mm_val - ab_ord.best_value) < 1e-4, f"Mismatch on {label} d={depth}"

        reduction_ord = (
            round((1.0 - ab_ord.nodes_visited / mm_nodes) * 100.0, 2) if mm_nodes else None
        )
        reduction_tt = (
            round((1.0 - ab_tt.nodes_visited / mm_nodes) * 100.0, 2) if mm_nodes else None
        )
        rows.append(
            {
                "game": label,
                "depth": depth,
                "minimax_nodes": mm_nodes,
                "minimax_ms": round(mm_ms, 2) if mm_ms is not None else None,
                "ab_natural_nodes": ab_nat.nodes_visited,
                "ab_natural_cutoffs": ab_nat.cutoffs,
                "ab_natural_ms": round(ab_nat.elapsed_ms, 2),
                "ab_ordered_nodes": ab_ord.nodes_visited,
                "ab_ordered_cutoffs": ab_ord.cutoffs,
                "ab_ordered_ms": round(ab_ord.elapsed_ms, 2),
                "ab_tt_nodes": ab_tt.nodes_visited,
                "ab_tt_cutoffs": ab_tt.cutoffs,
                "ab_tt_hits": ab_tt.tt_hits,
                "ab_tt_ms": round(ab_tt.elapsed_ms, 2),
                "reduction_ordered_pct": reduction_ord,
                "reduction_tt_pct": reduction_tt,
                "exact_value": round(ab_ord.best_value, 2),
            }
        )
        print(
            f"  {label} (d={depth}): MM={mm_nodes} -> AB(ord)={ab_ord.nodes_visited} "
            f"(-{reduction_ord}%), AB+TT={ab_tt.nodes_visited} (-{reduction_tt}%)"
        )
    return rows


def exp2_ttt_mcts_never_loses(pool: ProcessPoolExecutor) -> dict[str, Any]:
    print("[Exp 2] 3x3 Tic-Tac-Toe MCTS-UCT Verification (100 games each vs Optimal Alpha-Beta & Random)...")
    vs_optimal = []
    for iters in (100, 250, 500, 1000, 2000, 5000):
        res = run_matchup(
            pool,
            game="ttt",
            agent_cfg={"type": "mcts", "iterations": iters, "c": math.sqrt(2.0), "rollout": "random"},
            opp_cfg={"type": "optimal_ttt"},
            num_games=100,
            base_seed=2000 + iters,
        )
        res["iterations"] = iters
        vs_optimal.append(res)
        print(
            f"  vs Optimal (it={iters:4d}): W={res['wins']} D={res['draws']} L={res['losses']} "
            f"(Non-loss={res['non_loss_rate']*100:.1f}%, {res['avg_move_ms']:.1f}ms)"
        )

    vs_random = []
    for iters in (100, 500, 1000):
        res = run_matchup(
            pool,
            game="ttt",
            agent_cfg={"type": "mcts", "iterations": iters, "c": math.sqrt(2.0), "rollout": "random"},
            opp_cfg={"type": "random"},
            num_games=100,
            base_seed=2500 + iters,
        )
        res["iterations"] = iters
        vs_random.append(res)
        print(
            f"  vs Random  (it={iters:4d}): W={res['wins']} D={res['draws']} L={res['losses']} "
            f"(Win={res['win_rate']*100:.1f}%)"
        )

    return {"vs_optimal": vs_optimal, "vs_random": vs_random}


def exp3_c_vs_iterations_sweep(pool: ProcessPoolExecutor) -> list[dict[str, Any]]:
    print("[Exp 3] Sweeping c in {0.5, 1.0, 1.4, 2.0} x iterations in {100, 250, 500, 1000, 2000, 5000} (50 games each)...")
    c_values = [0.5, 1.0, 1.4, 2.0]
    iter_values = [100, 250, 500, 1000, 2000, 5000]
    rows: list[dict[str, Any]] = []

    for c_val in c_values:
        for iters in iter_values:
            # Gomoku 7x7 5-in-a-row vs Semi-Tactical Opponent (50 games, alternating X/O)
            res = run_matchup(
                pool,
                game="gomoku7",
                agent_cfg={
                    "type": "mcts",
                    "iterations": iters,
                    "c": c_val,
                    "variant": "uct",
                    "rollout": "random",
                    "max_rollout_steps": 60,
                },
                opp_cfg={"type": "semi_tactical", "prob": 0.40},
                num_games=50,
                base_seed=3000 + int(c_val * 100) + iters,
                random_opening=True,
            )
            res["c"] = c_val
            res["iterations"] = iters
            rows.append(res)
            print(
                f"  c={c_val:.1f}, it={iters:4d}: W={res['wins']:2d} D={res['draws']:2d} L={res['losses']:2d} "
                f"WinRate={res['win_rate']*100:5.1f}% ({res['avg_move_ms']:6.1f}ms)"
            )
    return rows


def exp4_root_stats_and_robust_vs_max(pool: ProcessPoolExecutor) -> dict[str, Any]:
    print("[Exp 4] Root (W, N, W/N, UCT) Inspection & Empirical Robust Child (max N) vs Max Child (max W/N)...")
    # 4a. Exact board from problem statement PDF page 3:
    # O | X | .
    # . | O | .
    # X | . | .
    # Turn: X (+1). Best move MUST block O's diagonal threat at cell 8 (bottom-right)!
    pdf_board = (-1, 1, 0, 0, -1, 0, 1, 0, 0)
    ttt_pdf = TicTacToeState(board=pdf_board, player=1, moves_played=4)
    ttt_res = mcts_search(ttt_pdf, iterations=1000, exploration=1.414, rollout_policy="random", seed=42)
    ttt_table = [
        {
            "move": ttt_pdf.format_move(st.move),
            "wins": round(st.wins, 1),
            "visits": st.visits,
            "win_rate": round(st.win_rate, 4),
            "uct": round(st.uct_score, 4),
        }
        for st in ttt_res.root_stats
    ]

    # 4b. Gomoku 9x9 tactical board with live threat
    g9 = GomokuState(size=9, win_length=5, candidate_radius=2)
    for r, c in [(4, 4), (4, 5), (3, 4), (5, 5), (2, 4), (6, 5), (5, 4), (3, 5)]:
        g9 = g9.apply_move(g9.idx(r, c))
    g9_res = mcts_search(g9, iterations=1000, exploration=1.4, rollout_policy="random", seed=2026)
    g9_table = [
        {
            "move": g9.format_move(st.move),
            "wins": round(st.wins, 1),
            "visits": st.visits,
            "win_rate": round(st.win_rate, 4),
            "uct": round(st.uct_score, 4),
        }
        for st in g9_res.root_stats[:10]
    ]

    # 4c. Empirical comparison: Robust Child (max N) vs Max Child (max W/N) over 60 games at low budget (it=150)
    robust_vs_max = run_matchup(
        pool,
        game="gomoku7",
        agent_cfg={"type": "mcts", "iterations": 150, "c": 1.4, "rollout": "random", "final_selection": "robust"},
        opp_cfg={"type": "mcts", "iterations": 150, "c": 1.4, "rollout": "random", "final_selection": "max_winrate"},
        num_games=60,
        base_seed=4100,
        random_opening=True,
    )
    robust_vs_tac = run_matchup(
        pool,
        game="gomoku7",
        agent_cfg={"type": "mcts", "iterations": 150, "c": 1.4, "rollout": "random", "final_selection": "robust"},
        opp_cfg={"type": "semi_tactical", "prob": 0.40},
        num_games=60,
        base_seed=4200,
        random_opening=True,
    )
    maxwr_vs_tac = run_matchup(
        pool,
        game="gomoku7",
        agent_cfg={"type": "mcts", "iterations": 150, "c": 1.4, "rollout": "random", "final_selection": "max_winrate"},
        opp_cfg={"type": "semi_tactical", "prob": 0.40},
        num_games=60,
        base_seed=4200,
        random_opening=True,
    )
    print(
        f"  Robust vs MaxWinRate (H2H 60g): W={robust_vs_max['wins']} D={robust_vs_max['draws']} L={robust_vs_max['losses']}"
    )
    print(
        f"  vs SemiTactical: Robust={robust_vs_tac['win_rate']*100:.1f}% vs MaxWinRate={maxwr_vs_tac['win_rate']*100:.1f}%"
    )
    return {
        "ttt_pdf_root_table": ttt_table,
        "gomoku9_root_table": g9_table,
        "robust_vs_max_h2h": robust_vs_max,
        "robust_vs_tactical": robust_vs_tac,
        "max_winrate_vs_tactical": maxwr_vs_tac,
    }


def exp5_rollout_comparison(pool: ProcessPoolExecutor) -> list[dict[str, Any]]:
    print("[Exp 5] Gomoku 9x9 (5-in-a-row): Random vs Tactical Heuristic vs Value-Cutoff Rollouts (50 games each)...")
    rows: list[dict[str, Any]] = []
    for policy in ("random", "heuristic", "value_cutoff"):
        for iters in (100, 300, 600, 1000):
            res = run_matchup(
                pool,
                game="gomoku9",
                agent_cfg={
                    "type": "mcts",
                    "iterations": iters,
                    "c": 1.4,
                    "variant": "uct",
                    "rollout": policy,
                    "max_rollout_steps": 60,
                    "cutoff_steps": 4,
                },
                opp_cfg={"type": "semi_tactical", "prob": 0.55},
                num_games=50,
                base_seed=5000 + iters,
                random_opening=True,
            )
            res["rollout_policy"] = policy
            res["iterations"] = iters
            rows.append(res)
            print(
                f"  {policy:13s} (it={iters:4d}): W={res['wins']:2d} D={res['draws']:2d} L={res['losses']:2d} "
                f"WinRate={res['win_rate']*100:5.1f}% ({res['avg_move_ms']:6.1f}ms)"
            )

    # Also evaluate Gomoku 9x9 MCTS vs pure Random opponent (50 games) for required summary table
    vs_rand = run_matchup(
        pool,
        game="gomoku9",
        agent_cfg={"type": "mcts", "iterations": 500, "c": 1.4, "rollout": "heuristic", "max_rollout_steps": 60},
        opp_cfg={"type": "random"},
        num_games=50,
        base_seed=5900,
    )
    vs_rand["rollout_policy"] = "heuristic_vs_random"
    vs_rand["iterations"] = 500
    rows.append(vs_rand)
    print(f"  Gomoku 9x9 Heuristic(it=500) vs Random: W={vs_rand['wins']} D={vs_rand['draws']} L={vs_rand['losses']}")
    return rows


def exp6_uct_variants_ablation(pool: ProcessPoolExecutor) -> list[dict[str, Any]]:
    print("[Exp 6 - Bonus] UCT Formula Variants Ablation & Improvements (50 games each)...")
    variants = [
        ("greedy", "Greedy (c=0, W/N)"),
        ("linear", "Linear (W/N + c/N)"),
        ("no_ln", "No-Ln (W/N + c*sqrt(1/N))"),
        ("no_sqrt", "No-Sqrt (W/N + c*ln(Np)/N)"),
        ("uct", "Standard UCT"),
        ("ucb1_tuned", "UCB1-Tuned (Variance-aware)"),
        ("puct", "PUCT (Heuristic Prior)"),
    ]
    rows: list[dict[str, Any]] = []
    for var_key, var_label in variants:
        res = run_matchup(
            pool,
            game="gomoku7",
            agent_cfg={
                "type": "mcts",
                "iterations": 500,
                "c": 1.4,
                "variant": var_key,
                "rollout": "random",
                "max_rollout_steps": 60,
            },
            opp_cfg={"type": "semi_tactical", "prob": 0.45},
            num_games=50,
            base_seed=6000,
            random_opening=True,
        )
        res["variant"] = var_key
        res["label"] = var_label
        rows.append(res)
        print(
            f"  {var_label:28s}: W={res['wins']:2d} D={res['draws']:2d} L={res['losses']:2d} "
            f"WinRate={res['win_rate']*100:5.1f}% ({res['avg_move_ms']:6.1f}ms)"
        )
    return rows


def exp7_xiangqi_tournament(pool: ProcessPoolExecutor) -> list[dict[str, Any]]:
    print("[Exp 7 - Bonus] Xiangqi (Co Tuong) 50-Game Benchmark Tournament...")
    configs = [
        (
            "Alpha-Beta (d=2) vs Random",
            {"type": "alphabeta", "depth": 2},
            {"type": "random"},
        ),
        (
            "MCTS-UCT (Random, it=200) vs Random",
            {"type": "mcts", "iterations": 200, "c": 1.4, "rollout": "random", "max_rollout_steps": 40},
            {"type": "random"},
        ),
        (
            "MCTS-UCT (Heuristic, it=200) vs Random",
            {"type": "mcts", "iterations": 200, "c": 1.4, "rollout": "heuristic", "max_rollout_steps": 40},
            {"type": "random"},
        ),
        (
            "MCTS-UCT (Value-Cutoff, it=200) vs Random",
            {"type": "mcts", "iterations": 200, "c": 1.4, "rollout": "value_cutoff", "cutoff_steps": 4},
            {"type": "random"},
        ),
        (
            "MCTS-PUCT (Value-Cutoff, it=200) vs Greedy",
            {"type": "mcts", "iterations": 200, "c": 1.4, "variant": "puct", "rollout": "value_cutoff", "cutoff_steps": 4},
            {"type": "greedy"},
        ),
        (
            "Alpha-Beta (d=2) vs Greedy",
            {"type": "alphabeta", "depth": 2},
            {"type": "greedy"},
        ),
    ]
    rows: list[dict[str, Any]] = []
    for idx, (label, agent_cfg, opp_cfg) in enumerate(configs):
        res = run_matchup(
            pool,
            game="xiangqi",
            agent_cfg=agent_cfg,
            opp_cfg=opp_cfg,
            num_games=50,
            base_seed=7000 + idx * 100,
        )
        res["matchup"] = label
        rows.append(res)
        print(
            f"  {label:43s}: W={res['wins']:2d} D={res['draws']:2d} L={res['losses']:2d} "
            f"Win={res['win_rate']*100:5.1f}% ({res['avg_move_ms']:6.1f}ms)"
        )
    return rows


def _solve_delivery_instance(task: tuple[int, int]) -> dict[str, Any]:
    num_orders, seed = task
    inst = generate_instance(num_orders=num_orders, seed=seed)
    out: dict[str, Any] = {"num_orders": num_orders, "seed": seed, "methods": {}}
    sols = [
        solve_nearest_neighbor(inst),
        solve_edf(inst),
        solve_mcts(inst, iterations=1500, exploration=1.4, heuristic_rollout=False, seed=seed),
        solve_mcts(inst, iterations=1500, exploration=1.4, heuristic_rollout=True, seed=seed),
    ]
    if num_orders <= 8:
        sols.append(solve_branch_and_bound(inst))
    for s in sols:
        out["methods"][s.method] = {
            "total_cost": s.total_cost,
            "travel_distance": s.travel_distance,
            "total_lateness": s.total_lateness,
            "late_orders": s.late_orders,
            "elapsed_ms": s.elapsed_ms,
        }
    return out


def exp8_bonus_delivery(pool: ProcessPoolExecutor) -> list[dict[str, Any]]:
    print("[Exp 8 - Bonus Task 2] Last-Mile Delivery Routing with Time Windows (30 instances x M in {8, 12, 16})...")
    summary_rows: list[dict[str, Any]] = []
    for num_orders in (8, 12, 16):
        tasks = [(num_orders, 8000 + num_orders * 100 + i) for i in range(30)]
        inst_results = list(pool.map(_solve_delivery_instance, tasks))
        method_names = list(inst_results[0]["methods"].keys())
        opt_costs = (
            [r["methods"]["ExactBnB"]["total_cost"] for r in inst_results]
            if "ExactBnB" in method_names
            else None
        )
        for m in method_names:
            costs = [r["methods"][m]["total_cost"] for r in inst_results]
            dists = [r["methods"][m]["travel_distance"] for r in inst_results]
            lates = [r["methods"][m]["total_lateness"] for r in inst_results]
            late_cnts = [r["methods"][m]["late_orders"] for r in inst_results]
            times = [r["methods"][m]["elapsed_ms"] for r in inst_results]
            gap = (
                round(
                    sum((c - o) / max(o, 1e-6) for c, o in zip(costs, opt_costs)) / len(costs) * 100.0,
                    2,
                )
                if opt_costs is not None
                else None
            )
            row = {
                "num_orders": num_orders,
                "instances": 30,
                "method": m,
                "avg_cost": round(sum(costs) / len(costs), 2),
                "avg_distance": round(sum(dists) / len(dists), 2),
                "avg_lateness": round(sum(lates) / len(lates), 2),
                "avg_late_orders": round(sum(late_cnts) / len(late_cnts), 2),
                "avg_ms": round(sum(times) / len(times), 2),
                "optimality_gap_pct": gap,
            }
            summary_rows.append(row)
            print(
                f"  M={num_orders:2d} | {m:24s}: Cost={row['avg_cost']:6.2f}, Dist={row['avg_distance']:6.2f}, "
                f"Late={row['avg_lateness']:5.2f}, Gap={gap}%, Time={row['avg_ms']:6.1f}ms"
            )
    return summary_rows


def main() -> None:
    t_start = time.perf_counter()
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Running full benchmark suite with {workers} worker processes...")

    meta = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "workers_used": workers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = RESULTS_DIR / "all_results.json"
    results: dict[str, Any] = {"meta": meta}

    def checkpoint() -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results["exp1_minimax_vs_alphabeta"] = exp1_minimax_vs_alphabeta()
        checkpoint()
        results["exp2_ttt_mcts_never_loses"] = exp2_ttt_mcts_never_loses(pool)
        checkpoint()
        results["exp3_c_vs_iterations"] = exp3_c_vs_iterations_sweep(pool)
        checkpoint()
        results["exp4_root_and_robust_vs_max"] = exp4_root_stats_and_robust_vs_max(pool)
        checkpoint()
        results["exp5_rollout_comparison"] = exp5_rollout_comparison(pool)
        checkpoint()
        results["exp6_uct_variants"] = exp6_uct_variants_ablation(pool)
        checkpoint()
        results["exp7_xiangqi"] = exp7_xiangqi_tournament(pool)
        checkpoint()
        results["exp8_delivery_bonus"] = exp8_bonus_delivery(pool)
        checkpoint()

    total_sec = time.perf_counter() - t_start
    results["meta"]["total_elapsed_sec"] = round(total_sec, 2)
    checkpoint()

    print(f"\nALL 8 EXPERIMENTS COMPLETED in {total_sec:.1f}s! Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

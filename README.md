# Adversarial Search & MCTS-UCT: Tic-Tac-Toe, 5-in-a-Row Gomoku, Xiangqi & Delivery Routing

Complete, verified implementation, reproducible experimental benchmarks, interactive CLI + Tkinter GUI demos, and an Overleaf-ready IEEE Conference Paper report for the Artificial Intelligence Midterm Project (2026–2027).

## Project Highlights (Mandatory + All Bonus Objectives)
- **Mandatory Core (Bài 1):**
  - **3×3 Tic-Tac-Toe** & **Configurable $n \times n$ 5-in-a-Row Gomoku** (with fast $O(k)$ incremental win detection and bug-free offensive/defensive threat evaluation).
  - **Plain Minimax**, **Alpha-Beta Pruning** (natural & heuristically ordered), and **Transposition Tables** — verified for exact minimax value equivalence.
  - **MCTS-UCT** (`W/N + c * sqrt(ln(N_p) / N)`) verified to achieve **0% loss rate over 100 alternating games** vs. Optimal Alpha-Beta on 3×3 Tic-Tac-Toe (`iterations ≥ 500`).
  - Full empirical sweep of $c \in \{0.5, 1.0, 1.4, 2.0\}$ across iterations $\{100, 250, 500, 1000, 2000, 5000\}$ (50 alternating games per configuration = 1,200 games).
  - Root $(W, N, W/N, \text{UCT})$ analysis & Robust Child (`max N`) vs. Max Child (`max W/N`).
  - **Random Rollout vs. Tactical Heuristic Rollout vs. Value-Cutoff Rollout** (bridging classical MCTS to AlphaZero's Value Network).
- **Bonus Extensions (Target: 10/10):**
  - **Mathematical Foundations:** Complete derivation of UCB1/UCT from Hoeffding's inequality and Multi-Armed Bandit theory (explaining $\sqrt{\cdot}$ and $\ln$).
  - **UCT Formula Ablation & Improvements:** 7 selection rules (`Greedy`, `Linear`, `No-Ln`, `No-Sqrt`, `Standard UCT`, variance-aware `UCB1-Tuned`, and heuristic-prior `PUCT`).
  - **Complete 9×10 Xiangqi (Cờ Tướng) Engine:** All 7 piece types, Flying General rule, Horse leg blocking, Elephant eye & river, Cannon screen jumping, strict check/checkmate/stalemate-loss legality, PST + MVV-LVA Alpha-Beta, and MCTS/PUCT tournaments.
  - **Bonus Task 2 (Real-World AI):** Single-Player MCTS for **Last-Mile Urban Delivery Routing with Time Windows (TSP-TW)** compared against Nearest-Neighbor, Earliest-Deadline-First, and Exact Branch-and-Bound across 90 city instances.

---

## Repository Structure

```text
btl-ai-complete/
├── src/
│   ├── games/
│   │   ├── base.py                 # Abstract GameState interface
│   │   ├── tictactoe.py            # 3x3 Tic-Tac-Toe + in-place fast rollout
│   │   ├── gomoku.py               # n x n 5-in-a-row Gomoku + threat patterns + fast rollout
│   │   └── xiangqi.py              # Full-rule 9x10 Xiangqi (Cờ Tướng) + PST & MVV-LVA
│   ├── algorithms/
│   │   ├── minimax.py              # Plain Minimax, Alpha-Beta, Move Ordering, Transposition Table
│   │   ├── mcts.py                 # MCTS + 7 UCT variants + 3 rollout policies + root stats
│   │   └── opponents.py            # Unified Agent wrappers
│   └── bonus/
│       └── delivery_scheduling.py  # Bonus Task 2: TSP-TW (NN, EDF, Exact BnB, SP-MCTS)
├── demo/
│   ├── cli.py                      # Interactive CLI demo with live W/N/UCT & node telemetry
│   └── gui.py                      # Tkinter GUI (TTT 3x3, Gomoku 9x9, Xiangqi 9x10)
├── experiments/
│   ├── run_all.py                  # Parallel 8-experiment benchmark runner (reproducible seeds)
│   ├── generate_charts.py          # Pure-Python SVG + PDF/PNG vector chart generator
│   └── results/all_results.json    # Complete verified experimental dataset
├── tests/
│   └── test_all.py                 # 13 comprehensive unit tests (rules, pruning, MCTS, bonus)
├── report/
│   ├── main.tex                    # Complete IEEE Conference Paper (pdflatex / Overleaf-ready)
│   ├── report_preview.html         # Instant browser-viewable formatted IEEE paper
│   ├── IEEE_Report_Preview.pdf     # Rendered PDF copy of the report
│   ├── figures/                    # 5 publication-grade vector figures (.pdf, .svg, .png)
│   ├── latexmkrc & OVERLEAF.md     # Overleaf instructions
├── DEFENSE_GUIDE_VI.md             # Vietnamese oral defense Q&A & live demo script
└── requirements.txt
```

---

## Quick Start

### 1. Run Unit Tests (Verifies All Rules & Algorithms)
```bash
python tests/test_all.py
```

### 2. Launch Interactive Tkinter GUI Demo (Best for Oral Defense!)
```bash
python -m demo.gui
```
Supports **Tic-Tac-Toe 3×3**, **Gomoku 9×9 (5-in-a-row)**, and **Xiangqi 9×10 (Cờ Tướng)** with live display of nodes visited, Alpha-Beta cutoffs, runtime, and Root `(W, N, W/N, UCT)` tables.

### 3. Run Interactive CLI Demos
```bash
# Play 3x3 Tic-Tac-Toe vs MCTS-UCT
python -m demo.cli --game tictactoe --algo mcts --iterations 1000 --human

# Play 9x9 Gomoku (5-in-a-row) vs MCTS-UCT (Value-Cutoff)
python -m demo.cli --game gomoku --size 9 --algo mcts --rollout value_cutoff --iterations 1000 --human

# Play Xiangqi (Cờ Tướng) vs Alpha-Beta (depth=2)
python -m demo.cli --game xiangqi --algo alphabeta --depth 2 --human

# Run Bonus Task 2: Last-Mile Delivery Routing (12 orders)
python -m demo.cli --game delivery --orders 12 --iterations 1500
```

### 4. Reproduce All 8 Experiments & Publication Figures
```bash
python -m experiments.run_all
python -m experiments.generate_charts
```

---

## Key Verified Results Summary

| Experiment | Key Finding (100% Measured) |
|---|---|
| **Exp 1: Alpha-Beta Pruning** | **-98.57%** nodes on TTT 3×3 (`549,946 -> 7,865`, `-99.61%` with TT); **-93.29%** on Gomoku 9×9 (`d=3`); **-93.44%** on Xiangqi Opening (`d=2`). |
| **Exp 2: TTT 3×3 Never-Losing** | **0 losses / 100 alternating games (100% non-loss)** vs. Optimal Alpha-Beta from `iterations ≥ 500` (`7.7 ms/move`). |
| **Exp 3: $c$ vs. Iterations** | Monotonic rise from `2–14%` (`100` it.) to `94%` (`c=0.5`) and `90%` with **0 losses** (`c=1.4`) at `5000` it.; `c=2.0` over-explores (`78%`). |
| **Exp 5: Rollouts (Gomoku 9×9)** | At `1000` it.: Random **16%** → Tactical Heuristic **34%** → Value-Cutoff **74%** (and **6.2× faster** than full heuristic!). |
| **Exp 6: UCT Variants** | No-Sqrt drops to **16%**; Standard UCT **28%**; UCB1-Tuned **42%**; Heuristic-Prior PUCT **98%** (49W/1D/0L). |
| **Exp 7: Xiangqi (50 games)** | Alpha-Beta (`d=2`): **88%W/0%L** vs. Random; MCTS Value-Cutoff: **96%W** vs. Random; PUCT Value-Cutoff: **90%W/0%L** vs. Greedy. |
| **Exp 8: Delivery Bonus (TSP-TW)** | At $M=8$: MCTS-Heuristic achieves **4.78% optimality gap** vs. Exact BnB (NN: `126.7%`). At $M=16$: **-43.6% cost** vs. Nearest-Neighbor. |

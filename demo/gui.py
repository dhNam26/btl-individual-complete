"""Interactive Tkinter GUI Demo for Oral Defense (Tic-Tac-Toe 3x3, Gomoku 5-in-a-row, Xiangqi).

Requires only Python's standard library (tkinter). Displays live search telemetry:
- Minimax / Alpha-Beta: value, nodes visited, Alpha-Beta cutoffs, runtime (ms).
- MCTS-UCT: root visits, tree nodes, runtime (ms), and Root (W, N, W/N, UCT) table!

Run:
    python -m demo.gui
"""

from __future__ import annotations

import math
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithms.mcts import mcts_search
from src.algorithms.minimax import alphabeta_search, minimax_search
from src.games.gomoku import GomokuState
from src.games.tictactoe import TicTacToeState
from src.games.xiangqi import COLS as XQ_COLS, PIECE_UNICODE, ROWS as XQ_ROWS, XiangqiState


class BoardGameGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Adversarial Search Demo — Minimax / Alpha-Beta / MCTS-UCT")
        self.root.geometry("1040x700")
        self.root.configure(bg="#1e222a")

        self.state: Any = None
        self.game_type = tk.StringVar(value="Gomoku 9x9 (5-in-a-row)")
        self.algo_type = tk.StringVar(value="MCTS-UCT (Heuristic)")
        self.depth_var = tk.IntVar(value=2)
        self.iter_var = tk.IntVar(value=1000)
        self.c_var = tk.DoubleVar(value=1.41)
        self.human_plays_first = tk.BooleanVar(value=True)
        self.ai_thinking = False
        self.selected_xq_cell: Optional[tuple[int, int]] = None

        self._build_ui()
        self.new_game()

    def _build_ui(self) -> None:
        # Top control bar
        ctrl = tk.Frame(self.root, bg="#282c34", padx=10, pady=8)
        ctrl.pack(fill=tk.X)

        tk.Label(ctrl, text="Game:", fg="#abb2bf", bg="#282c34").pack(side=tk.LEFT, padx=(0, 4))
        game_cb = ttk.Combobox(
            ctrl,
            textvariable=self.game_type,
            values=["Tic-Tac-Toe 3x3", "Gomoku 9x9 (5-in-a-row)", "Xiangqi 9x10 (Co Tuong)"],
            state="readonly",
            width=23,
        )
        game_cb.pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(ctrl, text="AI:", fg="#abb2bf", bg="#282c34").pack(side=tk.LEFT, padx=(0, 4))
        algo_cb = ttk.Combobox(
            ctrl,
            textvariable=self.algo_type,
            values=[
                "Alpha-Beta",
                "Minimax (Plain)",
                "MCTS-UCT (Random)",
                "MCTS-UCT (Heuristic)",
                "MCTS-UCT (Value Cutoff)",
                "MCTS-UCB1-Tuned",
                "MCTS-PUCT",
            ],
            state="readonly",
            width=22,
        )
        algo_cb.pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(ctrl, text="Depth:", fg="#abb2bf", bg="#282c34").pack(side=tk.LEFT)
        tk.Spinbox(ctrl, from_=1, to=9, textvariable=self.depth_var, width=3).pack(side=tk.LEFT, padx=(2, 10))

        tk.Label(ctrl, text="Iters:", fg="#abb2bf", bg="#282c34").pack(side=tk.LEFT)
        tk.Spinbox(ctrl, from_=50, to=5000, increment=100, textvariable=self.iter_var, width=5).pack(side=tk.LEFT, padx=(2, 10))

        tk.Label(ctrl, text="c:", fg="#abb2bf", bg="#282c34").pack(side=tk.LEFT)
        tk.Spinbox(ctrl, from_=0.0, to=3.0, increment=0.1, format="%.2f", textvariable=self.c_var, width=4).pack(side=tk.LEFT, padx=(2, 10))

        tk.Checkbutton(
            ctrl,
            text="Human First",
            variable=self.human_plays_first,
            fg="#e5c07b",
            bg="#282c34",
            selectcolor="#1e222a",
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(
            ctrl,
            text="New Game",
            command=self.new_game,
            bg="#61afef",
            fg="#1e222a",
            font=("Segoe UI", 9, "bold"),
            padx=10,
        ).pack(side=tk.LEFT)

        # Main workspace: Canvas left + Telemetry right
        body = tk.Frame(self.root, bg="#1e222a", padx=12, pady=12)
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, width=560, height=600, bg="#e8cfa0", highlightthickness=2, highlightbackground="#4b5263")
        self.canvas.pack(side=tk.LEFT, padx=(0, 14))
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        right = tk.Frame(body, bg="#282c34", padx=10, pady=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(
            right,
            text="Ready",
            fg="#98c379",
            bg="#282c34",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        )
        self.status_label.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            right,
            text="Live AI Search Telemetry (Nodes / Cutoffs / Root W, N, W/N, UCT):",
            fg="#61afef",
            bg="#282c34",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X)

        self.log_text = tk.Text(
            right,
            bg="#1e222a",
            fg="#dcdfe4",
            font=("Consolas", 9),
            wrap=tk.NONE,
            relief=tk.FLAT,
            padx=8,
            pady=8,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def new_game(self) -> None:
        self.ai_thinking = False
        self.selected_xq_cell = None
        gt = self.game_type.get()
        if gt.startswith("Tic-Tac-Toe"):
            self.state = TicTacToeState()
        elif gt.startswith("Gomoku"):
            self.state = GomokuState(size=9, win_length=5)
        else:
            self.state = XiangqiState()

        self.log_text.delete("1.0", tk.END)
        self._log(f"Started {gt} | AI: {self.algo_type.get()}\n")
        self.draw_board()
        self._update_status()
        if not self.human_plays_first.get():
            self.trigger_ai_move()

    def _log(self, msg: str) -> None:
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)

    def _human_side(self) -> int:
        return 1 if self.human_plays_first.get() else -1

    def _update_status(self) -> None:
        w = self.state.winner()
        if w is not None:
            if w == 0:
                self.status_label.config(text="Game Over: DRAW!", fg="#e5c07b")
            else:
                self.status_label.config(text=f"Game Over: {self.state.player_name(w)} WINS!", fg="#e06c75")
            return
        pname = self.state.player_name(self.state.current_player)
        who = "Human" if self.state.current_player == self._human_side() else "AI"
        self.status_label.config(text=f"Turn: {pname} ({who})", fg="#98c379")

    def draw_board(self) -> None:
        self.canvas.delete("all")
        if isinstance(self.state, TicTacToeState):
            self._draw_ttt()
        elif isinstance(self.state, GomokuState):
            self._draw_gomoku()
        elif isinstance(self.state, XiangqiState):
            self._draw_xiangqi()

    def _draw_ttt(self) -> None:
        margin, cell = 70, 140
        for i in range(4):
            self.canvas.create_line(margin + i * cell, margin, margin + i * cell, margin + 3 * cell, width=3, fill="#3b2f2f")
            self.canvas.create_line(margin, margin + i * cell, margin + 3 * cell, margin + i * cell, width=3, fill="#3b2f2f")
        for idx, val in enumerate(self.state.board):
            r, c = divmod(idx, 3)
            cx = margin + c * cell + cell // 2
            cy = margin + r * cell + cell // 2
            if val == 1:
                self.canvas.create_line(cx - 40, cy - 40, cx + 40, cy + 40, width=7, fill="#c0392b")
                self.canvas.create_line(cx + 40, cy - 40, cx - 40, cy + 40, width=7, fill="#c0392b")
            elif val == -1:
                self.canvas.create_oval(cx - 42, cy - 42, cx + 42, cy + 42, width=7, outline="#2471a3")

    def _draw_gomoku(self) -> None:
        n = self.state.size
        margin = 45
        cell = 470 // (n - 1)
        for i in range(n):
            p = margin + i * cell
            self.canvas.create_line(margin, p, margin + (n - 1) * cell, p, fill="#5d4037", width=1.5)
            self.canvas.create_line(p, margin, p, margin + (n - 1) * cell, fill="#5d4037", width=1.5)
            self.canvas.create_text(margin - 22, p, text=str(i), fill="#4e342e", font=("Consolas", 9))
            self.canvas.create_text(p, margin - 22, text=str(i), fill="#4e342e", font=("Consolas", 9))
        rad = max(12, cell // 2 - 4)
        for idx, val in enumerate(self.state.board):
            if val == 0:
                continue
            r, c = divmod(idx, n)
            cx, cy = margin + c * cell, margin + r * cell
            color = "#1a1a1a" if val == 1 else "#fafafa"
            outline = "#000000" if val == 1 else "#333333"
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill=color, outline=outline, width=2)
            if idx == self.state.last_move:
                self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#e74c3c", outline="")

    def _draw_xiangqi(self) -> None:
        mx, my, cell = 45, 35, 58
        # Horizontal lines
        for r in range(XQ_ROWS):
            y = my + r * cell
            self.canvas.create_line(mx, y, mx + (XQ_COLS - 1) * cell, y, fill="#4e342e", width=1.5)
        # Vertical lines (broken at river for inner files)
        for c in range(XQ_COLS):
            x = mx + c * cell
            if c in (0, XQ_COLS - 1):
                self.canvas.create_line(x, my, x, my + (XQ_ROWS - 1) * cell, fill="#4e342e", width=1.5)
            else:
                self.canvas.create_line(x, my, x, my + 4 * cell, fill="#4e342e", width=1.5)
                self.canvas.create_line(x, my + 5 * cell, x, my + 9 * cell, fill="#4e342e", width=1.5)
        # Palaces X lines
        for r0 in (0, 7):
            self.canvas.create_line(mx + 3 * cell, my + r0 * cell, mx + 5 * cell, my + (r0 + 2) * cell, fill="#4e342e")
            self.canvas.create_line(mx + 5 * cell, my + r0 * cell, mx + 3 * cell, my + (r0 + 2) * cell, fill="#4e342e")
        # River label
        self.canvas.create_text(mx + 4 * cell, my + 4.5 * cell, text="楚 河        漢 界", font=("Segoe UI", 14, "bold"), fill="#6d4c41")
        # Pieces
        for idx, piece in enumerate(self.state.board):
            if piece == 0:
                continue
            r, c = divmod(idx, XQ_COLS)
            cx, cy = mx + c * cell, my + r * cell
            is_sel = self.selected_xq_cell == (r, c)
            fg = "#b71c1c" if piece > 0 else "#1a237e"
            bg = "#fff8dc" if not is_sel else "#ffe082"
            self.canvas.create_oval(cx - 22, cy - 22, cx + 22, cy + 22, fill=bg, outline=fg, width=2.5 if not is_sel else 4)
            self.canvas.create_text(cx, cy, text=PIECE_UNICODE[piece], fill=fg, font=("SimSun", 16, "bold"))

    def on_canvas_click(self, event: tk.Event) -> None:
        if self.ai_thinking or self.state.is_terminal():
            return
        if self.state.current_player != self._human_side():
            return

        move = None
        if isinstance(self.state, TicTacToeState):
            margin, cell = 70, 140
            c = (event.x - margin) // cell
            r = (event.y - margin) // cell
            if 0 <= r < 3 and 0 <= c < 3:
                cand = r * 3 + c
                if cand in self.state.legal_moves():
                    move = cand
        elif isinstance(self.state, GomokuState):
            n = self.state.size
            margin = 45
            cell = 470 // (n - 1)
            c = round((event.x - margin) / cell)
            r = round((event.y - margin) / cell)
            if 0 <= r < n and 0 <= c < n:
                cand = r * n + c
                if self.state.board[cand] == 0:
                    move = cand
        elif isinstance(self.state, XiangqiState):
            mx, my, cell = 45, 35, 58
            c = round((event.x - mx) / cell)
            r = round((event.y - my) / cell)
            if 0 <= r < XQ_ROWS and 0 <= c < XQ_COLS:
                if self.selected_xq_cell is None:
                    if self.state.board[r * XQ_COLS + c] * self.state.current_player > 0:
                        self.selected_xq_cell = (r, c)
                        self.draw_board()
                        return
                else:
                    r1, c1 = self.selected_xq_cell
                    cand = (r1, c1, r, c)
                    if cand in self.state.legal_moves():
                        move = cand
                        self.selected_xq_cell = None
                    elif self.state.board[r * XQ_COLS + c] * self.state.current_player > 0:
                        self.selected_xq_cell = (r, c)
                        self.draw_board()
                        return
                    else:
                        self.selected_xq_cell = None
                        self.draw_board()
                        return

        if move is None:
            return
        m_str = self.state.format_move(move)
        self._log(f"Human plays {m_str}\n")
        self.state = self.state.apply_move(move)
        self.draw_board()
        self._update_status()
        if not self.state.is_terminal():
            self.trigger_ai_move()

    def trigger_ai_move(self) -> None:
        self.ai_thinking = True
        self.status_label.config(text="AI is thinking...", fg="#e5c07b")
        state_snap = self.state
        algo = self.algo_type.get()
        depth = self.depth_var.get()
        iters = self.iter_var.get()
        c_val = self.c_var.get()

        def worker() -> None:
            if algo == "Minimax (Plain)":
                res = minimax_search(state_snap, depth=depth)
                log_msg = (
                    f"[Minimax d={depth}] move={state_snap.format_move(res.best_move)} "
                    f"val={res.best_value:.1f} nodes={res.nodes_visited} time={res.elapsed_ms:.1f}ms\n"
                )
                chosen = res.best_move
            elif algo == "Alpha-Beta":
                res = alphabeta_search(state_snap, depth=depth, use_move_ordering=True, use_tt=True)
                log_msg = (
                    f"[AlphaBeta d={depth}] move={state_snap.format_move(res.best_move)} "
                    f"val={res.best_value:.1f} nodes={res.nodes_visited} "
                    f"cutoffs={res.cutoffs} time={res.elapsed_ms:.1f}ms\n"
                )
                chosen = res.best_move
            else:
                variant_map = {
                    "MCTS-UCT (Random)": ("uct", "random"),
                    "MCTS-UCT (Heuristic)": ("uct", "heuristic"),
                    "MCTS-UCT (Value Cutoff)": ("uct", "value_cutoff"),
                    "MCTS-UCB1-Tuned": ("ucb1_tuned", "heuristic"),
                    "MCTS-PUCT": ("puct", "value_cutoff"),
                }
                variant, rollout = variant_map[algo]
                res = mcts_search(
                    state_snap,
                    iterations=iters,
                    exploration=c_val,
                    uct_variant=variant,
                    rollout_policy=rollout,
                )
                lines = [
                    f"[{algo} c={c_val:.2f} it={iters}] move={state_snap.format_move(res.best_move)} "
                    f"nodes={res.tree_nodes} time={res.elapsed_ms:.1f}ms"
                ]
                lines.append(f"  {'Move':<14} {'W':>7} {'N':>5} {'W/N':>7} {'Score':>7}")
                for st in res.root_stats[:6]:
                    lines.append(
                        f"  {state_snap.format_move(st.move):<14} {st.wins:7.1f} {st.visits:5d} "
                        f"{st.win_rate:7.3f} {st.uct_score:7.3f}"
                    )
                log_msg = "\n".join(lines) + "\n\n"
                chosen = res.best_move

            self.root.after(0, lambda: self._apply_ai_result(chosen, log_msg))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_ai_result(self, move: Any, log_msg: str) -> None:
        self.ai_thinking = False
        self._log(log_msg)
        self.state = self.state.apply_move(move)
        self.draw_board()
        self._update_status()
        if self.state.is_terminal():
            w = self.state.winner()
            msg = "It's a Draw!" if w == 0 else f"{self.state.player_name(w)} wins!"
            messagebox.showinfo("Game Over", msg)


def main() -> None:
    root = tk.Tk()
    BoardGameGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""Generate publication-grade vector SVG, PDF, and PNG figures from `all_results.json`.

Uses pure Python SVG generation (IEEE-style clean typography, crisp axes, gridlines,
legends, value annotations) and PyMuPDF (`pymupdf`) to render vector PDFs (for Overleaf
`pdflatex`) and high-DPI PNGs (for GitHub README / previews). Zero matplotlib dependency!

Usage:
    python -m experiments.generate_charts
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "experiments" / "results" / "all_results.json"
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
EXP_FIG_DIR = ROOT / "experiments" / "results" / "figures"
EXP_FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "blue": "#1f4e79",
    "teal": "#008080",
    "orange": "#d95f02",
    "red": "#b2182b",
    "purple": "#7570b3",
    "green": "#1b9e77",
    "gray": "#666666",
    "light_grid": "#e2e8f0",
    "axis": "#1e293b",
}

SUPERSCRIPT = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶"}


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _save_svg_and_exports(svg: str, stem: str) -> None:
    for out_dir in (FIG_DIR, EXP_FIG_DIR):
        (out_dir / f"{stem}.svg").write_text(svg, encoding="utf-8")

    try:
        import pymupdf

        svg_bytes = svg.encode("utf-8")
        doc = pymupdf.open(stream=svg_bytes, filetype="svg")
        pdf_bytes = doc.convert_to_pdf()
        pdf_doc = pymupdf.open("pdf", pdf_bytes)
        for out_dir in (FIG_DIR, EXP_FIG_DIR):
            pdf_doc.save(str(out_dir / f"{stem}.pdf"))
        pix = pdf_doc[0].get_pixmap(dpi=220)
        for out_dir in (FIG_DIR, EXP_FIG_DIR):
            pix.save(str(out_dir / f"{stem}.png"))
        pdf_doc.close()
        doc.close()
    except Exception as exc:
        print(f"  [warn] PDF/PNG conversion skipped for {stem}: {exc}")


def _svg_header(w: int, h: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<style>",
        '  text { font-family: "Times New Roman", Georgia, serif; fill: #1e293b; }',
        "  .title { font-size: 13.5px; font-weight: bold; }",
        "  .axis-label { font-size: 11.5px; font-weight: bold; }",
        "  .tick { font-size: 10px; fill: #334155; }",
        "  .legend { font-size: 10px; }",
        "  .val { font-size: 9.5px; font-weight: bold; }",
        "</style>",
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
    ]


# ---------------------------------------------------------------------------
# Figure 1: Iterations vs Win Rate for c in {0.5, 1.0, 1.4, 2.0}
# ---------------------------------------------------------------------------

def plot_fig1_c_vs_iterations(data: dict[str, Any]) -> None:
    rows = data["exp3_c_vs_iterations"]
    w, h = 640, 380
    ml, mr, mt, mb = 64, 28, 40, 54
    pw, ph = w - ml - mr, h - mt - mb

    iters = sorted({r["iterations"] for r in rows})
    x_min, x_max = math.log10(min(iters)), math.log10(max(iters))

    def tx(it: int) -> float:
        return ml + (math.log10(it) - x_min) / (x_max - x_min) * pw

    def ty(pct: float) -> float:
        return mt + ph - (pct / 100.0) * ph

    svg = _svg_header(w, h)
    svg.append(
        f'<text x="{w/2:.1f}" y="22" text-anchor="middle" class="title">'
        "Effect of Exploration Constant c and MCTS Iterations on Win Rate (Gomoku 5-in-a-Row)</text>"
    )

    for yv in range(0, 101, 20):
        yy = ty(yv)
        svg.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{PALETTE["light_grid"]}" stroke-width="1"/>')
        svg.append(f'<text x="{ml-8}" y="{yy+4:.1f}" text-anchor="end" class="tick">{yv}%</text>')

    for it in iters:
        xx = tx(it)
        svg.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="{PALETTE["light_grid"]}" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{xx:.1f}" y="{mt+ph+17}" text-anchor="middle" class="tick">{it}</text>')

    svg.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="{PALETTE["axis"]}" stroke-width="1.3"/>')
    svg.append(f'<text x="{ml+pw/2:.1f}" y="{h-12}" text-anchor="middle" class="axis-label">MCTS Iterations (log scale)</text>')
    svg.append(
        f'<text x="18" y="{mt+ph/2:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 18 {mt+ph/2:.1f})">Win Rate (%)</text>'
    )

    styles = [
        (0.5, PALETTE["orange"], "6,3", "circle"),
        (1.0, PALETTE["teal"], "none", "square"),
        (1.4, PALETTE["blue"], "none", "diamond"),
        (2.0, PALETTE["red"], "2,2", "triangle"),
    ]

    for c_val, color, dash, marker in styles:
        pts = sorted([r for r in rows if abs(r["c"] - c_val) < 1e-6], key=lambda r: r["iterations"])
        coords = [(tx(r["iterations"]), ty(r["win_rate"] * 100.0)) for r in pts]
        d_str = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        stroke_w = "2.6" if abs(c_val - 1.4) < 1e-6 else "2.0"
        svg.append(f'<path d="{d_str}" fill="none" stroke="{color}" stroke-width="{stroke_w}"{dash_attr}/>')
        for x, y in coords:
            if marker == "circle":
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" stroke="#fff" stroke-width="1"/>')
            elif marker == "square":
                svg.append(f'<rect x="{x-3.6:.1f}" y="{y-3.6:.1f}" width="7.2" height="7.2" fill="{color}" stroke="#fff" stroke-width="1"/>')
            elif marker == "diamond":
                svg.append(f'<polygon points="{x:.1f},{y-5:.1f} {x+5:.1f},{y:.1f} {x:.1f},{y+5:.1f} {x-5:.1f},{y:.1f}" fill="{color}" stroke="#fff" stroke-width="1"/>')
            else:
                svg.append(f'<polygon points="{x:.1f},{y-5:.1f} {x+4.5:.1f},{y+4:.1f} {x-4.5:.1f},{y+4:.1f}" fill="{color}" stroke="#fff" stroke-width="1"/>')

    # Clean generously sized legend box (bottom right)
    lx, ly, lw, lh = ml + pw - 178, mt + ph - 94, 168, 84
    svg.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="#ffffff" fill-opacity="0.95" stroke="#94a3b8" stroke-width="1" rx="3"/>')
    for idx, (c_val, color, dash, _) in enumerate(styles):
        yy = ly + 17 + idx * 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        svg.append(f'<line x1="{lx+10}" y1="{yy}" x2="{lx+34}" y2="{yy}" stroke="{color}" stroke-width="2.4"{dash_attr}/>')
        suffix = " (theory)" if abs(c_val - 1.4) < 1e-6 else ""
        svg.append(f'<text x="{lx+42}" y="{yy+3.5}" class="legend">c = {c_val:.1f}{suffix}</text>')

    svg.append("</svg>")
    _save_svg_and_exports("\n".join(svg), "fig1_c_vs_iterations")


# ---------------------------------------------------------------------------
# Figure 2: Rollout Policies (Random vs Tactical Heuristic vs Value-Cutoff)
# ---------------------------------------------------------------------------

def plot_fig2_rollout_comparison(data: dict[str, Any]) -> None:
    rows = [r for r in data["exp5_rollout_comparison"] if r["rollout_policy"] in ("random", "heuristic", "value_cutoff")]
    w, h = 720, 355
    svg = _svg_header(w, h)
    svg.append(
        f'<text x="{w/2:.1f}" y="22" text-anchor="middle" class="title">'
        "Gomoku 9x9 (5-in-a-Row): Random vs. Heuristic vs. Value-Cutoff Rollouts</text>"
    )

    policies = [
        ("random", "Random Rollout", PALETTE["gray"], "5,3"),
        ("heuristic", "Tactical Heuristic", PALETTE["orange"], "none"),
        ("value_cutoff", "Value-Cutoff (Precursor)", PALETTE["blue"], "none"),
    ]
    iters = sorted({r["iterations"] for r in rows})
    x_min, x_max = min(iters), max(iters)

    # Panel (a): Win Rate vs Iterations
    ml, mt, pw, ph = 58, 52, 275, 232
    tx1 = lambda it: ml + (it - x_min) / (x_max - x_min) * pw
    ty1 = lambda pct: mt + ph - pct / 100.0 * ph

    svg.append(f'<text x="{ml+pw/2:.1f}" y="{mt-10}" text-anchor="middle" class="axis-label">(a) Win Rate vs. Iterations</text>')
    for yv in range(0, 101, 20):
        yy = ty1(yv)
        svg.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{PALETTE["light_grid"]}"/>')
        svg.append(f'<text x="{ml-6}" y="{yy+3.5:.1f}" text-anchor="end" class="tick">{yv}%</text>')
    for it in iters:
        xx = tx1(it)
        svg.append(f'<text x="{xx:.1f}" y="{mt+ph+16}" text-anchor="middle" class="tick">{it}</text>')
    svg.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')
    svg.append(f'<text x="{ml+pw/2:.1f}" y="{h-14}" text-anchor="middle" class="axis-label">Iterations</text>')
    svg.append(f'<text x="16" y="{mt+ph/2:.1f}" text-anchor="middle" class="axis-label" transform="rotate(-90 16 {mt+ph/2:.1f})">Win Rate (%)</text>')

    for pol, _, color, dash in policies:
        pts = sorted([r for r in rows if r["rollout_policy"] == pol], key=lambda r: r["iterations"])
        coords = [(tx1(r["iterations"]), ty1(r["win_rate"] * 100.0)) for r in pts]
        d_str = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        svg.append(f'<path d="{d_str}" fill="none" stroke="{color}" stroke-width="2.3"{dash_attr}/>')
        for x, y in coords:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="1"/>')

    # Panel (b): Avg Move Time (ms) vs Iterations
    ml2 = 415
    max_ms = 8000.0
    tx2 = lambda it: ml2 + (it - x_min) / (x_max - x_min) * pw
    ty2 = lambda ms: mt + ph - (ms / max_ms) * ph

    svg.append(f'<text x="{ml2+pw/2:.1f}" y="{mt-10}" text-anchor="middle" class="axis-label">(b) Avg. Time per Move (ms)</text>')
    for yv in (0, 2000, 4000, 6000, 8000):
        yy = ty2(yv)
        svg.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw}" y2="{yy:.1f}" stroke="{PALETTE["light_grid"]}"/>')
        svg.append(f'<text x="{ml2-6}" y="{yy+3.5:.1f}" text-anchor="end" class="tick">{yv}</text>')
    for it in iters:
        xx = tx2(it)
        svg.append(f'<text x="{xx:.1f}" y="{mt+ph+16}" text-anchor="middle" class="tick">{it}</text>')
    svg.append(f'<rect x="{ml2}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')
    svg.append(f'<text x="{ml2+pw/2:.1f}" y="{h-14}" text-anchor="middle" class="axis-label">Iterations</text>')
    svg.append(f'<text x="{ml2-46}" y="{mt+ph/2:.1f}" text-anchor="middle" class="axis-label" transform="rotate(-90 {ml2-46} {mt+ph/2:.1f})">Time / Move (ms)</text>')

    for pol, _, color, dash in policies:
        pts = sorted([r for r in rows if r["rollout_policy"] == pol], key=lambda r: r["iterations"])
        coords = [(tx2(r["iterations"]), ty2(r["avg_move_ms"])) for r in pts]
        d_str = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        svg.append(f'<path d="{d_str}" fill="none" stroke="{color}" stroke-width="2.3"{dash_attr}/>')
        for x, y in coords:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="1"/>')

    # Clean generously sized legend box inside panel (b) top-left
    lx, ly = ml2 + 10, mt + 10
    svg.append(f'<rect x="{lx}" y="{ly}" width="212" height="64" fill="#fff" fill-opacity="0.95" stroke="#94a3b8" rx="3"/>')
    for idx, (_, label, color, dash) in enumerate(policies):
        yy = ly + 15 + idx * 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        svg.append(f'<line x1="{lx+8}" y1="{yy}" x2="{lx+30}" y2="{yy}" stroke="{color}" stroke-width="2.4"{dash_attr}/>')
        svg.append(f'<text x="{lx+38}" y="{yy+3.5}" class="legend">{_esc(label)}</text>')

    svg.append("</svg>")
    _save_svg_and_exports("\n".join(svg), "fig2_rollout_comparison")


# ---------------------------------------------------------------------------
# Figure 3: UCT Formula Variants Ablation & Improvements
# ---------------------------------------------------------------------------

def plot_fig3_uct_variants(data: dict[str, Any]) -> None:
    rows = data["exp6_uct_variants"]
    w, h = 680, 330
    ml, mr, mt, mb = 235, 55, 42, 46
    pw, ph = w - ml - mr, h - mt - mb

    svg = _svg_header(w, h)
    svg.append(
        f'<text x="{w/2:.1f}" y="22" text-anchor="middle" class="title">'
        "UCT Selection Formula Ablation &amp; Improvements (Gomoku 5-in-a-Row, 500 Iterations)</text>"
    )

    for xv in range(0, 101, 20):
        xx = ml + (xv / 100.0) * pw
        svg.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="{PALETTE["light_grid"]}"/>')
        svg.append(f'<text x="{xx:.1f}" y="{mt+ph+17}" text-anchor="middle" class="tick">{xv}%</text>')

    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="{PALETTE["axis"]}" stroke-width="1.3"/>')
    svg.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="{PALETTE["axis"]}" stroke-width="1.3"/>')
    svg.append(f'<text x="{ml+pw/2:.1f}" y="{h-10}" text-anchor="middle" class="axis-label">Win Rate (% over 50 alternating games)</text>')

    n = len(rows)
    step = ph / n
    bar_h = step * 0.62
    colors = {
        "greedy": "#94a3b8",
        "linear": "#d97706",
        "no_ln": "#dc2626",
        "no_sqrt": "#ea580c",
        "uct": PALETTE["blue"],
        "ucb1_tuned": PALETTE["teal"],
        "puct": PALETTE["green"],
    }

    for idx, r in enumerate(rows):
        yc = mt + (idx + 0.5) * step
        pct = r["win_rate"] * 100.0
        bw = (pct / 100.0) * pw
        col = colors.get(r["variant"], PALETTE["blue"])
        svg.append(f'<rect x="{ml}" y="{yc-bar_h/2:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="{col}" rx="2"/>')
        svg.append(f'<text x="{ml-8}" y="{yc+3.5:.1f}" text-anchor="end" class="tick">{_esc(r["label"])}</text>')
        svg.append(f'<text x="{ml+bw+6:.1f}" y="{yc+3.5:.1f}" class="val">{pct:.1f}%</text>')

    svg.append("</svg>")
    _save_svg_and_exports("\n".join(svg), "fig3_uct_variants")


# ---------------------------------------------------------------------------
# Figure 4: Minimax vs Alpha-Beta Node Counts (Log Scale)
# ---------------------------------------------------------------------------

def plot_fig4_alphabeta_pruning(data: dict[str, Any]) -> None:
    rows = [r for r in data["exp1_minimax_vs_alphabeta"] if r["minimax_nodes"] is not None]
    selected = [rows[0], rows[2], rows[3], rows[4], rows[5]]
    short_names = [
        "TTT 3x3\n(d=9)",
        "Gomoku 9x9\n(d=2)",
        "Gomoku 9x9\n(d=3)",
        "Xiangqi Std\n(d=2)",
        "Xiangqi Skirm.\n(d=2)",
    ]

    w, h = 680, 390
    ml, mr, mt, mb = 66, 24, 72, 60
    pw, ph = w - ml - mr, h - mt - mb

    max_log, min_log = 6.0, 1.0
    ty = lambda v: mt + ph - (max(min_log, math.log10(max(v, 10))) - min_log) / (max_log - min_log) * ph

    svg = _svg_header(w, h)
    svg.append(
        f'<text x="{w/2:.1f}" y="22" text-anchor="middle" class="title">'
        "Nodes Visited: Plain Minimax vs. Alpha-Beta Pruning &amp; Transposition Table (Log Scale)</text>"
    )

    series = [
        ("minimax_nodes", "Plain Minimax", PALETTE["red"]),
        ("ab_natural_nodes", "Alpha-Beta (Natural)", PALETTE["orange"]),
        ("ab_ordered_nodes", "Alpha-Beta (Ordered)", PALETTE["blue"]),
        ("ab_tt_nodes", "Alpha-Beta + TT", PALETTE["green"]),
    ]

    # Clean legend strip ABOVE the plot frame (zero bar overlap!)
    lx, ly = 42, 34
    svg.append(f'<rect x="{lx}" y="{ly}" width="600" height="24" fill="#f8fafc" stroke="#cbd5e1" rx="3"/>')
    x_positions = [54, 182, 344, 510]
    for xx, (_, label, col) in zip(x_positions, series):
        svg.append(f'<rect x="{xx}" y="{ly+6.5}" width="11" height="11" fill="{col}"/>')
        svg.append(f'<text x="{xx+16}" y="{ly+15.5}" class="legend">{_esc(label)}</text>')

    for exp in range(1, 7):
        yy = ty(10**exp)
        svg.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{PALETTE["light_grid"]}"/>')
        svg.append(f'<text x="{ml-8}" y="{yy+3.5:.1f}" text-anchor="end" class="tick">10{SUPERSCRIPT[str(exp)]}</text>')

    svg.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')
    svg.append(f'<text x="18" y="{mt+ph/2:.1f}" text-anchor="middle" class="axis-label" transform="rotate(-90 18 {mt+ph/2:.1f})">Nodes Visited (log scale)</text>')

    group_w = pw / len(selected)
    bar_w = group_w * 0.17

    for g_idx, (row, name) in enumerate(zip(selected, short_names)):
        gx = ml + (g_idx + 0.5) * group_w
        parts = name.split("\n")
        svg.append(f'<text x="{gx:.1f}" y="{mt+ph+16}" text-anchor="middle" class="tick">{_esc(parts[0])}</text>')
        svg.append(f'<text x="{gx:.1f}" y="{mt+ph+30}" text-anchor="middle" class="tick">{_esc(parts[1])}</text>')

        start_x = gx - 2 * bar_w
        for s_idx, (key, _, col) in enumerate(series):
            val = row[key]
            bx = start_x + s_idx * bar_w
            by = ty(val)
            bh = (mt + ph) - by
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w-2:.1f}" height="{bh:.1f}" fill="{col}" rx="1"/>')

    svg.append("</svg>")
    _save_svg_and_exports("\n".join(svg), "fig4_alphabeta_pruning")


# ---------------------------------------------------------------------------
# Figure 5: Bonus Results — Xiangqi Win/Score Rate & Delivery Routing Cost
# ---------------------------------------------------------------------------

def plot_fig5_bonus_results(data: dict[str, Any]) -> None:
    xq_rows = data["exp7_xiangqi"]
    del_rows = data["exp8_delivery_bonus"]

    w, h = 760, 365
    svg = _svg_header(w, h)
    svg.append(
        f'<text x="{w/2:.1f}" y="22" text-anchor="middle" class="title">'
        "Bonus Extensions: (a) Xiangqi AI Strength &amp; (b) Last-Mile Delivery Routing Cost</text>"
    )

    # Panel (a): Xiangqi Win/Draw stacked bars
    ml, mt, pw, ph = 158, 52, 180, 232
    svg.append(f'<text x="{ml+pw/2:.1f}" y="{mt-10}" text-anchor="middle" class="axis-label">(a) Xiangqi Win / Draw Rate (%)</text>')
    short_xq = [
        "AB(d=2) vs Rand",
        "MCTS-Rand vs Rand",
        "MCTS-Heur vs Rand",
        "MCTS-Val vs Rand",
        "PUCT-Val vs Greedy",
        "AB(d=2) vs Greedy",
    ]
    for xv in (0, 50, 100):
        xx = ml + (xv / 100.0) * pw
        svg.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="{PALETTE["light_grid"]}"/>')
        svg.append(f'<text x="{xx:.1f}" y="{mt+ph+16}" text-anchor="middle" class="tick">{xv}%</text>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')
    svg.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')

    step = ph / len(xq_rows)
    for idx, (r, lbl) in enumerate(zip(xq_rows, short_xq)):
        yc = mt + (idx + 0.5) * step
        bh = step * 0.55
        win_w = r["win_rate"] * pw
        draw_w = r["draw_rate"] * pw
        svg.append(f'<rect x="{ml}" y="{yc-bh/2:.1f}" width="{win_w:.1f}" height="{bh:.1f}" fill="{PALETTE["blue"]}"/>')
        svg.append(f'<rect x="{ml+win_w:.1f}" y="{yc-bh/2:.1f}" width="{draw_w:.1f}" height="{bh:.1f}" fill="#93c5fd"/>')
        svg.append(f'<text x="{ml-6}" y="{yc+3.5:.1f}" text-anchor="end" class="tick">{_esc(lbl)}</text>')
        svg.append(f'<text x="{ml+win_w+draw_w+5:.1f}" y="{yc+3.5:.1f}" class="val">{r["win_rate"]*100:.0f}%W</text>')

    svg.append(f'<rect x="{ml+15}" y="{mt+ph+30}" width="10" height="10" fill="{PALETTE["blue"]}"/>')
    svg.append(f'<text x="{ml+29}" y="{mt+ph+39}" class="tick">Win</text>')
    svg.append(f'<rect x="{ml+80}" y="{mt+ph+30}" width="10" height="10" fill="#93c5fd"/>')
    svg.append(f'<text x="{ml+94}" y="{mt+ph+39}" class="tick">Draw</text>')

    # Panel (b): Delivery Routing Total Cost across M in {8, 12, 16}
    ml2, pw2 = 472, 265
    svg.append(f'<text x="{ml2+pw2/2:.1f}" y="{mt-10}" text-anchor="middle" class="axis-label">(b) Delivery Total Cost (Lower = Better)</text>')

    ms = [8, 12, 16]
    methods = [
        ("NearestNeighbor", "Nearest Neighbor", PALETTE["red"]),
        ("EarliestDeadline", "EDF Heuristic", PALETTE["orange"]),
        ("MCTS-Random(it=1500)", "MCTS-Random", PALETTE["purple"]),
        ("MCTS-Heuristic(it=1500)", "MCTS-Heuristic", PALETTE["green"]),
    ]
    max_cost = 4000.0
    ty2 = lambda c: mt + ph - (c / max_cost) * ph

    for yv in (0, 1000, 2000, 3000, 4000):
        yy = ty2(yv)
        svg.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw2}" y2="{yy:.1f}" stroke="{PALETTE["light_grid"]}"/>')
        svg.append(f'<text x="{ml2-6}" y="{yy+3.5:.1f}" text-anchor="end" class="tick">{yv}</text>')
    svg.append(f'<rect x="{ml2}" y="{mt}" width="{pw2}" height="{ph}" fill="none" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')
    svg.append(f'<text x="{ml2-46}" y="{mt+ph/2:.1f}" text-anchor="middle" class="axis-label" transform="rotate(-90 {ml2-46} {mt+ph/2:.1f})">Total Cost</text>')

    gw = pw2 / len(ms)
    bw = gw * 0.18
    for g_idx, m_ord in enumerate(ms):
        gx = ml2 + (g_idx + 0.5) * gw
        svg.append(f'<text x="{gx:.1f}" y="{mt+ph+16}" text-anchor="middle" class="tick">M={m_ord}</text>')
        sx = gx - 2 * bw
        for s_idx, (m_key, _, col) in enumerate(methods):
            rec = next(r for r in del_rows if r["num_orders"] == m_ord and r["method"] == m_key)
            bx = sx + s_idx * bw
            by = ty2(rec["avg_cost"])
            bh = (mt + ph) - by
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw-1.5:.1f}" height="{bh:.1f}" fill="{col}" rx="1"/>')

    # Clean 2x2 legend under panel (b) with generous spacing
    for idx, (_, label, col) in enumerate(methods):
        col_i = idx % 2
        row_i = idx // 2
        xx = ml2 + col_i * 145
        yy = mt + ph + 34 + row_i * 18
        svg.append(f'<rect x="{xx}" y="{yy-8}" width="10" height="10" fill="{col}"/>')
        svg.append(f'<text x="{xx+15}" y="{yy+0.5}" class="tick">{_esc(label)}</text>')

    svg.append("</svg>")
    _save_svg_and_exports("\n".join(svg), "fig5_bonus_results")


def main() -> None:
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"Run experiments first: {RESULTS_JSON} not found.")
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    plot_fig1_c_vs_iterations(data)
    plot_fig2_rollout_comparison(data)
    plot_fig3_uct_variants(data)
    plot_fig4_alphabeta_pruning(data)
    plot_fig5_bonus_results(data)
    print(f"Generated all 5 polished publication figures in {FIG_DIR} and {EXP_FIG_DIR}!")


if __name__ == "__main__":
    main()

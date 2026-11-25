"""Build standalone 2-3 page IEEE HTML & PDF preview for the Xiangqi Bonus Report."""

from __future__ import annotations

from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
HTML_PATH = REPORT_DIR / "xiangqi_bonus_preview.html"
PDF_PATH = REPORT_DIR / "Xiangqi_Bonus_Report.pdf"

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Bonus Appendix: Xiangqi (Chinese Chess) — Alpha-Beta &amp; MCTS-UCT</title>
<style>
  @page { size: letter; margin: 0.58in 0.56in; }
  body {
    font-family: "Times New Roman", Georgia, serif;
    font-size: 9.6pt;
    line-height: 1.32;
    color: #111;
    max-width: 8.5in;
    margin: 0 auto;
    padding: 20px 28px;
    background: #fff;
  }
  .header { text-align: center; margin-bottom: 14px; border-bottom: 1.2px solid #cbd5e1; padding-bottom: 10px; }
  .title { font-size: 16.5pt; font-weight: bold; line-height: 1.18; margin-bottom: 6px; }
  .author { font-size: 10pt; font-style: italic; color: #334155; }
  .columns { column-count: 2; column-gap: 0.26in; text-align: justify; }
  h2 {
    font-size: 10.2pt;
    text-transform: uppercase;
    text-align: center;
    letter-spacing: 0.4px;
    margin: 11px 0 5px;
    break-after: avoid;
  }
  h3 { font-size: 9.7pt; font-style: italic; margin: 7px 0 3px; break-after: avoid; }
  p { margin: 0 0 5.5px; }
  .abstract { font-size: 9pt; font-weight: bold; margin-bottom: 7px; }
  .abstract span { font-weight: normal; }
  .keywords { font-size: 8.8pt; font-style: italic; margin-bottom: 9px; }
  .eq {
    text-align: center;
    margin: 6px 0;
    font-style: italic;
    font-size: 9.4pt;
    background: #f8fafc;
    padding: 4.5px 7px;
    border-left: 2.5px solid #1f4e79;
    break-inside: avoid;
  }
  table { width: 100%; border-collapse: collapse; font-size: 8.2pt; margin: 6px 0 9px; break-inside: avoid; }
  th, td { border-bottom: 0.6px solid #cbd5e1; padding: 3px 4px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  thead th { border-top: 1.3px solid #1e293b; border-bottom: 1px solid #1e293b; font-weight: bold; background: #f1f5f9; }
  tbody tr:last-child td { border-bottom: 1.3px solid #1e293b; }
  caption { font-size: 8.5pt; font-weight: bold; text-align: center; margin-bottom: 3px; color: #1e293b; }
  ul, ol { margin: 3px 0 6px 15px; padding: 0; }
  li { margin-bottom: 2.5px; }
  .refs { font-size: 8.2pt; }
  .refs p { padding-left: 16px; text-indent: -16px; margin-bottom: 2.5px; }
</style>
</head>
<body>

<div class="header">
  <div class="title">Bonus Appendix: Mathematical Modeling and Adversarial Search<br>
  for Xiangqi (Chinese Chess) via Alpha&ndash;Beta Pruning and MCTS&ndash;UCT</div>
  <div class="author">Student Author &mdash; Artificial Intelligence Midterm Project (Bonus Extension, 2026&ndash;2027)</div>
</div>

<div class="columns">

<div class="abstract">Abstract&mdash;<span>This standalone bonus report extends our adversarial search study from Tic-Tac-Toe and Gomoku to Xiangqi (Chinese Chess). Emphasizing formal domain knowledge and mathematical foundations, we model Xiangqi's asymmetric board topology, piece-blocking constraints, Flying General rule, and checkmate/stalemate-as-loss semantics. We then apply two search paradigms sequentially: (1) Minimax with Alpha&ndash;Beta pruning backed by Material + Piece-Square Table (PST) evaluation and MVV&ndash;LVA move ordering, and (2) Monte Carlo Tree Search (MCTS) with UCT, logistic value-cutoff simulation, and heuristic-prior PUCT. Verified benchmarks confirm a 93.44% node reduction on the 44-move opening via ordered Alpha&ndash;Beta, while Value-Cutoff MCTS lifts win rate against Random from 66.0% to 96.0% at an 8.4&times; speedup, and PUCT + Value-Cutoff defeats a 1-ply Greedy opponent 90.0%W / 10.0%D / 0.0%L.</span></div>

<div class="keywords"><b>Index Terms</b>&mdash;Xiangqi, Chinese Chess, Formal Game Modeling, Alpha&ndash;Beta Pruning, MVV&ndash;LVA, MCTS, UCT, PUCT, Value-Cutoff Simulation.</div>

<h2>I. Brief Introduction</h2>
<p>While 3&times;3 Tic-Tac-Toe is solvable exactly and 5-in-a-row Gomoku uses homogeneous stones, <b>Xiangqi (Chinese Chess)</b> introduces heterogeneous pieces, directional constraints, and non-local blocking rules on a 10&times;9 grid. With state-space complexity &sim;10<sup>40</sup>, game-tree complexity &sim;10<sup>150</sup>, and branching factor <i>b</i> &approx; 38 (44 legal moves at the opening), Xiangqi is an ideal bonus benchmark for examining how formal domain knowledge powers both exact tree pruning (Minimax + Alpha&ndash;Beta) and simulation-based planning (MCTS + UCT/PUCT).</p>

<h2>II. Formal Domain Knowledge &amp; Mathematical Model</h2>
<h3>A. Board Topology and State Space</h3>
<p>Let board intersections be &Omega; = {0,&hellip;,9} &times; {0,&hellip;,8} (|&Omega;| = 90). A state is <i>s</i> = (<i>B</i>, <i>p</i>) with active player <i>p</i> &isin; {+1, &minus;1} (Red +1 moves first; Black &minus;1 second) and piece map <i>B</i>: &Omega; &rarr; {&minus;7,&hellip;,0,&hellip;,+7}, where &tau; = |<i>B</i>(<i>x</i>)| encodes: <b>1</b>: General (<i>K</i>), <b>2</b>: Advisor (<i>A</i>), <b>3</b>: Elephant (<i>E</i>), <b>4</b>: Horse (<i>H</i>), <b>5</b>: Chariot (<i>R</i>), <b>6</b>: Cannon (<i>C</i>), <b>7</b>: Soldier (<i>P</i>).</p>
<p>Board topology partitions &Omega; into two <i>palaces</i> and two <i>river halves</i>:</p>
<div class="eq">P<sub>+1</sub> = {7,8,9}&times;{3,4,5}, &nbsp; P<sub>&minus;1</sub> = {0,1,2}&times;{3,4,5}<br>
H<sub>+1</sub> = {5..9}&times;{0..8}, &nbsp; H<sub>&minus;1</sub> = {0..4}&times;{0..8}</div>

<h3>B. Piece Kinematic &amp; Blocking Constraints</h3>
<p>For an orthogonal segment between <i>u</i> = (<i>r</i><sub>1</sub>, <i>c</i><sub>1</sub>) and <i>v</i> = (<i>r</i><sub>2</sub>, <i>c</i><sub>2</sub>), define the number of strictly intervening occupied squares:</p>
<div class="eq">m(u, v) = |{ w &isin; open_seg(u, v) : B(w) &ne; 0 }|</div>
<p>A pseudo-legal move <i>u</i> &rarr; <i>v</i> by player <i>p</i> requires <i>p</i>&middot;<i>B</i>(<i>u</i>) &gt; 0, <i>p</i>&middot;<i>B</i>(<i>v</i>) &le; 0 (no self-capture), and satisfies:</p>
<ul>
  <li><b>General (<i>K</i>):</b> <i>v</i> &isin; <i>P<sub>p</sub></i> and &Vert;<i>v</i> &minus; <i>u</i>&Vert;<sub>1</sub> = 1.</li>
  <li><b>Advisor (<i>A</i>):</b> <i>v</i> &isin; <i>P<sub>p</sub></i> and |&Delta;<i>r</i>| = |&Delta;<i>c</i>| = 1.</li>
  <li><b>Elephant (<i>E</i>):</b> <i>v</i> &isin; <i>H<sub>p</sub></i> (cannot cross river), |&Delta;<i>r</i>| = |&Delta;<i>c</i>| = 2, and unblocked <i>eye</i>: <i>B</i>((<i>r</i><sub>1</sub>+<i>r</i><sub>2</sub>)/2, (<i>c</i><sub>1</sub>+<i>c</i><sub>2</sub>)/2) = 0.</li>
  <li><b>Horse (<i>H</i>):</b> {|&Delta;<i>r</i>|, |&Delta;<i>c</i>|} = {1, 2} and unblocked orthogonal <i>leg</i>: <i>B</i>(<i>r</i><sub>1</sub>+&ell;<sub><i>r</i></sub>, <i>c</i><sub>1</sub>+&ell;<sub><i>c</i></sub>) = 0, where &ell;<sub><i>r</i></sub> = sgn(&Delta;<i>r</i>)&middot;<b>1</b><sub>|&Delta;<i>r</i>|=2</sub>, &ell;<sub><i>c</i></sub> = sgn(&Delta;<i>c</i>)&middot;<b>1</b><sub>|&Delta;<i>c</i>|=2</sub>.</li>
  <li><b>Chariot (<i>R</i>):</b> Orthogonal move with <i>m</i>(<i>u</i>, <i>v</i>) = 0.</li>
  <li><b>Cannon (<i>C</i>):</b> Orthogonal with <i>m</i>(<i>u</i>, <i>v</i>) = 0 if <i>B</i>(<i>v</i>) = 0 (non-capture), and <b><i>m</i>(<i>u</i>, <i>v</i>) = 1</b> if <i>p</i>&middot;<i>B</i>(<i>v</i>) &lt; 0 (must jump exactly one screen piece to capture).</li>
  <li><b>Soldier (<i>P</i>):</b> Before river (<i>u</i> &isin; <i>H<sub>p</sub></i>): (&Delta;<i>r</i>, &Delta;<i>c</i>) = (&minus;<i>p</i>, 0) only. After crossing (<i>u</i> &notin; <i>H<sub>p</sub></i>): (&Delta;<i>r</i>, &Delta;<i>c</i>) &isin; {(&minus;<i>p</i>, 0), (0, &minus;1), (0, +1)}.</li>
</ul>

<h3>C. Check, Flying General &amp; Terminal Semantics</h3>
<p>Let <i>x<sub>K</sub></i>(<i>p</i>) &isin; <i>P<sub>p</sub></i> be player <i>p</i>'s General. State <i>s</i> has player <i>p</i> in check, Check(<i>s</i>, <i>p</i>), iff an opponent piece pseudo-legally attacks <i>x<sub>K</sub></i>(<i>p</i>) or the <b>Flying General</b> rule is violated:</p>
<div class="eq">FlyingGen(s) &hArr; c<sub>K</sub>(+1) = c<sub>K</sub>(&minus;1) &and; m(x<sub>K</sub>(+1), x<sub>K</sub>(&minus;1)) = 0</div>
<p>The legal move set filters out self-checking moves: <i>A</i>(<i>s</i>) = { <i>a</i> &isin; <i>A</i><sub>pseudo</sub>(<i>s</i>) : &not;Check(<i>T</i>(<i>s</i>, <i>a</i>), <i>p</i>) }. Crucially, unlike Western Chess, Xiangqi scores <b>both checkmate and stalemate as a loss</b> for the side without legal moves:</p>
<div class="eq">A(s) = &empty; &rArr; u(s, p) = &minus;1, &nbsp; u(s, &minus;p) = +1</div>
<p>Draws (<i>u</i> = 0) occur on insufficient attacking material (&forall;<i>x</i>, |<i>B</i>(<i>x</i>)| &notin; {<i>R</i>, <i>C</i>, <i>H</i>, <i>P</i>}) or at the ply limit <i>t</i><sub>max</sub>.</p>

<h2>III. Algorithm 1: Minimax + Alpha&ndash;Beta Pruning</h2>
<h3>A. Exact Recurrence &amp; Pruning Complexity</h3>
<p>For root perspective <i>r</i> and depth <i>d</i>, Minimax computes <i>V<sub>d</sub></i>(<i>s</i>) = Eval(<i>s</i>, <i>r</i>) at leaves, max<sub><i>a</i></sub> <i>V</i><sub><i>d</i>&minus;1</sub>(<i>T</i>(<i>s</i>, <i>a</i>)) at MAX nodes, and min<sub><i>a</i></sub> <i>V</i><sub><i>d</i>&minus;1</sub>(<i>T</i>(<i>s</i>, <i>a</i>)) at MIN nodes. Alpha&ndash;Beta maintains guaranteed bounds [&alpha;, &beta;] along the recursion stack and prunes remaining siblings whenever &alpha; &ge; &beta; without altering <i>V<sub>d</sub></i>(<i>s</i><sub>0</sub>) [1]. Complexity drops from &Theta;(<i>b<sup>d</sup></i>) to &Theta;(<i>b</i><sup><i>d</i>/2</sup>) under strong move ordering: for Xiangqi (<i>b</i> &approx; 38, <i>d</i> = 2), leaves shrink from 38<sup>2</sup> &approx; 1,444 toward <i>O</i>(38).</p>

<h3>B. Domain Evaluation &amp; MVV&ndash;LVA Ordering</h3>
<p>Non-terminal leaves are evaluated by Material + Piece-Square Tables (PST) + river-crossing bonus:</p>
<div class="eq">Eval(s, r) = r &Sigma;<sub>B(x)&ne;0</sub> sgn(B(x)) [ w<sub>|B(x)|</sub> + PST<sub>|B(x)|</sub>(&phi;<sub>sgn</sub>(x)) + &delta;<sub>riv</sub>(x) ]</div>
<p>with material weights <i>w</i> = (10000, 200, 200, 400, 900, 450, 100) for (<i>K</i>, <i>A</i>, <i>E</i>, <i>H</i>, <i>R</i>, <i>C</i>, <i>P</i>), symmetry map &phi;<sub>+1</sub>(<i>r</i>, <i>c</i>) = (<i>r</i>, <i>c</i>), &phi;<sub>&minus;1</sub>(<i>r</i>, <i>c</i>) = (9&minus;<i>r</i>, <i>c</i>), and &delta;<sub>riv</sub> = 80 for crossed Soldiers. Terminal wins score &plusmn;(10<sup>5</sup> + 2(<i>t</i><sub>max</sub> &minus; <i>t</i>)) to prefer faster checkmates.</p>
<p>To maximize early cutoffs, legal moves <i>u</i> &rarr; <i>v</i> are sorted descending by <b>MVV&ndash;LVA</b> (Most Valuable Victim &minus; Least Valuable Attacker) plus central-file bonus:</p>
<div class="eq">h<sub>ord</sub>(u &rarr; v) = 10&middot;w<sub>|B(v)|</sub>&middot;1<sub>B(v)&ne;0</sub> &minus; w<sub>|B(u)|</sub> + 2(4 &minus; |c<sub>v</sub> &minus; 4|)</div>

<table>
  <caption>TABLE I. Verified Minimax vs. Alpha&ndash;Beta Pruning on Xiangqi (9&times;10)</caption>
  <thead>
    <tr><th>Position</th><th>d</th><th>Minimax</th><th>AB Nat.</th><th>AB Ord.</th><th>Red.%</th><th>ms</th></tr>
  </thead>
  <tbody>
    <tr><td>Std. Opening (b=44)</td><td>2</td><td>1,965</td><td>1,116</td><td><b>129</b></td><td><b>93.44%</b></td><td>20.7</td></tr>
    <tr><td>Skirmish (b&approx;28)</td><td>2</td><td>768</td><td>230</td><td><b>180</b></td><td><b>76.56%</b></td><td>23.6</td></tr>
    <tr><td>Skirmish (b&approx;28)</td><td>3</td><td>&mdash;</td><td>2,337</td><td><b>1,159</b></td><td>&mdash;</td><td>117.0</td></tr>
  </tbody>
</table>

<p>On the 44-move opening (Table I), MVV&ndash;LVA ordering cuts unordered Alpha&ndash;Beta nodes by 8.65&times; (1,116 &rarr; <b>129 nodes</b>), achieving a <b>93.44% reduction</b> over plain Minimax (1,965 nodes, 16.3&times; speedup) with exact value equivalence (&minus;66.0).</p>

<h2>IV. Algorithm 2: MCTS&ndash;UCT, Value-Cutoff &amp; PUCT</h2>
<h3>A. Bandit Foundation &amp; Sparse Rollout Challenge</h3>
<p>Modeling each internal node as a <i>K</i>-armed bandit with rewards <i>r</i> &isin; [0, 1], Hoeffding's inequality <i>P</i>(&mu;<sub><i>i</i></sub> &minus; X&#772;<sub><i>i</i></sub> &ge; &epsilon;) &le; exp(&minus;2<i>N<sub>i</sub></i>&epsilon;<sup>2</sup>) with failure probability &delta; = <i>N<sub>p</sub></i><sup>&minus;4</sup> yields canonical UCT [2], [3]:</p>
<div class="eq">UCT(s, a<sub>i</sub>) = W<sub>i</sub>/N<sub>i</sub> + c &middot; &radic;(ln N<sub>p</sub> / N<sub>i</sub>), &nbsp; c<sub>theory</sub> = &radic;2</div>
<p>Backpropagation tracks <i>player_just_moved</i> so Red nodes accumulate <i>r</i><sub>+1</sub> and Black nodes accumulate 1 &minus; <i>r</i><sub>+1</sub>; final selection picks the <i>robust child</i> argmax<sub><i>i</i></sub> <i>N<sub>i</sub></i>. However, pure random rollouts struggle in Xiangqi: with <i>b</i> &approx; 38, random moves rarely deliver checkmate within 40&ndash;60 plies, causing sparse draw signals (0.5) and slow search (1727 ms/move at 200 iterations).</p>

<h3>B. Domain-Knowledge Mathematical Enhancements</h3>
<ol>
  <li><b>Capture-Biased Heuristic Rollout:</b> With probability 0.85 pick argmax<sub><i>a</i></sub> <i>w</i><sub>|<i>B</i>(<i>v<sub>a</sub></i>)|</sub> among captures; else sample uniformly.</li>
  <li><b>Value-Cutoff Simulation (<i>L</i> = 4, Value-Network Precursor):</b> Play <i>L</i> = 4 tactical plies, then map Eval(<i>s</i>, +1) into an estimated win probability in (0, 1) via a logistic sigmoid:
    <div class="eq">v&#770;(s) = &sigma;(Eval(s, +1) / &tau;) = 1 / (1 + exp(&minus;Eval(s, +1) / 600))</div>
  </li>
  <li><b>PUCT with MVV&ndash;LVA Rank Prior [4]:</b>
    <div class="eq">PUCT(s, a) = Q(s, a) + c &middot; P(s, a) &middot; &radic;N(s) / (1 + N(s, a))</div>
    where <i>P</i>(<i>s</i>, <i>a</i>) &prop; exp(&minus;0.35&middot;rank<sub><i>h</i><sub>ord</sub></sub>(<i>a</i>)) focuses expansion on sharp tactical moves.
  </li>
</ol>

<table>
  <caption>TABLE II. Verified Xiangqi 50-Game Alternating Tournament (200 MCTS Iters)</caption>
  <thead>
    <tr><th>Matchup</th><th>W / D / L</th><th>Win%</th><th>Score%</th><th>ms/mv</th></tr>
  </thead>
  <tbody>
    <tr><td>Alpha&ndash;Beta (d=2) vs. Random</td><td>44 / 6 / 0</td><td>88.0%</td><td>94.0%</td><td>47.1</td></tr>
    <tr><td>MCTS-Random vs. Random</td><td>33 / 17 / 0</td><td>66.0%</td><td>83.0%</td><td>1727.4</td></tr>
    <tr><td>MCTS-Heuristic vs. Random</td><td>44 / 6 / 0</td><td>88.0%</td><td>94.0%</td><td>1062.2</td></tr>
    <tr><td><b>MCTS-ValueCut vs. Random</b></td><td><b>48 / 0 / 2</b></td><td><b>96.0%</b></td><td><b>96.0%</b></td><td><b>206.1</b></td></tr>
    <tr><td>Alpha&ndash;Beta (d=2) vs. Greedy</td><td>25 / 25 / 0</td><td>50.0%</td><td>75.0%</td><td>39.3</td></tr>
    <tr><td><b>PUCT-ValueCut vs. Greedy</b></td><td><b>45 / 5 / 0</b></td><td><b>90.0%</b></td><td><b>95.0%</b></td><td><b>185.6</b></td></tr>
  </tbody>
</table>

<p>Table II demonstrates dramatic gains from domain knowledge: against Random, Value-Cutoff MCTS raises win rate from <b>66.0% &rarr; 96.0%</b> while running <b>8.4&times; faster</b> than random rollouts (206.1 vs. 1727.4 ms). Against 1-ply Greedy, <b>PUCT + Value-Cutoff</b> wins <b>90.0% with 0 losses</b>, outperforming fixed-depth Alpha&ndash;Beta (<i>d</i> = 2, 50.0%W / 50.0%D).</p>

<h2>V. Comparative Synthesis &amp; Conclusion</h2>
<p>Applying both paradigms to Xiangqi exposes a clear mathematical trade-off: <b>Alpha&ndash;Beta + MVV&ndash;LVA</b> delivers ultra-fast exact tactical bounds (20&ndash;47 ms/mv, 93.4% pruning) but scales as <i>O</i>(<i>b</i><sup><i>d</i>/2</sup>) and suffers from fixed-depth horizon effects; <b>MCTS&ndash;PUCT + Value-Cutoff</b> scales linearly <i>O</i>(<i>T</i>&middot;<i>L</i>), asymmetrically deepens forcing lines, and achieves 90%&ndash;96% win rates&mdash;directly demonstrating the mathematical bridge from classical UCT to modern AlphaZero architectures [4].</p>

<h2>References</h2>
<div class="refs">
  <p>[1] D. E. Knuth and R. W. Moore, "An analysis of alpha-beta pruning," <i>Artificial Intelligence</i>, vol. 6, no. 4, pp. 293&ndash;326, 1975.</p>
  <p>[2] P. Auer, N. Cesa-Bianchi, and P. Fischer, "Finite-time analysis of the multiarmed bandit problem," <i>Machine Learning</i>, vol. 47, pp. 235&ndash;256, 2002.</p>
  <p>[3] L. Kocsis and C. Szepesv&aacute;ri, "Bandit based Monte-Carlo planning," in <i>Proc. ECML</i>, Springer, 2006, pp. 282&ndash;293.</p>
  <p>[4] D. Silver et al., "Mastering the game of Go without human knowledge," <i>Nature</i>, vol. 550, pp. 354&ndash;359, 2017.</p>
</div>

</div>
</body>
</html>
"""

if __name__ == "__main__":
    HTML_PATH.write_text(HTML_CONTENT, encoding="utf-8")
    print(f"Wrote Xiangqi Bonus HTML: {HTML_PATH}")

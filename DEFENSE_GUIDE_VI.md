# Hướng Dẫn Vấn Đáp, Giải Thích Code & Kịch Bản Demo (10–15 Phút)

Tài liệu này giúp bạn tự tin trả lời mọi câu hỏi **vấn đáp lý thuyết + hỏi code + chạy demo trực tiếp** để đạt mức điểm tối đa (9–10 điểm).

---

## 1. Kịch Bản Demo Trực Tiếp (3–4 Phút)

1. **Mở giao diện Tkinter GUI** (không cần cài thư viện ngoài):
   ```bash
   python -m demo.gui
   ```
2. **Demo 1 — Tic-Tac-Toe 3×3:**
   - Chọn `Game: Tic-Tac-Toe 3x3`, `AI: MCTS-UCT (Random)`, `Iters: 1000`, `c: 1.41`.
   - Đánh thử tạo bẫy 2 hướng (fork) → cho giảng viên xem bảng **Telemetry bên phải** hiển thị `(W, N, W/N, Score)` của từng ô trống và thấy máy luôn chặn chính xác, không bao giờ thua!
   - Đổi sang `Alpha-Beta` (`Depth: 9`) → chỉ ra số nút giảm từ **549,946** xuống chỉ còn vài nghìn nút mà giá trị vẫn bằng `0.0` (hòa tối ưu).
3. **Demo 2 — Gomoku 9×9 (5 quân thẳng hàng):**
   - Chọn `Game: Gomoku 9x9 (5-in-a-row)`, `AI: MCTS-PUCT` hoặc `MCTS-UCT (Value Cutoff)`, `Iters: 600`.
   - Đánh 3–4 nước liên tiếp → cho thấy máy tự động chặn 3 mở / 4 quân và tấn công mạnh mẽ trong <1 giây/nước, đồng thời in rõ bảng `W, N, W/N` của top-6 nước ứng viên.
4. **Demo 3 — Bonus Cờ Tướng 9×10:**
   - Chọn `Game: Xiangqi 9x10 (Co Tuong)`, `AI: Alpha-Beta`, `Depth: 2`.
   - Đi thử Pháo đầu hoặc Mã → chỉ cho giảng viên thấy luật cản mã, ăn pháo qua ngòi, lộ mặt tướng được cài đặt đầy đủ.
5. **Demo 4 — Bonus Bài 2 (Lập lịch giao hàng thực tế) qua terminal:**
   ```bash
   python -m demo.cli --game delivery --orders 12 --iterations 1500
   ```
   → Cho thấy ngay bảng so sánh tổng chi phí: Nearest-Neighbor (`~1309`) vs EDF (`~1855`) vs MCTS-Heuristic (`~742`, giảm >43% chi phí!).

---

## 2. Bản Đồ Mã Nguồn (Nếu Giảng Viên Hỏi "Chỗ Này Code Ở Đâu?")

| Chức năng | File & Hàm chính |
|---|---|
| Luật 3×3 Tic-Tac-Toe & kiểm tra thắng 8 đường | `src/games/tictactoe.py` → `TicTacToeState._compute_winner()`, `fast_rollout()` |
| Luật Gomoku 5 quân thẳng hàng (4 hướng) | `src/games/gomoku.py` → `GomokuState._check_win_at()`, `is_winning_move()` |
| Hàm lượng giá Gomoku (đếm chuỗi + số đầu mở) | `src/games/gomoku.py` → `GomokuState.evaluate()`, `PATTERN_SCORES` |
| Cờ Tướng: Lộ mặt tướng, cản mã, mắt tượng, ngòi pháo | `src/games/xiangqi.py` → `is_in_check()`, `_pseudo_legal_moves()`, `legal_moves()` |
| Minimax thuần & Alpha-Beta + Move Ordering + TT | `src/algorithms/minimax.py` → `minimax_search()`, `alphabeta_search()` |
| 4 bước MCTS (Selection, Expansion, Simulation, Backprop) | `src/algorithms/mcts.py` → `mcts_search()` |
| Công thức UCT + 6 biến thể (`no_sqrt`, `no_ln`, `ucb1_tuned`, `puct`, ...) | `src/algorithms/mcts.py` → `MCTSNode.selection_score()` |
| Rollout ngẫu nhiên / Heuristic / Value-Cutoff | `src/algorithms/mcts.py` → `_simulate()`, `GomokuState.fast_rollout()` |
| Bài 2 Bonus: Lập lịch giao hàng TSP-TW (Exact BnB & SP-MCTS) | `src/bonus/delivery_scheduling.py` → `solve_branch_and_bound()`, `solve_mcts()` |
| Toàn bộ 8 thực nghiệm song song & vẽ 5 biểu đồ | `experiments/run_all.py`, `experiments/generate_charts.py` |

---

## 3. Bộ 12 Câu Hỏi Vấn Đáp Trọng Tâm & Trả Lời Chuẩn

### Q1. Tại sao Alpha-Beta cắt tỉa được rất nhiều nút mà không làm sai kết quả Minimax?
- **Trả lời:** $\alpha$ là điểm thấp nhất mà MAX đã chắc chắn có trên đường đi hiện tại; $\beta$ là điểm cao nhất mà MIN đã chắc chắn ép được. Khi duyệt một nút mà xảy ra $\alpha \ge \beta$, nghĩa là nút tổ tiên đã có một lựa chọn khác tốt hơn (hoặc bằng) bất kỳ giá trị nào nhánh hiện tại có thể trả về. Do đó cắt bỏ các nút con còn lại không bao giờ làm thay đổi giá trị Minimax ở gốc. Thực nghiệm của em kiểm chứng **sai số = 0** giữa Minimax và Alpha-Beta trên cả 3 trò chơi, trong khi giảm **98.57%** số nút trên TTT 3×3 (99.61% khi thêm Transposition Table) và **93.44%** trên khai cuộc Cờ Tướng.

### Q2. Tại sao thứ tự nước đi (Move Ordering) lại ảnh hưởng lớn đến Alpha-Beta?
- **Trả lời:** Trường hợp xấu nhất Alpha-Beta vẫn tốn $O(b^d)$, nhưng nếu luôn xét nước tốt nhất trước thì giảm xuống $O(b^{d/2})$ (gấp đôi độ sâu tìm kiếm). Minh chứng từ thực nghiệm Exp 1: trên khai cuộc Cờ Tướng (44 nước, $d=2$), Alpha-Beta không sắp xếp duyệt **1,116 nút**, nhưng chỉ cần thêm MVV-LVA (ưu tiên ăn quân giá trị cao bằng quân giá trị thấp) giảm mạnh xuống **129 nút** (nhanh gấp 8.6 lần).

### Q3. Tại sao công thức UCT lại có căn bậc hai ($\sqrt{\cdot}$) và logarit ($\ln N_p$)?
- **Trả lời:** UCT xuất phát từ bài toán Multi-Armed Bandit và **Bất đẳng thức Hoeffding**: với phần thưởng $X \in [0, 1]$, xác suất trung bình thật vượt quá trung bình thực nghiệm một lượng $\varepsilon$ bị chặn bởi $P(\mu_i - \bar{X}_i \ge \varepsilon) \le e^{-2 N_i \varepsilon^2}$.
  - Đặt mức sai số giảm theo $N_p^{-4}$ rồi giải tìm $\varepsilon$: ta lấy $\ln$ hai vế (xuất hiện $\ln N_p$) rồi khai căn (xuất hiện $\sqrt{\cdot}$), thu được $\varepsilon = \sqrt{\frac{2 \ln N_p}{N_i}}$, chính là $c = \sqrt{2} \approx 1.414$.
  - **Vai trò của $\sqrt{\cdot}$:** độ lệch chuẩn của trung bình mẫu giảm theo $1/\sqrt{N_i}$. Nếu bỏ căn (`No-Sqrt`), số hạng khám phá $\ln N_p / N_i$ quá lớn với nút ít thăm, khiến thuật toán khám phá lan man (tỷ lệ thắng tụt từ 28% xuống **16%** trong Exp 6).
  - **Vai trò của $\ln N_p$:** tăng rất chậm về $\infty$, đảm bảo không nhánh nào bị bỏ quên vĩnh viễn nếu lỡ thua xui vài ván đầu, đồng thời chỉ tốn $O(\ln T)$ lần thử cho nhánh kém (tiệm cận hội tụ về nước tối ưu).

### Q4. Tại sao cuối cùng MCTS chọn nước có $N_i$ lớn nhất (Robust Child) thay vì $W_i/N_i$ lớn nhất (Max Child)?
- **Trả lời:** Có 3 lý do:
  1. **Tránh nhiễu mẫu nhỏ:** Một nước dở mới thử $N_i = 1$ hoặc $2$ lần có thể ăn may thắng hết → $W_i/N_i = 1.0$, trong khi sai số chuẩn cực lớn ($\approx 0.5/\sqrt{N_i}$).
  2. **$N_i$ đã tích hợp sẵn chất lượng:** Để UCT liên tục quay lại một nhánh hàng trăm lần, bản thân nhánh đó phải duy trì tỷ lệ thắng cao trước hàng trăm phản đòn của đối thủ.
  3. **Chứng minh thực nghiệm (Exp 4):** Trên thế cờ TTT 3×3 của đề bài, nước chặn bắt buộc `(2,2)` có $N = 795$ (chiếm 79.5% tổng ngân sách), $W/N = 0.611$, áp đảo hoàn toàn các nước thua ($N = 42..61$). Đối đầu trực tiếp 60 ván ở ngân sách thấp (150 iters), Robust Child thắng Max Child **32–28**.

### Q5. Tại sao trên Gomoku, $c = 0.5$ lại thắng cao hơn $c = 1.4$ ở ngân sách thấp (100–1000 vòng), nhưng ở 5000 vòng $c = 1.4$ lại đạt 0% thua?
- **Trả lời:** Đây là phát hiện thực nghiệm rất hay (khớp với ghi chú trong đề: *"Giá trị lý thuyết là $\sqrt{2}$; thực tế cần tinh chỉnh"*). Trên Gomoku branching factor lớn (~20–30 nước ứng viên), khi ngân sách ít (100–1000 vòng), $c$ lớn khiến MCTS dàn trải đều các vòng lặp sang các ô rìa vô hại thay vì đào sâu nhánh chiến thuật, nên $c = 0.5$ tập trung khai thác tốt hơn (74% vs 58% ở 1000 vòng). Nhưng khi ngân sách đủ lớn (5000 vòng), $c = 1.4$ vừa đủ khám phá để không sót đòn phản công, đạt **90% thắng và 0/50 trận thua**, trong khi $c = 2.0$ vẫn khám phá quá mức (chỉ 78%).

### Q6. Khi cập nhật ngược (Backpropagation) trong MCTS, xử lý góc nhìn 2 người chơi như thế nào?
- **Trả lời:** Mỗi nút `MCTSNode` lưu trường `player_just_moved` (người vừa đi nước tạo ra nút đó). Khi simulation trả về phần thưởng `reward_p1` theo góc nhìn người `+1` (`1.0` nếu `+1` thắng, `0.5` hòa, `0.0` nếu `-1` thắng), vòng lặp Backpropagation cộng `reward_p1` nếu `node.player_just_moved == +1` và cộng `1.0 - reward_p1` nếu `node.player_just_moved == -1`. Nhờ vậy, tại nút cha (đến lượt người `p`), mọi nút con đều có `player_just_moved == p`, nên `child.wins / child.visits` luôn là tỷ lệ thắng **có lợi cho người đang chọn nước tại nút cha**!

### Q7. So sánh Random Rollout, Tactical Heuristic Rollout và Value-Cutoff Rollout? Liên hệ gì với Value Network?
- **Trả lời:** Trên Gomoku 9×9 (Exp 5, 1000 vòng lặp):
  - **Random Rollout:** nhanh (`225 ms/nước`) nhưng quá nhiễu, bỏ qua đòn thắng/chặn hiển nhiên → chỉ thắng **16%**.
  - **Tactical Heuristic Rollout:** biết thắng ngay / chặn ngay / tạo chuỗi mở → tăng hơn gấp đôi lên **34%**, nhưng vì chạy heuristic ở mọi bước tới hết ván nên rất chậm (**6,645 ms/nước**, chậm gấp 29.5 lần).
  - **Value-Cutoff Rollout ($L=4$ bước + hàm giá trị sigmoid):** chỉ mô phỏng 4 bước chiến thuật rồi đánh giá trực tiếp bằng hàm lượng giá $\sigma(\text{Eval}(s)/\tau)$ → vừa tăng tỷ lệ thắng vọt lên **74%**, vừa nhanh hơn **6.2 lần** so với heuristic rollout đầy đủ (`1,077 ms/nước`).
  - **Bài học cốt lõi:** Đây chính là **tiền thân của Value Network trong AlphaGo / AlphaZero** — thay vì mô phỏng ngẫu nhiên dài và nhiễu đến cuối ván, ta cắt ngắn rollout và dùng một hàm đánh giá (học sâu) để ước lượng xác suất thắng ngay tại nút lá!

### Q8. Hai cải tiến công thức UCT (UCB1-Tuned và PUCT) hoạt động ra sao?
- **Trả lời:**
  - **UCB1-Tuned:** lưu thêm tổng bình phương phần thưởng $\sum r^2$ tại mỗi nút để ước lượng phương sai thực nghiệm $\hat{\sigma}_i^2$, giúp thu hẹp bán kính tin cậy thay vì dùng cận xấu nhất $1/4$. Kết quả tăng tỷ lệ thắng từ 28% lên **42%** mà hầu như không tăng thời gian.
  - **PUCT (kiểu AlphaZero):** gắn phân phối xác suất tiên nghiệm $P(s,a)$ từ heuristic sắp xếp nước đi vào số hạng khám phá $c \cdot P(s,a) \frac{\sqrt{N_p}}{1+N_i}$. Kết quả giúp cây mở rộng tập trung ngay vào các nước chiến thuật sắc bén, đẩy tỷ lệ thắng lên **98% (49W / 1D / 0L)** ở cùng 500 vòng lặp!

### Q9. Trong Cờ Tướng (Xiangqi), làm sao kiểm tra hợp lệ các luật đặc thù như Lộ mặt tướng, Cản mã, Ngòi pháo?
- **Trả lời:** Trong `src/games/xiangqi.py`:
  - **Cản mã:** kiểm tra ô kề trực giao (chân mã) phải bằng `0` trước khi sinh 2 bước nhảy chéo tương ứng.
  - **Mắt tượng:** kiểm tra trung điểm chéo bằng `0` và giới hạn hàng không vượt sông (`row >= 5` cho Đỏ, `row <= 4` cho Đen).
  - **Pháo:** trượt theo 4 tia trực giao; trước khi gặp vật cản (`jumped=False`) chỉ đi vào ô trống; sau khi gặp đúng 1 quân làm ngòi (`jumped=True`), quân địch đầu tiên gặp trên tia là nước ăn hợp lệ rồi dừng tia.
  - **Lộ mặt tướng & Chiếu tướng:** hàm `is_in_check(player)` quét ngược từ Tướng theo 4 tia trực giao — nếu quân đầu tiên gặp là Xe hoặc **Tướng địch** (không có quân chắn giữa) thì coi là đang bị chiếu (bất hợp lệ). Mọi nước đi giả hợp lệ đều được thử đặt tạm, gọi `is_in_check(player)` rồi hoàn tác.
  - **Luật kết thúc Cờ Tướng:** khác Cờ Vua, nếu đến lượt mà không còn nước đi hợp lệ nào (kể cả chiếu bí lẫn hết nước đi / stalemate) thì bên đến lượt bị xử **THUA**.

### Q10. Bài toán thực tế Bonus (Lập lịch giao hàng TSP-TW) chuyển sang MCTS như thế nào khi chỉ có 1 người chơi?
- **Trả lời:** Dùng **Single-Player MCTS (SP-MCTS)**: trạng thái là lộ trình đã đi qua và tập đơn hàng còn lại; hành động là chọn đơn hàng kế tiếp; phần thưởng chuẩn hóa về $(0,1]$ theo chi phí so với mốc Nearest-Neighbor $r = \min(1, e^{-\max(0, J/J_{\text{NN}} - 0.5)})$, đồng thời lưu lộ trình có tổng chi phí (quãng đường + phạt giao trễ) nhỏ nhất từng thấy. Thực nghiệm trên 90 bài toán (Exp 8): ở $M=8$ đơn hàng, MCTS-Heuristic đạt độ lệch chỉ **4.78%** so với nghiệm tối ưu tuyệt đối Branch-and-Bound (trong khi tham lam Nearest-Neighbor lệch tới **126.7%**); ở $M=16$ đơn ($16! \approx 2\times 10^{13}$ trạng thái), MCTS-Heuristic giảm **43.6%** tổng chi phí so với Nearest-Neighbor chỉ trong `185 ms`.

### Q11. MCTS có thật sự "không bao giờ thua" trên Tic-Tac-Toe 3×3 không? Điều kiện là gì?
- **Trả lời:** Về lý thuyết (định lý Kocsis & Szepesvári 2006), khi $T \to \infty$ UCT hội tụ về quyết định Minimax tối ưu (hòa). Trong thực nghiệm với số vòng lặp hữu hạn (Exp 2, 100 ván đổi lượt với Alpha-Beta tối ưu), ở `100` và `250` vòng MCTS vẫn thua lần lượt 9% và 5%; từ **`iterations ≥ 500` trở lên** (500, 1000, 2000, 5000), MCTS đạt **0/100 trận thua (100% không thua)**. Điều kiện là ngân sách đủ lớn và rollout/luật cài đặt không có lỗi.

### Q12. Hạn chế của chương trình hiện tại và hướng phát triển?
- **Trả lời:** (1) Viết bằng Python thuần nên tốc độ thấp hơn C++/Rust ~50×; (2) Gomoku dùng bán kính ứng viên $r=2$ để giảm nhánh; (3) Cờ Tướng chưa cài luật cấm chiếu lặp/bắt quân lặp liên hoàn của Liên đoàn Cờ Tướng châu Á. Hướng phát triển: viết lại lõi bằng C++/Cython và huấn luyện mạng Policy + Value nhỏ bằng Self-Play (AlphaZero-lite) cho Gomoku & Cờ Tướng.

# Báo cáo tiếng Việt

- `main.tex`: mã LaTeX UTF-8, bố cục A4 một cột.
- `figures/`: 7 hình, mỗi hình có SVG và PDF vector tương ứng.
- `charts.py`: mã Python/Matplotlib để dựng lại hình.

## Biên dịch
Trên Overleaf, tải các tệp lên cùng cấu trúc thư mục và chọn XeLaTeX trong Compiler. Trên máy tính, cài TeX Live và font Noto Serif, Noto Sans, rồi chạy `xelatex main.tex` hai lần. Các hình PDF vector được nhúng trực tiếp nên không cần chuyển SVG khi biên dịch. Không cần chạy Python trừ khi muốn sửa hình; cần matplotlib và numpy nếu chạy `python charts.py`.

Số liệu lấy từ báo cáo và bản dịch trong chat, chưa kiểm chứng độc lập. Các hình được dựng lại từ số liệu đã nêu, không phải bản sao đầy đủ các hình gốc.

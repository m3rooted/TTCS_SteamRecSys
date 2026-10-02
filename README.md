# Hệ khuyến nghị trò chơi Steam

Huấn luyện **Popularity**, **Item-kNN** và **BPR-MF** trên Steam-200k. Đọc cơ sở lý thuyết, công thức và cách đánh giá trong [LY_THUYET.md](LY_THUYET.md).

Phần khuyến nghị, đánh giá, so sánh cấu hình và kiểm thử đã được để trống để Lương Đức Khải An thực hiện theo [bảng phân công](PHAN_CONG_CONG_VIEC.md).

## Cấu trúc

```text
TTCS/
|-- main.py                  # Điểm chạy duy nhất
|-- LY_THUYET.md             # Tài liệu lý thuyết tiếng Việt
|-- README.md                # Hướng dẫn chạy
|-- requirements.txt         # Thư viện Python
|-- config/experiment.json   # Cấu hình thí nghiệm
|-- data/raw/                # Dữ liệu gốc
|-- src/steam_recsys/        # Xử lý dữ liệu và chạy thí nghiệm
|   `-- models/             # popularity.py, item_knn.py, bpr.py
|-- tests/                   # File chờ An viết kiểm thử
|-- reports/                 # Số liệu CSV, thống kê và biểu đồ
`-- artifacts/              # Mô hình đã huấn luyện
```

## Cài đặt và chạy

Python 3.12 trở lên. Chạy trong thư mục `TTCS` bằng PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py download
.\.venv\Scripts\python.exe main.py train
```

Đã có `.venv` và dữ liệu thì chạy trực tiếp lệnh `train`. Lệnh này huấn luyện mô hình, lưu `artifacts/bpr_model.npz`, thống kê dữ liệu và loss huấn luyện; chưa tính chỉ số đánh giá hay sinh khuyến nghị.

File BPR lưu cả tham số mô hình, ID người dùng, tên game và cấu hình BPR. Các hàng của `user_factors` tương ứng với `user_ids`; các hàng của `item_factors` và `item_bias` tương ứng với `item_titles`. Nạp bằng `numpy.load(..., allow_pickle=False)`. Ngưỡng `min_user_interactions` phải từ 3 trở lên để mỗi người dùng còn ít nhất một game train sau khi giữ lại validation/test.

Kiểm tra các lỗi hồi quy của phần huấn luyện bằng:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Phần chờ thực hiện

| File | Công việc của Lương Đức Khải An |
|---|---|
| `src/steam_recsys/recommend.py` | Khuyến nghị cho một người dùng |
| `src/steam_recsys/evaluation.py` | Tính chỉ số đánh giá |
| `src/steam_recsys/tuning.py` | So sánh cấu hình trên validation |
| `tests/test_data.py` | Kiểm thử xử lý dữ liệu |
| `tests/test_models.py` | Kiểm thử mô hình và pipeline |

Các file trên chỉ còn ghi chú TODO. Lệnh `tune` và `recommend` sẽ được nối lại khi có phần cài đặt.

Đã bỏ hàm vẽ biểu đồ chưa sử dụng và phụ thuộc Matplotlib. Khi triển khai đánh giá, phần báo cáo có thể được bổ sung riêng. Các chỉ số và biểu đồ so sánh hiện có trong `reports/` là kết quả cũ, không được lệnh `train` hiện tại cập nhật; lệnh này chỉ ghi lại thống kê dữ liệu và loss BPR.

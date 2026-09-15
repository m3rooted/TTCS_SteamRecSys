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
|-- src/steam_recsys/        # Xử lý dữ liệu, mô hình, đánh giá
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

## Phần chờ thực hiện

| File | Công việc của Lương Đức Khải An |
|---|---|
| `src/steam_recsys/recommend.py` | Khuyến nghị cho một người dùng |
| `src/steam_recsys/evaluation.py` | Tính chỉ số đánh giá |
| `src/steam_recsys/tuning.py` | So sánh cấu hình trên validation |
| `tests/test_data.py` | Kiểm thử xử lý dữ liệu |
| `tests/test_models.py` | Kiểm thử mô hình và pipeline |

Các file trên chỉ còn ghi chú TODO. Lệnh `tune` và `recommend` sẽ được nối lại khi có phần cài đặt.

Phần biểu đồ/báo cáo dùng chung vẫn được giữ trong `experiment.py`; hàm vẽ biểu đồ chưa được gọi khi chưa có đánh giá. Các số liệu và biểu đồ hiện có trong `reports/` là kết quả từ lần chạy trước khi xóa code, không được lệnh `train` hiện tại cập nhật.

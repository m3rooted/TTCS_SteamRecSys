# Hệ khuyến nghị trò chơi Steam

Dự án dùng [Steam Games Dataset của FronkonGames trên Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) để gợi ý game theo nội dung. Dataset này có một dòng cho mỗi game và không có lịch sử tương tác user-game, nên pipeline hiện tại biểu diễn game từ metadata rồi tìm các game tương tự.

## Dữ liệu

Tải CSV từ Kaggle và đặt tại `data/raw/games.csv` (hoặc truyền đường dẫn khác bằng `--config`). Bộ đọc nhận tên cột dạng Kaggle như `AppID`, `Release date`, `Estimated owners`, `About the game`, `Supported languages`, `Average playtime forever`, đồng thời nhận tên snake_case. Các trường bắt buộc:

`AppID`, `Name`, `Release date`, `Estimated owners`, `Peak CCU`, `Required age`, `Price`, `About the game`, `Supported languages`, `Reviews`, `Windows`, `Positive`, `Recommendations`, `Average playtime forever`, `Developers`, `Publishers`, `Categories`, `Genres`, `Tags`.

`AppID` là khóa game, `Name` để tra cứu và hiển thị. Các trường còn lại được dùng làm đặc trưng. Văn bản được mã hóa TF-IDF; ngày phát hành, số chủ sở hữu ước lượng, CCU, giá, số đánh giá tích cực, đề xuất, thời lượng chơi, độ tuổi và hỗ trợ Windows được chuẩn hóa thành đặc trưng số. Tên cột `Supported language` cũng được chấp nhận.

Steam-200k cũ vẫn nằm trong `data/raw/steam-200k.csv` nhưng không còn là đầu vào mặc định.

## Cài đặt và sử dụng

Python 3.12 trở lên:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tạo chỉ mục đặc trưng và thống kê catalog:

```powershell
.\.venv\Scripts\python.exe main.py train
```

Gợi ý 10 game tương tự theo AppID hoặc tên game:

```powershell
.\.venv\Scripts\python.exe main.py recommend --game "Portal 2" --top-k 10
```

Kết quả `train` lưu ma trận thưa tại `artifacts/game_features.npz`, danh mục AppID/tên tại `artifacts/game_index.csv` và thống kê tại `reports/dataset_stats.json`. Lệnh `recommend` nhận game mẫu và dựng đặc trưng từ CSV cấu hình hiện tại.

Có thể đổi đường dẫn CSV trong `config/experiment.json`, hoặc tạo config khác rồi truyền `--config path/to/config.json`.

## Cấu trúc

```text
main.py                         # Lệnh train và recommend
config/experiment.json          # Cấu hình đường dẫn dataset
data/raw/                       # Dataset tải về
src/steam_recsys/content_based.py # Đọc catalog, tạo đặc trưng, gợi ý
artifacts/                      # Ma trận đặc trưng và chỉ mục game
reports/                        # Thống kê catalog
```

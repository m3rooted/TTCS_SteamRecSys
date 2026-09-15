# Cơ sở lý thuyết hệ khuyến nghị trò chơi Steam

Tài liệu phục vụ đề tài tìm hiểu, cài đặt và đánh giá hệ khuyến nghị game. Nội dung đi từ bài toán và dữ liệu đến thuật toán, thiết kế thực nghiệm và cách đọc kết quả.

**Trạng thái mã nguồn:** phần khuyến nghị, đánh giá, so sánh cấu hình và kiểm thử đã được để trống cho Lương Đức Khải An thực hiện. Các mô tả quy trình dưới đây là thiết kế tham khảo; bảng kết quả ở mục 7 thuộc lần chạy trước khi xóa code.

## 1. Hệ khuyến nghị và bài toán đề tài

Hệ khuyến nghị (recommender system) lựa chọn các đối tượng có khả năng phù hợp với một người dùng. Trong đề tài này, đối tượng là trò chơi trên Steam và đầu ra là danh sách K game chưa có trong lịch sử tương tác, được sắp theo mức độ phù hợp dự đoán.

Ví dụ: người dùng đã chơi một số game chiến thuật; hệ thống tìm các game có liên hệ với lịch sử đó và xếp hạng chúng. Liên hệ có thể đến từ đặc điểm game hoặc hành vi của cộng đồng người chơi.

Các ký hiệu dùng trong tài liệu:

| Ký hiệu | Ý nghĩa |
|---|---|
| U, I | Tập người dùng và tập trò chơi |
| u, i | Một người dùng và một trò chơi |
| I_u | Các game người dùng u đã tương tác trong train |
| R | Ma trận tương tác: hàng là người dùng, cột là game |
| s(u, i) | Điểm mô hình dự đoán cho cặp người dùng và game |
| L_u | Danh sách game được khuyến nghị cho u |
| K | Số game được lấy ở đầu danh sách, chẳng hạn 10 |

Hai bài toán thường gặp là dự đoán điểm đánh giá và xếp hạng top-K. Đề tài chọn xếp hạng vì Steam-200k ghi hành vi mua/chơi, không cung cấp điểm yêu thích để dự đoán trực tiếp.

## 2. Dữ liệu phản hồi và đặc điểm Steam

### 2.1. Phản hồi tường minh và phản hồi ẩn

**Phản hồi tường minh** là đánh giá trực tiếp như số sao hoặc thích/không thích. **Phản hồi ẩn** được suy ra từ hành vi như mua hàng, nhấp chuột và chơi game. Các cách biểu diễn này được giới thiệu trong [tài liệu collaborative filtering của Google](https://developers.google.com/machine-learning/recommendation/collaborative/basics).

Với Steam, mua hoặc chơi game là tín hiệu quan tâm, nhưng chưa chứng minh người dùng thích game. Chơi lâu có thể do game có thời lượng dài hoặc đang chạy nền. Không có tương tác cũng không có nghĩa là không thích: người dùng có thể chưa biết đến game đó.

### 2.2. Bộ Steam-200k trong dự án

File [steam-200k.csv](data/raw/steam-200k.csv) gồm 200.000 dòng hành vi, không phải 200.000 người dùng. Thống kê trên file đã tải: 12.393 người dùng, 5.155 game, 129.511 dòng `purchase` và 70.489 dòng `play`.

| Trường | Ý nghĩa |
|---|---|
| user_id | Mã người dùng |
| game | Tên trò chơi |
| behavior | `purchase` hoặc `play` |
| value | Bằng 1 khi mua; là số giờ khi chơi |
| Cột cuối | Cột phụ không dùng trong mô hình |

Một cặp user-game có thể xuất hiện ở cả hai loại hành vi. Cần gộp cặp này trước khi chia dữ liệu, tránh để dòng mua ở train và dòng chơi cùng game ở test. Giá trị 1 của dòng mua không được cộng thành một giờ chơi.

Bộ dữ liệu này không có timestamp, thể loại, mô tả hay Steam app ID. Khi cần content-based, phải bổ sung metadata và kiểm tra việc ghép tên game; không tự suy diễn thể loại từ tên.

### 2.3. Ma trận thưa và trọng số tương tác

Ma trận nhị phân R có giá trị 1 tại cặp đã tương tác; các ô còn lại biểu thị chưa quan sát. Do mỗi người chỉ tiếp xúc với một phần nhỏ catalog, ma trận thường rất thưa. Lưu bằng dạng sparse như CSR giúp giảm bộ nhớ so với ma trận đặc.

Dự án gán trọng số cho tương tác đã quan sát bằng:

$$w_{ui} = 1 + \ln(1 + h_{ui})$$

Trong đó h là số giờ chơi. Phép log làm giảm chênh lệch giữa người chơi rất lâu và người chơi ít. Cặp chỉ có hành vi mua nhận trọng số 1. Đây là trọng số do người thiết kế lựa chọn, không phải điểm đánh giá hay xác suất yêu thích.

## 3. Các phương pháp khuyến nghị cơ bản

### 3.1. Khuyến nghị theo độ phổ biến: Popularity

Gọi n_i là số người dùng đã tương tác với game i trong train:

$$s(i) = \ln(1 + n_i)$$

Logarit không thay đổi thứ tự theo n_i. Mô hình xếp các game nhiều người tương tác lên trước, rồi loại game đã biết của từng người dùng. Đây là baseline để kiểm tra xem cá nhân hóa có mang lại cải thiện hay không.

Ưu điểm là nhanh và có thể dùng cho người dùng mới. Hạn chế là không học sở thích riêng và dễ tập trung vào một số game nổi tiếng.

### 3.2. Lọc dựa trên nội dung: Content-based

Mỗi game được biểu diễn bằng vector đặc trưng như thể loại, thẻ hoặc mô tả. Hồ sơ người dùng được tổng hợp từ các game họ đã quan tâm; game mới được chấm điểm theo độ giống với hồ sơ đó. Cách tiếp cận này được mô tả trong [tài liệu content-based của Google](https://developers.google.com/machine-learning/recommendation/content-based/basics).

Có thể mã hóa thể loại bằng vector nhị phân nhiều nhãn và mô tả bằng TF-IDF. TF-IDF tăng trọng số của từ đặc trưng trong một tài liệu, đồng thời giảm ảnh hưởng của từ xuất hiện ở nhiều tài liệu.

Độ tương đồng cosine giữa hai vector a và b khác vector không:

$$\operatorname{cosine}(a,b)=\frac{a^Tb}{\|a\|_2\|b\|_2}$$

Content-based có thể khuyến nghị game mới có metadata, nhưng dễ lặp lại các nội dung giống lịch sử. Người dùng hoàn toàn mới vẫn cần cung cấp sở thích ban đầu. Phần này là kiến thức liên quan; dự án hiện chưa cài đặt vì Steam-200k thiếu metadata.

### 3.3. Lọc cộng tác: Collaborative filtering

Lọc cộng tác học từ quan hệ giữa người dùng và game trong lịch sử tương tác, không bắt buộc biết thể loại hay nội dung game.

| Hướng tiếp cận | Ý tưởng |
|---|---|
| User-based kNN | Tìm người có lịch sử tương tự rồi đề xuất game họ đã tương tác |
| Item-based kNN | Tìm game thường được cùng một nhóm người tương tác |

Trong Item-kNN, cột R_i biểu diễn tập người dùng của game i. Tính cosine giữa các cột để xác định độ giống nhau. Điểm khuyến nghị trong mã nguồn là:

$$s(u,i)=\sum_{j\in I_u} w_{uj}\,\widetilde{S}_{ji}$$

S là ma trận cosine; mỗi hàng chỉ giữ các hàng xóm gần nhất, các phần tử khác bằng 0. Vì việc giữ hàng xóm thực hiện theo từng hàng, ma trận sau cắt có thể không còn đối xứng. Cấu hình hiện tại giữ 80 hàng xóm cho mỗi game.

Item-kNN dễ giải thích qua các game đã biết đóng góp vào điểm của ứng viên. Tuy nhiên, nó gặp khó khi người dùng hoặc game chưa có tương tác. Độ giống tính từ rất ít người cũng có thể thiếu ổn định.

## 4. Phương pháp nâng cao

### 4.1. Phân rã ma trận: Matrix factorization

Mỗi người dùng được biểu diễn bằng vector p_u và mỗi game bằng vector q_i trong không gian d chiều. Điểm phù hợp có thể viết:

$$s(u,i)=p_u^Tq_i+b_i$$

b_i là bias của game. Các vector được học từ dữ liệu và thường gọi là embedding hoặc nhân tố ẩn. Các chiều ẩn không tự động mang ý nghĩa rõ ràng như “hành động” hay “chiến thuật”.

Số chiều d quyết định sức biểu diễn và số tham số. Quá ít chiều có thể không học đủ quan hệ; quá nhiều chiều có thể học cả nhiễu. Phân rã ma trận là họ mô hình; cách xác định hàm mất mát và cách tối ưu là những lựa chọn riêng.

### 4.2. Bayesian Personalized Ranking: BPR-MF

BPR học quan hệ ưu tiên giữa hai game của cùng người dùng. Với bộ ba (u, i, j), i đã được tương tác trong train, còn j được lấy mẫu từ các game chưa có trong train của u. Mục tiêu là làm điểm của i cao hơn j. Xem [bài báo BPR của Rendle và cộng sự, UAI 2009](https://arxiv.org/abs/1205.2618).

Đặt chênh lệch điểm x = s(u,i) - s(u,j) và hàm sigmoid:

$$\sigma(x)=\frac{1}{1+e^{-x}}$$

Một dạng hàm mất mát BPR có điều chuẩn là:

$$\mathcal{L}=-\sum_{(u,i,j)}\ln\sigma(s(u,i)-s(u,j))+\lambda\|\Theta\|_2^2$$

Theta gồm các tham số cần học; lambda kiểm soát mức phạt tham số lớn. Gradient descent cập nhật tham số để giảm loss. BPR tối ưu một mục tiêu xếp hạng theo cặp, không trực tiếp tối ưu NDCG@K.

Trong dự án, mô hình dùng mini-batch, giảm learning rate qua epoch và lấy negative chỉ dựa trên train. Negative có thể là game người dùng thực sự thích nhưng chưa quan sát; đó là giới hạn của phản hồi ẩn.

Các tham số cần hiểu:

| Tham số | Vai trò |
|---|---|
| factors | Số chiều của vector người dùng và game |
| epochs | Số lượt học qua tập tương tác |
| batch_size | Số bộ ba xử lý trong một lần cập nhật |
| learning_rate | Độ lớn bước cập nhật |
| regularization | Mức điều chuẩn để hạn chế quá khớp |
| seed | Cố định khởi tạo và lấy mẫu để hỗ trợ tái lập |

### 4.3. Các hướng mở rộng

Hệ lai (hybrid) kết hợp thông tin nội dung và lịch sử tương tác. Một cách đơn giản là cộng có trọng số hai loại điểm sau khi chuẩn hóa chúng về thang đo phù hợp, rồi chọn trọng số trên validation.

Các hướng nâng cao khác gồm ALS cho phản hồi ẩn, mô hình mạng nơ-ron và mô hình học chuỗi hành vi. Chúng cần giao thức đánh giá phù hợp; mô hình chuỗi còn cần thứ tự hoặc thời gian tương tác. Đề tài hiện chọn BPR-MF làm phương pháp nâng cao để giữ phạm vi gọn.

## 5. Tiền xử lý và thiết kế thực nghiệm

Quy trình hiện tại:

1. Gộp các dòng theo cặp người dùng và game; tách giờ chơi khỏi hành vi mua.
2. Lọc lặp theo ngưỡng: tối thiểu 5 game/người dùng và 3 người dùng/game.
3. Mã hóa người dùng, game thành chỉ số và tạo ma trận sparse.
4. Giữ ngẫu nhiên một game validation và một game test cho mỗi người dùng; phần còn lại dùng train.
5. Huấn luyện trên train; so sánh cấu hình BPR theo NDCG trên validation.
6. Chạy cấu hình được chọn trên test, lưu kết quả và sinh danh sách khuyến nghị.

Bước 2 thường gọi là lọc k-core. Phải lặp vì xóa người dùng có thể làm một game rơi dưới ngưỡng, và ngược lại. Các ngưỡng áp dụng trước khi chia dữ liệu, không bảo đảm số tương tác tối thiểu đó còn nguyên trong train. Mã nguồn bỏ khỏi đánh giá các positive validation/test không có game tương ứng trong train.

**Train** dùng học tham số; **validation** dùng chọn cấu hình; **test** dùng báo cáo chất lượng sau lựa chọn. Việc điều chỉnh nhiều lần dựa vào test có thể làm kết quả quá lạc quan. Loss train giảm cũng không bảo đảm chất lượng khuyến nghị tăng.

Dự án dùng phép chia ngẫu nhiên vì không có timestamp. Nếu có thời gian, nên kiểm tra thêm temporal split: dùng quá khứ để dự đoán tương lai. Lọc k-core trước chia tập và xác định catalog trên toàn dữ liệu là thiết lập có biết trước tập đối tượng; kết quả chủ yếu phản ánh người dùng/game đã có lịch sử, không đại diện cho cold-start.

## 6. Đánh giá hệ khuyến nghị

Gọi T_u là tập game liên quan trong test, L_u là danh sách K game khuyến nghị. Các chỉ số chất lượng được tính cho từng người dùng rồi lấy trung bình. [Tổng quan đánh giá hệ khuyến nghị của Microsoft Research](https://www.microsoft.com/en-us/research/publication/evaluating-recommender-systems/) bàn về việc lựa chọn nhiều tiêu chí phù hợp với ứng dụng.

### 6.1. Độ chính xác và thứ hạng

$$Precision@K=\frac{|L_u\cap T_u|}{K},\qquad Recall@K=\frac{|L_u\cap T_u|}{|T_u|}$$

Precision cho biết tỷ lệ game đúng trong danh sách; Recall cho biết tỷ lệ game đúng được tìm thấy trong số cần tìm. HitRate bằng 1 nếu danh sách có ít nhất một game đúng, bằng 0 nếu không có.

Với mức liên quan nhị phân rel_r tại vị trí r:

$$DCG@K=\sum_{r=1}^{K}\frac{rel_r}{\log_2(r+1)},\qquad NDCG@K=\frac{DCG@K}{IDCG@K}$$

IDCG là DCG của thứ tự lý tưởng. NDCG thưởng nhiều hơn khi game đúng xuất hiện gần đầu danh sách.

Ví dụ: có một game test và game đó đứng thứ 3 trong top-10. Precision@10 = 0,1; Recall@10 = HitRate@10 = 1; NDCG@10 = 1/log2(4) = 0,5. Do mỗi người dùng trong dự án chỉ có một positive test, Precision@10 không vượt 0,1 và Recall luôn bằng HitRate.

### 6.2. Độ phủ và mức độ mới lạ

**Coverage** là tỷ lệ game khác nhau xuất hiện trong các danh sách khuyến nghị trên tổng số game của catalog:

$$Coverage@K=\frac{|\bigcup_{u\in U}L_u|}{|I|}$$

**Novelty** trong dự án là trung bình của -log2(p_i) trên các game được đề xuất, với p_i = (n_i + 1)/(tổng n_j + số game). Game ít phổ biến đóng góp giá trị lớn hơn. Chỉ số này đo mới lạ theo độ phổ biến, không chứng minh game thực sự mới đối với người dùng.

Coverage cao không đồng nghĩa danh sách của từng người đa dạng về thể loại. Để đo tính đa dạng trong một danh sách cần thêm thông tin hoặc độ tương đồng phù hợp giữa các game.

### 6.3. Tập ứng viên và giới hạn của số đo

Dự án tính Precision, Recall, HitRate và NDCG bằng cách xếp một positive cùng 100 game chưa quan sát được lấy mẫu. Các mô hình nhận cùng tập ứng viên. Coverage và Novelty được tính riêng trên top-10 của toàn catalog sau khi loại lịch sử người dùng.

Chỉ số trên ứng viên lấy mẫu không tương đương xếp hạng toàn catalog, thậm chí thứ tự tốt/xấu giữa các mô hình có thể thay đổi. Kết quả chỉ được so sánh trong cùng giao thức, như phân tích trong [nghiên cứu về đánh giá dưới lấy mẫu của Rendle](https://arxiv.org/abs/1912.02263).

## 7. Đối chiếu với phần cài đặt

| Nội dung | Tình trạng trong dự án |
|---|---|
| Popularity | Đã cài đặt; đếm tương tác nhị phân trong train |
| Item-kNN | Đã cài đặt; cosine nhị phân, hồ sơ người dùng có trọng số giờ chơi |
| BPR-MF | Đã cài đặt; học từ cặp tương tác nhị phân, không dùng giờ chơi làm trọng số loss |
| Content-based, User-kNN, hybrid | Trình bày lý thuyết, chưa cài đặt |
| Tuning BPR | Chờ An thực hiện; `reports/bpr_validation_search.csv` là kết quả cũ |
| Đánh giá và biểu đồ | Phần đánh giá chờ An thực hiện; `reports/metrics.csv` và `reports/model_comparison.png` là kết quả cũ |

Sau lọc, bộ thực nghiệm có 3.744 người dùng, 3.338 game và 112.694 cặp tương tác; mật độ khoảng 0,902%. Kết quả đã chạy với seed 42:

| Mô hình | Recall@10 lấy mẫu | NDCG@10 lấy mẫu | Coverage toàn catalog |
|---|---:|---:|---:|
| Popularity | 0,7118 | 0,4934 | 0,0171 |
| Item-kNN | 0,8069 | 0,6262 | 0,3933 |
| BPR-MF | 0,7746 | 0,5428 | 0,0953 |

Item-kNN tốt nhất trong thí nghiệm này. Các số liệu chưa đủ chứng minh nguyên nhân hoặc kết luận thuật toán luôn tốt hơn BPR. Cần thêm nhiều seed, khoảng tin cậy và đánh giá toàn catalog nếu muốn đưa ra kết luận rộng hơn. Khi thay dữ liệu/cấu hình, xem file kết quả mới thay vì coi bảng tĩnh này là số liệu cập nhật tự động.

## 8. Những điểm cần nắm khi trình bày đề tài

- Vì sao dùng ranking: đầu ra là danh sách game, dữ liệu không có điểm đánh giá trực tiếp.
- Vì sao cần baseline: phải kiểm chứng giá trị của cá nhân hóa so với game phổ biến.
- Item-kNN và content-based khác nhau ở nguồn tạo độ tương đồng: lịch sử cộng đồng và đặc điểm game.
- Vì sao BPR phù hợp: học thứ tự ưu tiên theo cặp từ phản hồi ẩn; negative vẫn là giả định lấy mẫu.
- Vì sao mô hình nâng cao có thể thua: kết quả phụ thuộc dữ liệu, cấu hình và giao thức đánh giá.
- Hạn chế chính: thiếu timestamp/metadata, lọc người ít tương tác và chưa kiểm chứng trên người dùng thật.

## Tài liệu đọc thêm

Các nguồn chuyên biệt được dẫn ngay tại nội dung tương ứng. Nguồn dữ liệu gốc là [Steam Video Games trên Kaggle](https://www.kaggle.com/datasets/tamber/steam-video-games); dự án tải [bản sao công khai trên GitHub](https://github.com/Michalos88/Game-Recommendation-System).

Hai kho người dùng cung cấp phù hợp để tra cứu kiến thức nền và tài liệu bổ sung: [Awesome Machine Learning Resources](https://github.com/ZhiningLiu1998/awesome-machine-learning-resources) và [Machine Learning Cheat Sheet](https://github.com/soulmachine/machine-learning-cheat-sheet).

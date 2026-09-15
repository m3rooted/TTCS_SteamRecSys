# Cơ sở lý thuyết hệ khuyến nghị game

## 1. Hệ khuyến nghị

Hệ khuyến nghị là hệ thống dự đoán những sản phẩm hoặc nội dung mà người dùng có khả năng quan tâm. Với Steam, sản phẩm được khuyến nghị là game và mục tiêu chính là sắp xếp các game phù hợp lên đầu danh sách.

Có hai dạng bài toán phổ biến:

- **Dự đoán đánh giá:** dự đoán điểm số người dùng dành cho một game.
- **Khuyến nghị xếp hạng:** sắp xếp các game theo mức độ phù hợp.

Đề tài này tập trung vào khuyến nghị xếp hạng vì hệ thống thực tế thường cần trả về một danh sách `Top-K`, thay vì dự đoán chính xác số giờ chơi.

## 2. Phản hồi tường minh và phản hồi ẩn

Phản hồi của người dùng có thể là:

- **Tường minh:** điểm sao, lượt thích hoặc đánh giá trực tiếp.
- **Ẩn:** hành vi mua, chơi hoặc thời gian chơi.

Dữ liệu Steam-200k chủ yếu cung cấp phản hồi ẩn. Việc người dùng mua hoặc chơi một game được xem là tín hiệu tích cực. Việc không thấy người dùng chơi một game không chắc chắn có nghĩa là họ không thích game đó; họ có thể chưa biết hoặc chưa từng được giới thiệu.

Do đó, bài toán được mô hình hóa như sau:

- Tập người dùng: $U$.
- Tập game: $I$.
- Tập tương tác quan sát được: $R^+ \subseteq U \times I$.
- Với mỗi người dùng, mô hình cần xếp hạng các game chưa từng tương tác.

Thời gian chơi có thể được dùng làm độ mạnh của tương tác. Trong dự án, giá trị này được biến đổi theo công thức:

$$
s(u,i) = 1 + \log(1 + \text{hours}(u,i))
$$

Phép biến đổi log giúp giảm ảnh hưởng của một số giá trị thời gian chơi quá lớn.

## 3. Lọc dữ liệu bằng k-core

K-core là kỹ thuật loại bỏ những người dùng hoặc game có quá ít tương tác. Quy trình lặp lại hai bước:

1. Giữ người dùng có ít nhất một số lượng tương tác tối thiểu.
2. Giữ game có ít nhất một số lượng người dùng tối thiểu.

Các bước được lặp lại đến khi dữ liệu ổn định. Cách này làm giảm dữ liệu quá thưa và giúp mỗi người dùng có đủ tương tác để chia thành tập huấn luyện, validation và test.

## 4. Các mô hình khuyến nghị

### 4.1. Popularity

Popularity xếp hạng game theo số lượng người dùng đã tương tác với game đó. Đây là mô hình nền, không cá nhân hóa.

Điểm của game $i$ có thể viết là:

$$
score(i) = \log(1 + count(i))
$$

Ưu điểm của mô hình là đơn giản, nhanh và có thể dùng khi chưa có nhiều thông tin về người dùng. Nhược điểm là mọi người dùng nhận gần như cùng một danh sách.

### 4.2. Item-kNN

Item-kNN là phương pháp lọc cộng tác dựa trên sự tương tự giữa các game. Hai game được xem là tương tự nếu có nhiều người dùng cùng tương tác với cả hai game.

Độ tương tự cosine giữa hai game $i$ và $j$ là:

$$
sim(i,j) = \frac{v_i \cdot v_j}{\lVert v_i \rVert \lVert v_j \rVert}
$$

Trong đó $v_i$ là vector người dùng đã tương tác với game $i$. Điểm của game ứng viên được tính từ các game người dùng đã tương tác và độ tương tự của chúng với game ứng viên:

$$
score(u,j) = \sum_{i \in I_u} sim(i,j)
$$

Item-kNN dễ giải thích vì có thể nói game được đề xuất dựa trên những game tương tự mà người dùng đã chơi. Tuy nhiên, mô hình bị ảnh hưởng bởi dữ liệu thưa và thường khó xử lý game mới.

### 4.3. BPR-MF

BPR-MF kết hợp Bayesian Personalized Ranking và Matrix Factorization. Mỗi người dùng và mỗi game được biểu diễn bằng một vector ẩn:

$$
p(u,i) = q_i^T x_u + b_i
$$

Trong đó:

- $x_u$ là vector của người dùng $u$.
- $q_i$ là vector của game $i$.
- $b_i$ là độ lệch của game.

BPR không học điểm tuyệt đối. Với một bộ ba $(u, i, j)$, trong đó $i$ là game dương tính và $j$ là game âm tính, mô hình học để:

$$
p(u,i) > p(u,j)
$$

Hàm mục tiêu thường được viết:

$$
L = -\sum_{(u,i,j)} \log \sigma\left(p(u,i) - p(u,j)\right) + \lambda \lVert \Theta \rVert^2
$$

Trong đó $\sigma$ là hàm sigmoid, còn $\lambda$ là hệ số điều chuẩn. BPR-MF có khả năng cá nhân hóa tốt hơn Popularity, nhưng cần lựa chọn siêu tham số và thời gian huấn luyện.

## 5. Chia dữ liệu và lấy mẫu âm

Với mỗi người dùng, một số tương tác được giữ lại làm validation và test; các tương tác còn lại dùng để huấn luyện. Cách chia `leave-two-out` phù hợp khi cần đánh giá khả năng xếp hạng một game chưa thấy trong quá trình huấn luyện.

Trong phản hồi ẩn, số lượng game không tương tác thường rất lớn. Vì vậy, khi đánh giá có thể lấy một game dương tính cùng một số game âm tính chưa tương tác. Dự án sử dụng 100 game âm tính cho mỗi game dương tính.

Kết quả trên tập mẫu âm phản ánh khả năng xếp hạng trong tập ứng viên đó, không hoàn toàn tương đương với việc xếp hạng trên toàn bộ kho game.

## 6. Các chỉ số đánh giá

### Recall@K

Recall@K đo tỷ lệ người dùng có game đúng xuất hiện trong $K$ vị trí đầu:

$$
Recall@K = \frac{\text{số người dùng có game đúng trong Top-K}}{\text{tổng số người dùng đánh giá}}
$$

Recall@K càng cao thì mô hình càng thường xuyên đưa được game mục tiêu vào danh sách đề xuất.

### Precision@K

Precision@K đo tỷ lệ vị trí trong Top-K chứa kết quả đúng. Với một game mục tiêu cho mỗi người dùng:

$$
Precision@K = \frac{\text{số kết quả đúng}}{K \times \text{số người dùng}}
$$

### NDCG@K

NDCG chú trọng vị trí của kết quả đúng. Game đúng ở vị trí càng cao thì điểm càng lớn:

$$
DCG@K = \frac{1}{\log_2(rank + 1)}
$$

Sau đó, DCG được chuẩn hóa theo điểm lý tưởng để thu được NDCG. Chỉ số nằm trong khoảng từ 0 đến 1; càng gần 1 càng tốt.

### Coverage

Coverage là tỷ lệ số game khác nhau từng xuất hiện trong các danh sách đề xuất trên tổng số game trong catalog:

$$
Coverage = \frac{\text{số game được đề xuất ít nhất một lần}}{\text{tổng số game}}
$$

Coverage cao cho thấy mô hình ít tập trung vào một nhóm game phổ biến, nhưng chưa chắc đồng nghĩa với độ chính xác cao hơn.

### Novelty

Novelty đo mức độ ít phổ biến của các game được đề xuất. Các game ít phổ biến thường có độ mới cao hơn đối với người dùng. Chỉ số này nên được xem cùng Recall và NDCG để cân bằng giữa độ chính xác và sự đa dạng.

## 7. Quy trình thực nghiệm

Quy trình tổng quát của đề tài gồm:

1. Đọc và làm sạch dữ liệu Steam-200k.
2. Gộp các dòng mua và chơi thành cặp người dùng-game.
3. Lọc dữ liệu bằng k-core.
4. Chia dữ liệu thành train, validation và test.
5. Huấn luyện Popularity, Item-kNN và BPR-MF.
6. Chọn siêu tham số trên validation.
7. Đánh giá cuối cùng trên test bằng các chỉ số xếp hạng.
8. So sánh chất lượng, độ bao phủ và độ mới của các mô hình.

## 8. Hạn chế lý thuyết và thực nghiệm

- Dữ liệu không có đánh giá sao nên không đo được mức độ yêu thích trực tiếp.
- Không có timestamp nên chưa đánh giá được sự thay đổi sở thích theo thời gian.
- Không có mô tả, thể loại hoặc thông tin nội dung của game nên chưa xử lý tốt cold-start cho game mới.
- Phản hồi ẩn có thể chứa nhiều nhiễu: mua nhưng không chơi hoặc chơi rất ít.
- Đánh giá bằng mẫu âm có thể khác với xếp hạng trên toàn bộ catalog.

Vì vậy, các chỉ số cần được diễn giải trong đúng giao thức thực nghiệm, đồng thời nên xem xét cả độ chính xác, độ bao phủ và độ mới thay vì chỉ dựa vào một chỉ số duy nhất.

## 9. Kết luận

Hệ khuyến nghị game trong đề tài là một bài toán lọc cộng tác với phản hồi ẩn. Popularity cung cấp đường cơ sở, Item-kNN khai thác sự tương tự giữa các game, còn BPR-MF học biểu diễn ẩn để tối ưu trực tiếp thứ tự ưu tiên. Việc dùng các chỉ số ranking như Recall@K và NDCG@K phù hợp với mục tiêu tạo danh sách Top-K cho người dùng Steam.

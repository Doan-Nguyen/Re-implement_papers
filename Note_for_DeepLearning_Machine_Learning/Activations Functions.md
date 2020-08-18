Chào, mình là Đoàn. Nếu như bạn nào tìm hiểu về DL, ML đều biết đến *activation functions*. Nó là 1 thành phần nhỏ (về kích thước) nhưng đảm đương trọng trách khá lớn trong mỗi mạng Neural. Cùng tìm hiểu với mình.

# 1. Activation Functions - Overview
+ Xét phương thức não bộ hoạt động nhé. Giả sử khi bạn chạm vào vật nóng, các dây thần kinh xúc giác phản hồi thông tin đến não bộ của bạn (coi là tín hiệu *x*), trong cơ thể chúng ta cùng 1 thời điểm có vô vàn tín hiệu trả về vậy não bộ sẽ xử thông tin nào trước ? chúng ta sẽ thụt tay lại hay chú ý tới màn hình máy tính ?. Từ nhu cầu phân loại mức độ ưu tiên chúng ta có tham số *weight* trong mạng. Ngoài ra tín hiệu còn được nhận thêm 1 tham số *bias*.

![Công thức chuẩn của Neural Network] LSTM](https://github.com/Doan-Nguyen/Reviewer---Writer/blob/master/Note%20for%20DeepLearning-Machine%20Learning/images/congthuc1.png)

+ Activation Functions đóng vai trò ở đâu ? nó sẽ quyết định độ ưu tiên của tín hiệu bỏng tay đó. Khi bạn bị nóng (*x signal*) xuất hiện nhưng khi đó bạn đang cầm bát canh cá vừa nấu *activation func* chi phối việc bạn sẽ không bỏ bát canh mà chịu nóng (trong các bài viết họ thường dùng từ *"fired"*)

## 1.1 Step function 
+ Có thể hiểu đây là ngưỡng cho Activation Functions được hoạt động (nhỏ hơn thì không được thực hiện).

# Very Deep Convolutional Networks For Large-Scale Image Recognition


## Ý tưởng bài báo 
+ Bài báo đề cập cải thiện mạng CNN bằng cách tăng độ sâu cho kiến trúc mạng. Ở đây, việc phát triển mạng bằng cách tăng độ sâu các tầng nhờ sử dụng kiến trúc các lớp *convolution filters có kích thước nhỏ* (3x3) & (1x1) thay cho (5x5) đã xuất hiện trong mạng **LeNet-5**.
+ Đây cũng đánh dấu tư tưởng thiết kế kiến trúc mạng theo khối blocks thay cho kiểu kiến trúc phân tầng trong **AlexNet**

## Thiết lập ConvNet
### 2.1 Kiến trúc mạng 
+ Khối VGG:
    - Khối cơ bản trong mạng tích chập cổ điển: [conv -> ReLU -> MaxPooling]
    - Khối đề xuất trong VGG:  
        - Chuỗi tầng tích chập 
        - MaxPooling (giảm chiều không gian: rộng & dài) nhưng vẫn giữ đặc trưng chung.
        -> Các CNN sâu hơn giúp mô hình học được nhiều đặc trưng hơn.
    - Sử dụng Fully Connected Layers ở những tầng cuối đề phân loại.
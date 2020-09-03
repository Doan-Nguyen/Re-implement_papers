# ImageNet Classification with Deep Convolutional Neural Networks - AlexNet 


## 0. Giải thích các khái niệm 

Phần này chúng ta sẽ đi làm rõ các kiến thức/nội dung nhỏ đề cập đến trong paper. Deeper !

### 0.1 Non-Saturating & Saturating neurons

+ Hầu hết các kiến trúc mạng neuron trong machine learning sử dụng hàm kích hoạt phi tuyến tính cho việc kết hợp các đầu vào tuyến tính. Công thức phổ biến mọi người biết đến: 

    <img src="https://render.githubusercontent.com/render/math?math=g(b_j + sum_{i}(w_i,_jx_j))">

![Activation functions](figures/Activation_functions.png)

+ Hiện tượng bão hòa xảy khi các hàm kích hoạt đạt giá trị cận biên trên/biên dưới ứng với mỗi vùng hoạt động của hàm kích hoạt. Ví dụ: tanh() có giá trị trong khoảng [-1, 1]; softmax() có gía trị trong khoảng [0, 1]. 
+ Hiện tượng bão hòa này gây ra tốn thời gian cập nhật trọng số  (update the weights) vì giá trị của gradient rất nhỏ.
+ Cải tiến lớn nhất của ReLU là khắc phục hiện tường bão hòa của gradient


### 0.2 Overfitting, Drop-out 

### 0.3 Highly-optimized GPU implementation 4 (section 3)

### 0.4 Several effective techniques for preventing overfitting (section 4)

### 0.5 Unusual features


## 1. Tóm tắt nội dung chính 

### 1.1 Tư tưởng chính 

+ Nhằm giảm thời gian training, bài báo đề xuất sử dụng non-saturating neurons.

+ Hạn chế hiện tượng over-fitting tại lớp fully-connected, tác giả sử dụng phương pháp regularization **dropout**. 

+ Cấu trúc mạng AlexNet khá tương đồng với LeNet:
    - Kích thước của convolution layers giảm dần. 
    - AlexNet có độ sâu lớn hơn LeNet
        - Với LeNet: [conv(5x5) -> AvgPool(2x2)] -> [conv(5x5) -> AvgPool(2x2)]
        - AlexNet: [conv(11x11, padding:4) -> MaxPool(3x3, padding:2)]

![LeNet vs AlexNet](figures/lenet_alexnet.png)

### 1.2 Các hướng giải quyết 

### 1.3 Cách thuật toán đề xuất 

### 1.4 Đóng góp chính của paper 

### 1.5 Kiến trúc mạng 

+  

### 1.6 Fine-turning model 


## 2. Re-implement

### 2.1 Dataloader 

### 2.2 Re-build & debug model

### 2.3 Training model 

### 2.4 Measure accuracy 

### 2.5 Fine-turning model 

### 2.6 Source code 


## Tham khảo

1. Saturated neural

+ [Why would a saturated neuron be a problem?](https://www.quora.com/Why-would-a-saturated-neuron-be-a-problem)

+ [What Is Saturating Gradient Problem](https://datascience.stackexchange.com/questions/27665/what-is-saturating-gradient-problem#:~:text=Saturating%20means%20that%20after%20some,a%20solution%20for%20this%20problem.)
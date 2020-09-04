# TernausNet: U-Net with VGG Encoder Pre-Trained on ImageNet for ImageSegmentation


## I.Các kiến thức cần làm rõ 

###  Các thành phần trong mạng U-Net 
+ En-coder & de-coder

### Ảnh hưởng của việc sử dụng Initialize Weights

### Kiến trúc VGG 

### Skip connections 


## II. Explain paper 

### Abstract

+ Mạng U-Net truyền thống được cấu tạo bởi 2 thành phần:
    - Encoders 
    - Decoders 
+ Mạng neural network sử dụng trọng số từ các pre-trained model đã được học từ các tập dữ liệu lớn như ImageNet sẽ đem lại kết kết quả khả tốt hơn việc cho model học từ đầu trên các tập dữ liệu nhỏ. 

### Introduction

+ Bài toán image segmentation thực hiện việc phân loại các vùng ảnh ở cấp độ pixel. Bài toán này thời gian gần đây được đưa vào xử lý ảnh trong y tế, nên đòi hỏi độ chính xác rất cao. 

+ Phương pháp được đề xuất gần đây là sử dụng 1 mạng CNN có thể thực hiện việc segmentation vùng ảnh từ ảnh đầu vào. 
    - Fully Convolutional Networks (FCN).
        - Thay thế lớp fully connected layers bằng lớp conv(1x1), chuyển từ đầu ra là classification score sang spatial [feature maps](http://kaiminghe.com/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf).
            - Feature maps = features (shaped) + locations 

+ U-Net sử dụng [*skip connections*](https://theaisummer.com/skip-connections/).

### Network Architecture 

+ U-Net gồm 2 phần:
    - Phần kết nối 
    - Phần mở rộng 


## Tài liệu tham khảo

1. [Image Segmentation](https://phamdinhkhanh.github.io/)
2. [What is skip architecture in CNN?](https://www.quora.com/What-is-skip-architecture-in-CNN)
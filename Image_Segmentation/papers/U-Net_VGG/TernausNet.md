# TernausNet: U-Net with VGG Encoder Pre-Trained on ImageNet for ImageSegmentation


## Các kiến thức cần làm rõ 

### En-coder & de-coder trong mạng U-Net 

### Ảnh hưởng của việc sử dụng Initialize Weights


## Abstract

+ Mạng U-Net truyền thống được cấu tạo bởi 2 thành phần:
    - Encoders 
    - Decoders 
+ Mạng neural network sử dụng trọng số từ các pre-trained model đã được học từ các tập dữ liệu lớn như ImageNet sẽ đem lại kết kết quả khả tốt hơn việc cho model học từ đầu trên các tập dữ liệu nhỏ. 
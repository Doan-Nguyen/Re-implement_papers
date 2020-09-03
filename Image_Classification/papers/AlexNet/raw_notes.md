[AlexNet: The Architecture that Challenged CNNs](https://towardsdatascience.com/alexnet-the-architecture-that-challenged-cnns-e406d5297951)

+ Problem: 
    - apply to high resolution images


[Kiến trúc các mạng CNN nổi tiếng (PHẦN 1): AlexNet](https://blogcuabuicaodoanh.wordpress.com/2019/12/31/kien-truc-cac-mang-cnn-noi-tieng-phan-1-alexnet/)

+ ACtivation function
    - Sigmoid thì giá trị đầu ra thu được dao động từ 0 đến 1, gradient của hàm Sigmoid trong khoảng này gần như bằng 0, nếu khởi tạo tham số không đúng cách rất có khả năng không thể áp dụng được backpropagation, khiến cho các parameters không được tiếp tục cập nhật.
    - Hàm ReLu khiến cho model được huấn luyện dễ dàng hơn bằng nhiều phương pháp khởi tạo tham số khác nhau, với gradient trên khoảng dương luôn luôn là 1.
    

+ LeNet sử dụng kỹ thuật Weight Decay thì AlexNet sử dụng kỹ thuật Dropout

- Dropout, hàm ReLU và tiền xử lý dữ liệu bằng data agumentation chính là chìa khóa cho sự hiệu quả của mạng kiến trúc này.


[Measuring Saturation in Neural Networks](https://www.researchgate.net/publication/301363359_Measuring_Saturation_in_Neural_Networks#:~:text=Abstract%20and%20Figures,ability%20of%20a%20neural%20network.)
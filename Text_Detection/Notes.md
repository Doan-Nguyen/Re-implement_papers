#   Phỏng vấn 

## Trình bài bài toán: 
    - Bài toán OCR cho tờ rơi bất động sản.
    
## Mô tả dữ liệu:
    - Đặc điểm dữ liệu:
        - 70-80 % nội dung nằm trong các bảng/xếp theo dãy hàng ngang.
        - 10 % text có kích thước lớn (tiêu đề)
        - 10 - 20% text có kích thước to + nhỏ đứng sát kề nhau (phần thông tin bổ sung)
    - Đặc điểm nội dung text: 
        - Chữ to + nhỏ đứng sát nhau.
        - Có dãy chữ nằm dọc.
        - Ví trí thông tin không theo format do tờ rơi user upload.

## Các bước pre-process:
    - Viết tool gán nhãn:
        - Libary: Tkinkter
        - Pretrained: EAST

## Model, loss function (phân tích các thành phần)

## Kết quả (metric dùng)
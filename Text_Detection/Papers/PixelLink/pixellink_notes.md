# PixelLink: Detecting Scence Text via Instance Segmentation

## 0. Các khái niệm cần làm rõ

### 0.1 Instance segmentation

+ Thực hiện nhận diện các đối tượng ở cấp độ điểm ảnh, mỗi đối tượng sẽ có 1 vùng các điểm ảnh riêng. Khác với *semantic segmentation* chỉ nhận diện 1 đối tượng duy nhất trong ảnh, *instance segmentation* sẽ nhận diện nhiều đối tượng trong cùng 1 ảnh.
        ![Semantic & instance segmentation](figures/sematic_vs_instance.png)
        [Nguồn](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)


### 0.2 Regression-based
+ Có thể hiểu là vị trí của vùng đối tượng được suy ra từ các bounding boxes mà mô hình dự đoán. 

### 0.3 Location Regression

### 0.4 minAreaRect



## 1. Sơ lược về bài toán 
+ Bài toán *text detection* hay thường biết đến với tên OCR là một bài toán phổ biến. Những thuật toán gần thời điểm bài báo ra đời thường tập trung detect dựa vào *bounding box regression* thông qua 2 output dự đoán: *text/non-text score* & *vị trí regression*.

+ Ở đây tác giả có đề cập đến việc, *text/non-text score* cũng trả về thông tin về vị trí của text, điều này khiến vai trò của *location regression* giảm bớt đi. Mình xin trích dẫn cấu trúc 1 mạng trong bài báo "Text/Non-text Image Classification in the Wild with Convolutional Neural Networks" [Link](https://www.researchgate.net/publication/311578437_TextNon-text_Image_Classification_in_the_Wild_with_Convolutional_Neural_Networks). Khi xác định *text/non-text score*, mô hình đã chia ảnh thành nhiều phân vùng, rồi từ đó xác định khả năng tồn tại của text:        
        
    ![Text/non-text](figures/text-nontext.png)


## 2. Ý tưởng, cải tiến của thuật toán
+ Các thuật toán trước:  
    - Ý tưởng đầu tiên **Regression-based Text Detection**, xác định vị trí text thông qua text proposals (được đề cập trong CTPN). Phát triển từ *anchor* trong bài toán *object detection* đề xác định một phần text & kết nối với nhau thông qua cơ chế liên kết (vd: RNN). Ở đây, số lượng anchor nhiều để có thể detect được các text có kích thước da dạng.
    - Ý tưởng khác **Segmentation-based Text Detection**, việc xác định vị trí text được coi như bài toán *semantic segmentation problem* thông qua dự đoán 3 tham số:
        - text/non text score
        - character classes
        - character linking orientations 
    - Nói thêm, với bài toán nhận diện chữ tiếng Nhật, detect orientations đóng vai trò quan trọng vì nó sẽ giải quyết chữ dọc. Với thuật toán CTPN mà mình đã phân tích, nó không giải quyết trường hợp này nhưng bù lại vì sử dụng *cơ chế vertical anchor* trong chuỗi thuật toán R-CNN nên có thể cải thiện về mặt tốc độ (thú thật cái này mình đoán nhé).

+ Ý tưởng thuật toán PixelLink đề xuất:
    - Vấn đề:
        - Khi các text quá sát nhau, việc *semantic segmentation* trở lên khó khăn để segmentation từ kí tự.
    - Đề xuất:
        - Việc xác định vị trí text bằng *instance segmentation* thay cho *bounding boxes regression*. Mô hình **Pixel Link** dự đoán 2 kết quả:
            - text/non-text prediction {các điểm ảnh trong vùng text xuất hiện được gán nhãn *positive* & ngược lại}
            - link prediction
                - Việc liên kết các điểm ảnh diễn ra giữa điểm ảnh đang xét với 8 điểm ảnh *hàng xóm*.
                - Xét 2 cặp điểm ảnh {điểm ảnh trung tâm, 1 trong 8 điểm ảnh}.
                    - Nếu nằm cùng một kí tự trong text (nguyên gốc: *same instance*) sẽ được đánh nhãn *positive* & ngược lại.
                - Các điểm ảnh được gán nhãn *positive*

## 3. Phân tích thuật toán 

### 3.1 Kiến trúc thuật toán 

### 3.2 Kết nối các Pixels 

### 

## 4. Tối ưu thuật toán

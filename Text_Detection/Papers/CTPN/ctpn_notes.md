# Detecting Text in Natural Image with Connectionist Text Proposal Network

## Đánh giá quá trình phát triền (để sau)
+ Quan điểm cá nhân chia các thuật toán Text Detection thành 3 nhóm:
    - Xác định text proposals bởi các mạng CNN như R-CNN với cơ chế **Region Proposal Network**.
    - 
    - 

## Sơ lược về thuật toán 
+ Ý tưởng chính của thuật toán chia nhỏ thành nhiều phần & dự đoán khả năng xuất hiện text trong từng phần. Từ các phần được dự đoán chứa text sẽ kết nối chúng lại thành 1 dòng. 

+ Xác định vị trí line text dựa trên feature maps. Ở đây, tác giả coi dòng text cần xác định như chuỗi các *fine-scale text proposal*, với mỗi proposal dự đoán các thành phần của dòng text đó.

+ Sử dụng cơ chế "vertical anchor" để dự đoán 2 yếu tố: *ví trí* & *khả năng tồn tại của text* trong mỗi proposal cố định về chiều rộng. Các dãy/chuỗi proposals được nối bởi mạng hồi tiếp (RNN).



## 1.Introduction

+ **Architecture of the CTPN**

    - VGG16 ~> feature maps

    - Bi-directional LSTM (~ the sequential windows)

        - RNN layer is connected to a 512D fully-connected layer => predicts: text/non-text scores; y-axis coordinates; side-refinement offsets of *k* anchor.

### 1.1 Contributions
    Bước 1: Xác định vị trí text từ chuỗi *fine-scale text proposals*. 
    - Sử dụng *Region Proposal Network* để tìm ra các text proposals (region proposals) nằm trên feature map. 
    - CTPN sử dụng *k-anchor* để dự đoán thông tin về vị trí + khả năng chứa text với mỗi text proposal.
    Bước 2: Kết nối các chuỗi text proposals
    - Bài báo sử dụng mạng hồi tiếp để kết nối thành chuỗi các text proposals.
    Bước 3: Sàng lọc biên (side-refinement)
    - Tích hợp 2 bước trên thu được mô hình end-to-end.


## 2. Connectionist Text Proposal Network

### 2.1 Detecting text in Fine-Scale Proposals

+ **A fully convolutional network** ~> allow input image of arbitrary size => output a sequence of fine-scale text proposals

+ **VGG16:**

    - kernel size: 3x3

    - Architecture:
        
        - the size of *conv5* feature maps is determined by the size of input img.

        - ? *the total stride & receptive field* are fixed as 16 & 228 pixels

        - *sliding-window* methods adopt multi-scale windows to detect objects with different sizes.

+ Design the *fine-scale* text proposals, that investigates each spatial location in the *conv5* densely.

    - Text proposals is defined **width of 16 pixels**

    - Design *k* vertical anchors to:

        - **? vertical & horizontal same width: 16 pixels**
        
        - predict y-coordinates for each proposals.

        - same horizontal location with a fixed width of 16 pixels but vertical locations are varied in k different heights.

        - Author used *k=10* anchors for each proposal



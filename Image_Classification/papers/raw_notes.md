+ Networks:

- Recursion Cellular Image Classification

    - seresnext50, 101, densenet, efficientnet 
    
    - Resnet

    - DenseNet201, ResNeXt101_32x8d, HRNet-W18, HRNet-W30

    - se-resnext50

+ **AdaptiveAvgPool2d**

    - Adaptive average pooling over an input signal

+ Logging form 
```
logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

logging.info('Start training process')
handler = logging.FileHandler('train_log.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)
```

+ 

    - few-shot classification

        - Triplet Network ~ feature extractor
            - một CNN base network để chiếu input lên không gian euclidean n chiều
            - để extract được cho những unseen classes thì bộ dữ liệu huấn luyện của bạn phải có số lượng nhãn lớn và đa dạng các đặc trưng. Cơ hội cover được đặc trưng của unseen data sẽ cao hơn

        - T-SNE để visualize feature

        - ? k và n trong n-way k-shot
            - n-way là n classes mà model chưa được train
            - k-shot là số ảnh được đánh nhãn trong n classes

        - **baselines như matching networks for fewshot hoặc siamese network for fewshot**

    - kỹ thuật zero-shot

    - imbalance classification là oversampling ~ opy paste cái images của rare class cho bằng số images của dominate class rồi train

    - data medical: capsule hoặc rotation equivariant model

    - n-shot learning có thể hiểu là bạn so sánh distance của feature vector (được extract từ pretrained models ) cho class mới ( không nằm trong training set ) để phân loại class đó mà không cần train lại với class mới này

    - https://phamdinhkhanh.github.io/2020/03/12/faceNetAlgorithm.html?fbclid=IwAR1_7JS5enWgkZ0khHVPHn7r82eQ2Jl2-NJJz5_xaC2Uc2yMIgDidBX0yu4#31-one-shot-learning

+ Regression problem
    - đầu ra là các số thực CÓ THỨ TỰ >< classification

    - FC để tìm kiếm mối liên hệ của Features và đầu ra nào đó. Hàm Loss nó tạo ra quan hệ tương hỗ hay không giữa các đầu ra, tương hỗ như thế nào, mà thành Classification hay Regression.

    https://hunch.net/~jl/projects/reductions/reductions.html?fbclid=IwAR1RaH5qzCONWizsHt6dKJI7xxg68FIJcThB7W9QVLdtp--OG_Dzmdo9t08

+ Trong TH của bạn nếu các giá trị phân phối xác suất quá gần nhau thì bạn nên lọc riêng dữ liệu ra để khảo sát.
    1. Nếu các mẫu có những đặc trưng phân biệt rõ ràng dựa trên phân tích về knownledge domain thì chứng tỏ thuật toán của bạn chưa tốt. Cần thay đổi một thuật toán khác.
    2. Nếu các mẫu có sự overlap về tính chất thì đây là những mẫu khó phân loại. Trường hợp này bạn nên thống kê lại tỷ lện mẫu khó phân loại/tổng mẫu. Nếu chiếm tỷ trọng lớn thì nên cân nhắc bổ sung biến. Nếu tỷ trọng nhỏ thì xem phân phối giữa các class để ưu tiên voting output cho các class thiểu số (việc dự báo đúng được 1 quan sát thiểu số lúc này có ý nghĩa hơn là dự báo đúng các class đa số).

+ up-sampling và down-sampling. up-sampling thì mình sẽ train lặp lại các mẫu có nhãn ít. down-sampling thì giảm số lượng các mẫu ở lớp nhiều dữ liệu đi

+ Inferrent: Pruning, Quantization, Knowledge Distillation, Huffman coding
 
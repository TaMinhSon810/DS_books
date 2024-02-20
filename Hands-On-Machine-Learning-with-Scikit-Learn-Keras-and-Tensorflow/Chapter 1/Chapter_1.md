# CHAPTER 1: The Machine Learning Landscape

## Day 1 (20/2/2024)

Bài đặt ra vấn đề khi hỏi **Machine Learning** chính xác là gì? Việc một machine mà learn something nghĩa là gì? Liệu tải một kho dữ liệu như Wiki về máy tính, máy tính có thể "learn" được không?
## What is Machine Learning?

**Machine Learning** is the science (and art) of programming computers so they can learn from data.

Ví dụ, bạn có một công cụ để lọc spam dựa trên **Machine Learning** có thể học từ việc đánh dấu những ví dụ các emails spam và các emails không phải là spam.\
Các ví dụ mà chương trình học gọi là ***training set***. Từng ví dụ là ***training instance*** (hoặc ***sample***).\
Phần mà **Machine Learning** học và tạo ra dự báo gọi là **model**. \
Khi mà máy dự đoán, ta có thể set một tỷ lệ dự đoán chính xác mà ta muốn máy dự đoán. Đó là **accuracy**.\
Quay lại vấn đề nên ở đầu bài, việc download dữ liệu Wiki giúp máy tính chúng ta có thêm nhiều data nhưng khôn giúp tốt lên. Đó không phải là **Machine Learning**.

## Why Use Machine Learning?
Quay trở lại ví dụ về làm một chương trình về lọc emails spam.\
Một technique truyền thống chúng ta có thể sử dụng khi làm chương trình đó là:
- Đầu tiên xem thử các email spam thì trông như nào. VD: Các email spam hay có chữ "4U", "credit card", "free" hay "amazing" trong nội dung email.
- Tiếp theo viết thuật toán để có thể detect được các pattern mà mình nhận ra, và chương trình sẽ đánh flag nếu một số pattern trong rule mình đưa ra được detect.
- Cuối cùng sẽ test chương trình, lặp lại bước 1 và 2 đến khi chương trình đủ tốt.

![alt text](image-2.png)

Nếu làm theo cách truyền thống này, chúng ta sẽ rất khó khăn khi viết được rules phù hợp và cover được các case.\
Chính vì vậy, **Machine Learning** sẽ giúp chương trình lọc spam email có thể tự động học từ bộ data mà mình dưa cho để tìm và đưa ra các pattern để phân biệt email nào spam và không spam. Chúng ta sẽ không cần phải viết rules cho chương trình nữa. Việc này sẽ nhanh hơn, dễ dàng maintain và chính xác hơn so cách truyền thống.\
![alt text](image-1.png)

Nếu spammers biết các email spam bị block do có chữ "4U" và chuyển sang thành "For U" thì sẽ như thế nào?\
Một chương trình truyền thống thì chúng ta cần phải kiểm tra và update thường xuyên để đánh dấu lại. Tuy nhiên, một chương trình sử dụng ML sẽ tự động học data cho thêm và nhận ra "For U" là một pattern cần detect để gắn flag cho việc spam email.\
![alt text](image-3.png)

Mở rộng hơn, **Machine Learning** có thể xử lý một số vấn đề mà thuật toán truyền thống không xử lý được. Ví dụ như speech recognition, ta chỉ muốn một chương trình đơn giản là khi nghe 1 âm thanh khi chương trình trả ra âm thanh là "one" hay "two". Cách xử lý tốt nhất là viết thuật toán và cho máy tự học dựa trên hàng ngàn, hàng triệu ví dụ khác nhau.\
Cuối cùng, **Machine Learning** có thể tự xem lại chúng đã học những gì. VD với trường hợp email spam, máy có thể phát hiện và đưa ra những từ và cụm từ là best pattern để gắn flag spam. Đôi lúc, chúng sẽ phát hiện cả những correlation hoặc các trends mới, từ đó giúp chúng ta hiểu thêm về bài toán.\
![alt text](image-4.png)\
Nói tóm lại, **Machine Learning** sử dụng tốt khi:
- Các vấn đề nếu dùng phương pháp truyền thống cần viết rules điều chỉnh nhiều hoặc rất dài.
- Các vấn đề khi sử dụng phương pháp truyền thống sẽ không tốt hoặc không xử lý được
- Dữ liệu mới thay đổi và cần update liên tục
- Lấy được insight từ các vấn đề phức tạp và bộ data lớn

## Types of Machine Learning Systems
Có rất nhiều loại **Machine Learning Systems** khác nhau mà chúng ta có thể phân loại dựa trên:
- Liệu chúng có được giám sát trong quá trình training không? (supervised, unsupervised, semi-supervised, self-supervised, and others)
- Liệu chúng có học thêm liên tục khi đang thực hiện? (online versus
batch learning)
- Liệu chúng học bằng cách so sánh new data points với các điểm data points đã biết, hay detect các pattern sử dụng tập training data và build một model dự đoán? (instance-based versus model-based learning)

Nói chung, các cateria này không riêng biệt nhau mà chúng ta có thể combine chúng theo bất cứ nào ta muốn. VD một bộ lọc spam có thể dạng online, model-based, supervised learning system.\
### Training Supervision
Như để cập ở trên, ML systems có thể phân loại dựa trên loại supervision mà chúng dùng trong quá trình training. Chúng ta sẽ tập trung vào: **supervised learning**, **unsupervised learning**, **selfsupervised learning**, **semi-supervised learning**, and **reinforcement learning**.
#### Supervised learning
Trong **Supervised learning**, bạn sẽ đưa 1 tập training bao gồm nhãn đưa vào thuật toán gọi là ***labels***.\
![alt text](image-5.png)\
Một task phổ biến trong **Supervised learning** là ***classification***. Ví dụ như chính trong spam email, máy sẽ được train với rất nhiều email gắn với ***class*** của chúng (spam hoặc không spam), và máy sẽ học để phân loại 1 email mới xem là spam hay không phải spam.\
Một task phổ biến khác đó là ***regression***. Máy sẽ dự đoán một biến ***target*** có thể là giá của xe, nhà,... dựa trên 1 số lượng ***features*** input (Số km đi, tuổi, hãng hiệu xe,...). Để dạy máy, ta cần phải đưa vô số ví dụ của xe gồm ***features*** và ***target*** của xe đó.\
***Regression*** dùng dự đoán một giá trị liên tục. Trong khi ***classification*** dự đoán xem thuộc từng loại đơn lẻ.\
***Note:*** Một số Regression models nhưng dùng trong bài toán ***classification*** và ngược lại. \
Ví dụ, logistics regression có output là một số (như 20% tỷ lệ bị spam) tuy nhiên lại sử dụng trong bài toán classification chẳng hạn nếu chúng ta muốn xác định positive hay negative khi tỷ lệ spam trên dưới 30%. Khi này 20% là positive.\
***Note:*** ***target*** và ***model*** được sử dụng đan xen và gần giống nhau trong  ***supervised learning***. Tuy nhiên, ***target*** hay sử dụng trong bài toán ***regression*** và ***model*** dùng trong bài toán ***classification***.
Hơn nữa, ***features*** cũng có thể được gọi là ***predictors*** hoặc ***attributes***.

#### Unsupervised learning

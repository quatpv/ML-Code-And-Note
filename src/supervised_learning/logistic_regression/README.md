#### 1. Mục đích sử dụng
- Thường sử dụng cho bài toán classification
- Boundary tạo bởi Logistic Regression có dạng tuyến tính do đó phù hợp với dữ liệu mà 2 class gần linearly separable (Không phù hợp với dữ liệu kiểu non linear). 
- Giả thiết là các điểm dữ liệu là độc lập tuyến tính với nhau

#### 2. Ghi chú
- Công thức của logistic regression có dạng <!-- $f(X) = \theta(W^TX)$ --> <img style="transform: translateY(0.2em); background: white;" src="https://render.githubusercontent.com/render/math?math=f(X)%20%3D%20%5Ctheta(W%5ETX)">
  trong đó <!-- $\theta$ --> <img style="transform: translateY(0.2em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ctheta"> là một activation function - thường là Sigmoid. 
- Lý do sử dụng hàm Sigmoid
    + Hàm liên tục, bị chặn trong khoảng (0, 1)
    + Có đạo hàm tại mọi điểm trong đoạn (0, 1)
- Loss function:
    <!-- $w = argmax_wP(y|X; w)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=w%20%3D%20argmax_wP(y%7CX%3B%20w)"> trong đó <!-- $P(\mathbf{y} \mid \mathbf{X} ; \mathbf{w})=\prod_{i=1}^{N} P\left(y_{i} \mid \mathbf{x}_{i} ; \mathbf{w}\right)=\prod_{i=1}^{N} z_{i}^{y_{i}}\left(1-z_{i}\right)^{1-y_{i}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(%5Cmathbf%7By%7D%20%5Cmid%20%5Cmathbf%7BX%7D%20%3B%20%5Cmathbf%7Bw%7D)%3D%5Cprod_%7Bi%3D1%7D%5E%7BN%7D%20P%5Cleft(y_%7Bi%7D%20%5Cmid%20%5Cmathbf%7Bx%7D_%7Bi%7D%20%3B%20%5Cmathbf%7Bw%7D%5Cright)%3D%5Cprod_%7Bi%3D1%7D%5E%7BN%7D%20z_%7Bi%7D%5E%7By_%7Bi%7D%7D%5Cleft(1-z_%7Bi%7D%5Cright)%5E%7B1-y_%7Bi%7D%7D">. Sử dụng phương pháp maximum likelihood estimate [2] để giải phương trình trên, ta sẽ cần đi tìm giá trị nhỏ nhất của hàm loss: <!-- $J(\mathbf{w})=-\log P(\mathbf{y} \mid \mathbf{X} ; \mathbf{w})=-\sum_{i=1}^{N}\left(y_{i} \log z_{i}+\left(1-y_{i}\right) \log \left(1-z_{i}\right)\right)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=J(%5Cmathbf%7Bw%7D)%3D-%5Clog%20P(%5Cmathbf%7By%7D%20%5Cmid%20%5Cmathbf%7BX%7D%20%3B%20%5Cmathbf%7Bw%7D)%3D-%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cleft(y_%7Bi%7D%20%5Clog%20z_%7Bi%7D%2B%5Cleft(1-y_%7Bi%7D%5Cright)%20%5Clog%20%5Cleft(1-z_%7Bi%7D%5Cright)%5Cright)"> - Hàm loss này gọi là cross entropy

- Có 2 cách để tối ưu cho hàm loss trên: 
    + Sử dụng gradient descent: Hiệu quả với bài toán có nhiều features
    ```python
    self.param -= self.learning_rate * -(y - y_pred).dot(X)
    ``` 
    + Sử dụng tính toán ma trận: Hiệu quả với bài toán có số lượng features ít hơn 1000
    ```python
    diag_gradient = make_diagonal(self.sigmoid.gradient(X @ self.param))
    self.param = np.linalg.pinv(X.T @ diag_gradient @ X) @ X.T @ (diag_gradient @ X @ self.param + y - y_pred)    
    ```


#### 3. Tài liệu tham khảo
[1] https://machinelearningcoban.com/2017/01/27/logisticregression/

[2] http://dangnguyenit.blogspot.com/2018/10/uoc-luong-hop-ly-cuc-aimaximum.html

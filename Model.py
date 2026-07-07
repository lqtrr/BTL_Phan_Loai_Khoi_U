import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        
        # --- Khối Tích chập 1 ---
        # Ảnh đầu vào: 3 kênh màu (RGB), kích thước 224x224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Khối Tích chập 2 ---
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Khối Tích chập 3 ---
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Khối Phân loại (Fully Connected) ---
        # Sau 3 lần MaxPool (giảm một nửa kích thước), ảnh 224x224 sẽ còn 28x28.
        # Số node đầu vào: 64 (kênh) * 28 * 28 = 50176
        self.fc1 = nn.Linear(in_features=64 * 28 * 28, out_features=512)
        self.dropout = nn.Dropout(p=0.5) # Chống Overfitting
        self.fc2 = nn.Linear(in_features=512, out_features=4) # 4 tương ứng với 4 loại u não
        
    def forward(self, x):
        # Truyền dữ liệu qua khối 1
        x = self.pool1(F.relu(self.conv1(x)))
        # Truyền dữ liệu qua khối 2
        x = self.pool2(F.relu(self.conv2(x)))
        # Truyền dữ liệu qua khối 3
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Làm phẳng (Flatten) ma trận 3D thành vector 1D để đưa vào lớp Linear
        x = torch.flatten(x, 1)
        
        # Truyền qua lớp phân loại
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Lớp cuối cùng không dùng ReLU
        
        return x

# Kiểm tra thử mô hình với một Tensor giả lập
if __name__ == "__main__":
    model = BrainTumorCNN()
    # Tạo một ảnh giả lập: 1 ảnh, 3 kênh màu, kích thước 224x224
    dummy_input = torch.randn(1, 3, 224, 224) 
    output = model(dummy_input)
    print(model)
    print("\nKích thước đầu ra dự đoán (Batch size, Số lớp):", output.shape)
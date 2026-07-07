from datasets import load_dataset
import torch
from torchvision import transforms

# 1. Tải dataset
print("Đang tải dữ liệu...")
ds = load_dataset("PranomVignesh/MRI-Images-of-Brain-Tumor")

# 2. Định nghĩa các phép biến đổi ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Ép về kích thước chuẩn 224x224
    transforms.ToTensor(),         # Chuyển ảnh thành Tensor với giá trị pixel từ 0-1
])

# 3. Hàm áp dụng biến đổi cho từng lô ảnh (batch)
def apply_transform(examples):
    # Đảm bảo ảnh ở hệ màu RGB (tránh lỗi ảnh xám đen trắng)
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

# 4. Áp dụng biến đổi cho toàn bộ dataset
print("Đang tiền xử lý ảnh (Resize & ToTensor)...")
ds = ds.map(apply_transform, batched=True)

# 5. Cấu hình định dạng đầu ra thành PyTorch Tensor
ds.set_format(type="torch", columns=["pixel_values", "label"])

# Kiểm tra kết quả sau khi xử lý
print("Kích thước Tensor của ảnh đầu tiên (Channels, Height, Width):", ds['train'][0]['pixel_values'].shape)
print("Quá trình tiền xử lý hoàn tất!")
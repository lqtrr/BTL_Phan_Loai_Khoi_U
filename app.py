import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. ĐỊNH NGHĨA LẠI KIẾN TRÚC MODEL (Bắt buộc)
# ==========================================
# Phải định nghĩa lại mạng CNN giống hệt lúc Train thì mới nạp được file .pth
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 4)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# 2. CÀI ĐẶT CÁC THÔNG SỐ CƠ BẢN
# ==========================================
# Các nhãn dựa theo thứ tự của bộ dữ liệu Hugging Face
LABELS = ['Glioma (U thần kinh đệm)', 'Meningioma (U màng não)', 'No-tumor (Không có u)', 'Pituitary (U tuyến yên)']

# Cấu hình tiền xử lý ảnh giống hệt lúc Train
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Hàm load model (sử dụng cache của Streamlit để không phải load lại nhiều lần)
@st.cache_resource
def load_model():
    model = BrainTumorCNN()
    # Dùng map_location='cpu' để web có thể chạy trên máy không có Card rời
    model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=torch.device('cpu')))
    model.eval() # Chuyển mô hình sang chế độ dự đoán (Inference)
    return model

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN WEB VỚI STREAMLIT
# ==========================================
st.set_page_config(page_title="AI Chẩn đoán U Não", page_icon="🧠", layout="centered")

st.title("🧠 Hệ thống Phân loại Khối u Não từ ảnh MRI")
st.markdown("Hệ thống sử dụng Trí tuệ nhân tạo (Custom CNN) để hỗ trợ chẩn đoán 4 loại u não.")

# Gọi hàm nạp mô hình
model = load_model()

# Tạo khu vực để người dùng upload ảnh
uploaded_file = st.file_uploader("Tải ảnh MRI của bệnh nhân lên (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Cột hiển thị ảnh và cột kết quả
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh MRI tải lên:")
        # Đọc ảnh bằng thư viện PIL
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Kết quả Phân loại:")
        with st.spinner('AI đang phân tích hình ảnh...'):
            # 1. Tiền xử lý ảnh tải lên
            input_tensor = transform(image).unsqueeze(0) # Thêm 1 chiều batch ở đầu -> [1, 3, 224, 224]
            
            # 2. Đưa qua mô hình để dự đoán
            with torch.no_grad():
                output = model(input_tensor)
                
                # Tính phần trăm độ tin cậy bằng Softmax
                probabilities = F.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                predicted_label = LABELS[predicted_idx.item()]
                confidence_score = confidence.item() * 100
                
            # 3. Hiển thị kết quả trực quan
            if "No-tumor" in predicted_label:
                st.success(f"**Dự đoán:** {predicted_label}")
            else:
                st.error(f"**Dự đoán:** {predicted_label}")
                
            st.info(f"**Độ tin cậy của AI:** {confidence_score:.2f}%")
            
            # Hiển thị thanh tiến trình (progress bar) cho độ tin cậy
            st.progress(int(confidence_score))
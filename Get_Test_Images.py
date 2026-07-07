from datasets import load_dataset
import os

print("Đang tải ảnh mẫu về máy...")
ds = load_dataset("PranomVignesh/MRI-Images-of-Brain-Tumor")
labels = ds['train'].features['label'].names

# Tạo thư mục chứa ảnh
os.makedirs("Anh_Test_Demo", exist_ok=True)

# Lấy đúng 1 ảnh cho mỗi loại u
saved_labels = set()
for sample in ds['train']:
    label_id = sample['label']
    label_name = labels[label_id]
    
    if label_name not in saved_labels:
        img = sample['image']
        # Lưu ảnh dưới dạng JPG
        img.save(f"Anh_Test_Demo/{label_name}.jpg")
        saved_labels.add(label_name)
        
    if len(saved_labels) == 4: # Đủ 4 loại thì dừng
        break

print("Đã lưu thành công 4 ảnh test vào thư mục 'Anh_Test_Demo'!")  
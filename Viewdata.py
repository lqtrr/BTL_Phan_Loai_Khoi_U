from datasets import load_dataset
import matplotlib.pyplot as plt

print("Đang kết nối tải dữ liệu để tạo lưới ảnh...")
ds = load_dataset("PranomVignesh/MRI-Images-of-Brain-Tumor")
labels = ds['train'].features['label'].names

# Số lượng ảnh bạn muốn xem cho MỖI loại u (ở đây chọn 5 ảnh/loại)
num_images_per_class = 5
images_by_class = {i: [] for i in range(len(labels))}

# Quét qua tập train để thu thập đủ 5 ảnh cho mỗi loại
for sample in ds['train']:
    label = sample['label']
    if len(images_by_class[label]) < num_images_per_class:
        images_by_class[label].append(sample['image'])
    
    # Dừng vòng lặp nếu đã thu thập đủ số lượng cho tất cả 4 loại
    if all(len(imgs) == num_images_per_class for imgs in images_by_class.values()):
        break

# Tạo khung vẽ lưới (Grid): 4 hàng (số loại u) x 5 cột (số ảnh)
fig, axes = plt.subplots(nrows=len(labels), ncols=num_images_per_class, figsize=(15, 10))
fig.suptitle("Đa dạng hình thái ảnh MRI theo từng loại khối u", fontsize=16, fontweight='bold')

# Bắt đầu vẽ từng ảnh lên lưới
for i, label_name in enumerate(labels):
    for j in range(num_images_per_class):
        ax = axes[i, j]
        img = images_by_class[i][j]
        
        ax.imshow(img, cmap='gray')
        ax.axis('off') # Tắt trục tọa độ
        
        # Ghi tên loại u ở bức ảnh đầu tiên của mỗi hàng để dễ nhìn
        if j == 0:
            ax.set_title(f"{label_name.upper()}\nKích thước: {img.size[0]}x{img.size[1]}", 
                         loc='left', color='red', fontweight='bold', fontsize=12)
        else:
            ax.set_title(f"{img.size[0]}x{img.size[1]}", fontsize=10, color='blue')

# Tự động căn chỉnh khoảng cách giữa các ảnh cho đẹp
plt.tight_layout()
plt.show()
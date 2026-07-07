import matplotlib.pyplot as plt
import seaborn as sns

# 1. Dữ liệu phân bố của bạn
labels = ['Glioma\n(U thần kinh đệm)', 'Meningioma\n(U màng não)', 'Pituitary\n(U tuyến yên)', 'No-tumor\n(Não khỏe mạnh)']
counts = [1610, 1342, 1342, 1075]

# 2. Tạo khung vẽ
fig, ax = plt.subplots(figsize=(10, 6))

# 3. Vẽ biểu đồ theo form mẫu (Màu xanh pastel dịu nhẹ, cột hẹp)
bars = ax.bar(labels, counts, 
              color='#aec7e8',       # Màu xanh lam nhạt giống hệt ảnh
              edgecolor='#7f7f7f',   # Viền xám mỏng
              linewidth=1.2, 
              width=0.6)             # Thu hẹp chiều rộng cột để có khoảng trống

# 4. Thêm con số trên đỉnh mỗi cột
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 15, 
            int(yval), 
            ha='center', va='bottom', 
            fontsize=11, color='black')

# 5. Cấu hình Tiêu đề và Trục (Font chữ cơ bản)
ax.set_title('Phân bố số lượng ảnh theo từng nhãn bệnh (Dataset Distribution)', fontsize=14, pad=15)
ax.set_ylabel('Số lượng ảnh (Images)', fontsize=12)
ax.set_xlabel('Nhãn Dữ liệu (Classes)', fontsize=12)

# 6. Cấu hình lưới (Chỉ kẻ ngang, nét đứt mờ)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
ax.set_axisbelow(True) # Đẩy đường lưới xuống dưới các cột
ax.set_ylim(0, 1800)

# Loại bỏ viền trên và viền phải của khung biểu đồ cho thoáng mắt
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 7. Hiển thị và Lưu ảnh
plt.tight_layout()
plt.savefig('Dataset_Distribution_Standard.png', dpi=300, bbox_inches='tight')
plt.show()

print("Đã vẽ biểu đồ thành công và lưu ảnh sắc nét!")
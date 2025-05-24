import os

# Đường dẫn thư mục chứa ảnh
input_folder = 'data/Train/Derain/rainy'
# Đường dẫn tệp txt để lưu danh sách ảnh
output_txt = 'data_dir/rainy/rainTrain.txt'

# Lấy danh sách file, sắp xếp và ghi vào file txt
with open(output_txt, 'w') as f:
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            f.write(f'rainy/{filename}\n')

print(f'Done. Written list to: {output_txt}')

import numpy as np
import glob
import os

# Đường dẫn tới folder chứa dữ liệu gốc
input_folder = '/Users/duc/Desktop/pythonProject3/PPG2BP/device_data'
output_folder = '/Users/duc/Desktop/pythonProject3/PPG2BP/device_data/normalized_data'
os.makedirs(output_folder, exist_ok=True)

# 1. Tìm min/max toàn bộ dữ liệu PPG (cột 0)
all_ppg = []
for file in glob.glob(os.path.join(input_folder, '*.npy')):
    data = np.load(file)
    if data.ndim == 2 and data.shape[1] >= 1:
        all_ppg.append(data[:, 0].flatten())
all_ppg = np.concatenate(all_ppg)
min_ppg = all_ppg.min()
max_ppg = all_ppg.max()
print(f"Min PPG: {min_ppg}, Max PPG: {max_ppg}")

# 2. Chuẩn hóa và cắt thành các đoạn 1024 điểm
for file in glob.glob(os.path.join(input_folder, '*.npy')):
    data = np.load(file)
    if data.ndim == 2 and data.shape[1] >= 1:
        ppg = data[:, 0]
        n_segments = len(ppg) // 1024
        for i in range(n_segments):
            segment = ppg[i*1024:(i+1)*1024]
            norm_segment = (segment - min_ppg) / (max_ppg - min_ppg)
            norm_segment = norm_segment.astype(np.float32)
            out_name = os.path.join(output_folder, f"{os.path.basename(file).replace('.npy','')}_seg{i}.npy")
            np.save(out_name, norm_segment)
            print(f"Đã lưu: {out_name}")
    else:
        print(f"Bỏ qua file {file} (không đúng định dạng 2D hoặc thiếu cột PPG)")

print("Hoàn thành chuẩn hóa và cắt đoạn toàn bộ dữ liệu!") 
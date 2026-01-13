import numpy as np
import os
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import glob

def preprocess_signal(data, target_length=1024, fs=250, min_segment_length=256):
    """
    Xử lý tín hiệu PPG thô và chuẩn hóa
    
    Args:
        data: Dữ liệu thô từ file .npy (shape: N, 4)
        target_length: Độ dài mục tiêu cho mỗi segment (mặc định 1024)
        fs: Tần số lấy mẫu (mặc định 250 Hz)
        min_segment_length: Độ dài tối thiểu cho segment (mặc định 256)
    
    Returns:
        List các segment đã được xử lý và chuẩn hóa
    """
    segments = []
    
    # Lấy cột đầu tiên (giả sử đây là tín hiệu PPG chính)
    ppg_signal = data[:, 0].astype(np.float64)
    
    # Loại bỏ nhiễu bằng bộ lọc Butterworth (chỉ khi đủ dữ liệu)
    if len(ppg_signal) > 100:  # Cần ít nhất 100 điểm để lọc
        nyquist = fs / 2
        low_cutoff = 0.5 / nyquist  # 0.5 Hz
        high_cutoff = 10 / nyquist  # 10 Hz
        
        try:
            b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            filtered_signal = signal.filtfilt(b, a, ppg_signal)
        except:
            # Nếu lọc thất bại, sử dụng tín hiệu gốc
            filtered_signal = ppg_signal
    else:
        filtered_signal = ppg_signal
    
    signal_length = len(filtered_signal)
    
    # Điều chỉnh target_length dựa trên độ dài tín hiệu
    if signal_length < target_length:
        if signal_length >= min_segment_length:
            # Nếu tín hiệu ngắn hơn target nhưng đủ dài, sử dụng toàn bộ làm một segment
            segment = filtered_signal
            
            # Chuẩn hóa về [0, 1]
            scaler = MinMaxScaler()
            normalized_segment = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
            
            # Resize về target_length bằng interpolation
            from scipy.interpolate import interp1d
            original_indices = np.linspace(0, 1, len(normalized_segment))
            target_indices = np.linspace(0, 1, target_length)
            interpolator = interp1d(original_indices, normalized_segment, kind='cubic')
            resized_segment = interpolator(target_indices)
            
            # Đảm bảo giá trị trong khoảng [0, 1]
            resized_segment = np.clip(resized_segment, 0.0, 1.0)
            
            segments.append(resized_segment.astype(np.float32))
        else:
            print(f"    Cảnh báo: Tín hiệu quá ngắn ({signal_length} điểm), bỏ qua")
    else:
        # Chia thành các segment có độ dài target_length
        num_segments = signal_length // target_length
        
        for i in range(num_segments):
            start_idx = i * target_length
            end_idx = start_idx + target_length
            
            segment = filtered_signal[start_idx:end_idx]
            
            # Chuẩn hóa segment về [0, 1]
            scaler = MinMaxScaler()
            normalized_segment = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
            
            segments.append(normalized_segment.astype(np.float32))
        
        # Xử lý phần dư nếu đủ dài
        remaining_length = signal_length % target_length
        if remaining_length >= min_segment_length:
            remaining_segment = filtered_signal[-remaining_length:]
            
            # Chuẩn hóa
            scaler = MinMaxScaler()
            normalized_segment = scaler.fit_transform(remaining_segment.reshape(-1, 1)).flatten()
            
            # Resize về target_length
            from scipy.interpolate import interp1d
            original_indices = np.linspace(0, 1, len(normalized_segment))
            target_indices = np.linspace(0, 1, target_length)
            interpolator = interp1d(original_indices, normalized_segment, kind='cubic')
            resized_segment = interpolator(target_indices)
            
            # Đảm bảo giá trị trong khoảng [0, 1]
            resized_segment = np.clip(resized_segment, 0.0, 1.0)
            
            segments.append(resized_segment.astype(np.float32))
    
    return segments

def process_single_file(input_file, output_dir):
    """
    Xử lý một file .npy và tạo các segment chuẩn hóa
    
    Args:
        input_file: Đường dẫn file .npy đầu vào
        output_dir: Thư mục đầu ra
    """
    try:
        # Đọc dữ liệu
        data = np.load(input_file)
        print(f"Đang xử lý {input_file} - Shape: {data.shape}")
        
        # Xử lý và tạo segments
        segments = preprocess_signal(data)
        
        # Tạo tên file đầu ra
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Lưu từng segment
        for i, segment in enumerate(segments):
            output_file = os.path.join(output_dir, f"{base_name}_seg{i}.npy")
            np.save(output_file, segment)
            print(f"  Đã lưu segment {i}: {output_file}")
        
        return len(segments)
        
    except Exception as e:
        print(f"Lỗi khi xử lý {input_file}: {str(e)}")
        return 0

def process_all_files(input_dir=".", output_dir="normalized_data"):
    """
    Xử lý tất cả các file .npy trong thư mục đầu vào
    
    Args:
        input_dir: Thư mục chứa file .npy thô
        output_dir: Thư mục đầu ra cho file đã chuẩn hóa
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm tất cả file .npy trong thư mục đầu vào (loại trừ thư mục con)
    pattern = os.path.join(input_dir, "maxim_*.npy")
    npy_files = glob.glob(pattern)
    
    if not npy_files:
        print(f"Không tìm thấy file .npy nào trong {input_dir}")
        return
    
    print(f"Tìm thấy {len(npy_files)} file .npy để xử lý")
    
    total_segments = 0
    processed_files = 0
    
    for npy_file in npy_files:
        # Bỏ qua các file trong thư mục con
        if os.path.dirname(npy_file) != input_dir:
            continue
            
        segments_count = process_single_file(npy_file, output_dir)
        if segments_count > 0:
            total_segments += segments_count
            processed_files += 1
    
    print(f"\n=== Kết quả xử lý ===")
    print(f"Đã xử lý: {processed_files} file")
    print(f"Tổng số segment tạo ra: {total_segments}")
    print(f"File đầu ra được lưu trong: {output_dir}")

if __name__ == "__main__":
    # Thực thi xử lý dữ liệu
    print("=== Bắt đầu xử lý dữ liệu PPG ===")
    process_all_files()
    print("=== Hoàn thành xử lý ===") 
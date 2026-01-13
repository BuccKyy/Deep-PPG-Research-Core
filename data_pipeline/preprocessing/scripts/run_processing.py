#!/usr/bin/env python3
"""
Script để xử lý tất cả dữ liệu PPG thô và tạo ra các file chuẩn hóa

Sử dụng: python run_processing.py
"""

from process_data import process_all_files
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Xử lý dữ liệu PPG thô thành dữ liệu chuẩn hóa')
    parser.add_argument('--input_dir', '-i', default='.', 
                       help='Thư mục chứa file .npy đầu vào (mặc định: thư mục hiện tại)')
    parser.add_argument('--output_dir', '-o', default='normalized_data_new', 
                       help='Thư mục đầu ra (mặc định: normalized_data_new)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Ghi đè thư mục đầu ra nếu đã tồn tại')
    
    args = parser.parse_args()
    
    # Kiểm tra thư mục đầu vào
    if not os.path.exists(args.input_dir):
        print(f"Lỗi: Thư mục đầu vào '{args.input_dir}' không tồn tại")
        return
    
    # Kiểm tra thư mục đầu ra
    if os.path.exists(args.output_dir) and not args.overwrite:
        response = input(f"Thư mục '{args.output_dir}' đã tồn tại. Tiếp tục? (y/n): ")
        if response.lower() != 'y':
            print("Đã hủy.")
            return
    
    print("=" * 50)
    print("🔄 BẮT ĐẦU XỬ LÝ DỮ LIỆU PPG")
    print("=" * 50)
    print(f"📁 Thư mục đầu vào: {args.input_dir}")
    print(f"📁 Thư mục đầu ra: {args.output_dir}")
    print("=" * 50)
    
    # Chạy quá trình xử lý
    process_all_files(input_dir=args.input_dir, output_dir=args.output_dir)
    
    print("=" * 50)
    print("✅ HOÀN THÀNH XỬ LÝ DỮ LIỆU")
    print("=" * 50)

if __name__ == "__main__":
    main() 
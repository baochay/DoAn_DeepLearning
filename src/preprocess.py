import os
import cv2
import numpy as np
from PIL import Image
import shutil

def preprocess_image(image_path, output_size=(28, 28)):
    """Tiền xử lý ảnh: chuyển sang grayscale, cắt viền trắng, resize"""
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Cắt viền trắng
    mask = gray < 255
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = gray[y0:y1, x0:x1]
    else:
        cropped = gray
        
    # Resize về kích thước chuẩn
    resized = cv2.resize(cropped, output_size)
    
    return resized

def process_directory(input_dir, output_dir):
    """Xử lý toàn bộ thư mục ảnh"""
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Xóa thư mục output cũ nếu tồn tại
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Duyệt qua từng thư mục con (mỗi ký tự)
    for char_dir in os.listdir(input_dir):
        input_char_dir = os.path.join(input_dir, char_dir)
        output_char_dir = os.path.join(output_dir, char_dir)
        
        # Bỏ qua nếu không phải thư mục
        if not os.path.isdir(input_char_dir):
            continue
            
        # Tạo thư mục cho ký tự
        os.makedirs(output_char_dir, exist_ok=True)
        
        # Xử lý từng ảnh trong thư mục
        for img_file in os.listdir(input_char_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            input_path = os.path.join(input_char_dir, img_file)
            output_path = os.path.join(output_char_dir, img_file)
            
            # Tiền xử lý ảnh
            processed = preprocess_image(input_path)
            if processed is not None:
                # Lưu ảnh đã xử lý
                cv2.imwrite(output_path, processed)
                print(f"Đã xử lý: {input_path} -> {output_path}")

def main():
    input_dir = "data/real_images"
    output_dir = "data/real_images_processed"
    
    print("Bắt đầu tiền xử lý dữ liệu...")
    process_directory(input_dir, output_dir)
    print("Hoàn thành tiền xử lý dữ liệu!")

if __name__ == "__main__":
    main() 
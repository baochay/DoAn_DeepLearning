from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os
from tqdm import tqdm
import shutil
from datetime import datetime

def crop_auto(img):
    # Cắt viền trắng tự động
    img_np = np.array(img)
    mask = img_np < 255  # vùng không phải trắng hoàn toàn
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # slice end is exclusive
        img_cropped = img.crop((x0, y0, x1, y1))
        return img_cropped
    else:
        return img

def enhance_image(img, contrast=1.2, brightness=1.1):
    """Tăng cường chất lượng ảnh"""
    # Tăng độ tương phản
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    
    # Tăng độ sáng
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness)
    
    return img

def backup_directory(src_dir, backup_dir):
    """Tạo bản sao lưu của thư mục"""
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(src_dir, backup_dir)
    print(f"Đã tạo bản sao lưu tại: {backup_dir}")

def preprocess_images(input_dir, output_dir, size=(28, 28), enhance=True):
    # Tạo bản sao lưu
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"data/backup_{backup_time}"
    backup_directory(input_dir, backup_dir)
    
    # Tạo thư mục output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Thống kê
    total_images = 0
    processed_images = 0
    error_images = []
    
    # Đếm tổng số ảnh
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if os.path.isdir(label_dir):
            total_images += len([f for f in os.listdir(label_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    print(f"Tổng số ảnh cần xử lý: {total_images}")
    
    # Xử lý từng thư mục với thanh tiến trình
    for label in tqdm(os.listdir(input_dir), desc="Xử lý các thư mục"):
        label_dir = os.path.join(input_dir, label)
        out_label_dir = os.path.join(output_dir, label)
        
        if not os.path.isdir(label_dir):
            continue
            
        os.makedirs(out_label_dir, exist_ok=True)
        
        # Xử lý từng ảnh
        for img_name in os.listdir(label_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            img_path = os.path.join(label_dir, img_name)
            try:
                # Đọc và xử lý ảnh
                img = Image.open(img_path).convert('L')
                img = crop_auto(img)
                if enhance:
                    img = enhance_image(img)
                img = img.resize(size)
                
                # Lưu ảnh đã xử lý
                img.save(os.path.join(out_label_dir, img_name))
                processed_images += 1
                
            except Exception as e:
                error_images.append((img_path, str(e)))
    
    # Báo cáo kết quả
    print("\nKết quả xử lý:")
    print(f"- Tổng số ảnh: {total_images}")
    print(f"- Đã xử lý thành công: {processed_images}")
    print(f"- Số ảnh lỗi: {len(error_images)}")
    
    if error_images:
        print("\nChi tiết lỗi:")
        for img_path, error in error_images:
            print(f"- {img_path}: {error}")

if __name__ == '__main__':
    print("Bắt đầu tiền xử lý ảnh...")
    preprocess_images('data/real_images', 'data/real_images_processed')
    print('Đã hoàn thành tiền xử lý ảnh!') 
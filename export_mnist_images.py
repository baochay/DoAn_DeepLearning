import pandas as pd
import numpy as np
from PIL import Image
import os

def export_images(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    for i, row in df.iterrows():
        label = row[0]
        pixels = row[1:].values.astype(np.uint8).reshape(28, 28)
        img = Image.fromarray(pixels, mode='L')
        label_dir = os.path.join(out_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img.save(os.path.join(label_dir, f'{i}.png'))
    print(f'Đã xuất {len(df)} ảnh vào {out_dir}')

if __name__ == '__main__':
    export_images('data/sign_mnist_train.csv', 'data/mnist_images_train')
    export_images('data/sign_mnist_test.csv', 'data/mnist_images_test') 
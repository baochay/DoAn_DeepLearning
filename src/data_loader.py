import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class SignLanguageMNIST(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.uint8)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_data(data_dir='data', batch_size=32):
    """
    Tải và chuẩn bị dữ liệu Sign Language MNIST
    """
    # Tạo thư mục data nếu chưa tồn tại
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Tải dữ liệu từ Kaggle
    # Lưu ý: Bạn cần tải dữ liệu từ Kaggle và đặt trong thư mục data
    train_data = pd.read_csv(os.path.join(data_dir, 'sign_mnist_train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'sign_mnist_test.csv'))

    # Tách features và labels
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    # Reshape dữ liệu thành hình ảnh 28x28
    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)

    # Chia tập validation từ tập train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Ánh xạ lại nhãn để liên tục từ 0 đến num_classes-1
    unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    def map_labels(y):
        return np.array([label_map[label] for label in y])
    y_train = map_labels(y_train)
    y_val = map_labels(y_val)
    y_test = map_labels(y_test)

    # Đếm số lớp
    num_classes = len(np.unique(y_train))

    # Định nghĩa các biến đổi
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Tạo datasets
    train_dataset = SignLanguageMNIST(X_train, y_train, transform=transform)
    val_dataset = SignLanguageMNIST(X_val, y_val, transform=transform)
    test_dataset = SignLanguageMNIST(X_test, y_test, transform=transform)

    # Tạo dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, num_classes

if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, num_classes = load_data()
    print("Data loaded successfully!")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Number of classes: {num_classes}") 
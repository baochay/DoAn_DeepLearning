import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from .data_loader import load_data, SignLanguageMNIST
from .model import get_model

# Bảng ánh xạ chữ cái sang label MNIST (bỏ J, Z)
CHAR_TO_LABEL = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'K': 9, 'L': 10, 'M': 11, 'N': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17, 'T': 18, 'U': 19, 'V': 20, 'W': 21, 'X': 22, 'Y': 23
}

class MappedImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        # target là chỉ số class theo thứ tự thư mục, cần ánh xạ lại nếu là chữ cái
        class_name = os.path.basename(os.path.dirname(path)).upper()
        mapped_target = CHAR_TO_LABEL.get(class_name, target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            mapped_target = self.target_transform(mapped_target)
        return sample, mapped_target

def main():
    # MNIST loader
    train_loader, val_loader, test_loader, num_classes = load_data()
    mnist_train_dataset = train_loader.dataset

    # Real images dataset (thư mục con là chữ cái)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    real_dataset = MappedImageFolder('data/real_images_processed', transform=transform)

    # Kết hợp dataset
    combined_dataset = ConcatDataset([mnist_train_dataset, real_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes, device)
    # Có thể load lại trọng số cũ nếu muốn transfer learning
    if os.path.exists('models/best_model.pth'):
        model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    num_epochs = 10

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        bar = tqdm(combined_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            bar.set_postfix({'loss': running_loss/(bar.n+1), 'acc': 100.*correct/total})
        print(f'Epoch {epoch+1}: Loss={running_loss/len(combined_loader):.4f}, Acc={100.*correct/total:.2f}%')
        # Lưu model tốt nhất
        torch.save(model.state_dict(), 'models/best_model_finetune.pth')

    print('Fine-tune xong! Model lưu tại models/best_model_finetune.pth')

if __name__ == '__main__':
    main() 
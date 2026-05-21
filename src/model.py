import torch
import torch.nn as nn
import torch.nn.functional as F

class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 28, 28)
        
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (batch_size, 32, 14, 14)
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (batch_size, 64, 7, 7)
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (batch_size, 128, 3, 3)
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_model(num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Tạo và trả về mô hình CNN
    """
    model = SignLanguageCNN(num_classes)
    model = model.to(device)
    return model

if __name__ == "__main__":
    # Test model
    model = get_model(24)
    x = torch.randn(32, 1, 28, 28)  # Batch size of 32
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 
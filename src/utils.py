import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_model
from data_loader import load_data

def predict_image(image_path, model_path='models/best_model.pth'):
    # Lấy số lớp từ data
    _, _, _, num_classes = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
    return predicted.item()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python src/utils.py <image_path>')
    else:
        label = predict_image(sys.argv[1])
        print(f'Predicted label: {label}') 
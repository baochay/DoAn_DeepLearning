import torch
from data_loader import load_data
from model import get_model

def test_model():
    _, _, test_loader, num_classes = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes, device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    test_model() 
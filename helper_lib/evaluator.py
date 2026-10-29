
import torch


def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            total_loss += criterion(predictions, labels).item() * images.size(0)
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            samples += images.size(0)
    return total_loss / samples, correct / samples

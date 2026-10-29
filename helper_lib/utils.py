
import torch


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

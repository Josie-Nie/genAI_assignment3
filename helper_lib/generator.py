"""Generate sample images from the latent space."""
import torch
from torchvision.utils import make_grid


def generate_samples(model, device='cpu', num_samples=10):
    """Return a grid tensor of generated images with values in [0, 1]."""
    if not 1 <= num_samples <= 100:
        raise ValueError('num_samples must be between 1 and 100')
    model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.z_dim, device=device)
        images = model(z)
    return make_grid((images + 1) / 2, nrow=min(10, num_samples))

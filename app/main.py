"""FastAPI inference endpoint for a GAN trained on MNIST."""
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from PIL import Image

from helper_lib.model import get_model
from helper_lib.utils import load_model

app = FastAPI(title='Assignment 3 - MNIST GAN')
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'mnist_gan_generator.pth'
_generator = None


def get_generator():
    global _generator
    if _generator is None:
        if not WEIGHTS_PATH.is_file():
            raise HTTPException(
                status_code=503,
                detail='Trained generator weights are missing. Run python train.py first.',
            )
        model = get_model('GENERATOR')
        try:
            _generator = load_model(model, WEIGHTS_PATH, device='cpu')
        except (RuntimeError, ValueError, OSError) as exc:
            raise HTTPException(status_code=503, detail='Cannot load trained generator weights.') from exc
    return _generator


@app.get('/')
def home():
    return {'message': 'MNIST GAN API', 'generate': '/generate?num_samples=10'}


@app.get('/generate')
def generate(num_samples: int = Query(default=10, ge=1, le=100)):
    """Return a PNG grid of images generated from random noise."""
    generator = get_generator()
    with torch.no_grad():
        noise = torch.randn(num_samples, generator.z_dim)
        images = generator(noise)
    # Create a row-based image grid without training in the API.
    from torchvision.utils import make_grid
    grid = make_grid((images + 1) / 2, nrow=min(num_samples, 10))
    pixels = (grid.mul(255).clamp(0, 255).byte()
              .permute(1, 2, 0).cpu().numpy())
    image = Image.fromarray(pixels, mode='RGB')
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return Response(content=buffer.getvalue(), media_type='image/png')

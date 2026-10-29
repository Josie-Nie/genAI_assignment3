"""Train the Assignment 3 GAN on MNIST and save its generator weights."""
import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torchvision.utils import save_image

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples
from helper_lib.utils import save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0002)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size and lr must be positive')
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on {device}')

    loader = get_data_loader('./data', batch_size=args.batch_size)
    generator, discriminator = get_model('GAN')
    criterion = nn.BCELoss()
    # Adam is also used in Module 4; a standard optimizer for BCE-based GANs.
    opt_g = optim.Adam(generator.parameters(), lr=args.lr)
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr)
    generator, discriminator = train_gan(
        (generator, discriminator), loader, criterion, (opt_g, opt_d),
        device=device, epochs=args.epochs
    )

    out = Path('models')
    out.mkdir(exist_ok=True)
    save_model(generator, out / 'mnist_gan_generator.pth')
    save_image(generate_samples(generator, device, num_samples=64), out / 'generated_samples.png')
    print('Saved models/mnist_gan_generator.pth and models/generated_samples.png')


if __name__ == '__main__':
    main()

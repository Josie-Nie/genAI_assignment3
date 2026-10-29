# Assignment 3 – MNIST GAN

## Overview

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate handwritten digits from the MNIST dataset. The trained generator is integrated into a FastAPI application for image generation.

## Model Architecture

- **Generator:** Linear layer, ConvTranspose2d, BatchNorm2d, ReLU, and Tanh.
- **Discriminator:** Conv2d, BatchNorm2d, LeakyReLU, and Linear.
- **Loss Function:** Binary Cross-Entropy (BCE).
- **Dataset:** MNIST handwritten digits (28×28 grayscale images).

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --epochs 20 --batch-size 64
```

The trained generator is saved to `models/mnist_gan_generator.pth`.

## Run FastAPI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API documentation: http://localhost:8000/docs

Generate handwritten digits:

```text
http://localhost:8000/generate?num_samples=10
```

## Docker Deployment

Build the Docker image:

```bash
docker build -t assignment3-gan .
```

Run the container:

```bash
docker run --rm -p 8000:8000 assignment3-gan
```

Then access the API at http://localhost:8000/docs.

## Technologies

Python, PyTorch, Torchvision, FastAPI, Uvicorn, Docker.

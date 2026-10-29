"""Shape and functionality checks for the exact specified architecture."""
import torch
from torch import nn
from helper_lib.model import Generator, Discriminator, get_model


def test_specified_layers():
    g, d = get_model('GAN')
    assert g.fc.in_features == 100 and g.fc.out_features == 7 * 7 * 128
    assert (g.deconv1.in_channels, g.deconv1.out_channels) == (128, 64)
    assert (g.deconv2.in_channels, g.deconv2.out_channels) == (64, 1)
    assert isinstance(g.bn1, nn.BatchNorm2d)
    assert isinstance(g.tanh, nn.Tanh)
    assert (d.conv1.in_channels, d.conv1.out_channels) == (1, 64)
    assert (d.conv2.in_channels, d.conv2.out_channels) == (64, 128)
    assert d.fc.in_features == 128 * 7 * 7 and d.fc.out_features == 1
    for layer in [g.deconv1, g.deconv2, d.conv1, d.conv2]:
        assert (layer.kernel_size, layer.stride, layer.padding) == ((4, 4), (2, 2), (1, 1))


def test_shapes_and_probabilities():
    g = Generator()
    d = Discriminator()
    with torch.no_grad():
        images = g(torch.randn(3, 100))
        probabilities = d(images)
    assert images.shape == (3, 1, 28, 28)
    assert images.min() >= -1 and images.max() <= 1
    assert probabilities.shape == (3, 1)
    assert probabilities.min() >= 0 and probabilities.max() <= 1

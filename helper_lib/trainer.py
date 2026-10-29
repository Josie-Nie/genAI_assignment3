"""Standard GAN training: alternate Discriminator and Generator updates."""
import torch


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """Train the (generator, discriminator) model pair.

    criterion: nn.BCELoss(); optimizer: (optimizer_G, optimizer_D).
    Returns the trained pair, matching the Module 6 class-activity interface.
    """
    generator, discriminator = model
    optimizer_g, optimizer_d = optimizer
    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        total_d = 0.0
        total_g = 0.0
        for real, _ in data_loader:
            real = real.to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Step 1: train Discriminator on real and fake images.
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size, generator.z_dim, device=device)
            fake = generator(noise)
            loss_real = criterion(discriminator(real), real_labels)
            loss_fake = criterion(discriminator(fake.detach()), fake_labels)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Step 2: train Generator to make fake images appear real.
            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, generator.z_dim, device=device)
            fake = generator(noise)
            loss_g = criterion(discriminator(fake), real_labels)
            loss_g.backward()
            optimizer_g.step()

            total_d += loss_d.item()
            total_g += loss_g.item()

        batches = len(data_loader)
        print(f'Epoch {epoch + 1}/{epochs} | D loss: {total_d/batches:.4f} | G loss: {total_g/batches:.4f}')
    return generator, discriminator

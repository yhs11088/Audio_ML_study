# source : https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

import matplotlib.pyplot as plt
import torch

def generate_from_truth(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for (x, _) in test_loader:
            B, C, H, W = x.shape
            x = x.view(B*C, H*W)
            x = x.to(device)

            x_hat, _, _ = model(x)

            return x_hat


def generate_from_noise(decoder, noise_shape, device):
    noise = torch.randn(*noise_shape).to(device)
    generated_images = decoder(noise)
    return generated_images

def show_image(x, idx, H = 28, W = 28):
    x = x.view(-1, H, W)

    fig = plt.figure()
    plt.imshow(x[idx].detach().numpy())
    plt.show()
    
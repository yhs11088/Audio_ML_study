# source : https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random

import torch
from torch.optim import Adam
from torchvision.utils import save_image

from data import load_dataset
from vae import Encoder, Decoder, VAE
from loss import VAEloss
from train import train_VAE
from test_generate import generate_from_noise, generate_from_truth, show_image


if __name__ == "__main__":

    dataset_path = "./datasets/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 784
    hidden_dim = 400
    latent_dim = 200

    epochs = 30
    batch_size = 64
    lr = 1e-3

    # load (or download) dataset
    train_loader, test_loader = load_dataset(dataset_path, batch_size)

    # VAE model
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)
    model = VAE(encoder, decoder).to(device)

    # path where trained model parameters are (or will be) saved
    trained_params_path = "./VAE.params.MNIST.pt"

    if not os.path.exists(trained_params_path):

        # optimizer
        optimizer = Adam(model.parameters(), lr = lr)

        # train
        train_VAE(
            model, optimizer, VAEloss, train_loader, device,
            batch_size, input_dim, epochs
        )

        # save model
        torch.save(model.state_dict(), trained_params_path)

    else:

        model.load_state_dict(torch.load(trained_params_path))

        # 1) generate image from test dataset
        print("1. Genrate images from test dataset")
        generated_images = generate_from_truth(model, test_loader, device)

        for _ in range(5):
            idx = random.choice(range(len(generated_images)))
            show_image(generated_images, idx)

        # 2) generate image from noise & save
        print("2. Genrate images from random noise")
        generated_images = generate_from_noise(decoder, (batch_size, latent_dim), device)
        save_image(generated_images.view(batch_size, 1, 28, 28), "generated_sample.png")
        
        for _ in range(5):
            idx = random.choice(range(len(generated_images)))
            show_image(generated_images, idx)





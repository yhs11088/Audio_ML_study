import sys
import torch
from torch.utils import data
from torch import optim

from cpc import CPC
from datareader import RawDataset
from training import train


def main():

    # Hyperparameters
    common_path = r"C:\Users\hyunsuk yoon\Desktop\GITHUB_PROJECTS\LibriSpeech\dev-clean" # <-- r"..." does not interpret escape character

    timestep = 12
    batch_size = 8
    audio_window = 20480
    output_dim = 256

    epochs = 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Train dataset
    train_dataset = RawDataset(common_path, audio_window)

    # Train data loader
    # NOTE. drop_last = True to drop last non-full batch
    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)

    # model
    model = CPC(
        audio_dim = 1, Z_dim = 512, C_dim = 256, pred_timestep = 12,
        kernels = [10, 8, 4, 4, 4], strides = [5, 4, 2, 2, 2], paddings = [3, 2, 1, 1, 1]
    )

    # optimizer
    optimizer = optim.Adam(
        params = filter(lambda p: p.requires_grad, model.parameters()),
        betas = (0.9, 0.98),
        eps = 1e-9,
        weight_decay = 1e-4,
        amsgrad = True
    )

    for epoch in range(1, epochs+1):
        train(model, device, train_loader, optimizer, epoch, batch_size)


if __name__ == "__main__":
    main()
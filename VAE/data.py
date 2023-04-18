# source : https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_dataset(dataset_path = "./", batch_size = 32):

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {
        "num_workers" : 1,
        "pin_memory" : True
    }

    train_dataset = MNIST(dataset_path, transform = mnist_transform, train = True, download = True)
    test_dataset = MNIST(dataset_path, transform = mnist_transform, train = False, download = True)

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, **kwargs)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True, **kwargs)

    return train_loader, test_loader
import torch
import torch.nn as nn

class Autoregressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers = 1):
        super(Autoregressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.autoregressor = nn.GRU(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True
        )

    def forward(self, Z, h):
        # Z : shape = (batch, Z_length, input_dim) = (8, 128, 512)
        # h : shape = (RNN_layers, batch, hidden_dim) = (1, 8, 256)

        # flatten RNN parameters (i.e. make their memory contiguous to each other...?)
        # - reference : https://www.facebook.com/groups/PyTorchKR/posts/1390778517728492/
        self.autoregressor.flatten_parameters()

        C, h = self.autoregressor(Z, h)
        return C, h
        
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernels, strides, paddings):
        super(Encoder, self).__init__()

        self.seq = nn.Sequential()
        for idx, (k, s, p) in enumerate(zip(kernels, strides, paddings)):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels = (input_dim if idx == 0 else hidden_dim),
                    out_channels = hidden_dim,
                    kernel_size = k,
                    stride = s,
                    padding = p,
                    bias = False
                ),
                nn.BatchNorm1d(num_features = hidden_dim),
                nn.ReLU(inplace = True)
            )
            self.seq.add_module(
                name = f"layer-{idx+1}",
                module = block
            )

    def forward(self, X):
        # X : shape = (batch, input_dim, audio_length) = (8, 1, 20480)
        return self.seq(X) # shape = (batch, hidden_dim, Z_length) = (8, 512, 128)
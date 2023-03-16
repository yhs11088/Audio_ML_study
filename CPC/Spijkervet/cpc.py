import torch
import torch.nn as nn

from encoder import Encoder
from autoregressor import Autoregressor
from infonce import InfoNCELoss

class CPC(nn.Module):
    def __init__(self, audio_dim, Z_dim, C_dim, pred_timestep, 
                 kernels, strides, paddings):
        super(CPC, self).__init__()

        self.encoder = Encoder(
            input_dim = audio_dim,
            hidden_dim = Z_dim,
            kernels = kernels,
            strides = strides,
            paddings = paddings
        )
        self.autoregressor = Autoregressor(
            input_dim = Z_dim,
            hidden_dim = C_dim,
            num_layers = 1
        )
        self.preds = nn.ModuleList([
            nn.Linear(
                in_features = C_dim, 
                out_features = Z_dim,
                bias = False                    # **** Masked when comparing with jefflai108
            ) \
            for _ in range(pred_timestep)
        ])

        self.loss = InfoNCELoss
    
    #def forward(self, X, h, t_samples):        # ***** When comparing with jeffali108
    def forward(self, X, h):
        '''
        X : shape = (batch, audio_dim, audio_length) = (8, 1, 20480)
        h : shape = (RNN_layers, batch, C_dim) = (1, 8, 256)
        '''

        # 1. Encoder
        Z = self.encoder(X) # shape = (batch, Z_dim, Z_length) = (8, 512, 128)

        # 2. Transpose
        Z = torch.transpose(Z, 1, 2) # shape = (batch, Z_length, Z_dim) = (8, 128, 512)

        # 3. Autoregressive
        C, h = self.autoregressor(Z, h) # C : shape = (batch, Z_length, C_dim) = (8, 128, 256)

        # 4. use all c_t to predict z_{t+k} & calculate InfoNCE loss
        infonce = 0.
        for k in range(1, len(self.preds)+1):

            # 4-1. Predict
            Wc = self.preds[k-1](C) # shape = (batch, Z_length, Z_dim) = (8, 128, 512)

            # 4-2. Calculate mean InfoNCE loss for k-timestep future prediction
            #infonce_k = self.loss(Wc[:,t_samples,:], Z[:,t_samples + k,:])         # ***** When comparing with jefflai108
            infonce_k = self.loss(Wc[:,:-k,:], Z[:,k:,:])
            

            # 4-3. Add infonce_k to infonce
            infonce += infonce_k

        # 4-4. Calculate total mean InfoNCE loss
        infonce /= len(self.preds)

        return infonce, h

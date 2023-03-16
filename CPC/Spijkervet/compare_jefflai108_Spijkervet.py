# ***** NOTE *****
# To compare CPC from jefflai108 & Spijkervet,
# you should modify some lines with comments like "# ***** When comparing with ...""
# in [../jefflai108/model.py] and [./cpc.py]
# ****************

import sys
import torch

from encoder import Encoder
from autoregressor import Autoregressor
from infonce import InfoNCELoss
from copy_CPC_parameters import kakashi
from cpc import CPC as my_CPC

sys.path.append(r"C:\Users\hyunsuk yoon\Desktop\GITHUB_PROJECTS\myCPC\jefflai108")
from model import CPC


def compare_InfoNCE_loss():

    X = torch.randn(8, 1, 20480)
    h = torch.randn(1, 8, 256)

    # answer
    model = CPC(timestep = 12, batch_size = 8, seq_len = 20480)
    _, nce, _, Z, t_samples = model(X, h)

    # test InfoNCE loss
    C, _ = model.predict(X, h)

    c_t = C[:,t_samples,:].squeeze() # shape = (8, 256)
    z_k = Z[:,t_samples+1:t_samples+12+1,:] # shape = (8, 12, 512)
    Wc = torch.empty(z_k.shape)
    for k in range(12):
        Wc[:,k,:] = model.Wk[k](c_t).squeeze()

    nce2 = InfoNCELoss(Wc, z_k)

    print(nce)
    print(nce2)


def compare_CPC():

    X = torch.randn(8, 1, 20480)
    h = torch.zeros(1, 8, 256)

    model1 = CPC(timestep = 12, batch_size = 8, seq_len = 20480)
    model2 = my_CPC(
        audio_dim = 1, Z_dim = 512, C_dim = 256, pred_timestep = 12,
        kernels = [10, 8, 4, 4, 4], strides = [5, 4, 2, 2, 2], paddings = [3, 2, 1, 1, 1]
    )

    # copy CPC1's parameter to CPC2
    kakashi(model1, model2)

    # compare forward
    acc, nce1, _, Z, t_samples = model1(X, h)
    nce2, _ = model2(X, h, t_samples)

    print(t_samples)
    print(nce1)
    print(nce2)

    # compare backward
    nce1.backward()
    nce2.backward()

    for i, ((k1, v1), (k2, v2)) in enumerate(zip(model1.named_parameters(), model2.named_parameters())):
        print(i, k1, v1.grad.mean().item(), v2.grad.mean().item())


if __name__ == "__main__":
    compare_InfoNCE_loss(); print()
    compare_CPC(); print()
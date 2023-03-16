# ***** NOTE *****
# To compare CPC from jefflai108 & Spijkervet,
# you should modify some lines with comments like "# ***** When comparing with ...""
# in [../jefflai108/model.py] and [./cpc.py]
# ****************

import sys
import torch

from copy_CPC_parameters import kakashi

sys.path.append(r"C:\Users\hyunsuk yoon\Desktop\Audio_ML_study_COMMIT\CPC")

# jefflai108 style CPC
from jefflai108.model import CPC1

# Spijkervet style CPC
from Spijkervet.infonce import InfoNCELoss
from Spijkervet.cpc import CPC as CPC2


def compare_InfoNCE_loss():

    X = torch.randn(8, 1, 20480)
    h = torch.randn(1, 8, 256)

    # answer (jefflai108)
    cpc1 = CPC1(timestep = 12, batch_size = 8, seq_len = 20480)
    _, nce, _, Z, t_samples = cpc1(X, h)

    # test Spijkervet's InfoNCE loss
    C, _ = cpc1.predict(X, h)

    c_t = C[:,t_samples,:].squeeze()        # shape = (8, 256)
    z_k = Z[:,t_samples+1:t_samples+12+1,:] # shape = (8, 12, 512)
    Wc = torch.empty(z_k.shape)
    for k in range(12):
        Wc[:,k,:] = cpc1.Wk[k](c_t).squeeze()

    nce2 = InfoNCELoss(Wc, z_k)

    print(nce)
    print(nce2)


def compare_CPC():

    X = torch.randn(8, 1, 20480)
    h = torch.zeros(1, 8, 256)

    # jefflai108 CPC
    cpc1 = CPC1(timestep = 12, batch_size = 8, seq_len = 20480)
    # Spijkervet CPC
    cpc2 = CPC2(
        audio_dim = 1, Z_dim = 512, C_dim = 256, pred_timestep = 12,
        kernels = [10, 8, 4, 4, 4], strides = [5, 4, 2, 2, 2], paddings = [3, 2, 1, 1, 1]
    )

    # copy CPC1's parameter to CPC2
    kakashi(cpc1, cpc2)

    # compare forward
    acc, nce1, _, Z, t_samples = cpc1(X, h)
    nce2, _ = cpc2(X, h, t_samples)

    print(t_samples)
    print(nce1)
    print(nce2)

    # compare backward
    nce1.backward()
    nce2.backward()

    for i, ((k1, v1), (k2, v2)) in enumerate(zip(cpc1.named_parameters(), cpc2.named_parameters())):
        print(i, k1, v1.grad.mean().item(), v2.grad.mean().item())


if __name__ == "__main__":
    compare_InfoNCE_loss(); print()
    compare_CPC(); print()
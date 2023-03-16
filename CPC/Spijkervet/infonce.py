import torch
import torch.nn as nn
import torch.nn.functional as F

def InfoNCELoss(Wc, Z):
    '''
    For k : fixed,
    Wc : k-timestep future predictions
         - shape = (batch, Z_length, Z_dim) = (8, 128, 512)
    Z : k-timestep future true representations 
        - shape = (batch, Z_length, Z_dim) = (8, 128, 512)

    for 0 <= i, j <= batch-1,
        for 0 <= t <= Z_lenght-1,
            (Wc[i,t,:], Z[j,t,:]) = (if i == j) positive sample
                                    (otherwise) negative sample
    '''
    
    Z_length = Wc.shape[1]

    # transpose to enable torch.bmm
    Wc = torch.transpose(Wc, 0, 1)  # shape = (Z_length, batch, Z_dim) = (128 - k, 8, 512)
    Z = torch.permute(Z, [1, 2, 0]) # shape = (Z_length, Z_dim, batch) = (128 - k, 512, 8)

    # dot product using torch.bmm
    # -> sim[t,i,j] = dot product between Wc[t,i,:] and Z[t,:,j]
    #               = (if i == j) similarity of positive sample
    #                 (otherwise) similarity of negative sample
    sim = torch.bmm(Wc, Z)   # shape = (Z_length, batch, batch) = (128 - k, 8, 8)

    # log softmax
    # -> lsoft_sim[t,i,i] = log softmax of positive sample similarity
    lsoft_sim = F.log_softmax(sim, dim = -1) # shape = (Z_length, batch, batch) = (128 - k, 8, 8)

    # batch diagonal of log softmax
    # i.e. diag_lsoft_sim[t,i] = lsoft_sim[t,i,i] = log softmax of positive sample similarity
    lsoft_sim_pos = torch.diagonal(lsoft_sim, dim1 = -2, dim2 = -1) # shape = (Z_length, batch) = (128 - k, 8)

    # mean InfoNCE loss for k-th future prediction
    infonce = -1. * torch.mean(lsoft_sim_pos) # shape = (,)

    return infonce

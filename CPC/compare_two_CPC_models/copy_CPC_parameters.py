import torch
import torch.nn as nn

def kakashi(CPC1, CPC2):
    '''
    copy model parameters of CPC1 to CPC2

    CPC1 : CPC from github of jeffali108
    CPC2 : CPC from github of Spijkervet
    '''

    # 1. Encoder
    # -> params1 = dictionary of CPC1's Encoder parameters whose keys match that of CPC2's Encoder parameters
    params1 = {}
    seq_idx = 1 # corresponding CPC2's sequential layer index
    for k, v in CPC1.encoder.state_dict().items():
        layer_idx, name = k.split(".")
        key = f"seq.layer-{seq_idx}.{0 if int(layer_idx) % 3 == 0 else 1}.{name}"
        params1[key] = v
        if name == "num_batches_tracked":
            seq_idx += 1
    CPC2.encoder.load_state_dict(params1)

    # 2. Autoregressor
    # -> params1 = dictionary of CPC1's GRU parameters whose keys match that of CPC2's GRU parameters
    params1 = {f"autoregressor.{k}":v for (k, v) in CPC1.gru.state_dict().items()} 
    CPC2.autoregressor.load_state_dict(params1)
    
    # 3. Linear predictions
    CPC2.preds.load_state_dict(CPC1.Wk.state_dict())
    

    
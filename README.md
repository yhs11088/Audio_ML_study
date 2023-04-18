# Audio_ML_study

Repository for simple audio-related machine learning models

<br/>
<br/>

## CPC

Directory for simple Contrastive Predictive Coding (CPC) models to learn representations of 1D audio signal
* reference paper : Oord, A. V. D., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.

<br/>

### (1) jefflai108

Directory for simple CPC models (reference : jefflai108's github (https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch))

**NOTE**
  * Encoder, autoregressive layers & InfoNCE loss are all defined in model.py
  * However, InfoNCE loss is calculated using only 1 time sample of representation Z

<br/>

### (2) Spijkervet

Directory for simple CPC models (reference : Spijkervet's github (https://github.com/Spijkervet/contrastive-predictive-coding))

**NOTE**
  * Encoder, autoregressive layers & InfoNCE loss are defined in separate files (encoder.py, autoregressor.py, infonce.py)
  * InfoNCE loss is calculated using all time steps of representation Z
  * Spijkervet's infonce.py contains complex positive & negative sampling.
    
    In my infonce.py, these operations are omitted for simplicity & InfoNCE loss is calculated in a way similar to that of jefflai108

<br/>

### (3) compare_two_CPC_models

Directory for simple comparison of jefflai108 & Spijkervet's CPC models


<br/>
<br/>


## VAE

Directory for simple Variational Auto-Encoder (VAE) model to learn representation of MNIST image and generation of new MNIST images
* reference paper : 

<br/>

### (1) Jackson-Kang

Directory for simple VAE models (refernece : Jackson-Kang's github)
(https://github.com/Jackson-Kang/Pytorch-VAE-tutorial)

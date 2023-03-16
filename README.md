# Audio_ML_study

Repository for simple audio-related machine learning models

## CPC

Directory for Simple Contrastive Predictive Coding (CPC) models

(1) **jefflai108**

Directory for Simple CPC models (reference : jefflai108's github (https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch))

Note.
  * Encoder, autoregressive layers & InfoNCE loss are all defined in model.py
  * However, InfoNCE loss is calculated using only 1 time sample of representation Z

(2) **Spijkervet**

Directory for Simple CPC models (reference : Spijkervet's github (https://github.com/Spijkervet/contrastive-predictive-coding))

Note.
  * Encoder, autoregressive layers & InfoNCE loss are defined in separate files (encoder.py, autoregressor.py, infonce.py)
  * InfoNCE loss is calcualted using all time steps of representation Z
  * Spijkervet's infonce.py contains complex positive & negative sampling - these are omitted in my infonce.py for simplicity

(3) **compare_two_CPC_models**

Directory for simple comparison of jefflai108 & Spijkervet's CPC models



# Audio_ML_study

Repository for simple audio-related machine learning models

<br/>
<br/>

## CPC

Directory for simple Contrastive Predictive Coding (CPC) models to learn representations of 1D audio signal
* reference paper : Oord, A. V. D., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. [[link](https://arxiv.org/abs/1807.03748)]

<br/>

### (1) jefflai108

Directory for simple CPC models (reference : [jefflai108's github](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch))

**NOTE**
  * Encoder, autoregressive layers & InfoNCE loss are all defined in `model.py`
  * However, InfoNCE loss is calculated using only 1 time sample of representation Z

<br/>

### (2) Spijkervet

Directory for simple CPC models (reference : [Spijkervet's github](https://github.com/Spijkervet/contrastive-predictive-coding))

**NOTE**
  * Encoder, autoregressive layers & InfoNCE loss are defined in separate files (`encoder.py`, `autoregressor.py`, `infonce.py`)
  * InfoNCE loss is calculated using all time steps of representation Z
  * Spijkervet's infonce.py contains complex positive & negative sampling.
    
    In my `infonce.py`, these operations are omitted for simplicity & InfoNCE loss is calculated in a way similar to that of jefflai108

<br/>

### (3) compare_two_CPC_models

Directory for simple comparison of jefflai108 & Spijkervet's CPC models


<br/>
<br/>


## VAE

Directory for simple Variational Auto-Encoder (VAE) model to learn representation of MNIST image and generation of new MNIST images
* reference paper : Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. [[link](https://arxiv.org/abs/1312.6114)]

<br/>

### (1) Jackson-Kang

Directory for simple VAE models (refernece : [Jackson-Kang's github](https://github.com/Jackson-Kang/Pytorch-VAE-tutorial))


<br/>
<br/>


## PPO

Directory for simple discrete Proximal Policy Optimization (PPO) model 
* reference paper : Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. [[link](https://arxiv.org/abs/1707.06347)]

<br/>

### (1) seungeunrho

Directory for simple discrete PPO model (reference : [seungeunrho's github](https://github.com/seungeunrho/minimalRL))

**NOTE**
  * seungeunrho's github contains other reinforcement learning (RL) algorithms
  * In this github directory, however, only discrete PPO model (i.e. PPO that works for environment with discrete action space) is implemented
  * Training result figures are saved in directory `./figures`
  * Pre-trained model parameters are saved in directory `./params`
  
  
<br/>
<br/>


## my_gym_env

Directory for running A2C/PPO algorihtms to simple custom gym environment
* main reference for custom gym environment : [gym documentation](https://www.gymlibrary.dev/content/environment_creation/)


<br/>

* directory structure
~~~bash
my_gym_env
├── veca
│   └── envs
│   │   ├── __init__.py
│   │   ├── playground_v0.py
│   │   └── playground_v1.py 
│   └── __init__.py
├── setup.py
├── main.py
├── simple_main.py
├── A2C_Playground-v0.zip
├── PPO_Playground-v0.zip
└── run.sh
~~~ 

<br/>


**NOTE**
  * Stable-baselines3 models seem to be incompatible with `gym==0.26.0`. Thus, to use stable-baselines3, `gym==0.21.0` is recommended
  * `./veca/envs/playground_v0.py` contains "PlaygroundEnv_v0" class, which is a simple custom environment and assumes `gym==0.21.0`
  * `./veca/envs/plyagounrd_v1.py` contains "PlaygroundEnv_v1" class, which is almost identical to "PlaygroundEnv_v0" except that it assumes `gym==0.26.0`
  * Pretrained A2C/PPO models are saved as zip files
  * You may
    
    (1) first install/register the custom environment to gym
    ~~~bash
    cd my_gym_env
    pip install -e .
    ~~~
    (2) run main.py 
    ~~~bash
    python main.py
    ~~~
    
    or
    
    (1) run simple_main.py
    ~~~bash
    python simple_main.py
    ~~~
    to run the custom environment & train/test RL algorihtms

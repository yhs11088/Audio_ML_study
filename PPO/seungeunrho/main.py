'''
Training & testing simple discrete PPO model
- source : https://github.com/seungeunrho/minimalRL
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.spaces import Discrete

import torch
from torch.distributions import Categorical

from ppo_discrete import PPO


###########################################################
# Main
###########################################################
def main(
        env_name = "CartPole-v1", 
        params_path = "./PPO.CartPole_v1.pt",
        figure_path = "./Train_history.CartPole-v1.png",
        train = True, 
        max_episode = 500,
        print_episode = 10,
        T_horizon = 20
    ):


    #--------------------
    # Environment
    #--------------------
    env = gym.make(env_name, render_mode = None if train else "human")
    s_dim = env.observation_space.shape[0]
    action_is_discrete = isinstance(env.action_space, Discrete)
    if action_is_discrete:
        a_dim = env.action_space.n

    #--------------------
    # PPO model
    #--------------------
    model = PPO(s_dim, a_dim, action_is_discrete)

    #--------------------
    # If testing, load pre-trained model
    #--------------------
    if not train and os.path.exists(params_path):
        model.load_state_dict(torch.load(params_path))
        model.eval()
    
    #--------------------
    # Train or test
    #--------------------
    logger = {"avg_score" : []}
    score = 0.0     # sum of scores at each timestep
    
    for n_epi in range(max_episode):
        s, _ = env.reset()
        done = False

        while not done:
            for t in range(T_horizon):

                # Sample action based on model's policy
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()

                # Go to new state
                s_prime, r, terminated, truncated, info = env.step(a)
                done = (terminated or truncated)

                # If training, put data in model
                if train:
                    model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                
                # Update state & score
                s = s_prime
                score += r

                if done:
                    break

        # If training, update model
        if train:
            model.train_net()

        # Print & record intermediate train/test results
        if n_epi % print_episode == 0 and n_epi != 0:
            print(f"Epsiode {n_epi}, avg score : {score / print_episode:.1f}")
            logger['avg_score'].append(score / print_episode)
            score = 0.0

    #--------------------
    # If training, save trained model & train history
    #--------------------
    if train:

        # Save model parameters
        torch.save(model.state_dict(), params_path)

        # Save train history figure
        x = np.arange(1, len(logger['avg_score'])+1) * print_episode
        fig = plt.figure()
        plt.plot(x, logger['avg_score'], "k-")
        plt.xlim(0, max(x) + print_episode)
        plt.grid()
        plt.xlabel("Episode", fontsize = 11)
        plt.title("Average Reward", fontsize = 14)
        plt.savefig(figure_path)

    #--------------------
    # Close environment
    #--------------------
    env.close()



###########################################################
# Main
###########################################################
if __name__ == "__main__":

    #--------------------
    # Hyperparmaeters
    #--------------------
    env_name = "Acrobot-v1"
    params_path = f"./params/PPO.{env_name}.pt"
    figure_path = f"./figures/Train_history.{env_name}.png"

    #--------------------
    # Train
    #--------------------
    print(f" Train ".center(20, "="))
    main(env_name, params_path, figure_path, train = True, max_episode = 500)
    
    #--------------------
    # Test
    #--------------------
    print("\n" + f" Test ".center(20, "="))
    main(env_name, params_path, figure_path = None, train = False, max_episode = 10)
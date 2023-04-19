'''
Code for simple discrete PPO model class
- source : https://github.com/seungeunrho/minimalRL
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PPO(nn.Module):
    def __init__(
            self, s_dim, a_dim,
            action_is_discrete = True,
            gamma = 0.98, lmbda = 0.95, eps_clip = 0.1,
            K_epoch = 3, lr = 5e-4
        ):
        
        super(PPO, self).__init__()

        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.action_is_discrete = action_is_discrete

        self.data = []

        self.fc1 = nn.Linear(s_dim, 256)
        self.fc_pi = nn.Linear(256, a_dim)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def pi(self, x, softmax_dim = 0):
        if self.action_is_discrete:
            x = F.relu(self.fc1(x))
            x = self.fc_pi(x)
            prob = F.softmax(x, dim = softmax_dim)
            return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_list, a_list, r_list, s_prime_list, prob_a_list, done_list = [], [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            prob_a_list.append([prob_a])
            done_list.append([0 if done else 1])

        s = torch.tensor(s_list, dtype = torch.float)
        a = torch.tensor(a_list)
        r = torch.tensor(r_list)
        s_prime = torch.tensor(s_prime_list, dtype = torch.float)
        prob_a = torch.tensor(prob_a_list)
        done_mask = torch.tensor(done_list, dtype = torch.float)

        self.data = []

        return s, a, r, s_prime, done_mask, prob_a
    
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        #print(s.shape, a.shape, r.shape, s_prime.shape, done_mask.shape, prob_a.shape)

        for i in range(self.K_epoch):

            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.tensor(advantage_list, dtype = torch.float)
            #print(advantage.shape)

            pi = self.pi(s, softmax_dim = 1)
            if self.action_is_discrete:
                pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

'''
quick test custom environment
'''

import gym
from veca.envs.playground_v0 import PlaygroundEnv_v0
#from veca.envs.playground_v1 import PlaygroundEnv_v1

from stable_baselines3 import A2C, PPO

env = PlaygroundEnv_v0(room_width = 15, room_height = 10)

model = A2C("MlpPolicy", env, verbose = True)
#model = PPO("MlpPolicy", env, verbose = True)
model.learn(total_timesteps = 15000)

env = PlaygroundEnv_v0(render_mode = "human", room_width = 15, room_height = 10)
obs = env.reset()

print_interval = 10
cum_reward = 0
cum_count = 0

for i in range(200):
    action, _states = model.predict(obs)
    action = int(action)
    #action = env.action_space.sample()

    obs, reward, done, info = env.step(action)
    env.render()

    cum_reward += reward
    cum_count += 1
    if (i+1) % print_interval == 0:
        print(f"Iteration {i+1} : recent mean reward = {cum_reward / cum_count:.3f}")
        cum_reward = 0
        cum_count = 0

    if done:
        obs = env.reset()
        if cum_count > 0:
            print(f"Iteration {i+1} : recent mean reward = {cum_reward / cum_count:.3f}")
        cum_reward = 0
        cum_count = 0
        print("-" * 20)

env.close()
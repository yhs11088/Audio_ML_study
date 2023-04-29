'''
main reference : main reference : https://www.gymlibrary.dev/content/environment_creation/
'''

from gym.envs.registration import register

register(
    id = "veca/Playground-v0",
    entry_point = "veca.envs:PlaygroundEnv_v0",
    max_episode_steps = 300
)
register(
    id = "veca/Playground-v1",
    entry_point = "veca.envs:PlaygroundEnv_v1",
    max_episode_steps = 300
)
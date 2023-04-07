import os
import gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

import numpy as np


# training in 4 envs at a time with multiprocessing

# env = DummyVecEnv([lambda: gym.make("Ant-v4") for _ in range(8)])

env = gym.make("Ant-v4")

# Automatically normalize the input features and reward

# env = VecNormalize(env, norm_obs=True, norm_reward=True,
#                    clip_obs=10.)

# instantiate an agent
model = PPO("MlpPolicy", env, verbose=1)


# train an agent
model.learn(1000)

# save the agent
model.save("/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/TRAINED_agents/NEW_ant_right_1000steps")

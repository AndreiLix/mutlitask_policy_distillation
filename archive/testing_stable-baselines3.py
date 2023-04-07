import os
import gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

env = DummyVecEnv([lambda: gym.make("Ant-v4") for _ in range(4)])
# Automatically normalize the input features and reward

env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)



model = PPO("MlpPolicy", env, verbose = 1)
model.learn(total_timesteps=10000)

obs = env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    # action = env.action_space.sample()
    obs, rewards, dones, info = env.step(action)
    env.render()
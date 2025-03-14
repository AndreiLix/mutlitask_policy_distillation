import gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import gym
import numpy as np

import torch as th
from torch import nn

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


from WORKING_folder_thesis.archive.bad_WRAPPED_train import AntDirectionTaskWrapper, CustomCombinedExtractor



def make_env(rank, seed=0):
    def _init():
        env = gym.make("Ant-v4")
        env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[1,0]]) # Single fixed goal for all environments
        #env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[1,0],[-1,0],[0,1],[0,-1]]) # Each environment samples a random cardinal direction
        #env = AntDirectionTaskWrapper(env, randomized_goal_directions=None) # Each environment samples a random direction from all possible directions. We are not going to use this for a while.

        env.reset(seed = seed + rank)
        return env
    set_random_seed(seed)
    return _init



print(__name__)
if __name__=="__main__":
    print("creating env")
    # env = SubprocVecEnv([make_env(i) for i in range(4)])
    env = gym.make("Ant-v4")
    env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[1,0]])

    # TODO: change this based on env


    policy_kwargs = dict(activation_fn=th.nn.Tanh, # Perhaps try ReLU
                            features_extractor_class=CustomCombinedExtractor,
                            features_extractor_kwargs=dict(state_embedding_mlp=[128, 128], task_embedding_mlp=[16, 32], activation=th.nn.Tanh),
                            net_arch=[dict(vf=[64, 64], pi=[64, 64])] )


    # TODO: change model path

    print("loading model")
    PATH_model = "/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/TRAINED_agents/Wrapped_right_10000steps"
    model = PPO.load( PATH_model, env = env )
    print("model loaded")

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    print(mean_reward)

    from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
    video_folder = "logs/videos/"
    video_length = 100
    env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"rvideo") 

    obs = env.reset()
    for i in range(1000):
        print(i)
        action, _states = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        obs, rewards, dones, info = env.step(action)
        #env.render()
    env.close()
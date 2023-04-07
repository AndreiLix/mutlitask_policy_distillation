
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


from WRAPPED_train import AntDirectionTaskWrapper, CustomCombinedExtractor




print(__name__)
if __name__=="__main__":
    print("creating env")
    # env = SubprocVecEnv([make_env(i) for i in range(4)])

    env = gym.make("Ant-v4")


    #TODO: select the appropriate env

    # env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[1,0]])  #-> right
    # env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[-1,0]])  #-> left
    # env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[0,1]])  #-> forward
    # env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[0,-1]])  #-> backward

    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)

    # TODO: change VecNormalize path
    env = VecNormalize.load("/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/ant_multitask/ant_multitask_vecnormalize.pkl", env)

    env.training = False
    env.norm_reward = False



    policy_kwargs = dict(activation_fn=th.nn.Tanh, # Perhaps try ReLU
                            features_extractor_class=CustomCombinedExtractor,
                            features_extractor_kwargs=dict(state_embedding_mlp=[256, 128], task_embedding_mlp=[16, 32], activation=th.nn.Tanh),
                            net_arch=[dict(vf=[128], pi=[128])] )
                         

    # TODO: change model path

    print("loading model")
    PATH_model = "/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/ant_multitask/ant_multitask_model.zip"
    model = PPO.load( PATH_model, env = env )
    print("model loaded")

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(mean_reward)

    #from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
    #video_folder = "logs/videos/"
    #video_length = 100
    #env = VecVideoRecorder(env, video_folder,
    #                   record_video_trigger=lambda x: x == 0, video_length=video_length,
    #                   name_prefix=f"rvideo") 

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        obs, rewards, dones, info = env.step(action)
        if np.any(dones):
            print("DONE", dones)
        env.render()
    env.close()

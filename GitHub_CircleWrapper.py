import os
import gym
import numpy as np
import math
import torch as th
from torch import nn

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



"""
This module contains the Ant-v4 environment adjusted for the Ant to learn to walk in a circle.

The Circle has a radius of 10 and its center is at ( x, y ) = ( 10, y )


"""


class CircleTaskWrapper(gym.Wrapper):

    """
    Notes:
    - everything related to coordinates needs to be defined as numpy arrays. this is for shorter code related to operations with positions.
    
    """

    def __init__(
        self,
        env: gym.Env,
        radius = 10,
        center = (10, 0)      # coordinates of the circle center
    ):
        
        super().__init__(env)
        self.center = np.array(center)
        self.radius = radius

        # self.randomized_goal_directions = randomized_goal_directions
        # if self.randomized_goal_directions is not None:
        #     # Normalize directions
        #     for i in range(len(self.randomized_goal_directions)):
        #         goal_norm = np.sqrt(self.randomized_goal_directions[i][0]**2+self.randomized_goal_directions[i][1]**2)
        #         if goal_norm < 1e-3:
        #             print("ERROR: goal_direction has 0 norm (",goal_norm,")")

                # self.randomized_goal_directions[i][0] /= goal_norm
                # self.randomized_goal_directions[i][1] /= goal_norm



        self.observation_space = gym.spaces.Dict(
            spaces={
                "state": self.env.observation_space
            }
        )


        from gym.envs.mujoco.ant_v4 import AntEnv
        def get_obs(self):
            return self._get_obs()
        AntEnv.get_obs = get_obs

        self.max_timesteps = 1000


        self.reset() # Sample first goal


    def step(self, action):

        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_v, y_v = xy_velocity

        # Hyperparameters
        x_pos, y_pos = xy_position_after[0], xy_position_after[1]
        d_xy = np.linalg.norm(xy_position_after, ord=2)
        d_o = 10
        x_lim = 3

        # Get reward
        reward1 = (- x_v * y_pos + y_v * x_pos) / (1 + np.abs(d_xy - d_o))

        # Get cost
        ctrl_cost = self.env.control_cost(action)

        contact_cost = self.env.contact_cost
        costs = ctrl_cost + contact_cost + np.float(np.abs(x_pos) > x_lim)

        reward = reward1 - costs
        terminated = self.env.terminated

        observation = self.env.get_obs()
        info = {
            #"circle_reward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            #"reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            #"x_velocity": x_velocity,
            #"y_velocity": y_velocity,
        }

        if self.env.render_mode == "human":
            self.env.render()

        self.num_timesteps += 1
        if self.num_timesteps >= self.max_timesteps:
            terminated = True

        # dict_obs = {"state": observation, "task":self.current_goal}
        dict_obs = {"state": observation}
       
        # return dict_obs, reward, terminated, False, info
        return dict_obs, reward, terminated, info
        # return observation, reward, terminated, False, info


    def reset(self, **kwargs):
        self.num_timesteps = 0

        observation = self.env.reset(**kwargs)

        dict_obs = {"state": observation}
        return dict_obs



class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, state_embedding_mlp=[128, 128], activation=th.nn.Tanh):
        self.state_embedding_mlp = state_embedding_mlp
        self.activation = activation
        self._features_dim = state_embedding_mlp[-1]

        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=self._features_dim)

        extractors = {}

        for key, subspace in observation_space.spaces.items():
            if key == "state":
                mlp = [observation_space['state'].shape[0]] + self.state_embedding_mlp
            layers = []
            for i in range(len(mlp)-1):
                layers.append( nn.Linear(mlp[i], mlp[i+1]) )
                layers.append( self.activation() )

            extractors[key] = nn.Sequential(*layers)

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


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




    def __init__(
        self,
        env: gym.Env,
        radius = 10,
        center = (10, 0)      # coordinates of the circle center
    ):
        
        super().__init__(env)

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

        ## From Ant-v3
        xy_position_before = self.env.get_body_com("torso")[:2].copy()
        self.env.do_simulation(action, self.env.frame_skip)
        xy_position_after = self.env.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.env.dt
        x_velocity, y_velocity = xy_velocity


        def circular_velocity_difference(current_pos, prev_pos, center_pos, radius):
            # Calculate the angle between the current position and the center of the circle
            angle = math.atan2(current_pos[1] - center_pos[1], current_pos[0] - center_pos[0])
            
            # Calculate the expected velocity of the agent moving along the circumference of the circle
            expected_velocity = [radius * math.cos(angle), radius * math.sin(angle)]
            
            # Calculate the actual velocity of the agent
            actual_velocity = [(current_pos[0] - prev_pos[0]), (current_pos[1] - prev_pos[1])]
            
            # Calculate the difference between the actual and expected velocities
            velocity_difference = [actual_velocity[0] - expected_velocity[0], actual_velocity[1] - expected_velocity[1]]
            
            magnitute_difference_vector = math.sqrt( velocity_difference[0] ** 2 + velocity_difference[1] ** 2 )

            return magnitute_difference_vector
        
        circle_reward = - circular_velocity_difference( xy_position_after, xy_position_before, self.center, self.radius )  



        ctrl_cost = self.env.control_cost(action)
        contact_cost = self.env.contact_cost


        healthy_reward = self.env.healthy_reward

        rewards = circle_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        terminated = self.env.terminated
        observation = self.env.get_obs()
        info = {
            "circle_reward": circle_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
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


    # def reset(self, **kwargs):
    #     self.num_timesteps = 0

    #     # Sample a new goal
    #     if self.randomized_goal_directions is None:
    #         # Sample a goal at random
    #         theta = np.random.random()*2*np.pi
    #         self.current_goal = [np.cos(theta), np.sin(theta)]
    #     else:
    #         if len(self.randomized_goal_directions) == 1:
    #             self.current_goal = self.randomized_goal_directions[0]
    #         else:
    #             self.current_goal = self.randomized_goal_directions[ np.random.randint(0, len(self.randomized_goal_directions)) ]

    #     observation = self.env.reset(**kwargs)

    #     dict_obs = {"state": observation, "task":self.current_goal}
    #     return dict_obs



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


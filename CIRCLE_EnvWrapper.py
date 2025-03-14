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

        ## From Ant-v3
        xy_position_before = self.env.get_body_com("torso")[:2].copy()
        self.env.do_simulation(action, self.env.frame_skip)
        xy_position_after = self.env.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.env.dt
        x_velocity, y_velocity = xy_velocity


        

        def get_proj( xy_position, center_circle, radius ):

            """
            Parameters:
            ------------

            xy_position: tupple-like shape (2,). The x, y coordinates of the Ant
            center_circle: tupple-like (2,). The x, y coordinates of the circle

            Returns:
            --------
            np array size (2,) with the x,y coordinates of the Ant's projection on the circle.
            """

            x, y = xy_position
            x_circle, y_circle = center_circle
            magnitude = math.sqrt( ( x - x_circle ) ** 2 + ( y - y_circle ) ** 2 )
            projection = (xy_position - center_circle) / magnitude * radius + center_circle

            return np.array(projection)
        
        def get_GoalPos( proj, radius, center_circle, angle_increment= 1):

            """
            A function that returns the goal position of the Ant. 
                The goal position is computed by getting the angle between the lines (center_circle, ant_proj_line) and (center_circle, positive_XAxis),
                    increasing the angle by angle_incremet, computing the sin and cos of the new angle, multiplying them by radius and adding center_circle.
                        The resulting coordinates are those of goal_pos - a step along the circle in the counterclockwise direction
        
            Parameters:
            -----------
            proj: np array size (2,): the x, y coordinates of the proj_lineection of the Ant.
            danger_zone: float between 0 and 1, default = 0.95.
                This is a fast and dirty implementation, so I have to micromanage at parts. When the Ant dets close to the intersections of the circle and the x axis, 
                    the y coordinate of goal_pos must change its sign.
            angle_incremet: float, the increment(degrees) added to the new angle used to compute goal_pos
            
            Returns:
            --------
            np array size (2,) = the x, y coordinates of the goal position

            """


            import math
            import numpy as np

            proj_line =  np.array([center_circle, proj ]).tolist()    # transforming np arrays to embedded lists, since np's array opperations perturb the algorithm
            x_axis_line = np.array([ center_circle, [ center_circle[0] + radius, center_circle[1]] ] ).tolist()

            try:
                slope1 = (x_axis_line[1][1] - x_axis_line[0][1]) / (x_axis_line[1][0] - x_axis_line[0][0])
            except ZeroDivisionError:
                slope1 = (x_axis_line[1][1] - x_axis_line[0][1]) / 1e-8

            try:
                slope2 = (proj_line[1][1] - proj_line[0][1]) / (proj_line[1][0] - proj_line[0][0])
            except ZeroDivisionError:
                slope2 = (proj_line[1][1] - proj_line[0][1]) / 1e-8

            # Calculate the angle between the two lines
            angle_rad = math.atan2(slope2 - slope1, 1 + slope1 * slope2)

            new_angle_rad = math.radians( math.degrees(angle_rad) + angle_increment )

            new_sin, new_cos = math.sin(new_angle_rad), math.cos(new_angle_rad)

            goal_pos = np.array( [new_cos * radius + center_circle[0], new_sin * radius + center_circle[1] ]) 

            return goal_pos
        

        xy_position_before = self.env.get_body_com("torso")[:2].copy()
        self.env.do_simulation(action, self.env.frame_skip)
        xy_position_after = self.env.get_body_com("torso")[:2].copy()

        # xy_velocity = (xy_position_after - xy_position_before) / self.env.dt         # redundant
        # x_velocity, y_velocity = xy_velocity

        AntPosBefore = np.array(xy_position_before)
        x_AntPos_after, y_AntPos_after = np.array(xy_position_after)

        proj = get_proj( AntPosBefore, center_circle= self.center, radius= self.radius )

        x_GoalPos, y_GoalPos = get_GoalPos( proj, radius= self.radius, center_circle= self.center, angle_increment=10 )
        
        distance_AntPosAfter_GoalPos = math.sqrt( ( x_AntPos_after - x_GoalPos) ** 2 +  (y_AntPos_after - y_GoalPos) ** 2 )

        forward_reward = - distance_AntPosAfter_GoalPos



        ctrl_cost = self.env.control_cost(action)
        contact_cost = self.env.contact_cost


        healthy_reward = self.env.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        terminated = self.env.terminated
        observation = self.env.get_obs()
        info = {
            "circle_reward": forward_reward,
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


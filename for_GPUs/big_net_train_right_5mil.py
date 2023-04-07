# https://stable-baselines3.readthedocs.io/en/master/


# needs gym 0.26.2


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


"""
NOTE: I could not test this code yet, but it should be mostly correct.
The difference between this wrapper and the older one is that this accepts options (in the constructor) for either generating 
environments with the same goal_direction, or environments that randomly select a new goal on each reset.

To perform multi-task learning in the first case, we would have to create separate lists of environments with each possible goal, 
and then concatenate them (to be used in SubprocVecEnv).
In the second case, however, we can simply create many identical wrapped environments, and let them randomize their task automatically on each reset.
"""





class AntDirectionTaskWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        randomized_goal_directions=[[1,0]] # list of [dx, dy] vectors of goal directions.
                                            # 1) If the list has a single vector, it will deterministically always
                                            #    yield the specified task (goal direction).
                                            # 2) If the list has more than one vector, the wrapper will randomly sample
                                            #    a task (goal direction) from the list on each reset.
                                            # 3) If randomized_goal_directions=None, a random vector will be sampled 
                                            #    uniformly at random from every possible direction
    ):
        super().__init__(env)

        self.randomized_goal_directions = randomized_goal_directions
        if self.randomized_goal_directions is not None:
            # Normalize directions
            for i in range(len(self.randomized_goal_directions)):
                goal_norm = np.sqrt(self.randomized_goal_directions[i][0]**2+self.randomized_goal_directions[i][1]**2)
                if goal_norm < 1e-3:
                    print("ERROR: goal_direction has 0 norm (",goal_norm,")")

                self.randomized_goal_directions[i][0] /= goal_norm
                self.randomized_goal_directions[i][1] /= goal_norm

        self.observation_space = gym.spaces.Dict(
            spaces={
                "state": self.env.observation_space,
                "task": gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
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

        ctrl_cost = self.env.control_cost(action)
        contact_cost = self.env.contact_cost

        ##forward_reward = x_velocity
        forward_reward = self.current_goal[0]*x_velocity + self.current_goal[1]*y_velocity

        healthy_reward = self.env.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        terminated = self.env.terminated
        observation = self.env.get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.env.render_mode == "human":
            self.env.render()

        self.num_timesteps += 1
        if self.num_timesteps >= self.max_timesteps:
            terminated = True

        dict_obs = {"state": observation, "task":self.current_goal}
        #return dict_obs, reward, terminated, False, info
        return dict_obs, reward, terminated, info


    def reset(self, **kwargs):
        self.num_timesteps = 0

        # Sample a new goal
        if self.randomized_goal_directions is None:
            # Sample a goal at random
            theta = np.random.random()*2*np.pi
            self.current_goal = [np.cos(theta), np.sin(theta)]
        else:
            if len(self.randomized_goal_directions) == 1:
                self.current_goal = self.randomized_goal_directions[0]
            else:
                self.current_goal = self.randomized_goal_directions[ np.random.randint(0, len(self.randomized_goal_directions)) ]

        observation = self.env.reset(**kwargs)

        dict_obs = {"state": observation, "task":self.current_goal}
        return dict_obs





# Note that sb3 will automatically concatenate the observation vectors, which we may not want. With the following code we can design separate MLPs for each dict observation

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, state_embedding_mlp=[128, 128], task_embedding_mlp=[16, 32], activation=th.nn.Tanh):
        self.state_embedding_mlp = state_embedding_mlp
        self.task_embedding_mlp = task_embedding_mlp
        self.activation = activation
        self._features_dim = state_embedding_mlp[-1]+task_embedding_mlp[-1]

        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=self._features_dim)

        extractors = {}

        for key, subspace in observation_space.spaces.items():
            if key == "state":
                mlp = [observation_space['state'].shape[0]] + self.state_embedding_mlp
            elif key == "task":
                mlp = [observation_space['task'].shape[0]] + self.task_embedding_mlp

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


if __name__=="__main__":
    print("TRAINING")
    env = SubprocVecEnv([make_env(i) for i in range(4)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., norm_obs_keys=["state"])


    # This defines a neural network as follows:  two embedding MLPs are run in parallel, one with the state as input, and the 
    # other with the task-vector as input. Their final layers are concatenated, and fed to the DRL part of the network.
    # The DRL part of the network takes the concatenated embedding as input, and attaches 2 output heads to it, each consisting of 2 layers of 64 units, followed by the appropriate output (n_action units for the policy head 'pi', and 1 output unit for the value function head 'vf'). activation_fn in the dictionary below is for the vf/pi mlps; 'activation' in features_extractor_kwargs is for the activation function of the embedding network.
    policy_kwargs = dict(activation_fn=th.nn.Tanh, # Perhaps try ReLU
                         features_extractor_class=CustomCombinedExtractor,
                         features_extractor_kwargs=dict(state_embedding_mlp=[256, 128], task_embedding_mlp=[16, 32], activation=th.nn.Tanh),
                         net_arch=[dict(vf=[128], pi=[128])] )
                         

    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=1024, batch_size=256, n_epochs=5, gamma=0.99, tensorboard_log="/home/u699081/FOLDER_thesis/checkpoints/big_net_ant_right/tensorboard/")

    print(model.policy) # If you want to see what the neural network looks like (very useful)!


    model.learn(5e6)

    model.save("/home/u699081/FOLDER_thesis/checkpoints/ant_right/big_net_ant_right_model")
    env.save("/home/u699081/FOLDER_thesis/checkpoints/ant_right/big_net_ant_right_vecnormalize.pkl")

    print(model.policy)

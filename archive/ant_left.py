import os
import gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO




# you did something shady -> check again how to make it go strictly left

# changing the reward to moving left

# ? maybe replace def reward with def step

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # forward_reward = x_velocity

        # rewarding movement left
        forward_reward = -x_velocity

        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = self.control_cost(action)

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return reward






# training in 4 envs at a time with multiprocessing



env = DummyVecEnv([lambda: gym.make("Ant-v4") for _ in range(4)])

env= RewardWrapper( env )


# Automatically normalize the input features and reward

env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)

# instantiate an agent
model = PPO("MlpPolicy", env, verbose = 1)


# train an agent
model.learn(total_timesteps=1000000)

# save the agent
model.save("TRAINED_agents/ant_left_1000000steps")

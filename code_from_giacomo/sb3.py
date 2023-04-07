# https://stable-baselines3.readthedocs.io/en/master/


import os
import gym
#import pybullet_envs

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import numpy as np

"""
General stable-baselines3 code, for reference.
"""




"""
env = gym.make("Ant-v3") # AntBulletEnv-v0

obs = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, rewards, dones, info = env.step(action)
    env.render('human')
"""



def make_env(rank, seed=0):
    def _init():
        env = gym.make("Ant-v4")
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__=="__main__":
    # env = SubprocVecEnv([make_env(i) for i in range(4)])
    env = gym.make("Ant-v4")
    #env = VecMonitor(env) # Useful for printing training information; note that we first wrap the environment with the monitor, and only after with VecNormalize. If we did the opposite, the rewards printed during training would be normalized, and not very meaningful for us.
    #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=256, gamma=0.999, tensorboard_log="tensorboard/")



    print(model.policy)



    model.learn(total_timesteps=20000)

    # log_dir = "/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/TRAINED_agents/"
    # model.save(log_dir + "ppo_ant2000")



    """
    model.learn(total_timesteps=2000)


    log_dir = "./"
    model.save(log_dir + "ppo_ant")
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)


    """


    # To demonstrate loading
    # del model, env

    # Load the saved statistics
    #env = SubprocVecEnv([lambda: gym.make("Ant-v3") for _ in range(4)])
    #env = VecNormalize.load(stats_path, env)
    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False

    # Load the agent
    #model = PPO.load(log_dir + "ppo_ant", env=env)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
    print(mean_reward, '+-', std_reward)

    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)        # action predicted 
        #action = env.action_space.sample()                          # random action
        obs, rewards, dones, info = env.step(action)
        if np.any(dones):
            print("DONE", dones)
        env.render()
    env.close()

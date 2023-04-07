import gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# TODO: change this based on env
env = gym.make("Ant-v4") 

# TODO: change model path

PATH_model = "/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/TRAINED_agents/NEW_ant_right_1000steps.zip"


model = PPO.load( PATH_model, env = env )

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(mean_reward)

obs = env.reset()

for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    # action = env.action_space.sample()
    obs, rewards, dones, info = env.step(action)
    env.render()
import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np


model = DQN('MlpPolicy', 'LunarLander-v3')


# Separate env for evaluation
eval_env = gym.make('LunarLander-v3')

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


# Train the agent
model.learn(total_timesteps=int(100))
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading

model = DQN.load("dqn_lunar")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
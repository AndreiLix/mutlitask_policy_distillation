from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import sys

sys.path.insert(0, "/home/u699081/FOLDER_thesis")  # very important - utils is in FOLDER_thesis

from utils.code_from_giacomo.multitask_distillation_SINGLE_TASK_DISTILLATION.multitask_distillation.ppd import ProximalPolicyDistillation
from utils.WRAPPED_train import AntDirectionTaskWrapper, CustomCombinedExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
import gym
import torch as th
from stable_baselines3.common.utils import set_random_seed




if __name__ == "__main__":


    def make_env(rank, seed=0):
        def _init():
            env = gym.make("Ant-v4")
            env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[0,1]]) # Single fixed goal for all environments
            #env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[1,0],[-1,0],[0,1],[0,-1]]) # Each environment samples a random cardinal direction
            #env = AntDirectionTaskWrapper(env, randomized_goal_directions=None) # Each environment samples a random direction from all possible directions. We are not going to use this for a while.

            env.reset(seed = seed + rank)
            return env
        set_random_seed(seed)
        return _init



    # train for running right

    env = SubprocVecEnv([make_env(i) for i in range(4)])
    env = VecMonitor(env)    
    
    # try commenting ths line
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., norm_obs_keys=["state"])
    

    #TODO put the right path here
    teacher_model = PPO.load('/home/u699081/FOLDER_thesis/checkpoints/ant_right/ant_right_model.zip', env=env)


    # # architecture as the multitask and single task big nets
    # policy_kwargs = dict(activation_fn=th.nn.Tanh, # Perhaps try ReLU
    #                         features_extractor_class=CustomCombinedExtractor,
    #                         features_extractor_kwargs=dict(state_embedding_mlp=[256, 128], task_embedding_mlp=[16, 32], activation=th.nn.Tanh),
    #                         net_arch=[dict(vf=[128], pi=[128])] )


    # same architecture as teacher
    policy_kwargs = dict(activation_fn=th.nn.Tanh, # Perhaps try ReLU
                         features_extractor_class=CustomCombinedExtractor,
                         features_extractor_kwargs=dict(state_embedding_mlp=[128, 128], task_embedding_mlp=[16, 32], activation=th.nn.Tanh),
                         net_arch=[dict(vf=[64, 64], pi=[64, 64])] )


    #student_model = ProximalPolicyDistillation("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs, n_steps=512, batch_size=256, n_epochs=5, learning_rate=2.5e-4, gamma=0.999, ent_coef=0.01, tensorboard_log="tensorboard/")
    student_model = ProximalPolicyDistillation("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=1024, batch_size=256, n_epochs=5, gamma=0.99, tensorboard_log="/home/u699081/FOLDER_thesis/checkpoints/big_net_ant_backward/tensorboard/")

    student_model.set_teacher(teacher_model, distill_lambda=1)

    student_model.learn(total_timesteps=20_000)

    student_model.save('/home/u699081/FOLDER_thesis/checkpoints/student_right/student_right_model')
    env.save("/home/u699081/FOLDER_thesis/checkpoints/student_right/student_right_vecnormalize.pkl")

    mean_reward, std_reward = evaluate_policy(student_model, student_model.get_env(), n_eval_episodes=10)
    print('Student reward: ', mean_reward, '+-', std_reward)



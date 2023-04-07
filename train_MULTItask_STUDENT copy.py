from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

from code_from_giacomo.multitask_distillation_UNTESTED.multitask_distillation.ppd import ProximalPolicyDistillation
from WRAPPED_train import AntDirectionTaskWrapper, CustomCombinedExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
import gym
import torch as th
from stable_baselines3.common.utils import set_random_seed



if __name__ == "__main__":


    def make_env(rank, seed=0):
        def _init():
            env = gym.make("Ant-v4")

            # TODO: choose an env
            env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[1,0]]) # Single fixed goal for all environments
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
    

    # TODO: select teacher
    teacher_model = PPO.load('/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/from_GPUs/ant_right/ant_right_model.zip', env=env)


    # same architecture as teacher
    policy_kwargs =dict(activation_fn=th.nn.Tanh, # Perhaps try ReLU
                         features_extractor_class=CustomCombinedExtractor,
                         features_extractor_kwargs=dict(state_embedding_mlp=[128, 128], task_embedding_mlp=[16, 32], activation=th.nn.Tanh),
                         net_arch=[dict(vf=[64, 64], pi=[64, 64])] )

    # TODO: fill in the model name
    model_name = "student_right_2k"

    student_model = ProximalPolicyDistillation( "MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=1024, batch_size=256, n_epochs=5, gamma=0.99, tensorboard_log= f"/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/local_trained/{model_name}/tensorboard_{model_name}/" )

    student_model.set_teachers( [teacher_model] , distill_lambda=1)

    student_model.learn(total_timesteps=200_000)

    student_model.save( f'/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/local_trained/{model_name}/model_{model_name}.zip' )
    env.save( f'/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/local_trained/{model_name}/env_{model_name}.pkl' )

    mean_reward, std_reward = evaluate_policy(student_model, student_model.get_env(), n_eval_episodes=10)
    print('Student reward: ', mean_reward, '+-', std_reward)



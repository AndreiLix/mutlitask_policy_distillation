from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
import gym
import torch as th
from stable_baselines3.common.utils import set_random_seed

from WRAPPED_train import AntDirectionTaskWrapper, CustomCombinedExtractor
# from code_from_giacomo.multitask_distillation_UNTESTED.multitask_distillation.ppd import ProximalPolicyDistillation
from code_from_giacomo.mt_pd_LatestCode.ppd import ProximalPolicyDistillation
#from CIRCLE_EnvWrapper import CircleTaskWrapper, CustomCombinedExtractor



if __name__ == "__main__":

    def make_env(rank, seed=0):
        def _init():
            env = gym.make("Ant-v4")


            # TODO: choose an env
            
            #env = CircleTaskWrapper(env)

            env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[1,0],[-1,0],[0,1],[0,-1]]) # Each environment samples a random cardinal direction
            #env = AntDirectionTaskWrapper(env, randomized_goal_directions=[[0,1]]) # Single fixed goal for all environments
            #env = AntDirectionTaskWrapper(env, randomized_goal_directions=None) # Each environment samples a random direction from all possible directions. We are not going to use this for a while.

            env.reset(seed = seed + rank)
            return env
        set_random_seed(seed)
        return _init


    # # these lines causes errors for circle
    # env = SubprocVecEnv([make_env(i) for i in range(4)])
    # env = VecMonitor(env)    

    # testing:
    env = gym.make("Ant-v4")
    env = AntDirectionTaskWrapper(env, [[1,0]])

    # # Use these for circle
    # env = gym.make("Ant-v4")
    # env = CircleTaskWrapper(env)


    

    # TODO: uncomment Normalization for desired env

    # # for Cartesian direcitons
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., norm_obs_keys=["state"])
    
    # for Circle
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., norm_obs_keys=["state"])
    # env = VecNormalize(env, norm_reward=True, clip_obs=10.)



    # # if PD, TODO: select teachers
    # teacher_model1 = PPO.load('/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/from_GPUs/ant_forward/ant_forward_model.zip', env=env)
    # teacher_model2 = PPO.load('/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/from_GPUs/ant_backward/ant_backward_model.zip', env=env)
    # teacher_model3 = PPO.load('/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/from_GPUs/ant_right/ant_right_model.zip', env=env)
    # teacher_model4 = PPO.load('/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/from_GPUs/ant_left/ant_left_model.zip', env=env)



    # TODO: uncomment desired architecture

    # # for trianing to walk along the CARDINAL axes

    policy_kwargs = dict(activation_fn=th.nn.Tanh, # Perhaps try ReLU
                         features_extractor_class=CustomCombinedExtractor,
                         features_extractor_kwargs=dict(state_embedding_mlp=[256, 128], task_embedding_mlp=[16, 32], activation=th.nn.Tanh),
                         net_arch=[dict(vf=[128], pi=[128])] )

    # for trianing to walk in a CIRCLE

    # policy_kwargs = dict(activation_fn=th.nn.Tanh, # Perhaps try ReLU
    #                      features_extractor_class=CustomCombinedExtractor,
    #                      net_arch=[dict(vf=[128], pi=[128])] )
    

    # TODO: fill in the model name
    model_name = "testing1"


    # TODO: uncomment the type of training desired

    # for REGULAR training
    # model = PPO("MultiInputPolicy", env ,policy_kwargs=policy_kwargs, verbose=1, n_steps=1024, batch_size=256, n_epochs=5, gamma=0.99, tensorboard_log= f"/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/local_trained/{model_name}/tensorboard_{model_name}/")
    model = PPO("MultiInputPolicy", env, verbose=1, n_steps=1024, batch_size=256, n_epochs=5, gamma=0.99, tensorboard_log= f"/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/local_trained/{model_name}/tensorboard_{model_name}/")

    model.learn(1000)

    # for PD
    # student_model = ProximalPolicyDistillation( "MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=1024, batch_size=256, n_epochs=5, gamma=0.99, tensorboard_log= f"/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/local_trained/{model_name}/tensorboard_{model_name}/" )
    # student_model.set_teacher( [teacher_model1, teacher_model2, teacher_model3, teacher_model4], distill_lambda=1)
    # student_model.learn(total_timesteps=6_000_000)

    model.save( f'/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/local_trained/{model_name}/model_{model_name}.zip' )
    #env.save( f'/home/andrei/Desktop/THESIS_multi_task_policy_distilation/WORKING_folder_thesis/checkpoints/local_trained/{model_name}/env_{model_name}.pkl' )

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print('Student reward: ', mean_reward, '+-', std_reward)



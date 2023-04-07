from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

from multitask_distillation.ppd import ProximalPolicyDistillation


if __name__ == "__main__":
    env = ...  # for multitask policy distillation, you should use the vecenv with ALL tasks (same as multitask learning case)
    
    teacher_model = PPO.load('teacher_model.ckpt', env=env)

    teacher_model = PPO.load('teacher_model.ckpt', env=env)
    teacher2_model = PPO.load('teacher2_model.ckpt', env=env)
    # ...

    student_model = ProximalPolicyDistillation("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs, n_steps=512, batch_size=256, n_epochs=5, learning_rate=2.5e-4, gamma=0.999, ent_coef=0.01, tensorboard_log="tensorboard/")

    # NOTE: the current code rescales the distillation loss by the number of teachers;  if you want to try to use the SUM of distillation losses for each teacher instead of their mean, just set distill_lambda=number_of_teachers
    student_model.set_teachers([teacher_model, teacher2_model], distill_lambda=1)

    student_model.learn(total_timesteps=2_000_000)

    student_model.save('student_model.ckpt')

    mean_reward, std_reward = evaluate_policy(student_model, student_model.get_env(), n_eval_episodes=10)
    print('Student reward: ', mean_reward, '+-', std_reward)




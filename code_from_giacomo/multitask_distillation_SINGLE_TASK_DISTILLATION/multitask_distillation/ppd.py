from typing import Callable
import numpy as np
from gym import spaces
import torch as th
from torch.nn import functional as F
from stable_baselines3.common import distributions
from stable_baselines3.common.utils import explained_variance
from stable_baselines3 import PPO

#from sb3_distill.core import PolicyDistillationAlgorithm


#class ProximalPolicyDistillation(PPO, PolicyDistillationAlgorithm):
class ProximalPolicyDistillation(PPO):

    """
    Proximal Policy Distillation (PPD) algorithm, based on the stable-baselines3 implementation of PPO.

    Paper: XXX

    Usage:
        model = ProximalPolicyDistillation(usual ppo arguments)
        model.set_teacher(teacher_model, distill_lambda=1.0)

        distill_lambda can be either a floating point or a function. If it is a function, it must take
        a `timestep' argument with the number of elapsed timesteps.
    """


    def set_teacher(self, teacher_model, distill_lambda=1.0):
        """
        Specify or replace teacher model to use for policy distillation.
        ProximalPolicyDistillation will create a separate policy for the student.

        :param teacher_model: SB3 [On/Off]PolicyAlgorithm object to use as teacher for distillation.
        :param distill_lambda: Coefficient of the distillation loss, to balance the student-rewards-based PPO loss.
        """
        self.teacher_model = teacher_model
        self.distill_lambda = distill_lambda

    # # commented by Andrei to debug
    # def save(self, path, exclude = None, include = None):
    #     super().save(path, exclude=exclude+["teacher_model"], include=include)


    def train(self):
        """
        The train() method of PPO is overridden to add the PPD loss. Please note that only a small part of the method has been modified. Parts that have been changed are marked as ### MODIFIED ####.
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True


        ### MODIFIED ###
        epoch_distillation_lambda = 0.0
        distillation_losses = []
        ################

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())


                ### MODIFIED ###
                # TODO: student policy is already run on the current observations, and its output is stored in log_prob; here we recompute the outputs for sake of code clarity, but please note that this runs an unnecessary extra call to the policy!

                lambda_ = self.distill_lambda
                if isinstance(lambda_, Callable):
                    lambda_ = lambda_(self.num_timesteps)
                    #print(self.num_timesteps, lambda_)
                epoch_distillation_lambda = lambda_

                if hasattr(self, 'teacher_model') and self.teacher_model is not None and lambda_>0.0:
                    #print("ERROR: must first call model.set_teacher(teacher_model)")
                    #return

                    teacher_act_distribution = self.teacher_model.policy.get_distribution(rollout_data.observations)
                    student_act_distribution = self.policy.get_distribution(rollout_data.observations)

                    # TODO: different papers do this differently;  in 'policy distillation' and DeepMind's football paper (when distilling individual policies), KL(teacher || student) is used; however, in that case the TEACHER policy is used to collect trajectories !
                    #       in DeepMind's football paper (learning final behavior, using mixture of skill-priors), they use KL(student || mixture-of-priors), and they use the STUDENT policy to collect trajectories

                    kl_divergence = distributions.kl_divergence(teacher_act_distribution, student_act_distribution) # trying to replicate the teacher; mean-seeking
                    #kl_divergence = distributions.kl_divergence(student_act_distribution, teacher_act_distribution) # trying to find the most probable action of the teacher; mode-seeking


                    # ## unclipped version:
                    # distillation_loss = th.mean(ratio * th.squeeze(kl_divergence))   # 'ratio' or 'th.clamp(ratio, 1 - clip_range, 1 + clip_range)'

                    # clipped version: note that both ratio (clipped or unclipped) and KL are always >=0; thus, contrary to the clip on the ratio*advantage, we do not have problems with changing signs.   max(r*kl, clip_r*kl) = max(r, clip(r, 1-e, 1+e))*kl = max(r, 1-e)*kl
                    #clipped_ratio = th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    clipped_ratio = th.clamp(ratio, 1-clip_range, None)

                    # for debugging
                    # print( "clipped_ratio: ", clipped_ratio, "shape: ", clipped_ratio.shape, "type: ", clipped_ratio.type)
                    # print( "kl_divergence: ", kl_divergence, "shape: ", kl_divergence.shape, "type:", kl_divergence.type)
                    # print( "squeezed kl_divergence", th.squeeze(kl_divergence), "shape", (th.squeeze(kl_divergence).shape))
                    
                    # # original line
                    # distillation_loss = th.mean(clipped_ratio * th.squeeze(kl_divergence))   # 'ratio' or ''
                    
                    # fix distillation_loss from Giacomo
                    if isinstance(teacher_act_distribution, distributions.DiagGaussianDistribution):
                        kl_divergence = distributions.sum_independent_dims(kl_divergence)

                    distillation_loss = th.mean(clipped_ratio * kl_divergence)   # 'ratio' or ''

                    ## remove
                    # clipped version: this is the most correct implementation, which prevents the policy from changing too much in between PPO iterations.
                    # Note that the clip term is now max instead of min, and without negative sign; that is because we are now minimizing a loss (maximizing -KL).
                    #distillation_loss_1 = th.squeeze(kl_divergence) * ratio
                    #distillation_loss_2 = th.squeeze(kl_divergence) * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    #distillation_loss = th.max(distillation_loss_1, distillation_loss_2).mean()
                    ## end remove

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + lambda_ * distillation_loss

                    distillation_losses.append(distillation_loss.item())
                else:
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                ################


                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        ### MODIFIED ###
        self.logger.record("train/distillation_lambda", epoch_distillation_lambda)
        self.logger.record("train/distillation_loss", np.mean(distillation_losses))
        ################
